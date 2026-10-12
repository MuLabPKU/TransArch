"""Track shared candidate pools across model layers and query slices.

The pool opens on the first eligible prefill and closes at model-forward
exit. Exceptions clear the pool before leaving the forward scope.
"""
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import os

from ..api import ForwardLedger

_CURRENT = ContextVar('lisa_model_forward', default=None)
COUNTERS = Counter()


def current_scope(framework):
    scope = _CURRENT.get()
    if scope is None or scope.framework != framework:
        raise RuntimeError('shared4 requires the model-forward patch as well as the indexer patch')
    return scope


class ForwardScope:
    def __init__(self, framework, model, *, unsupported=None):
        self.framework = framework
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)
        self.unsupported = unsupported
        self.entries = {}

    @staticmethod
    def _step(context, rows):
        return (context.query_start, rows), (context.request_id, context.sequence_length, rows)

    def before(self, adapter, context, rows):
        if self.unsupported:
            raise RuntimeError(self.unsupported)
        if adapter.indexer.num_layers != self.num_layers:
            raise ValueError('adapter layer count does not match the model config')
        chunk, layout = self._step(context, rows)
        entry = self.entries.get(adapter)
        if entry is None:
            ledger = ForwardLedger(self.num_layers)
            ledger.begin()
            # Validate the first layer before allocating/copying any tensors.
            ledger.check(context.layer_id, chunk, layout)
            adapter.begin_forward(context.request_id)
            self.entries[adapter] = entry = (ledger, context.request_id)
        ledger, request = entry
        if context.request_id != request:
            raise RuntimeError('request changed inside one shared4 model forward')
        ledger.check(context.layer_id, chunk, layout)

    def after(self, adapter, context, rows):
        chunk, layout = self._step(context, rows)
        self.entries[adapter][0].commit(context.layer_id, chunk, layout)

    def finish(self):
        # Validate every device/adapter before closing any core generation.
        for ledger, _ in self.entries.values():
            ledger.finish()
        for adapter in self.entries:
            adapter.end_forward()
        if self.entries:
            COUNTERS[self.framework + ':shared_prefill_forwards_completed'] += 1
        else:
            COUNTERS[self.framework + ':forwards_without_shared_prefill'] += 1

    def abort(self, error):
        for adapter, (ledger, _) in self.entries.items():
            ledger.abort()
            try:
                adapter.abort_forward()
            except Exception as cleanup_error:
                # A CUDA assertion may also break synchronization during cleanup;
                # preserve the original exception and mark the worker unusable.
                COUNTERS[self.framework + ':cleanup_failures'] += 1
                if hasattr(error, 'add_note'):
                    error.add_note(f'LISA cleanup also failed: {cleanup_error}')
        COUNTERS[self.framework + ':model_forwards_aborted'] += 1


def model_forward(framework):
    if framework not in ('vllm', 'sglang'):
        raise ValueError('unknown framework')

    def decorate(forward):
        @wraps(forward)
        def wrapped(model, *args, **kwargs):
            if os.environ.get('LISA_INDEXER') != 'shared4':
                return forward(model, *args, **kwargs)
            if _CURRENT.get() is not None:
                raise RuntimeError('nested shared4 model forwards are unsupported')
            num_layers = int(model.config.num_hidden_layers)
            unsupported = None
            if int(model.start_layer) != 0 or int(model.end_layer) != num_layers:
                unsupported = 'shared4_pipeline_parallel'
            if framework == 'sglang':
                batch = kwargs.get('forward_batch', args[2] if len(args) > 2 else None)
                if batch is not None and getattr(batch, 'can_run_tbo', False):
                    unsupported = 'shared4_two_batch_overlap'
            scope = ForwardScope(framework, model, unsupported=unsupported)
            token = _CURRENT.set(scope)
            COUNTERS[framework + ':model_forwards_entered'] += 1
            try:
                result = forward(model, *args, **kwargs)
                scope.finish()
                return result
            except BaseException as error:
                scope.abort(error)
                raise
            finally:
                _CURRENT.reset(token)
        return wrapped
    return decorate
