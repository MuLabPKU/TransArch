"""trainer.py — DSATrainer: Hugging Face Trainer with DSA warmup/sparse losses.

compute_loss behavior by ``dsa_stage``:
  * "warmup": backbone frozen. We run the DECODER STACK ONLY (model.model, no LM
    head — avoids the [B,L,vocab] logits that would blow up at long context). Each
    layer's attention computes its indexer KL (stashed in the module REGISTRY).
    Loss = mean over layers of that KL. Only indexer params require grad, so
    autograd builds no backbone gradient; FlashKL keeps the teacher in no_grad, so
    memory is O(L·H) per layer. This trains the indexer to imitate the model's own
    attention distribution.
  * "sparse": backbone trainable. Same decoder-stack-only trick as warmup, then a
    CHUNKED lm_head + CE loss (each chunk under torch.utils.checkpoint) so the
    [B, L_local, vocab] fp32 logits — 10 GB/GPU at 200k with CP=8 — is only
    materialized per-chunk and released before the next. Loss = LM_loss + λ · KL.

cu_seqlens (per-doc boundaries from the packing collator) are pushed onto the
attention modules so FlashKL uses the correct causal supports.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Some CUDA images ship an incomplete ``apex`` package without ``apex.amp``.
# Transformers detects the package by name and then imports ``amp`` in Trainer,
# so mark Apex unavailable when that optional path is broken.
try:
    from apex import amp as _apex_amp  # noqa: F401
except Exception:
    try:
        import transformers.utils.import_utils as _tf_import_utils
        _tf_import_utils._apex_available = False
    except Exception:
        pass

from transformers import Trainer

from ..modeling import (dsa_loss_registry, set_cu_seqlens, set_dsa_mode,
                        set_log_recall, set_log_kl, indexer_parameters,
                        expert_parameters, nonexpert_parameters)


def _decoder_stack(model):
    """Return the base decoder module (runs layers, no LM head), unwrapping any
    DDP/FSDP wrapper first."""
    node = model
    for _ in range(6):
        if hasattr(node, "layers"):
            return node
        nxt = getattr(node, "model", None) or getattr(node, "module", None)
        if nxt is None:
            break
        node = nxt
    raise AttributeError("cannot find decoder stack (model.model)")


def _lm_head(model):
    """Return the LM head module, unwrapping DDP/FSDP wrappers."""
    node = model
    for _ in range(4):
        head = getattr(node, "lm_head", None)
        if head is not None:
            return head
        nxt = getattr(node, "module", None)
        if nxt is None:
            break
        node = nxt
    raise AttributeError("cannot find lm_head")


def _chunked_shift_ce(hidden, labels, lm_head_module, vocab_size, chunk_size, shift=True):
    """Chunked shift-CE that avoids materializing the [B, L_local-1, vocab] fp32
    logits tensor (10 GB at 200k/CP=8). Each chunk runs lm_head + CE under
    torch.utils.checkpoint so logits are freed after the forward chunk and
    recomputed during backward — trading a small recomputation for peak memory
    O(chunk_size * vocab) instead of O(L_local * vocab).

    ``shift=True`` (default, non-CP): ``labels`` is aligned with ``hidden`` and the
    next-token target of position i is ``labels[i+1]`` (standard causal shift).
    ``shift=False`` (zigzag CP): ``labels`` is ALREADY the per-position target
    (``labels[i]`` is the target for ``hidden[i]``, with -100 on invalid slots), so
    no local shift is applied — required because zigzag-local tokens are not globally
    contiguous, so a local shift would cross the chunk seam. The caller builds those
    targets from the pre-shard full labels via global positions (exact, no comm).

    Loss reduction matches HF's default CrossEntropyLoss (mean over valid tokens),
    computed as sum-over-chunks / count-of-valid.
    """
    B, L, H = hidden.shape
    Ls = (L - 1) if shift else L
    if Ls <= 0:
        return hidden.new_zeros(())

    def _ce_chunk(h_c, y_c):
        # h_c: [B, chunk, H]; y_c: [B, chunk]. Cast logits to fp32 for CE stability,
        # matching HF DS-V2's ``logits = self.lm_head(hidden_states).float()``.
        logits = lm_head_module(h_c).float()
        return F.cross_entropy(logits.reshape(-1, vocab_size), y_c.reshape(-1),
                               ignore_index=-100, reduction="sum")

    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    n_valid = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, Ls, chunk_size):
        end = min(start + chunk_size, Ls)
        h_c = hidden[:, start:end, :]
        y_c = labels[:, start + 1:end + 1] if shift else labels[:, start:end]
        n_valid = n_valid + (y_c != -100).sum().float()
        loss_sum = loss_sum + checkpoint(_ce_chunk, h_c, y_c, use_reentrant=False)
    return loss_sum / n_valid.clamp_min(1.0)


def _cp_shifted_targets(full_labels, gpos):
    """Per-local-position next-token targets for the zigzag-CP LM loss.

    ``full_labels`` [B, L] is the pre-shard full-sequence label tensor (every CP rank
    holds it); ``gpos`` [Lloc] gives each local token's global position. The target of
    local token i is the full label at global position ``gpos[i]+1`` (next token), or
    -100 when that would run past the sequence end. This reproduces the exact global
    causal shift for tokens that are not globally contiguous under zigzag sharding.
    """
    L = full_labels.shape[1]
    nxt = (gpos + 1).clamp(max=L - 1)
    tgt = full_labels[:, nxt]                       # [B, Lloc]
    tgt = torch.where((gpos + 1 < L).unsqueeze(0), tgt, torch.full_like(tgt, -100))
    return tgt


class DSATrainer(Trainer):
    def __init__(self, *args, dsa_stage: str = "warmup", indexer_loss_coeff: float = 1.0,
                 log_recall_every: int = 0, log_kl_every: int = 0, cp_size: int = 1,
                 zero2: bool = False, lm_chunk_size: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.dsa_stage = dsa_stage
        # NOTE: indexer_loss_coeff applies to the SPARSE stage only, where the total
        # loss is lm_loss + coeff * indexer_KL. In WARMUP the backbone is frozen and
        # the loss is the pure indexer KL (no coeff, no lm_loss), so this is ignored.
        self.indexer_loss_coeff = indexer_loss_coeff
        self.log_recall_every = log_recall_every
        self.log_kl_every = log_kl_every
        self._step = 0
        self._cp = cp_size
        self._zero2 = zero2 and dsa_stage == "sparse"
        # LM head + CE chunk size in tokens (per shift-position). Applied only in
        # sparse compute_loss to bound peak [chunk, vocab] fp32 logits memory.
        self._lm_chunk_size = lm_chunk_size
        set_dsa_mode(self.model, dsa_stage)
        if cp_size > 1:
            from ..modeling import set_cp_size
            set_cp_size(self.model, cp_size)
        if self._zero2:
            # Grad clipping needs a distributed-aware norm over the SHARDED grad;
            # v1 disables HF's clip rather than doing it wrong (see zero2.py header).
            if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                import warnings
                warnings.warn(f"Zero2Adam v1 does not support grad clipping; forcing "
                              f"max_grad_norm from {self.args.max_grad_norm} to 0.")
                self.args.max_grad_norm = 0.0

    def _indexer_metrics_for_log(self, indexer_loss, entropy, device):
        """Return (indexer_loss, true_kl) using the same cross-rank reduction as the
        indexer loss itself.

        CP: each rank stores a contribution normalized by the global query count, so
        metrics must be summed across the CP group. DDP: each rank has a data shard
        mean, so average for a stable global log value.
        """
        if indexer_loss is None:
            return None, None
        loss_f = float(indexer_loss.detach() if torch.is_tensor(indexer_loss) else indexer_loss)
        ent_f = 0.0 if entropy is None else float(entropy)
        vals = torch.tensor([loss_f, ent_f], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from ..dist import cp_group, cp_size
            if cp_size() > 1:
                dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=cp_group())
            else:
                dist.all_reduce(vals, op=dist.ReduceOp.SUM)
                vals /= dist.get_world_size()
        loss_f, ent_f = [float(x) for x in vals.cpu().tolist()]
        true_kl = None if entropy is None else loss_f - ent_f
        return loss_f, true_kl

    def _dsa_num_layers(self, model=None) -> int:
        node = model if model is not None else self.model
        for _ in range(8):
            n = getattr(node, "_dsa_num_layers", None)
            if n:
                return int(n)
            nxt = getattr(node, "module", None) or getattr(node, "model", None)
            if nxt is None:
                break
            node = nxt
        return 1

    def create_optimizer(self):
        """Sparse+ZeRO-2 path: build a Zero2Adam over (nonexpert=sharded, expert=local).
        Everything else falls through to HF's default single-AdamW builder."""
        if self.optimizer is None and self._zero2:
            from ..dist import cp_group
            from .zero2 import Zero2Adam
            sharded = nonexpert_parameters(self.model)
            local = expert_parameters(self.model)
            self.optimizer = Zero2Adam(
                sharded_params=sharded,
                local_params=local,
                group=cp_group(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def _get_train_sampler(self, *a, **k):
        """Under CP every rank must see the SAME sequence (then shard it locally), so
        we use a plain sampler with a FIXED seed shared across ranks — all ranks draw
        the identical example order; the DP sharding HF would normally do is replaced
        by CP sequence-sharding in compute_loss."""
        if self._cp > 1:
            from torch.utils.data import RandomSampler
            g = torch.Generator()
            g.manual_seed(int(self.args.seed))
            return RandomSampler(self.train_dataset, generator=g)
        return super()._get_train_sampler(*a, **k)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        reg = dsa_loss_registry()
        reg.reset()

        from ..dist import cp_size, local_shard
        cp = cp_size()

        cu_list = inputs.get("cu_seqlens_list", None)
        if cu_list is not None:
            # cu_seqlens are FULL-sequence doc boundaries; under CP the attention CP
            # branch consumes them as-is (global), so push row 0's cu to all modules.
            set_cu_seqlens(model, cu_list[0])

        want_recall = (self.log_recall_every > 0 and self._step % self.log_recall_every == 0)
        want_kl = (
            want_recall or
            (self.log_kl_every > 0 and self._step % self.log_kl_every == 0)
        )
        # True-KL logging requires a teacher entropy pass. It is computed under
        # no_grad only on requested diagnostic steps.
        set_log_recall(model, want_recall)
        set_log_kl(model, want_kl)
        self._step += 1

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        # CP: every rank receives the SAME full sequence; shard it along the seq dim
        # so each rank runs its L/cp local tokens (the attention CP branch gathers
        # keys). Sharding is ZIGZAG (load-balanced), so local tokens are NOT globally
        # contiguous — keep the full labels + this rank's global positions to build
        # exact next-token targets for the sparse LM loss below (a local shift would
        # cross the zigzag seam).
        cp_targets = None
        if cp > 1:
            from ..dist import zigzag_local_gpos
            full_labels = labels
            gpos_cp = zigzag_local_gpos(input_ids.shape[1], input_ids.device)
            input_ids = local_shard(input_ids, dim=1)
            if attention_mask is not None:
                attention_mask = local_shard(attention_mask, dim=1)
            if labels is not None:
                labels = local_shard(labels, dim=1)
                cp_targets = _cp_shifted_targets(full_labels, gpos_cp)

        if self.dsa_stage == "warmup":
            # Warmup loss is collected as a side-effect in the module REGISTRY, not
            # from the model's returned output — so DDP's autograd reducer cannot
            # track the indexer params correctly (it double-marks them). We instead
            # run the unwrapped decoder stack (also skips the [B,L,vocab] LM head)
            # and manually all-reduce the tiny indexer grads in training_step. Only
            # indexer params require grad, so no backbone gradient is built and
            # FlashKL keeps the teacher in no_grad -> O(L·H) memory.
            #
            # In eager-backward mode (set by training_step) each layer's indexer loss
            # is backwarded the moment it's produced (inside REGISTRY.add), freeing
            # that layer's graph before the next layer runs — peak activation drops
            # from ~num_layers·O(L·H) to ~1·O(L·H). We then return a DETACHED scalar
            # and training_step must NOT backward again.
            dec = _decoder_stack(model)
            _ = dec(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False)
            rec = reg.mean_recall()
            if rec is not None:
                self.log({"indexer_recall": rec})
            # true KL = trainable loss - teacher entropy (both summed over layers).
            # The logged indexer_loss is a cross-entropy (KL + teacher entropy); the
            # entropy is an incompressible constant, so this is the number that
            # actually reflects indexer quality (should trend to ~0). Logging-only.
            ent = reg.entropy_sum()
            if reg._eager:
                loss_val = reg.eager_loss_sum()
                loss_log, true_kl = self._indexer_metrics_for_log(loss_val, ent, input_ids.device)
                n_layers = self._dsa_num_layers(model)
                log_d = {
                    "indexer_ce": loss_log,
                    "indexer_ce_per_layer": loss_log / n_layers,
                }
                if true_kl is not None:
                    log_d["indexer_true_kl"] = true_kl
                    log_d["indexer_true_kl_per_layer"] = true_kl / n_layers
                self.log(log_d)
                loss = torch.tensor(loss_val, device=input_ids.device)
                return (loss, None) if return_outputs else loss
            loss = reg.total()
            if loss is not None:
                loss_log, true_kl = self._indexer_metrics_for_log(loss, ent, input_ids.device)
                n_layers = self._dsa_num_layers(model)
                log_d = {
                    "indexer_ce": loss_log,
                    "indexer_ce_per_layer": loss_log / n_layers,
                }
                if true_kl is not None:
                    log_d["indexer_true_kl"] = true_kl
                    log_d["indexer_true_kl_per_layer"] = true_kl / n_layers
                self.log(log_d)
            return (loss, None) if return_outputs else loss

        # sparse stage: run decoder stack ONLY (skip HF's built-in LM head +
        # CE, which materializes a [B, L_local, vocab] fp32 logits tensor —
        # 10 GB/GPU at 200k with CP=8). We then do chunked lm_head + CE with
        # grad-ckpt inside ``_chunked_shift_ce``, keeping peak logits memory to
        # O(chunk_size * vocab). Under CP, the LM loss is a local-token mean and must
        # be divided by cp before CP grad summation. The indexer KL is already
        # normalized by the global query count inside sparse_kl_chunked(nrow_global),
        # so it must NOT be divided by cp again.
        dec = _decoder_stack(model)
        head = _lm_head(model)
        dec_out = dec(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False)
        hidden = dec_out[0] if isinstance(dec_out, tuple) else dec_out.last_hidden_state
        vocab = self.model.config.vocab_size
        if cp_targets is not None:
            # zigzag CP: targets are pre-aligned to global next-token, no local shift
            lm_loss = _chunked_shift_ce(hidden, cp_targets, head, vocab,
                                        self._lm_chunk_size, shift=False)
        else:
            lm_loss = _chunked_shift_ce(hidden, labels, head, vocab, self._lm_chunk_size)
        idx_loss = reg.total()
        idx_term = 0.0 if idx_loss is None else self.indexer_loss_coeff * idx_loss
        if cp > 1:
            loss = lm_loss / cp + idx_term
        else:
            loss = lm_loss + idx_term
        if idx_loss is not None:
            idx_log, true_kl = self._indexer_metrics_for_log(idx_loss, reg.entropy_sum(),
                                                             input_ids.device)
            n_layers = self._dsa_num_layers(model)
            lm_loss_f = float(lm_loss.detach())
            lm_loss_scaled_f = lm_loss_f / cp if cp > 1 else lm_loss_f
            log_d = {
                "total_loss": float(loss.detach()),
                "lm_loss": lm_loss_f,
                "lm_loss_cp_scaled": lm_loss_scaled_f,
                "indexer_ce": idx_log,
                "indexer_ce_per_layer": idx_log / n_layers,
            }
            # true KL = indexer_loss - teacher entropy (both per-layer sums). The
            # logged indexer_ce is dominated by the incompressible teacher entropy
            # (~7/layer); true KL is the real indexer-quality signal. It is present
            # only when entropy diagnostics are enabled for this step.
            if true_kl is not None:
                log_d["indexer_true_kl"] = true_kl
                log_d["indexer_true_kl_per_layer"] = true_kl / n_layers
            self.log(log_d)
        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Warmup uses PER-LAYER eager backward (bounded memory) + manual indexer
        grad all-reduce, bypassing DDP; other stages use the standard HF path.

        Warmup: enable eager backward on the REGISTRY (scale = 1/grad_accum to match
        HF's loss scaling), run compute_loss which triggers the per-layer backwards
        as a side-effect, then all-reduce the indexer grads once. We do NOT call the
        HF backward (the graph is already consumed) — just return the detached loss.
        """
        if self.dsa_stage != "warmup":
            # sparse CP+EP: we hand-reduce grads (non-expert over CP, experts local),
            # so DDP's own gradient all-reduce must be suppressed. Run the HF backward
            # inside no_sync() when the model is DDP-wrapped, then reduce ourselves.
            if self._cp > 1 and hasattr(model, "no_sync"):
                import contextlib
                with model.no_sync():
                    loss = super().training_step(model, inputs, num_items_in_batch)
            else:
                loss = super().training_step(model, inputs, num_items_in_batch)
            if self._cp > 1:
                self._reduce_sparse_grads(model)
            return loss

        model.train()
        inputs = self._prepare_inputs(inputs)
        reg = dsa_loss_registry()
        scale = 1.0 / self.args.gradient_accumulation_steps
        reg.set_eager_backward(True, scale=scale)
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        finally:
            reg.set_eager_backward(False)

        self._allreduce_indexer_grads(model)
        return loss.detach()

    def _allreduce_indexer_grads(self, model):
        """Reduce indexer grads across ranks (warmup bypasses DDP's reducer).

        CP mode (cp_size>1): each rank's FlashKL loss is already normalized by the
        GLOBAL query count (nrow_global), so per-shard grads must be **summed** to get
        the true global gradient — reduce over the CP group with op=SUM, no /world.
        DDP mode (cp_size==1, data-parallel ranks): each rank sees different data, so
        **average** (SUM / world)."""
        from ..dist import cp_size, cp_group, is_dist
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        params = [p for p in indexer_parameters(model) if p.grad is not None]
        if cp_size() > 1:
            grp = cp_group()
            for p in params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=grp)
        else:
            world = dist.get_world_size()
            for p in params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

    def _reduce_sparse_grads(self, model):
        """Sparse CP+EP grad reduction (no FSDP/DDP). Non-expert (replicated) params
        get all-reduce(SUM) over the CP group — each rank contributed its sequence
        shard's grad (loss pre-scaled by 1/cp), so the sum is the global-mean grad.
        Expert params are unique per rank (grads already complete via the MoE
        all-to-all) and are left untouched.

        NO-OP when ZeRO-2 is on: the non-expert grad reduction is fused with the
        opt-state shard step inside ``Zero2Adam.step()`` (a single reduce_scatter
        replaces this all_reduce). Expert grads still need no reduction."""
        if self._zero2:
            return
        from ..dist import cp_group, is_dist
        from ..modeling import expert_parameters
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        expert_ids = set(id(p) for p in expert_parameters(model))
        grp = cp_group()
        for p in model.parameters():
            if p.grad is not None and id(p) not in expert_ids:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=grp)
