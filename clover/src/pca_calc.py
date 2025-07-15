import torch
import gc
import inspect
import logging
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )

def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)
        
def map_tensors(obj, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj
    
@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module, pad_token_id: int | None, testloader: DataLoader[dict[str, torch.Tensor]]
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in tqdm(testloader):
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, model.model.embed_tokens.weight.device)
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()

def insert_hooks(model):
    input_layernorm_hooks = []
    post_attention_layernorm_hooks = []
    input_layernorm_outputs = {}
    post_attention_layernorm_outputs = {}
    embed_tokens_outputs = []

    def embed_tokens_hook_fn(module, input, output):
        embed_tokens_outputs.append(output.to('cpu'))

    def input_layernorm_hook_fn(module, input, output, index):
        if index not in input_layernorm_outputs:
            input_layernorm_outputs[index] = []
        input_layernorm_outputs[index].append(output.to('cpu'))

    def post_attention_layernorm_hook_fn(module, input, output, index):
        if index not in post_attention_layernorm_outputs:
            post_attention_layernorm_outputs[index] = []
        post_attention_layernorm_outputs[index].append(output.to('cpu'))
        
    embed_tokens_hook = model.model.embed_tokens.register_forward_hook(lambda module, input, output: embed_tokens_hook_fn(module, input, output))
    for idx, layer in enumerate(model.model.layers):
        input_layernorm_hook = layer.input_layernorm.register_forward_hook(lambda module, input, output, idx=idx: input_layernorm_hook_fn(module, input, output, idx))
        input_layernorm_hooks.append(input_layernorm_hook)
        post_attention_layernorm_hook = layer.post_attention_layernorm.register_forward_hook(lambda module, input, output, idx=idx: post_attention_layernorm_hook_fn(module, input, output, idx))
        post_attention_layernorm_hooks.append(post_attention_layernorm_hook)
    
    return embed_tokens_hook, input_layernorm_hooks, post_attention_layernorm_hooks, embed_tokens_outputs, input_layernorm_outputs, post_attention_layernorm_outputs

@torch.no_grad()
def get_calibrate_outputs(
    model: torch.nn.Module, trainloader: DataLoader[dict[str, torch.Tensor]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    """

    start_time = time.time()

    model.eval()
    embed_tokens_hook, input_layernorm_hooks, post_attention_layernorm_hooks, embed_tokens_outputs, input_layernorm_outputs, post_attention_layernorm_outputs = insert_hooks(model)
    ignore_masks = []
    logging.info("Training perplexity...")
    for batch in tqdm(trainloader):
        batch = map_tensors(batch, model.model.embed_tokens.weight.device)
        ignore_masks.append(batch["attention_mask"].to('cpu'))
        model(**batch)

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    embed_tokens_hook.remove()
    for hook in input_layernorm_hooks:
        hook.remove()
    for hook in post_attention_layernorm_hooks:
        hook.remove()

    for idx, X_batch in enumerate(embed_tokens_outputs):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

    for value in input_layernorm_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0

    for value in post_attention_layernorm_outputs.values():
        for idx, X_batch in enumerate(value):
            if ignore_masks:
                X_batch[ignore_masks[idx] == 0] = 0


    outputs = {
        "embed_tokens": embed_tokens_outputs,
        "input_layernorm": input_layernorm_outputs,
        "post_attention_layernorm": post_attention_layernorm_outputs,
    }
    return outputs

@torch.no_grad()
def layer_pca_calc(
    X: list[torch.Tensor], device: str | torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        X_batch = X_batch.double().to(device=device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eigen_vec = X_eig[1][:, index]
    return eigen_vec

def model_pca_calc(model, outputs, device):
    emb_Q = layer_pca_calc(outputs['embed_tokens'], device)
    attn_Q = []
    mlp_Q = []
    for idx in range(len(outputs['input_layernorm'].keys())):
        attn_Q.append(layer_pca_calc(outputs['input_layernorm'][idx], device))
        mlp_Q.append(layer_pca_calc(outputs['post_attention_layernorm'][idx], device))
    return emb_Q, attn_Q, mlp_Q