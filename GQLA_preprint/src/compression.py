"""MLA -> GQLA conversion: per-group K/V PCA on neighbor head groups + weight absorption.

Per layer, per head group g (gs = H / G heads per group):

  1. Forward calibration hidden_states through the source kv_a chain
     (kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj). Split per-head
     kv_b_proj output into stacked K (gs * qk_nope) and V (gs * v_dim) vectors
     per group. Stream-accumulate the (G, F, F) Gram + first moment in fp64 —
     memory is O(G * F^2), independent of calibration size.

  2. Fit per-group PCA on K and V covariances (retained rank = qk_nope / v_dim);
     1% Tikhonov damping on the diagonal stabilises eigh under sparse calib.

  3. Build the compressed kv_b_proj (G * (qk_nope + v_dim) rows) by rotating
     the original K/V rows with U^g.T. Absorb the rotation into Q (nope rows)
     and O (per-head v_dim cols): rotations are square (d_k x d_k / d_v x d_v),
     so Q/O shapes don't change. Result is mathematically equivalent to the
     source MLA up to the rank-(d_k, d_v) PCA truncation per group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm


# ----------------------------- layout -----------------------------


@dataclass
class GqlaLayout:
    num_heads: int       # H
    num_kv_heads: int    # G
    group_size: int      # gs = H / G
    qk_nope: int         # d_k
    qk_rope: int
    v_dim: int           # d_v
    kv_lora: int

    @classmethod
    def from_config(cls, config, num_kv_heads: int) -> "GqlaLayout":
        H = config.num_attention_heads
        if H % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({H}) not divisible by num_kv_heads ({num_kv_heads})"
            )
        return cls(
            num_heads=H,
            num_kv_heads=num_kv_heads,
            group_size=H // num_kv_heads,
            qk_nope=config.qk_nope_head_dim,
            qk_rope=config.qk_rope_head_dim,
            v_dim=config.v_head_dim,
            kv_lora=config.kv_lora_rank,
        )


# ----------------------------- calibration data -----------------------------


def get_dataset(name: str):
    """Load wikitext2 / alpaca / pg19 as HF datasets, normalised to a single 'text' column."""
    import datasets
    if name == "wikitext2":
        return datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    if name == "alpaca":
        ds = datasets.load_dataset("tatsu-lab/alpaca")
        ds = ds.remove_columns(["input", "output", "instruction"])
        first = ds["train"].train_test_split(test_size=0.2, seed=42)
        second = first["test"].train_test_split(test_size=0.5, seed=42)
        return datasets.DatasetDict({
            "train": first["train"], "test": second["train"], "validation": second["test"],
        })
    if name == "pg19":
        ds = datasets.load_dataset("emozilla/pg19-test", split="test")
        ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
        return datasets.DatasetDict({"train": ds, "test": ds, "validation": ds})
    raise ValueError(f"unsupported dataset: {name!r}")


def prepare_calibration_inputs(
    tokenizer, dataset: str, nsamples: int, seqlen: int, seed: int = 42, split: str = "train",
) -> list[torch.Tensor]:
    """Pack ``nsamples`` chunks of exactly ``seqlen`` tokens from random rows."""
    ds = get_dataset(dataset)[split]
    col = ds.column_names[0]
    ds = ds.filter(lambda r: r[col] is not None and len(r[col]) > 0)
    gen = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    pbar = tqdm(total=nsamples, desc=f"Packing {dataset}", unit="sample")
    while len(out) < nsamples:
        text = ""
        for _ in range(10000):
            idx = int(torch.randint(0, len(ds), (1,), generator=gen).item())
            text = ds[idx][col] if not text else f"{text}\n\n{ds[idx][col]}"
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if ids.numel() >= seqlen:
                out.append(ids[:seqlen].contiguous())
                pbar.update(1)
                break
        else:
            raise RuntimeError(f"failed to pack {seqlen} tokens from {dataset}")
    pbar.close()
    return out


@torch.no_grad()
def gather_calibration_hidden_states(model, batches: list[torch.Tensor], device):
    """Capture each layer's attention input (CPU bf16) via forward pre-hooks. Independent mode."""
    n_layers = model.config.num_hidden_layers
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def make_hook(li):
        def _hook(_m, args, kwargs):
            x = kwargs.get("hidden_states") if kwargs else None
            if x is None and args:
                x = args[0]
            captured[li].append(x.detach().to("cpu"))
        return _hook

    hooks = [
        layer.self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        for batch in tqdm(batches, desc="Calibration forward", unit="batch"):
            model(input_ids=batch.unsqueeze(0).to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return captured


# ----------------------------- per-group cov + PCA -----------------------------


@torch.no_grad()
def collect_kv_grams(
    attn, hidden_list: list[torch.Tensor], layout: GqlaLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream the kv_a chain over calibration; accumulate per-group K/V covariance (fp64)."""
    G, gs = layout.num_kv_heads, layout.group_size
    per_head = layout.qk_nope + layout.v_dim
    f_k, f_v = gs * layout.qk_nope, gs * layout.v_dim
    dev = next(attn.parameters()).device

    sum_k = torch.zeros(G, f_k,      dtype=torch.float64, device=dev)
    sum_v = torch.zeros(G, f_v,      dtype=torch.float64, device=dev)
    H_k   = torch.zeros(G, f_k, f_k, dtype=torch.float64, device=dev)
    H_v   = torch.zeros(G, f_v, f_v, dtype=torch.float64, device=dev)
    n = 0
    for hidden in hidden_list:
        h = hidden.to(device=dev, dtype=next(attn.parameters()).dtype)
        ckv = attn.kv_a_proj_with_mqa(h)
        kv_x, _ = ckv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(kv_x))            # (B, S, H * per_head)
        flat = kv.reshape(-1, G, gs, per_head)                    # (N, G, gs, per_head)
        k = flat[..., :layout.qk_nope].reshape(-1, G, f_k).to(torch.float64)
        v = flat[..., layout.qk_nope:].reshape(-1, G, f_v).to(torch.float64)
        n += k.shape[0]
        sum_k += k.sum(dim=0)
        sum_v += v.sum(dim=0)
        H_k   += torch.einsum("ngf,ngd->gfd", k, k)
        H_v   += torch.einsum("ngf,ngd->gfd", v, v)
    denom = max(n - 1, 1)
    mean_k = sum_k / max(n, 1)
    mean_v = sum_v / max(n, 1)
    cov_k = (H_k - n * torch.einsum("gf,gd->gfd", mean_k, mean_k)) / denom
    cov_v = (H_v - n * torch.einsum("gf,gd->gfd", mean_v, mean_v)) / denom
    return cov_k, cov_v


def fit_per_group_pca(cov: torch.Tensor, retained_dim: int) -> torch.Tensor:
    """Per-group PCA via eigh with 1% Tikhonov diagonal damping. Returns (G, full_dim, retained_dim)."""
    src_device = cov.device
    H = cov.to(dtype=torch.float64).clone()
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    damp = 0.01 * diag.mean(dim=-1)
    eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
    H = H + damp.view(-1, 1, 1) * eye
    _, evecs = torch.linalg.eigh(H)                               # ascending
    top = evecs[..., -retained_dim:].flip(-1)                     # (G, d, r) desc
    return top.to(device=src_device, dtype=torch.float32).contiguous()


# ----------------------------- compress + absorb -----------------------------


@torch.no_grad()
def compress_and_absorb(
    attn, layout: GqlaLayout, u_k: torch.Tensor, u_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build compressed kv_b_proj + absorbed q_b_proj (nope rows) and o_proj (per-head v cols)."""
    H, G, gs = layout.num_heads, layout.num_kv_heads, layout.group_size
    qk_nope, qk_rope, v_dim, kv_lora = layout.qk_nope, layout.qk_rope, layout.v_dim, layout.kv_lora
    qk_head_dim = qk_nope + qk_rope
    per_head_kv = qk_nope + v_dim

    dev = attn.kv_b_proj.weight.device
    u_k = u_k.to(device=dev, dtype=torch.float32)
    u_v = u_v.to(device=dev, dtype=torch.float32)

    # New kv_b_proj: rotate per-group K/V rows of the original by U.T.
    w_kv = attn.kv_b_proj.weight.data.to(torch.float32)           # (H * per_head_kv, kv_lora)
    w_grp = w_kv.view(G, gs, per_head_kv, kv_lora)
    w_k = w_grp[:, :, :qk_nope, :].reshape(G, gs * qk_nope, kv_lora)
    w_v = w_grp[:, :, qk_nope:, :].reshape(G, gs * v_dim, kv_lora)
    new_k = torch.einsum("gfr,gfk->grk", u_k, w_k)                # (G, qk_nope, kv_lora)
    new_v = torch.einsum("gfr,gfk->grk", u_v, w_v)                # (G, v_dim, kv_lora)
    new_kv_b = torch.cat([new_k, new_v], dim=1).reshape(G * per_head_kv, kv_lora).contiguous()

    # Absorb K rotation into q_b_proj nope rows: per head h, new_q[nope] = U_h.T @ old_q[nope].
    w_q = attn.q_b_proj.weight.data.to(torch.float32)             # (H * qk_head_dim, q_lora)
    new_q = w_q.clone()
    u_k_per_head = u_k.view(G, gs, qk_nope, qk_nope)              # (G, gs, d_k, d_k)
    for h in range(H):
        g, hl = h // gs, h % gs
        s = h * qk_head_dim
        new_q[s:s + qk_nope, :] = u_k_per_head[g, hl].T @ w_q[s:s + qk_nope, :]

    # Absorb V rotation into o_proj cols: per head h, new_o[:, h] = old_o[:, h] @ U_h.
    w_o = attn.o_proj.weight.data.to(torch.float32)               # (hidden, H * v_dim)
    new_o = w_o.clone()
    u_v_per_head = u_v.view(G, gs, v_dim, v_dim)
    for h in range(H):
        g, hl = h // gs, h % gs
        s = h * v_dim
        new_o[:, s:s + v_dim] = w_o[:, s:s + v_dim] @ u_v_per_head[g, hl]

    return {
        "kv_b_proj.weight": new_kv_b,
        "q_b_proj.weight":  new_q.contiguous(),
        "o_proj.weight":    new_o.contiguous(),
    }


@torch.no_grad()
def inplace_apply_compressed(attn, compressed: dict[str, torch.Tensor], layout: GqlaLayout) -> None:
    """Replace kv_b_proj with the shrunken Linear; copy absorbed q/o weights; flip kv_groups."""
    dev = attn.kv_b_proj.weight.device
    dt = attn.kv_b_proj.weight.dtype
    new_kv_b = nn.Linear(
        layout.kv_lora,
        layout.num_kv_heads * (layout.qk_nope + layout.v_dim),
        bias=False,
    ).to(device=dev, dtype=dt)
    new_kv_b.weight.data.copy_(compressed["kv_b_proj.weight"].to(device=dev, dtype=dt))
    attn.kv_b_proj = new_kv_b

    for name in ("q_b_proj", "o_proj"):
        mod = getattr(attn, name)
        mod.weight.data.copy_(
            compressed[f"{name}.weight"].to(device=mod.weight.device, dtype=mod.weight.dtype)
        )
    attn.num_key_value_groups = layout.group_size


# ----------------------------- PPL eval (optional) -----------------------------


def prepare_ppl_dataloader(
    tokenizer, dataset: str, seqlen: int, batch_size: int,
    split: str = "test", max_chunks: int | None = None,
):
    """Fixed-length PPL loader (lm-eval-harness wikitext protocol: concat-and-chunk)."""
    ds = get_dataset(dataset)[split]
    col = ds.column_names[0]
    full = "\n\n".join(r[col] for r in ds if r[col])
    ids = tokenizer(full, return_tensors="pt", truncation=False).input_ids[0]
    n_chunks = ids.numel() // seqlen
    if n_chunks == 0:
        raise ValueError(f"{dataset}/{split} too short for seqlen={seqlen}")
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)
    chunks = ids[: n_chunks * seqlen].view(n_chunks, seqlen).contiguous()

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return chunks.shape[0]
        def __getitem__(self, i):
            return {"input_ids": chunks[i],
                    "attention_mask": torch.ones(seqlen, dtype=torch.long)}
    return torch.utils.data.DataLoader(_DS(), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_ppl(model, dataloader, loss_chunk: int = 2048) -> float:
    """Mean per-sequence next-token NLL, exp'd; logits chunked along seq axis for memory."""
    model.eval()
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    nll_means = []
    for batch in tqdm(dataloader, desc="PPL", unit="batch"):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        shift_logits = logits[:, :-1, :]
        shift_targets = ids[:, 1:]
        S = shift_logits.shape[1]
        parts = []
        for s in range(0, S, loss_chunk):
            e = min(s + loss_chunk, S)
            parts.append(loss_fn(
                shift_logits[:, s:e, :].float().permute(0, 2, 1),
                shift_targets[:, s:e],
            ))
        nll = torch.cat(parts, dim=1)
        del logits, shift_logits
        m = mask[:, 1:].float()
        nll_means.append(((nll * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)).cpu())
    return torch.exp(torch.cat(nll_means).mean()).item()


__all__ = [
    "GqlaLayout",
    "get_dataset",
    "prepare_calibration_inputs",
    "gather_calibration_hidden_states",
    "collect_kv_grams",
    "fit_per_group_pca",
    "compress_and_absorb",
    "inplace_apply_compressed",
    "prepare_ppl_dataloader",
    "evaluate_ppl",
]
