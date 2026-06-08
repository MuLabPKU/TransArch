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
import torch.nn.functional as F
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
def gather_calibration_hidden_states(
    model, batches: list[torch.Tensor], device, capture_final: bool = False,
):
    """Capture each layer's attention input (CPU bf16) via forward pre-hooks. Independent mode.

    If ``capture_final=True``, also captures the OUTPUT of the last decoder layer
    (= input to ``model.model.norm``) — used by hessian-PCA's NLL token weights.
    In that case returns ``(captured, final_hiddens)`` instead of ``captured``.
    """
    n_layers = model.config.num_hidden_layers
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    final_hiddens: list[torch.Tensor] | None = [] if capture_final else None

    def make_hook(li):
        def _hook(_m, args, kwargs):
            x = kwargs.get("hidden_states") if kwargs else None
            if x is None and args:
                x = args[0]
            captured[li].append(x.detach().to("cpu"))
        return _hook

    def _final_hook(_m, args, kwargs):
        x = kwargs.get("hidden_states") if kwargs else None
        if x is None and args:
            x = args[0]
        final_hiddens.append(x.detach().to("cpu"))

    hooks = [
        layer.self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(model.model.layers)
    ]
    if capture_final:
        hooks.append(
            model.model.norm.register_forward_pre_hook(_final_hook, with_kwargs=True)
        )
    try:
        for batch in tqdm(batches, desc="Calibration forward", unit="batch"):
            model(input_ids=batch.unsqueeze(0).to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return (captured, final_hiddens) if capture_final else captured


# ----------------------------- per-group cov + PCA -----------------------------


@torch.no_grad()
def collect_kv_grams(
    attn, hidden_list: list[torch.Tensor], layout: GqlaLayout,
    token_weights: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream the kv_a chain over calibration; accumulate per-group K/V covariance (fp64).

    If ``token_weights`` is provided (one ``(S,)`` fp32 tensor per sample in
    ``hidden_list``), cov becomes ``sum_t w_t * x_t @ x_t.T`` (= ``X.T @ diag(w) @ X``)
    — the Hessian-weighted PCA cov from SparseGPT / OBS. Mean and denominator use
    effective sample size ``sum(w)`` to keep cov scale comparable to the
    uniform-weight legacy path (mean ``w`` ≈ 1 after per-sample normalisation
    upstream).
    """
    G, gs = layout.num_kv_heads, layout.group_size
    per_head = layout.qk_nope + layout.v_dim
    f_k, f_v = gs * layout.qk_nope, gs * layout.v_dim
    dev = next(attn.parameters()).device
    pdtype = next(attn.parameters()).dtype

    sum_k = torch.zeros(G, f_k,      dtype=torch.float64, device=dev)
    sum_v = torch.zeros(G, f_v,      dtype=torch.float64, device=dev)
    H_k   = torch.zeros(G, f_k, f_k, dtype=torch.float64, device=dev)
    H_v   = torch.zeros(G, f_v, f_v, dtype=torch.float64, device=dev)
    n = 0
    n_eff = 0.0
    for idx, hidden in enumerate(hidden_list):
        h = hidden.to(device=dev, dtype=pdtype)
        ckv = attn.kv_a_proj_with_mqa(h)
        kv_x, _ = ckv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(kv_x))            # (B, S, H * per_head)
        flat = kv.reshape(-1, G, gs, per_head)                    # (N, G, gs, per_head)
        k = flat[..., :layout.qk_nope].reshape(-1, G, f_k).to(torch.float64)
        v = flat[..., layout.qk_nope:].reshape(-1, G, f_v).to(torch.float64)
        if token_weights is not None:
            sqrt_w = token_weights[idx].to(device=dev, dtype=torch.float64).sqrt().reshape(-1)
            k = k * sqrt_w.view(-1, 1, 1)
            v = v * sqrt_w.view(-1, 1, 1)
            n_eff += float((sqrt_w * sqrt_w).sum().item())
        n += k.shape[0]
        sum_k += k.sum(dim=0)
        sum_v += v.sum(dim=0)
        H_k   += torch.einsum("ngf,ngd->gfd", k, k)
        H_v   += torch.einsum("ngf,ngd->gfd", v, v)
    if token_weights is None:
        n_norm = max(n, 1)
        denom = max(n - 1, 1)
    else:
        n_norm = max(n_eff, 1.0)
        denom = max(n_eff - 1.0, 1.0)
    mean_k = sum_k / n_norm
    mean_v = sum_v / n_norm
    cov_k = (H_k - n_norm * torch.einsum("gf,gd->gfd", mean_k, mean_k)) / denom
    cov_v = (H_v - n_norm * torch.einsum("gf,gd->gfd", mean_v, mean_v)) / denom
    return cov_k, cov_v


@torch.no_grad()
def collect_full_kv_grams(
    attn, hidden_list: list[torch.Tensor], layout: GqlaLayout,
    token_weights: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-pair K/V covariance ``(H * d, H * d)``, fp64. Drives data-driven head grouping.

    Same kv_a chain as ``collect_kv_grams``; differs only in that K/V are flattened
    across heads (instead of stacked per group) so cross-head covariance is retained.
    Optional ``token_weights`` apply the same SparseGPT/OBS-style row scaling.
    """
    H = layout.num_heads
    per_head = layout.qk_nope + layout.v_dim
    f_k, f_v = H * layout.qk_nope, H * layout.v_dim
    dev = next(attn.parameters()).device
    pdtype = next(attn.parameters()).dtype

    sum_k = torch.zeros(f_k, dtype=torch.float64, device=dev)
    sum_v = torch.zeros(f_v, dtype=torch.float64, device=dev)
    H_k = torch.zeros(f_k, f_k, dtype=torch.float64, device=dev)
    H_v = torch.zeros(f_v, f_v, dtype=torch.float64, device=dev)
    n = 0
    n_eff = 0.0
    for idx, hidden in enumerate(hidden_list):
        h = hidden.to(device=dev, dtype=pdtype)
        ckv = attn.kv_a_proj_with_mqa(h)
        kv_x, _ = ckv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(kv_x))            # (B, S, H * per_head)
        flat = kv.reshape(-1, H, per_head)
        k = flat[..., :layout.qk_nope].reshape(-1, f_k).to(torch.float64)
        v = flat[..., layout.qk_nope:].reshape(-1, f_v).to(torch.float64)
        if token_weights is not None:
            sqrt_w = token_weights[idx].to(device=dev, dtype=torch.float64).sqrt().reshape(-1)
            k = k * sqrt_w.unsqueeze(-1)
            v = v * sqrt_w.unsqueeze(-1)
            n_eff += float((sqrt_w * sqrt_w).sum().item())
        n += k.shape[0]
        sum_k += k.sum(dim=0)
        sum_v += v.sum(dim=0)
        H_k += k.T @ k
        H_v += v.T @ v
    if token_weights is None:
        n_norm = max(n, 1)
        denom = max(n - 1, 1)
    else:
        n_norm = max(n_eff, 1.0)
        denom = max(n_eff - 1.0, 1.0)
    mean_k = sum_k / n_norm
    mean_v = sum_v / n_norm
    cov_k = (H_k - n_norm * torch.outer(mean_k, mean_k)) / denom
    cov_v = (H_v - n_norm * torch.outer(mean_v, mean_v)) / denom
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


# ----------------------------- data-driven head grouping (similarity) -----------------------------


def compute_head_similarity(
    full_cov_K: torch.Tensor, full_cov_V: torch.Tensor, layout: GqlaLayout,
    w_k: float = 1.0, w_v: float = 1.0,
) -> torch.Tensor:
    """Nuclear-norm-of-cross-cov head similarity, weighted K + V contributions.

    For each pair ``(h, h')``, ``||Sigma^{h,h'}||_*`` is the negated optimal-
    Procrustes residual (up to constants); we trace-normalise by
    ``sqrt(tr(Sigma_h) * tr(Sigma_h'))`` so per-head activation magnitudes
    don't dominate. Returns an ``(H, H)`` fp64 CPU tensor with the diagonal
    forced to ``-inf`` so a head never self-pairs.

    ``w_k``, ``w_v`` let the caller bias K vs V; default ``(1, 1)`` sums them.
    """
    H = layout.num_heads
    d_k, d_v = layout.qk_nope, layout.v_dim
    K = full_cov_K.to(dtype=torch.float64)
    V = full_cov_V.to(dtype=torch.float64)
    blocks_K = K.view(H, d_k, H, d_k).permute(0, 2, 1, 3).contiguous()
    blocks_V = V.view(H, d_v, H, d_v).permute(0, 2, 1, 3).contiguous()
    diag_K = torch.diagonal(blocks_K, dim1=0, dim2=1).permute(2, 0, 1)   # (H, d_k, d_k)
    diag_V = torch.diagonal(blocks_V, dim1=0, dim2=1).permute(2, 0, 1)   # (H, d_v, d_v)
    tr_k = torch.diagonal(diag_K, dim1=-2, dim2=-1).sum(dim=-1)
    tr_v = torch.diagonal(diag_V, dim1=-2, dim2=-1).sum(dim=-1)
    nuc_K = torch.linalg.svdvals(blocks_K.view(H * H, d_k, d_k)).sum(dim=-1).view(H, H)
    nuc_V = torch.linalg.svdvals(blocks_V.view(H * H, d_v, d_v)).sum(dim=-1).view(H, H)
    denom_k = (tr_k.unsqueeze(0) * tr_k.unsqueeze(1)).clamp_min(1e-30).sqrt()
    denom_v = (tr_v.unsqueeze(0) * tr_v.unsqueeze(1)).clamp_min(1e-30).sqrt()
    S = (w_k * nuc_K / denom_k + w_v * nuc_V / denom_v)
    S.fill_diagonal_(float("-inf"))
    return S.cpu()


def greedy_balanced_grouping(
    sim: torch.Tensor, num_groups: int, group_size: int,
) -> list[list[int]]:
    """Seed-and-grow balanced grouping: pick the best unassigned pair to seed each
    group, then iteratively add the unassigned head with highest sum-similarity to
    current members until the group is full. Deterministic tiebreak by lower head
    index. Returns a list of ``num_groups`` lists, each of length ``group_size``.
    """
    H = sim.shape[0]
    if H != num_groups * group_size:
        raise ValueError(f"H={H} != num_groups({num_groups}) * group_size({group_size})")
    sim_f = sim.to(torch.float64)
    unassigned = list(range(H))
    groups: list[list[int]] = []
    while unassigned:
        if len(unassigned) >= 2 and group_size >= 2:
            best_pair, best_s = None, -float("inf")
            for i, h in enumerate(unassigned):
                for hp in unassigned[i + 1:]:
                    s = sim_f[h, hp].item()
                    if s > best_s:
                        best_pair, best_s = (h, hp), s
            group = [best_pair[0], best_pair[1]]
            unassigned.remove(best_pair[0])
            unassigned.remove(best_pair[1])
        else:
            group = [unassigned[0]]
            unassigned.pop(0)
        while len(group) < group_size and unassigned:
            best_h, best_score = None, -float("inf")
            for h in unassigned:
                score = sum(sim_f[h, g].item() for g in group)
                if score > best_score:
                    best_h, best_score = h, score
            group.append(best_h)
            unassigned.remove(best_h)
        groups.append(group)
    return groups


@torch.no_grad()
def assemble_per_group_covs_from_full(
    full_cov_K: torch.Tensor, full_cov_V: torch.Tensor,
    perm: list[int], layout: GqlaLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice per-group ``(G, gs*d, gs*d)`` covs from the all-pair ``(H*d, H*d)`` cov.

    For group ``g``, the in-group head order is ``perm[g*gs:(g+1)*gs]``; block
    ``(i, j)`` is the original ``Sigma[h_i*d:(h_i+1)*d, h_j*d:(h_j+1)*d]``. The
    resulting per-group cov is what the per-group PCA fit consumes.
    """
    G, gs = layout.num_kv_heads, layout.group_size
    d_k, d_v = layout.qk_nope, layout.v_dim
    fcK = full_cov_K.to(torch.float64)
    fcV = full_cov_V.to(torch.float64)
    dev = fcK.device
    cov_k = torch.zeros(G, gs * d_k, gs * d_k, dtype=torch.float64, device=dev)
    cov_v = torch.zeros(G, gs * d_v, gs * d_v, dtype=torch.float64, device=dev)
    for g in range(G):
        for i in range(gs):
            hi = perm[g * gs + i]
            for j in range(gs):
                hj = perm[g * gs + j]
                cov_k[g, i * d_k:(i + 1) * d_k, j * d_k:(j + 1) * d_k] = (
                    fcK[hi * d_k:(hi + 1) * d_k, hj * d_k:(hj + 1) * d_k]
                )
                cov_v[g, i * d_v:(i + 1) * d_v, j * d_v:(j + 1) * d_v] = (
                    fcV[hi * d_v:(hi + 1) * d_v, hj * d_v:(hj + 1) * d_v]
                )
    return cov_k, cov_v


# ----------------------------- hessian-aware (per-token NLL) weights -----------------------------


@torch.no_grad()
def compute_token_weights_nll(
    model,
    final_hiddens: list[torch.Tensor],   # list of (1, S, d_model) -- output of last backbone layer
    input_ids_list: list[torch.Tensor],  # list of (1, S) or (S,) -- token ids
) -> list[torch.Tensor]:
    """Per-token NLL weights from the cached last-layer hidden states.

    Standard per-layer PCA accumulates ``Sigma_X = X.T @ X`` with uniform weights,
    implicitly assuming every token contributes equally to downstream loss. Rare
    / surprising tokens actually carry far more LM-loss gradient than easy ones.
    The SparseGPT / OBS prescription is ``Sigma_X = X.T @ diag(w) @ X``, with
    ``w_t = NLL_teacher(token_{t+1} | tokens_{<=t})`` approximating the diagonal
    of the empirical Hessian of the next-token loss at the final hidden state.

    Implementation: apply ``model.model.norm`` + ``model.lm_head`` + cross-entropy
    to the cached final hidden states; weight token ``t`` by the NLL of token
    ``t+1`` (last position gets weight 0). Per-sample mean-1 normalisation
    preserves the cov scale of the downstream PCA. Works under HF
    ``device_map="auto"``: norm / head stay on whichever GPU(s) HF placed them.
    """
    norm = model.model.norm
    head = model.lm_head
    norm_param = next(norm.parameters())
    head_param = next(head.parameters())
    norm_device = norm_param.device
    norm_dtype = norm_param.dtype
    head_device = head_param.device

    weights: list[torch.Tensor] = []
    for h, ids in zip(final_hiddens, input_ids_list):
        h = h.to(device=norm_device, dtype=norm_dtype)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids_h = ids.to(head_device)
        h_n = norm(h)
        if norm_device != head_device:
            h_n = h_n.to(head_device)
        logits = head(h_n).float()                              # (1, S, V)
        shift_logits = logits[:, :-1, :]
        shift_targets = ids_h[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="none",
        ).view(1, -1)                                           # (1, S-1)
        S = ids_h.shape[-1]
        w = torch.zeros(S, dtype=torch.float32, device=head_device)
        w[:-1] = nll[0].float()
        mean_w = w[:-1].mean().clamp_min(1e-12)
        w = w / mean_w
        weights.append(w.detach().cpu())
    return weights


def diagnose_weights(weights: list[torch.Tensor]) -> dict:
    """Quick stats for logging Hessian-PCA token weights."""
    flat = torch.cat([w.float() for w in weights])
    return {
        "n_tokens": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


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
def compose_compressed_with_perm(
    attn, layout: GqlaLayout, groups: list[list[int]],
    u_k: torch.Tensor, u_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compress + absorb for arbitrary head groupings (e.g. similarity-driven).

    For each new GQLA head ``h_new = g*gs + i`` with original head
    ``orig_h = groups[g][i]``:

        slot_k = u_k[g, i*d_k:(i+1)*d_k, :]                       # (d_k, d_k)
        slot_v = u_v[g, i*d_v:(i+1)*d_v, :]                       # (d_v, d_v)
        new_W_q[h_new, nope] = slot_k.T @ W_q[orig_h, nope]
        new_W_q[h_new, rope] = W_q[orig_h, rope]                  # rope just relabelled
        new_W_o[:, h_new]    = W_o[:, orig_h] @ slot_v
        new_kv_b_K[g]        = sum_i slot_k.T @ W_k[orig_h]
        new_kv_b_V[g]        = sum_i slot_v.T @ W_v[orig_h]

    Reduces to ``compress_and_absorb`` when ``groups = [[g*gs, g*gs+1, ...]]``
    (neighbor grouping) up to a (negligible) different reduction order.
    """
    H, G, gs = layout.num_heads, layout.num_kv_heads, layout.group_size
    d_k, d_rope, d_v, kv_lora = layout.qk_nope, layout.qk_rope, layout.v_dim, layout.kv_lora
    qk_head_dim = d_k + d_rope
    per_head_kv = d_k + d_v

    dev = attn.kv_b_proj.weight.device
    u_k_d = u_k.to(device=dev, dtype=torch.float32)
    u_v_d = u_v.to(device=dev, dtype=torch.float32)

    w_kv = attn.kv_b_proj.weight.data.to(torch.float32)            # (H * per_head_kv, kv_lora)
    w_per_head = w_kv.view(H, per_head_kv, kv_lora)
    w_k_per_head = w_per_head[:, :d_k, :]                          # (H, d_k, kv_lora)
    w_v_per_head = w_per_head[:, d_k:, :]                          # (H, d_v, kv_lora)

    new_kv_b = torch.zeros(G, per_head_kv, kv_lora, dtype=torch.float32, device=dev)
    for g, grp in enumerate(groups):
        acc_k = torch.zeros(d_k, kv_lora, dtype=torch.float32, device=dev)
        acc_v = torch.zeros(d_v, kv_lora, dtype=torch.float32, device=dev)
        for i, orig_h in enumerate(grp):
            slot_k = u_k_d[g, i * d_k:(i + 1) * d_k, :]
            slot_v = u_v_d[g, i * d_v:(i + 1) * d_v, :]
            acc_k = acc_k + slot_k.T @ w_k_per_head[orig_h]
            acc_v = acc_v + slot_v.T @ w_v_per_head[orig_h]
        new_kv_b[g, :d_k, :] = acc_k
        new_kv_b[g, d_k:, :] = acc_v
    new_kv_b = new_kv_b.reshape(G * per_head_kv, kv_lora).contiguous()

    w_q = attn.q_b_proj.weight.data.to(torch.float32)              # (H * qk_head_dim, q_lora)
    new_w_q = torch.empty_like(w_q)
    for g, grp in enumerate(groups):
        for i, orig_h in enumerate(grp):
            h_new = g * gs + i
            slot_k = u_k_d[g, i * d_k:(i + 1) * d_k, :]
            s_old = orig_h * qk_head_dim
            s_new = h_new * qk_head_dim
            new_w_q[s_new:s_new + d_k, :] = slot_k.T @ w_q[s_old:s_old + d_k, :]
            new_w_q[s_new + d_k:s_new + qk_head_dim, :] = w_q[s_old + d_k:s_old + qk_head_dim, :]

    w_o = attn.o_proj.weight.data.to(torch.float32)                # (hidden, H * v_dim)
    new_w_o = torch.empty_like(w_o)
    for g, grp in enumerate(groups):
        for i, orig_h in enumerate(grp):
            h_new = g * gs + i
            slot_v = u_v_d[g, i * d_v:(i + 1) * d_v, :]
            s_old = orig_h * d_v
            s_new = h_new * d_v
            new_w_o[:, s_new:s_new + d_v] = w_o[:, s_old:s_old + d_v] @ slot_v

    return {
        "kv_b_proj.weight": new_kv_b,
        "q_b_proj.weight":  new_w_q.contiguous(),
        "o_proj.weight":    new_w_o.contiguous(),
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
    "collect_full_kv_grams",
    "compute_head_similarity",
    "greedy_balanced_grouping",
    "assemble_per_group_covs_from_full",
    "compute_token_weights_nll",
    "diagnose_weights",
    "fit_per_group_pca",
    "compress_and_absorb",
    "compose_compressed_with_perm",
    "inplace_apply_compressed",
    "prepare_ppl_dataloader",
    "evaluate_ppl",
]
