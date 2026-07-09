"""test_patch_integration.py — GPU integration tests for the DSA model patch.

Run:  CUDA_VISIBLE_DEVICES=0 python tests/test_patch_integration.py

Builds a tiny (2-layer) DeepSeek-V2-Lite, patches it with DSA, and checks:
  1. warmup: only indexer params trainable; forward produces per-layer KL loss;
     backward flows to indexer only.
  2. sparse: LM loss flows through sparse attention to the backbone; indexer also
     trains; forward+backward run.
  3. numerical: FlashKL warmup loss/grads match the dense reference on REAL
     DeepSeek-V2 teacher activations (fp32).
"""
import _path  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoConfig

from opendsa.modeling import (
    patch_model_with_dsa, IndexerConfig, set_dsa_mode,
    freeze_backbone_train_indexer, unfreeze_all, indexer_parameters,
    dsa_loss_registry, set_log_recall,
)
from opendsa.modeling.dsa_attention import _compute_teacher_qk, _indexer_rope_cossin
from opendsa.ops import flashkl_warmup_loss, dense_warmup_reference

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def _tiny(dtype, nlayers=2):
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.num_hidden_layers = nlayers
    cfg.first_k_dense_replace = 1
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                         attn_implementation="eager").to("cuda").to(dtype)
    return cfg, m


def test_stages():
    cfg, model = _tiny(torch.bfloat16)
    patch_model_with_dsa(model, IndexerConfig(n_heads=8, head_dim=128, topk=64, tile=256),
                         mode="warmup")
    freeze_backbone_train_indexer(model)
    set_log_recall(model, True)
    B, L = 1, 128
    ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
    reg = dsa_loss_registry(); reg.reset()
    out = model(input_ids=ids)
    loss = reg.total(); loss.backward()
    g_idx = sum(p.grad.abs().sum().item() for p in indexer_parameters(model) if p.grad is not None)
    bb = [p for n, p in model.named_parameters() if 'indexer' not in n and p.requires_grad]
    assert g_idx > 0 and len(bb) == 0, "warmup grad isolation failed"
    print(f"[warmup] loss={loss.item():.4f} idx-grad={g_idx:.1f} backbone-trainable={len(bb)} OK")
    model.zero_grad(set_to_none=True)

    unfreeze_all(model); set_dsa_mode(model, "sparse")
    reg.reset()
    out = model(input_ids=ids, labels=ids)
    (out.loss + 0.1 * reg.total()).backward()
    oproj_g = next(p.grad.abs().sum().item() for n, p in model.named_parameters()
                   if p.grad is not None and 'self_attn.o_proj' in n)
    idx_g = next(p.grad.abs().sum().item() for n, p in model.named_parameters()
                 if p.grad is not None and 'indexer.wq' in n)
    assert oproj_g > 0 and idx_g > 0, "sparse grad flow failed"
    print(f"[sparse] lm={out.loss.item():.4f} idx={reg.total().item():.4f} "
          f"oproj-grad={oproj_g:.0f} idx-grad={idx_g:.1f} OK")


def test_numerical():
    cfg, model = _tiny(torch.float32, nlayers=1)
    patch_model_with_dsa(model, IndexerConfig(n_heads=8, head_dim=128, topk=64, tile=64),
                         mode="warmup", dtype=torch.float32)
    attn = model.model.layers[0].self_attn
    B, L = 1, 96
    hs = torch.randn(B, L, cfg.hidden_size, device="cuda")
    pos = torch.arange(L, device="cuda").unsqueeze(0)
    qs, ks_, vs, cos, sin = _compute_teacher_qk(attn, hs, pos)
    Qm = qs.transpose(1, 2)[0].detach(); Km = ks_.transpose(1, 2)[0].detach()
    c, s = _indexer_rope_cossin(attn, cos, sin, pos)
    q0, k0, w0 = attn.indexer(hs, hs, c, s)
    q = q0[0].detach().clone().requires_grad_(True)
    k = k0[0].detach().clone().requires_grad_(True)
    w = w0[0].detach().clone().requires_grad_(True)
    cu = torch.tensor([0, L], device="cuda")
    st = float(attn.softmax_scale); si = attn.indexer.softmax_scale
    Lfl = flashkl_warmup_loss(q, k, w, Qm, Km, cu, sm_scale_teacher=st, sm_scale_index=si, tile=16)
    Lfl.backward()
    gfl = {n: v.grad.clone() for n, v in [("q", q), ("k", k), ("w", w)]}
    q.grad = k.grad = w.grad = None
    full, trn = dense_warmup_reference(q, k, w, Qm, Km, cu, sm_scale_teacher=st, sm_scale_index=si)
    full.backward()
    gd = {n: v.grad.clone() for n, v in [("q", q), ("k", k), ("w", w)]}
    rel = lambda a, b: (a - b).norm().item() / (b.norm().item() + 1e-30)
    errs = {n: rel(gfl[n], gd[n]) for n in gfl}
    assert max(errs.values()) < 1e-5, f"numerical mismatch {errs}"
    print(f"[numerical] FlashKL vs dense on real DS-V2 teacher: "
          + " ".join(f"{n}={v:.1e}" for n, v in errs.items()) + " OK")


if __name__ == "__main__":
    try:
        test_stages()
        test_numerical()
        print("ALL INTEGRATION TESTS PASSED")
    finally:
        _path.shutdown_dist()
