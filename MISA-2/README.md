# MISA-2: ResMiSA: Residual Mixture of Sparse Attention for Long-Context LLM Inference [Preprint]

This folder hosts MISA-2; the accompanying paper is titled **ResMiSA**.

ResMiSA extends MISA with an always-active shared expert that fuses the linear contributions of all indexer heads, while routed residual experts supply nonlinear corrections. The shared path preserves a direct gradient path through every head. Combined with LiteTopK, ResMiSA achieves a 2.43× end-to-end time-to-first-token speedup over DSA at 1M context.

[[Paper (PDF)]](resmisa.pdf)

## Code

**Code coming soon.**

## Authors

Ruijie Zhou, Fanxu Meng, Zhixin Pan, Jinbao Xue, Ke Zhang, Jun Yu, Wenjie Pei
