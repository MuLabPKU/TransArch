# LISA: Head-Fused Linear Indexing for Efficient Sparse Attention [Preprint]

LISA is a training-free indexer that fuses the linear contributions of all query heads into one dot product per token. It reduces full-context scoring cost while preserving token-level granularity. LISA† refines candidates with the original multi-head scorer, and LISA‡ shares candidate pools across layers with layer-specific refinement. On DeepSeek-V3.2 at 1M context, LISA achieves a 4.13× speedup over DSA in single-request offline time to first token, while LISA‡ with four-layer groups achieves 5.20×.

[[Paper (PDF)]](lisa.pdf)

## Code

**Code coming soon.**

## Authors

Zhixin Pan, Fanxu Meng, Zhaohui Wang, Taosong Fang, Muhan Zhang
