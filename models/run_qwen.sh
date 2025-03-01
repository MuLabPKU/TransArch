CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-4x128_rope_repeat \
--num-kv-groups 4 \
--head-dim 128 \
--version v2 \
--test \
--rope repeat

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-2x256_rope_repeat \
--num-kv-groups 2 \
--head-dim 256 \
--version v2 \
--test \
--rope repeat

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-1x512_rope_repeat \
--num-kv-groups 1 \
--head-dim 512 \
--version v2 \
--test \
--rope repeat

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-4x128_rope_repeat_absorb \
--num-kv-groups 4 \
--head-dim 128 \
--version v2 \
--test \
--rope repeat \
--absorb

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-2x256_rope_repeat_absorb \
--num-kv-groups 2 \
--head-dim 256 \
--version v2 \
--test \
--rope repeat \
--absorb

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-1x512_rope_repeat_absorb \
--num-kv-groups 1 \
--head-dim 512 \
--version v2 \
--test \
--rope repeat \
--absorb

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-4x128_rope_extend \
--num-kv-groups 4 \
--head-dim 128 \
--version v2 \
--test \
--rope extend

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-2x256_rope_extend \
--num-kv-groups 2 \
--head-dim 256 \
--version v2 \
--test \
--rope extend

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-1x512_rope_extend \
--num-kv-groups 1 \
--head-dim 512 \
--version v2 \
--test \
--rope extend

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-4x128_rope_extend_absorb \
--num-kv-groups 4 \
--head-dim 128 \
--version v2 \
--test \
--rope extend \
--absorb

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-2x256_rope_extend_absorb \
--num-kv-groups 2 \
--head-dim 256 \
--version v2 \
--test \
--rope extend \
--absorb

CUDA_VISIBLE_DEVICES=4,5,6,7 python transfer_qwen2_to_mla.py \
--model /root/mfx/huggingface/Qwen/Qwen2.5-7B/ \
--output qwen-7b-1x512_rope_extend_absorb \
--num-kv-groups 1 \
--head-dim 512 \
--version v2 \
--test \
--rope extend \
--absorb