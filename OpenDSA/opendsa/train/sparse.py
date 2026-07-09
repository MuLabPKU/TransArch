"""sparse.py — Stage 2: sparse continued-training (backbone + indexer).

Loads warmup-trained indexer weights, unfreezes the backbone, and continues
training under top-k sparse MLA. Loss = LM + λ·indexer_KL(over selected keys).

Run (8 GPU):
    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        -m opendsa.train.sparse --data-dir data_cache/pack_64k \
        --indexer-init runs/warmup/indexer.pt --output-dir runs/sparse \
        --lr 2e-5 --topk 2048
"""
from __future__ import annotations

from transformers import HfArgumentParser, TrainingArguments

from .build import DSAArgs, build_model, build_tokenizer, load_packed
from .trainer import DSATrainer
from .convert_stage import load_indexer, save_indexer, save_ep_merged_model
from ..data import DSADataCollator


def main():
    parser = HfArgumentParser(DSAArgs)
    (args,) = parser.parse_args_into_dataclasses()
    args.stage = "sparse"
    if args.lr >= 1e-3:  # warmup default is too high for backbone; use a sane default
        args.lr = 2e-5

    # context + expert parallel (CP=EP=world on single node) when launched multi-proc
    from ..dist import init_parallel, cp_size, ep_size, cp_rank
    init_parallel()
    cp = cp_size()

    model = build_model(args, mode="sparse")
    if args.indexer_init:
        load_indexer(model, args.indexer_init, strict=True)
    if ep_size() > 1:
        from ..modeling import patch_model_with_ep
        patch_model_with_ep(model, ep_size())
    tok = build_tokenizer(args)
    ds = load_packed(args)
    collator = DSADataCollator(pad_token_id=tok.pad_token_id or 0)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # HF's default step checkpoint is incomplete under EP because rank0 does
        # not own all routed experts. Save through save_ep_merged_model below.
        save_strategy="no" if ep_size() > 1 else ("steps" if args.save_final else "no"),
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        label_names=["labels"],
    )
    trainer = DSATrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator,
        dsa_stage="sparse", indexer_loss_coeff=args.indexer_loss_coeff, cp_size=cp,
        log_recall_every=args.log_recall_every, log_kl_every=args.log_kl_every,
        zero2=(cp > 1),
    )
    trainer.train()
    if args.save_final:
        if cp_rank() == 0:
            save_indexer(model, f"{args.output_dir}/indexer.pt")
        if ep_size() > 1:
            save_ep_merged_model(model, f"{args.output_dir}/final", tokenizer=tok)
        elif cp_rank() == 0:
            trainer.save_model(f"{args.output_dir}/final")
        if cp_rank() == 0:
            print("[sparse] done.")


if __name__ == "__main__":
    main()
