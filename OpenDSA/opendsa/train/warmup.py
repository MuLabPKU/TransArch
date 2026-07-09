"""warmup.py — Stage 1: train the Lightning Indexer to imitate MLA attention.

Backbone frozen; only indexer params train, via FlashKL O(L·H) KL distillation.

Run (single GPU debug):
    python -m opendsa.train.warmup --data-dir data_cache/pack_8k --num-layers 4 \
        --max-steps 20 --output-dir runs/warmup_smoke

Run (8 GPU, full):
    accelerate launch --config_file configs/accelerate_ddp.yaml \
        -m opendsa.train.warmup --data-dir data_cache/pack_32k --output-dir runs/warmup
"""
from __future__ import annotations

from transformers import HfArgumentParser, TrainingArguments

from .build import DSAArgs, build_model, build_tokenizer, load_packed
from .trainer import DSATrainer
from .convert_stage import save_indexer
from ..data import DSADataCollator


def main():
    parser = HfArgumentParser(DSAArgs)
    (args,) = parser.parse_args_into_dataclasses()
    args.stage = "warmup"

    # context parallel: CP=world (single-node) when launched with >1 process
    from ..dist import init_parallel, cp_size, cp_rank
    init_parallel()
    cp = cp_size()

    model = build_model(args, mode="warmup")
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
        save_strategy="steps" if args.save_final else "no",
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=False,   # handled manually in build_model
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        label_names=["labels"],
    )

    trainer = DSATrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator,
        dsa_stage="warmup", indexer_loss_coeff=args.indexer_loss_coeff,
        log_recall_every=args.log_recall_every, log_kl_every=args.log_kl_every,
        cp_size=cp,
    )
    trainer.train()

    # save just the indexer weights for the sparse stage + full model (rank 0 only)
    if args.save_final and cp_rank() == 0:
        save_indexer(model, f"{args.output_dir}/indexer.pt")
        trainer.save_model(f"{args.output_dir}/final")
        print("[warmup] done.")


if __name__ == "__main__":
    main()
