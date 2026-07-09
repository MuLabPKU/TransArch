"""long_pack.py — tokenize + pack UltraData-SFT chat data into training sequences.

Source: a chat dataset laid out as ``<bucket>/<Category>/*.jsonl``; each line is
a JSON object with a ``messages`` = [{role, content}] field.

We render each conversation with the model's chat template, tokenize, and greedily
pack conversations into sequences of exactly ``seq_len`` tokens. Each packed
sequence carries ``cu_seqlens`` (document boundaries) so attention/loss can be
masked per-conversation — this matches the ProLong finding that disabling
cross-document attention helps both short and long context and speeds training.

Labels: standard causal-LM shift with -100 on pad; assistant-only masking is
optional (``assistant_only=True``) to only train on assistant spans.

Output: an on-disk HF dataset (Arrow) with columns:
    input_ids [seq_len] int32, labels [seq_len] int64, cu_seqlens [n+1] int32
Because cu_seqlens is variable length, it is stored as a list; the collator pads
per batch.

CLI:
    python -m opendsa.data.long_pack --seq-len 32768 --out data_cache/pack_32k \
        --data /path/to/UltraData-SFT-2605-split/32k-200k
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
from typing import List, Optional

import numpy as np

DEFAULT_DATA = os.environ.get("OPENDSA_DATA", "")


def _render_and_tokenize(messages, tokenizer, assistant_only):
    """Return (token_ids, label_ids) for one conversation using the chat template.
    If assistant_only, non-assistant tokens get label -100."""
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=False)
    except Exception:
        # fallback: concatenate role: content
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    ids = list(ids)
    if not assistant_only:
        return ids, list(ids)
    # assistant-only masking: re-render incrementally to find assistant spans
    labels = [-100] * len(ids)
    prefix = []
    cursor = 0
    for m in messages:
        prefix.append(m)
        try:
            pref_ids = tokenizer.apply_chat_template(
                prefix, tokenize=True, add_generation_prompt=False)
        except Exception:
            return ids, list(ids)  # fallback: train on all
        if m["role"] == "assistant":
            start = cursor
            end = len(pref_ids)
            for j in range(start, min(end, len(labels))):
                labels[j] = ids[j]
        cursor = len(pref_ids)
    return ids, labels


def _iter_ultradata_messages(data_dir):
    """Yield ``messages`` lists from the UltraData-SFT split. ``data_dir`` may be a
    directory (recursively globs ``<Category>/*.jsonl``), a single JSONL file, or a
    glob pattern. Each line is a JSON object with a ``messages`` field."""
    if os.path.isdir(data_dir):
        files = sorted(_glob.glob(os.path.join(data_dir, "**", "*.jsonl"),
                                  recursive=True))
    else:
        files = sorted(_glob.glob(data_dir))
    for fn in files:
        with open(fn) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = d.get("messages")
                if msgs:
                    yield msgs


def _iter_conversations(data_dir, tokenizer, assistant_only, max_seqs):
    count = 0
    for msgs in _iter_ultradata_messages(data_dir):
        ids, labels = _render_and_tokenize(msgs, tokenizer, assistant_only)
        if len(ids) < 8:
            continue
        yield ids, labels
        count += 1
        if max_seqs and count >= max_seqs:
            return


def pack_dataset(
    out_dir: str,
    seq_len: int,
    model: str = "deepseek-ai/DeepSeek-V2-Lite-Chat",
    data_dir: Optional[str] = None,
    assistant_only: bool = False,
    max_seqs: Optional[int] = None,
    max_packed: Optional[int] = None,
    drop_last: bool = True,
):
    from transformers import AutoTokenizer
    from datasets import Dataset

    data_dir = data_dir or DEFAULT_DATA
    if not data_dir:
        raise ValueError("pass --data or set OPENDSA_DATA to the source JSONL directory/glob")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    buf_ids: List[int] = []
    buf_lab: List[int] = []
    buf_cu: List[int] = [0]
    packed = {"input_ids": [], "labels": [], "cu_seqlens": []}

    def flush_full():
        # emit as many full seq_len windows as buffer holds
        while len(buf_ids) >= seq_len:
            chunk_ids = buf_ids[:seq_len]
            chunk_lab = buf_lab[:seq_len]
            # cu boundaries within [0, seq_len]
            cu = [c for c in buf_cu if c <= seq_len]
            if cu[-1] != seq_len:
                cu = cu + [seq_len]
            packed["input_ids"].append(np.asarray(chunk_ids, dtype=np.int32))
            packed["labels"].append(np.asarray(chunk_lab, dtype=np.int64))
            packed["cu_seqlens"].append(np.asarray(cu, dtype=np.int32))
            # carry the remainder; shift doc boundaries
            del buf_ids[:seq_len]
            del buf_lab[:seq_len]
            newcu = [0] + [c - seq_len for c in buf_cu if c > seq_len]
            buf_cu.clear(); buf_cu.extend(newcu)

    n_conv = 0
    for ids, labels in _iter_conversations(data_dir, tok, assistant_only, max_seqs):
        # truncate a single conversation that alone exceeds seq_len
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            labels = labels[:seq_len]
        buf_ids.extend(ids)
        buf_lab.extend(labels)
        buf_cu.append(len(buf_ids))
        n_conv += 1
        flush_full()
        if max_packed and len(packed["input_ids"]) >= max_packed:
            break

    if not drop_last and len(buf_ids) > 0:
        pad = seq_len - len(buf_ids)
        buf_ids.extend([tok.pad_token_id or 0] * pad)
        buf_lab.extend([-100] * pad)
        cu = [c for c in buf_cu] + [seq_len]
        packed["input_ids"].append(np.asarray(buf_ids, dtype=np.int32))
        packed["labels"].append(np.asarray(buf_lab, dtype=np.int64))
        packed["cu_seqlens"].append(np.asarray(cu, dtype=np.int32))

    ds = Dataset.from_dict({
        "input_ids": [a.tolist() for a in packed["input_ids"]],
        "labels": [a.tolist() for a in packed["labels"]],
        "cu_seqlens": [a.tolist() for a in packed["cu_seqlens"]],
    })
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"[long_pack] seq_len={seq_len}  conversations={n_conv}  "
          f"packed_sequences={len(ds)}  data={data_dir}  -> {out_dir}")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="UltraData-SFT split directory (recursive *.jsonl), a single "
                         "JSONL file, or a glob. Defaults to OPENDSA_DATA. Records must "
                         "have a `messages` field.")
    ap.add_argument("--assistant-only", action="store_true")
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="cap #conversations consumed (for quick builds)")
    ap.add_argument("--max-packed", type=int, default=None,
                    help="cap #packed sequences emitted")
    args = ap.parse_args()
    pack_dataset(args.out, args.seq_len, args.model, data_dir=args.data,
                 assistant_only=args.assistant_only,
                 max_seqs=args.max_seqs, max_packed=args.max_packed)


if __name__ == "__main__":
    main()
