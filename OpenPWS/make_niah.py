"""Rebuild Appendix C.1's full 128K protocol; unpublished text/seed are explicit.

Output uses uint32 mmap prompts plus JSONL metadata, independently retokenizing
every assembled prompt. V4.1 uses ten blocks of the exact published histograms.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
from paths import MODEL_V4 as MODEL, FILLER_SOURCE
H = [{8: 1, 16: 52, 24: 150, 32: 52, 40: 1}, {9: 2, 17: 68, 25: 146, 33: 40},
     {10: 4, 18: 84, 26: 140, 34: 28}, {11: 7, 19: 101, 27: 129, 35: 19},
     {12: 12, 20: 116, 28: 116, 36: 12}]
H += [{48-k: v for k, v in H[r].items()} for r in (3, 2, 1)]
QUERY = ('Now answer the question using the key-value pairs above. What is the value for key '
         '{key}? Answer with only the value. The value for key {key} is')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks', type=int, choices=[1, 10], required=True)
    p.add_argument('--seed', type=int, default=260936322)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int, help='Preparation smoke test only; never marks a partial dataset validated')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert a.batch_size > 0
    assert a.limit is None or a.limit > 0
    if (a.output/'validated.json').exists():
        raise RuntimeError('Validated output already exists; choose another output directory.')
    if (a.output/'metadata.jsonl').exists() and not a.resume:
        raise RuntimeError('Partial output exists; use --resume to preserve completed rows.')
    tok = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    candidates = json.loads((ROOT/'prompts/candidates.json').read_text())
    ids = np.array([c['id'] for c in candidates], dtype=np.uint32)
    strings = {c['id']: c['text'] for c in candidates}
    for c in candidates:
        assert enc(c['text']) == [c['id']]
        assert enc('。'+c['text']+':'+c['text']+'。') == [320,c['id'],28,c['id'],320]
        assert len(enc(QUERY.format(key=c['text']))) == 33
    # Each slot is (block, residue, binding, prefill length).
    slots = []
    by_length = defaultdict(list)
    for block in range(a.blocks):
        for residue, histogram in enumerate(H):
            lengths = [k for k, n in histogram.items() for _ in range(n)]
            rng.shuffle(lengths)
            assert len(lengths) == 256 and sum(lengths) == 256*24
            assert sum(k*k for k in lengths) == 256*604
            for binding, length in enumerate(lengths):
                by_length[length].append(len(slots))
                slots.append((block, residue, binding, length))
    need = {l: len(ix) for l, ix in by_length.items()}
    # Newlines are included in the reconstructed sentence segment lengths.
    with gzip.open(FILLER_SOURCE, 'rt') as f:
        sentences = list(dict.fromkeys(s.strip() for s in f if s.strip()))
    rng.shuffle(sentences)
    pools = defaultdict(list)
    for start in range(0, len(sentences), 4096):
        texts = [s+'\n' for s in sentences[start:start+4096]]
        for s, e in zip(texts, tok.encode_batch(texts, add_special_tokens=False)):
            l = len(e.ids)
            if l in need and len(pools[l]) < need[l] and enc('。'+s+'。') == [320]+e.ids+[320]:
                pools[l].append(s)
        if all(len(pools[l]) == n for l, n in need.items()):
            break
    assert all(len(pools[l]) == n for l, n in need.items()), {l:(len(pools[l]),n) for l,n in need.items()}
    fillers = {}
    for l in range(8, 25):
        left = by_length[l].copy()
        rng.shuffle(left)
        if l == 24:
            right, left = left[len(left)//2:], left[:len(left)//2]
        else:
            right = by_length[48-l].copy()
            rng.shuffle(right)
        assert len(left) == len(right)
        for i, j in zip(left, right):
            s, t = pools[l].pop(), pools[48-l].pop()
            fillers[i], fillers[j] = (s,t), (t,s)
    assert all(not pool for pool in pools.values())
    # The paper does not publish the intro, body padding or tail text.
    intro = ('Some special key-value pairs are hidden in the text below. '
             'Each record has the form key:value。. Remember the pairs and answer '
             'the final question with only the value. '
             'Examples: apple:red。sky:blue。grass:green。\n')
    while len(enc(intro)) % 8:
        intro += '。'
    intro_ids = enc(intro)
    background = enc('\n'.join(sentences[:16000]))
    def padding(n):
        # Both ends have explicit delimiters; retain natural text in the middle.
        for start in range(50):
            s = '。'+dec(background[start:start+n-2])+'。'
            if len(enc(s)) == n:
                return s
        raise RuntimeError('Cannot make stable padding')
    prefix = padding(62984-len(intro_ids))
    assert len(enc(prefix)) % 8 == 0
    # The tail's final Chinese full stop is already an unambiguous delimiter.
    # Adding a newline here would merge into the tokenizer's single 。\n token.
    separator = ''
    tail = padding(128000-(62984+64000+48+len(enc(separator))+33))
    fixed = [intro, prefix, tail, separator]
    fixed_ids = [enc(s) for s in fixed]
    fixed_text=dict(zip(['intro','body_prefix','tail','query_separator'],fixed))
    if a.resume:
        assert json.loads((a.output/'fixed_text.json').read_text())==fixed_text
    else:
        (a.output/'fixed_text.json').write_text(json.dumps(fixed_text,ensure_ascii=False))
    total = len(slots)
    arr = np.lib.format.open_memmap(a.output/'input_ids.npy', mode='r+' if a.resume else 'w+', dtype=np.uint32, shape=(total,128000))
    assert arr.shape==(total,128000)
    # Store bindings independently so all eight residue comparisons are auditable.
    bindings = np.empty((a.blocks,256,16000,2),dtype=np.uint32)
    for b in range(a.blocks):
        for k in range(256):
            bindings[b,k,:,0] = nrng.choice(ids,16000,replace=False)
            bindings[b,k,:,1] = nrng.choice(ids,16000,replace=False)
    if a.resume:
        assert np.array_equal(np.load(a.output/'bindings.npy',mmap_mode='r'),bindings)
    else:
        np.save(a.output/'bindings.npy',bindings)
    bodies = {}
    digest = hashlib.sha256()
    previous=[json.loads(s) for s in (a.output/'metadata.jsonl').read_text().splitlines()] if a.resume else []
    for i,row in enumerate(previous):
        block,residue,binding,l=slots[i]
        assert row['row']==i and (row['block'],row['residue'],row['binding'],row['prefill_tokens'])==(block,residue,binding,l)
        assert (row['prefill'],row['postfill'])==fillers[i]
        digest.update(arr[i].tobytes())
    stop=min(total,len(previous)+a.limit) if a.limit else total
    print(f'Preparing rows {len(previous)}..{stop-1}, tokenizer batch size {a.batch_size}',flush=True)
    with (a.output/'metadata.jsonl').open('a' if a.resume else 'w') as meta:
        pending=[]
        def flush_batch():
            encoded=tok.encode_batch([item[0] for item in pending],add_special_tokens=False)
            for (_,expected,row),encoded_row in zip(pending,encoded):
                actual=encoded_row.ids
                i=row['row']
                assert actual==expected, f'Full prompt retokenization mismatch at row {i}'
                arr[i]=actual
                digest.update(arr[i].tobytes())
                meta.write(json.dumps(row,ensure_ascii=False)+'\n')
            meta.flush()
            if pending:
                print(f'Validated {pending[-1][2]["row"]+1}/{total} prompts',flush=True)
            pending.clear()
        for i, (block,residue,binding,l) in enumerate(slots):
            if i<len(previous):
                continue
            if i>=stop:
                break
            key = (block,binding)
            if key not in bodies:
                pairs = bindings[block,binding]
                body_ids = np.empty((16000,4),dtype=np.uint32)
                body_ids[:,0],body_ids[:,1],body_ids[:,2],body_ids[:,3] = pairs[:,0],28,pairs[:,1],320
                body_text = ''.join(strings[int(k)]+':'+strings[int(v)]+'。' for k,v in pairs)
                bodies[key] = (body_text,body_ids.flatten().tolist())
            body, body_ids = bodies[key]
            pre, post = fillers[i]
            target_key, target_value = map(int,bindings[block,binding,248])
            query = QUERY.format(key=strings[target_key])
            parts = [intro,pre,prefix,body,post,tail,separator,query]
            component_ids = [intro_ids,enc(pre),fixed_ids[1],body_ids,enc(post),fixed_ids[2],fixed_ids[3],enc(query)]
            expected = [v for component in component_ids for v in component]
            assert len(expected)==128000 and len(component_ids[1])==l and len(component_ids[4])==48-l
            source = 63976+l
            body_start = len(intro_ids)+l+len(fixed_ids[1])
            assert body_start+248*4 == source and source%8==residue
            assert expected[source:source+4] == [target_key,28,target_value,320]
            assert len(component_ids[-1]) == 33
            boundaries = np.cumsum([0]+[len(x) for x in component_ids]).tolist()
            row = dict(id=f'niah_b{block:02d}_{binding:03d}_r{residue}',row=i,
                       block=block,binding=binding,residue=residue,prefill_tokens=l,
                       source_position=source,query_position=127967,gold=strings[target_value],
                       gold_id=target_value,key=strings[target_key],boundaries=boundaries,
                       prefill=pre,postfill=post)
            pending.append((''.join(parts),expected,row))
            if len(pending)>=a.batch_size:
                flush_batch()
            if (i+1)%2048==0:
                bodies.clear()
        flush_batch()
    arr.flush()
    if stop<total:
        print(f'Partial preparation only: {stop}/{total}; rerun with --resume',flush=True)
        return
    manifest = dict(paper='2609.36322 Appendix C.1',prompts=total,bindings=a.blocks*256,
                    tokens_per_prompt=128000,total_input_tokens=total*128000,
                    seed=a.seed,histograms=H,input_ids_sha256=digest.hexdigest(),
                    tokenizer_sha256=hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest(),
                    validation='Every prompt independently retokenized; every component and all records checked',
                    fidelity='Protocol reconstruction; author seed, intro, padding and exact sampled records unavailable',
                    choices=['Unique keys and independently sampled unique values per binding',
                             'Sentence segment token lengths include trailing newline',
                             'Natural-language body prefix and tail use fixed shuffled source text',
                             'Raw direct-answer prompt; no additional chat template'])
    (a.output/'validated.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
