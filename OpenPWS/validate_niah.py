"""Audit saved NIAH arrays, position histograms, bindings and tokenizer equivalence."""
import argparse
from paths import MODEL_V41
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from make_niah import H, MODEL, QUERY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path)
    a=p.parse_args()
    manifest=json.loads((a.data/'validated.json').read_text())
    rows=[json.loads(s) for s in (a.data/'metadata.jsonl').read_text().splitlines()]
    x=np.load(a.data/'input_ids.npy',mmap_mode='r')
    bindings=np.load(a.data/'bindings.npy',mmap_mode='r')
    assert x.shape==(manifest['prompts'],128000) and len(rows)==len(x)
    assert len({r['id'] for r in rows})==len(rows)
    ta=json.loads((MODEL/'tokenizer.json').read_text())
    tb=json.loads((MODEL_V41/'tokenizer.json').read_text())
    for key in ta.keys()|tb.keys():
        if key!='added_tokens':
            assert ta.get(key)==tb.get(key),(key,'base tokenization differs')
    ad={v['id']:v for v in ta['added_tokens']}
    bd={v['id']:v for v in tb['added_tokens']}
    excluded={d[i]['content'] for i in ad.keys()|bd.keys() if ad.get(i)!=bd.get(i) for d in (ad,bd) if i in d}
    tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    enc=lambda s:tok.encode(s,add_special_tokens=False).ids
    # All records use purely alphabetic tokens, so none can contain these special strings.
    assert all(any(not c.isalpha() and c not in ':。' for c in s) for s in excluded)
    fixed=json.loads((a.data/'fixed_text.json').read_text())
    fixed_ids={key:enc(s) for key,s in fixed.items()}
    for s in fixed.values():
        assert not any(e in s for e in excluded)
    for pairs in bindings.reshape(-1,16000,2):
        assert len(set(pairs[:,0].tolist()))==16000
        assert len(set(pairs[:,1].tolist()))==16000
    sentence_counts=Counter()
    pair_counts=Counter()
    hist=defaultdict(Counter)
    binding_residues=defaultdict(set)
    digest=hashlib.sha256()
    for i,row in enumerate(rows):
        assert row['row']==i
        prefix,post=row['prefill'],row['postfill']
        for s in (prefix,post):
            assert not any(e in s for e in excluded)
        assert len(enc(prefix))==row['prefill_tokens']
        assert len(enc(post))==48-row['prefill_tokens']
        sentence_counts.update([prefix,post])
        pair_counts[prefix,post]+=1
        block,binding,r=row['block'],row['binding'],row['residue']
        hist[block,r][row['prefill_tokens']]+=1
        assert r not in binding_residues[block,binding]
        binding_residues[block,binding].add(r)
        source=row['source_position']
        assert source==63976+row['prefill_tokens'] and source%8==r
        assert row['query_position']==127967
        b=row['boundaries']
        assert b[0]==0 and b[-1]==128000 and b[-2]==127967
        body=x[i,b[3]:b[4]].reshape(16000,4)
        assert np.array_equal(body[:,[0,2]],bindings[block,binding])
        assert np.all(body[:,1]==28) and np.all(body[:,3]==320)
        assert source==b[3]+248*4
        assert enc(row['key'])==[int(x[i,source])]
        assert enc(row['gold'])==[int(x[i,source+2])]
        assert enc(QUERY.format(key=row['key']))==x[i,127967:].tolist()
        for label,lo,hi in [('intro',b[0],b[1]),('body_prefix',b[2],b[3]),
                            ('tail',b[5],b[6]),('query_separator',b[6],b[7])]:
            assert fixed_ids[label]==x[i,lo:hi].tolist()
        digest.update(x[i].tobytes())
    assert all(set(rs)==set(range(8)) for rs in binding_residues.values())
    assert len(binding_residues)==manifest['bindings']
    assert all(dict(count)==H[r] for (_,r),count in hist.items())
    assert len(sentence_counts)==len(rows) and set(sentence_counts.values())=={2}
    assert all(n==1 and pair_counts[t,s]==1 for (s,t),n in pair_counts.items())
    assert digest.hexdigest()==manifest['input_ids_sha256']
    result=dict(prompts=len(rows),total_tokens=x.size,bindings=len(binding_residues),
                unique_sentences=len(sentence_counts),all_checks_passed=True,
                input_ids_sha256=digest.hexdigest(),
                tokenizer_equivalence='Identical tokenizer model, normalizer, pretokenizer, decoder and processor; all differing added-token spellings excluded from every component')
    (a.data/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
