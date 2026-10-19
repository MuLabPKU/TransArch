"""Explicit reconstructions of unpublished D/B.3 prompt sweeps, not exact replicas."""
import itertools
from paths import MODEL_V41
import json
from pathlib import Path
from tokenizers import Tokenizer
from make_niah import MODEL,ROOT
from make_code_prompts import INSTRUCTION


def main():
    t=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    other=Tokenizer.from_file(str(MODEL_V41/'tokenizer.json'))
    enc=lambda s:t.encode(s,add_special_tokens=False).ids
    source=(MODEL/'inference/kernel.py').read_text()
    marker='T.Cast(compute_dtype, T.Cast(FP'
    code=enc(source[:source.index(marker)+len(marker)])
    head=t.decode(code[:974],skip_special_tokens=False)
    ending=t.decode(code[974:],skip_special_tokens=False)
    fillers={}
    for subject,verb,objects,purpose,extra in itertools.product(
        ['This file','This module','The following source file','The following module'],
        ['defines','contains','provides implementations of'],
        ['TileLang kernels for block-wise FP8 quantization and the related low-precision operations',
         'optimized tensor kernels and low-precision numerical operations',
         'tensor operations and carefully optimized low-precision kernels'],
        ['for model inference','for the model inference runtime','used by the model during inference'],
        ['',' with explicit scaling',' with explicit scaling and type conversions',
         ' with explicit scaling and type conversions for numerical stability',
         ' with explicit scaling and type conversions for numerical stability and efficient execution']):
        s=f'# {subject} {verb} {objects} {purpose}{extra}.'
        l=len(enc(s))
        fillers.setdefault(l,s)
    published='# This file defines TileLang kernels for block-wise FP8 quantization and the related low-precision operations for model inference.'
    assert len(enc(published))==24
    fillers[24]=published
    assert all(l in fillers for l in range(23,40)),sorted(fillers)
    rows=[]
    def add(family,length,filler,instruction,body,prefill,gold,competitor,fidelity):
        prompt='<｜begin▁of▁sentence｜><｜User｜>'+instruction+'\n\n'+filler+'\n\n'+body+'<｜Assistant｜></think>'+prefill
        ids=enc(prompt)
        assert ids==other.encode(prompt,add_special_tokens=False).ids
        assert len(enc(gold))==len(enc(competitor))==1
        rows.append(dict(id=f'{family}_{length}_reconstructed',experiment=family,family=family,
                         filler_tokens=length,target_position=len(ids),prompt=prompt,input_ids=ids,
                         gold=gold,gold_id=enc(gold)[0],competitor=competitor,
                         competitor_id=enc(competitor)[0],fidelity=fidelity))
    for l in range(23,39):
        add('D',l,fillers[l],INSTRUCTION,head,ending,'8','32',
            'published_24_token_endpoint' if l==24 else 'reconstructed_unpublished_natural_paraphrase')
    shell=(ROOT/'sources/lid-close-test.sh').read_text()
    cursor='setup_scenario docked '
    shell=shell[:shell.index(cursor)+len(cursor)]
    record=json.loads((ROOT/'sources/sql_record_2113.json').read_text())
    sql=record['sql']
    assert sql.count(' + sin(')==1
    replay=sql[:sql.index(' + sin(')+1]
    # Frozen reconstruction: bare +/- are scored, but models often emit space-prefixed
    # operators. README reports this boundary limitation and a separate diagnostic.
    control=sql.replace(' + sin(',' - sin(')
    sql_body=('-- Source request: '+record['sql_prompt']+'\n'
              '-- At the next arithmetic slot, write exactly + or -; only one reproduces q_replay.\n\n'
              +record['sql_context']+'\n\n'
              '-- q_control captured row multiset (different label):\n-- [[5626.688008276917]]\n'
              '-- q_control:\n'+control+'\n\n'
              '-- q_replay captured row multiset (authoritative):\n-- [[0.38403957581396597]]\n'
              '-- q_replay:\n'+replay)
    for family,body,instruction,gold,competitor in [
        ('omarchy',shell,'Continue the shell test with the correct next argument, preserving the stated scenario.','0','1'),
        ('sql',sql_body,'Continue q_replay with exactly + or - to reproduce its authoritative recorded output without changing the remaining SQL.','+','-')]:
        # Task-appropriate natural comments, lengths verified independently.
        options={}
        for noun,verb,end in itertools.product(
            ['This example','The following example','This source file','The following source file'],
            ['contains the context needed to continue the code correctly',
             'provides the complete context for the code completion task',
             'describes the behavior expected from the implementation'],
            ['.',' and should be read carefully.',' with comments explaining the intended behavior.',
             ' with comments explaining the intended behavior and the relevant assumptions.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve the documented behavior.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve documented behavior.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve the documented behavior of this code.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve the behavior described by the surrounding source code.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve exactly the behavior described by the surrounding source code.',
             ' with comments explaining the intended behavior and the relevant assumptions for this particular example. Preserve exactly the behavior described by the surrounding source code and its documentation.']):
            comment=('# ' if family=='omarchy' else '-- ')+noun+' '+verb+end
            options.setdefault(len(enc(comment)),comment)
        assert all(l in options for l in range(24,40)),(family,sorted(options))
        # Put the final line in the assistant prefill, preceding source in user context.
        split=body.rfind('\n')+1
        for l in range(24,40):
            add(family,l,options[l],instruction,body[:split],body[split:],gold,competitor,
                'reconstructed_full_prompt_and_natural_comments_from_pinned_published_source')
    path=ROOT/'prompts/additional_reconstructed.jsonl'
    temp=path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    temp.replace(path)
    temp.write_text(json.dumps(fillers,indent=2,ensure_ascii=False))
    temp.replace(ROOT/'prompts/additional_fillers.json')
    print(f'{len(rows)} additional reconstructed prompts; not bit-identical author inputs',flush=True)


if __name__=='__main__':
    main()
