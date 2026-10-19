"""Score code prompts with vLLM using full-vocabulary probabilities; resume by ID."""
import argparse
import json
import math
from pathlib import Path
import time

from inference import add_arguments, append_result, digest, engine_options, identity, load_engine, pending_rows
from paths import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--prompts',type=Path,default=ROOT/'prompts/all_code.jsonl')
    args=parser.parse_args()
    options=engine_options(args,2048)
    run_identity=identity(args,options,prompt_sha256=digest(args.prompts))
    rows=[json.loads(line) for line in args.prompts.read_text().splitlines()]
    rows=pending_rows(rows,args.output,run_identity,args.limit)
    if not rows:
        print('All selected code prompts already completed',flush=True)
        return
    llm=load_engine(args,options,run_identity,__file__,temperature=0,max_tokens=1,
                    probabilities='full-vocabulary softmax, not candidate normalization')
    from vllm import SamplingParams
    for row in rows:
        start=time.monotonic()
        params=SamplingParams(temperature=0,max_tokens=1,
                              logprob_token_ids=[row['gold_id'],row['competitor_id']])
        out=llm.generate([{'prompt_token_ids':row['input_ids']}],params,use_tqdm=False)[0].outputs[0]
        lp=out.logprobs[0]
        gold=math.exp(lp[row['gold_id']].logprob)
        competitor=math.exp(lp[row['competitor_id']].logprob)
        top1=out.token_ids[0]
        result={k:v for k,v in row.items() if k not in ('prompt','input_ids')}
        result.update(p_gold=gold,p_competitor=competitor,delta=gold-competitor,
                      top1_id=top1,correct=top1==row['gold_id'],seconds=time.monotonic()-start)
        append_result(args.output,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
