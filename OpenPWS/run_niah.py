"""Evaluate validated 128K inputs through vLLM with greedy eight-token answers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from inference import add_arguments, append_result, engine_options, identity, load_engine, pending_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    options=engine_options(args,128008)
    manifest=json.loads((args.data/'validated.json').read_text())
    audit=json.loads((args.data/'independent_validation.json').read_text())
    if not audit['all_checks_passed'] or audit['input_ids_sha256']!=manifest['input_ids_sha256']:
        raise ValueError('NIAH inputs have not passed independent validation')
    run_identity=identity(args,options,data_sha256=manifest['input_ids_sha256'])
    rows=[json.loads(s) for s in (args.data/'metadata.jsonl').read_text().splitlines()]
    ids=np.load(args.data/'input_ids.npy',mmap_mode='r')
    if ids.shape!=(manifest['prompts'],128000) or len(rows)!=manifest['prompts']:
        raise ValueError('NIAH shape or metadata count mismatch')
    rows=pending_rows(rows,args.output,run_identity,args.limit)
    if not rows:
        print('All selected NIAH prompts already completed',flush=True)
        return
    llm=load_engine(args,options,run_identity,__file__,data_manifest=manifest,
                    temperature=0,max_tokens=8,metric='completion.strip().startswith(gold)')
    from vllm import SamplingParams
    params=SamplingParams(temperature=0,max_tokens=8)
    total=len(rows)
    for index,row in enumerate(rows,1):
        start=time.monotonic()
        out=llm.generate([{'prompt_token_ids':ids[row['row']].tolist()}],params,use_tqdm=False)[0].outputs[0]
        result=dict(row,completion=out.text,output_ids=list(out.token_ids),
                    correct=out.text.strip().startswith(row['gold']),seconds=time.monotonic()-start)
        append_result(args.output,result)
        print(f"{index}/{total} {row['id']} correct={result['correct']} seconds={result['seconds']:.2f}",flush=True)


if __name__=='__main__':
    main()
