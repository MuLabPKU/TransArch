"""Implement the fully specified Appendix B.2 FP8 A/B/C prompt construction."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
from paths import MODEL_V4, MODEL_V41
INSTRUCTION = ('Continue the function without changing its block-wise quantization semantics; '
               'avoid a redundant nested cast')


def build():
    model_path = MODEL_V4
    tok = AutoTokenizer.from_pretrained(model_path)
    other = AutoTokenizer.from_pretrained(MODEL_V41)
    source = (model_path / 'inference/kernel.py').read_text()
    marker = 'T.Cast(compute_dtype, T.Cast(FP'
    source_prefix = source[:source.index(marker) + len(marker)]
    code_ids = tok.encode(source_prefix, add_special_tokens=False)
    assert len(code_ids) == 983 and code_ids[-1] == 20199
    assert tok.encode('8', add_special_tokens=False) == [26]
    assert tok.encode('32', add_special_tokens=False) == [2111]
    head, prefill = tok.decode(code_ids[:974]), tok.decode(code_ids[974:])
    assert prefill == 'compute_dtype, T.Cast(FP'
    rows = []
    for family in 'ABC':
        for length in range(23, 39):
            count = length - 15 if family in 'AB' else length - 8
            if family == 'A':
                filler = ('"""This module contains optimized tensor kernels used by the model inference runtime.\n'
                          + ' =' * count + '\n"""')
            elif family == 'B':
                filler = ('# This module contains optimized tensor kernels used by the model inference runtime.\n#'
                          + ' =' * count)
            else:
                filler = '# Internal model inference kernel implementations.\n#' + ' *' * count
            assert len(tok.encode(filler, add_special_tokens=False)) == length
            prompt = ('<｜begin▁of▁sentence｜><｜User｜>' + INSTRUCTION + '\n\n' + filler
                      + '\n\n' + head + '<｜Assistant｜></think>' + prefill)
            ids = tok.encode(prompt, add_special_tokens=False)
            assert len(ids) == 1004 + length and ids[-1] == 20199
            assert ids == other.encode(prompt, add_special_tokens=False)
            assert tok.encode(tok.decode(ids), add_special_tokens=False) == ids
            rows.append({'id': f'fp8_{family}_{length}', 'experiment': 'fp8',
                         'family': family, 'filler_tokens': length, 'target_position': len(ids),
                         'prompt': prompt, 'input_ids': ids, 'gold': '8', 'gold_id': 26,
                         'competitor': '32', 'competitor_id': 2111,
                         'fidelity': 'matches_all_published_text_and_token_coordinates'})
    output = ROOT / 'prompts/fp8_abc.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    temp.replace(output)
    metadata = {'paper': '2609.36322 Appendix B.2', 'prompts': len(rows),
                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'tokenizer_sha256': hashlib.sha256((model_path/'tokenizer.json').read_bytes()).hexdigest(),
                'same_input_ids_for_both_models': True, 'thinking': False,
                'score': 'full-vocabulary softmax probabilities, P(8)-P(32), and full-vocabulary top-1',
                'missing_published_material': ['The 15 family D rephrasings other than the published 24-token endpoint',
                    'Complete adapted Omarchy and SQL prompts and filler sweeps']}
    temp.write_text(json.dumps(metadata, indent=2)+'\n')
    temp.replace(output.with_suffix('.metadata.json'))
    print(f'Validated {len(rows)} prompts; positions 1027..1042; identical model token IDs.')


if __name__ == '__main__':
    build()
