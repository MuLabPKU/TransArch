"""Download pinned public resources and build the exact inputs used in this repo."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from paths import ROOT, MODEL_V4, MODEL_V41, FILLER_SOURCE

RESOURCES = json.loads((ROOT / 'resources.json').read_text())
MODELS = {'v4': MODEL_V4, 'v41': MODEL_V41}


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{path}: SHA256 mismatch ({actual}, expected {expected})')


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def download_models(selected, weights):
    from huggingface_hub import snapshot_download
    # Both tokenizers are needed to verify that the models see identical inputs.
    for name, path in MODELS.items():
        spec = RESOURCES['models'][name]
        patterns = list(spec['sha256']) + ['generation_config.json', 'chat_template.jinja', 'LICENSE']
        if weights and name in selected:
            patterns += ['*.safetensors', 'encoding/*', '*.py']
        snapshot_download(spec['repo_id'], revision=spec['revision'],
                          local_dir=path, allow_patterns=patterns)
        for filename, digest in spec['sha256'].items():
            check(path / filename, digest)
        if weights and name in selected:
            index = json.loads((path / 'model.safetensors.index.json').read_text())
            missing = {s for s in index['weight_map'].values() if not (path / s).is_file()}
            if missing:
                raise ValueError(f'Missing model shards: {sorted(missing)}')


def download_filler():
    spec = RESOURCES['filler']
    if not FILLER_SOURCE.exists():
        FILLER_SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temp = FILLER_SOURCE.with_suffix('.download')
        urllib.request.urlretrieve(spec['url'], temp)
        check(temp, spec['sha256'])
        temp.replace(FILLER_SOURCE)
    check(FILLER_SOURCE, spec['sha256'])


def prepare_code():
    for name, digest in RESOURCES['sources'].items():
        check(ROOT / name, digest)
    run('make_code_prompts.py')
    run('make_extra_prompts.py')
    dest = ROOT / 'prompts/all_code.jsonl'
    # Replace atomically; do not modify old hard-linked datasets in place.
    temp = dest.with_suffix('.tmp')
    temp.write_bytes((ROOT / 'prompts/fp8_abc.jsonl').read_bytes()
                     + (ROOT / 'prompts/additional_reconstructed.jsonl').read_bytes())
    check(temp, RESOURCES['inputs']['all_code_sha256'])
    temp.replace(dest)


def make_candidates():
    from tokenizers import Tokenizer
    from make_niah import QUERY
    tok = Tokenizer.from_file(str(MODEL_V4/'tokenizer.json'))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    result = []
    for i in range(10000, tok.get_vocab_size()):
        s = tok.decode([i], skip_special_tokens=False)
        if not s.isalpha() or len(s)>16 or enc(s)!=[i]:
            continue
        if enc('。'+s+':'+s+'。') != [320,i,28,i,320]:
            continue
        query = enc(QUERY.format(key=s))
        if len(query)!=33 or query.count(i)!=2:
            continue
        result.append(dict(id=i,text=s,query_ids=query))
    path=ROOT/'prompts/candidates.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False))
    temp.replace(path)
    print(f'{len(result)} stable candidates')


def prepare_niah(selected):
    download_filler()
    make_candidates()
    check(ROOT / 'prompts/candidates.json', RESOURCES['inputs']['candidates_sha256'])
    for name in selected:
        dest = ROOT / f'prompts/niah_{name}'
        if not (dest / 'validated.json').exists():
            args = ['make_niah.py', '--blocks', '1' if name == 'v4' else '10', '--output', dest]
            if (dest / 'metadata.jsonl').exists():
                args += ['--resume']
            run(*args)
        manifest = json.loads((dest / 'validated.json').read_text())
        if manifest['input_ids_sha256'] != RESOURCES['inputs']['niah_sha256'][name]:
            raise ValueError(f'{dest}: input manifest differs from the pinned experiment')
        # Independently hashes the actual array, not just its manifest.
        run('validate_niah.py', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['v4', 'v41', 'both'], default='both')
    parser.add_argument('--stage', choices=['all', 'code', 'niah'], default='all')
    parser.add_argument('--weights', action='store_true', help='Also download selected model weights')
    parser.add_argument('--offline', action='store_true', help='Verify and reuse already downloaded resources')
    args = parser.parse_args()
    selected = list(MODELS) if args.model == 'both' else [args.model]
    if args.offline:
        for name, path in MODELS.items():
            for filename, digest in RESOURCES['models'][name]['sha256'].items():
                check(path / filename, digest)
        if args.weights:
            parser.error('--weights and --offline cannot be combined')
    else:
        download_models(selected, args.weights)
    (ROOT / 'prompts').mkdir(exist_ok=True)
    if args.stage in ['all', 'code']:
        prepare_code()
    if args.stage in ['all', 'niah']:
        if args.offline and not FILLER_SOURCE.exists():
            raise FileNotFoundError('Offline mode requires assets/filler-train.txt.gz')
        prepare_niah(selected)
    print('Preparation and input checks completed.', flush=True)


if __name__ == '__main__':
    main()
