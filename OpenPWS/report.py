"""Rebuild README tables from saved evidence; optionally export local results/plots.

The default command needs only Python's standard library and the tracked reports/.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from paths import ROOT

PAPER = {'v4':[80.47,88.67,98.05,96.48,81.25,90.23,99.61,96.09],
         'v41':[95.00,89.22,95.12,89.69,94.69,90.31,95.31,89.73]}


def wilson(k,n):
    z=1.96
    p=k/n
    c=(p+z*z/(2*n))/(1+z*z/n)
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [c-h,c+h]


START = '<!-- REPORT:START -->'
END = '<!-- REPORT:END -->'
REPORTS = ROOT / 'reports'
FAMILIES = ['A', 'B', 'C', 'D', 'omarchy', 'sql']


def read_rows(path):
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    # A running evaluator may be in the middle of appending its last row.
    if text and not text.endswith('\n'):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'{path}: duplicate result IDs')
    return rows


def expected_ids(task, model):
    if task == 'all_code':
        return {f'fp8_{f}_{n}' if f in 'ABC' else f'{f}_{n}_reconstructed'
                for f in FAMILIES for n in (range(23, 39) if f in 'ABCD' else range(24, 40))}
    blocks = 1 if model == 'v4' else 10
    return {f'niah_b{b:02d}_{i:03d}_r{r}' for b in range(blocks) for i in range(256) for r in range(8)}


def validate(rows, task, model):
    allowed = expected_ids(task, model)
    if not {r['id'] for r in rows}.issubset(allowed):
        raise ValueError(f'{model} {task}: unexpected IDs')
    for row in rows:
        if task == 'all_code':
            family, n = row['family'], row['filler_tokens']
            expected = f'fp8_{family}_{n}' if family in 'ABC' else f'{family}_{n}_reconstructed'
            correct = row['top1_id'] == row['gold_id']
        else:
            expected = f"niah_b{row['block']:02d}_{row['binding']:03d}_r{row['residue']}"
            correct = row['completion'].strip().startswith(row['gold'])
            if row['source_position'] % 8 != row['residue']:
                raise ValueError('Incorrect phase label')
        if row['id'] != expected or row['correct'] != correct:
            raise ValueError(f'Inconsistent result: {row["id"]}')
    return len(rows) == len(allowed)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if k not in ['host', 'hostname']}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and value.startswith('/'):
        return '<local>/' + Path(value).name
    return value


def export_results(source):
    for model in ['v4', 'v41']:
        for task in ['all_code', 'niah']:
            src = source / f'{model}_{task}.jsonl'
            rows = read_rows(src)
            if not rows:
                continue
            complete = validate(rows, task, model)
            provenance = json.loads(src.with_suffix('.provenance.json').read_text())
            if task == 'all_code':
                expected_hash = json.loads((ROOT / 'resources.json').read_text())['inputs']['all_code_sha256']
                if provenance.get('prompt_sha256') != expected_hash:
                    raise ValueError('Report requires the pinned 96 code inputs')
            else:
                expected_hash = json.loads((ROOT / 'resources.json').read_text())['inputs']['niah_sha256'][model]
                if provenance.get('data_sha256') != expected_hash:
                    raise ValueError('Report requires the pinned NIAH inputs')
            dest = REPORTS / src.name
            # Do not silently replace a complete published run with a smoke test.
            old = read_rows(dest)
            if old and validate(old, task, model) and not complete:
                raise ValueError(f'Refusing to replace complete {dest.name} with partial results')
            compact = [{k: v for k, v in r.items() if k not in ['prefill', 'postfill', 'boundaries']}
                       for r in rows]
            atomic_text(dest, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in compact))
            provenance = sanitize(provenance)
            provenance.update(evidence_status='current_vllm_run', complete=complete)
            atomic_text(dest.with_suffix('.provenance.json'), json.dumps(provenance, indent=2) + '\n')


def score(rows):
    return f"{sum(r['correct'] for r in rows)}/{len(rows)} ({sum(r['correct'] for r in rows)/len(rows):.2%})" if rows else 'Not run'


def build_report():
    code = {m: read_rows(REPORTS / f'{m}_all_code.jsonl') for m in ['v4', 'v41']}
    old = read_rows(REPORTS / 'reference/v4_official_code.jsonl')
    stats = {}
    lines = ['## Experimental results', '',
             'These tables are computed from the saved per-example results by `python3 report.py`. Incomplete runs show progress only; missing observations are never replaced with values from the paper.', '',
             '### Code completion (96 prompts per model)', '',
             '| Run | Completed | Full-vocabulary top-1 accuracy | Median time per prompt |',
             '|---|---:|---:|---:|']
    for label, model, rows in [('V4 official backend, TP4 (historical reference)', 'v4_official', old),
                               ('V4 vLLM TP8', 'v4', code['v4']), ('V4.1 vLLM TP8', 'v41', code['v41'])]:
        if rows:
            complete = validate(rows, 'all_code', 'v41' if model == 'v41' else 'v4')
            median = statistics.median(r['seconds'] for r in rows)
            stats[model] = dict(n=len(rows), correct=sum(r['correct'] for r in rows), complete=complete,
                                median_seconds=median, first_seconds=rows[0]['seconds'],
                                scoring_seconds=sum(r['seconds'] for r in rows))
            lines += [f'| {label} | {len(rows)}/96 | {score(rows)} | {median:.4f} s |']
        else:
            lines += [f'| {label} | 0/96 | Not run | — |']
    if (REPORTS / 'v41_all_code.provenance.json').exists():
        provenance = json.loads((REPORTS / 'v41_all_code.provenance.json').read_text())
        if provenance.get('evidence_status') == 'historical_vllm_run':
            lines += ['', 'The V4.1 entries are from an earlier completed vLLM run. They have not yet been rerun as part of this backend migration; the full reproduction command recomputes them.']
    lines += ['', '| Prompt family | V4 official reference | V4 vLLM | V4.1 vLLM |', '|---|---:|---:|---:|']
    for family in FAMILIES:
        lines += ['| ' + family + ' | ' + ' | '.join(score([r for r in rows if r['family'] == family])
                                                    for rows in [old, code['v4'], code['v41']]) + ' |']
    lines += ['', '**SQL token-boundary limitation:** this reconstruction scores bare `+` (ID 13) and `-` (ID 15), '
              'whereas the saved predictions use space-prefixed ` +` (ID 940) and ` -` (ID 565). The resulting zero strict token accuracy '
              'does not mean that every predicted operator is semantically wrong. The diagnostic below covers these known spellings only; it neither replaces the original metric nor estimates unrecorded probabilities.', '',
              '| Model | Top-1 is bare `+` or ` +` | Interpretation |', '|---|---:|---|']
    for model, rows in code.items():
        sql = [r for r in rows if r['family'] == 'sql']
        if sql:
            plus = sum(r['top1_id'] in [13, 940] for r in sql)
            stats[model]['sql_boundary_diagnostic'] = dict(n=len(sql), bare_or_space_prefixed_plus=plus,
                                                          accepted_token_ids=[13, 940])
            lines += [f'| {model} | {plus}/{len(sql)} | Full prompt and token boundary are not aligned with the original author inputs; no quantitative replication claim |']
    refs = json.loads((ROOT / 'paper_code_reference.json').read_text())['deltas']
    lines += ['', '**Direct comparison with the published A/B/C construction (48 prompts)**', '',
              '| Model | Paper: reference token selected | This run: reference token selected | Matching signs of Δ | Mean absolute difference in Δ |',
              '|---|---:|---:|---:|---:|']
    for model, rows in code.items():
        abc = [r for r in rows if r['family'] in 'ABC']
        if len(abc) != 48:
            continue
        paper_deltas = [refs[model][r['family']][str(r['filler_tokens'])] for r in abc]
        sign = lambda v: (v > 0) - (v < 0)
        agreement = sum(sign(r['delta']) == sign(d) for r, d in zip(abc, paper_deltas))
        mae = statistics.mean(abs(r['delta'] - d) for r, d in zip(abc, paper_deltas))
        lines += [f"| {model} | {sum(d > 0 for d in paper_deltas)}/48 | {sum(r['correct'] for r in abc)}/48 | {agreement}/48 | {mae:.4f} |"]
        stats[model]['paper_abc'] = dict(n=48, sign_agreement=agreement, delta_mae=mae)
    lines += ['', 'The paper rounds Δ to three decimal places, which limits the precision of this comparison. Δ=0 is treated as a tie; top-1 accuracy uses the actual generated token. The other D paraphrases and the full Omarchy and SQL prompts are unpublished, '
              'so those supplementary experiments are excluded from this per-example agreement calculation. The paper reports a predominantly four-token cycle for V4 and odd/even alternation for V4.1 in A/B/C. The plots below compare the observed probability differences.']
    lines += ['', 'The published **D endpoint at L=24**:', '', '| Model | Paper Δ | Observed Δ | Observed top-1 correct |', '|---|---:|---:|---|']
    for model, rows in code.items():
        endpoint = next((r for r in rows if r['family'] == 'D' and r['filler_tokens'] == 24), None)
        if endpoint:
            lines += [f"| {model} | {refs[model]['D']['24']:+.3f} | {endpoint['delta']:+.6f} | {'Yes' if endpoint['correct'] else 'No'} |"]
    if old and validate(code['v4'], 'all_code', 'v4'):
        previous = {r['id']: r for r in old}
        changed = [r for r in code['v4'] if previous[r['id']]['top1_id'] != r['top1_id']]
        agreement = len(code['v4']) - len(changed)
        ratio = stats['v4_official']['median_seconds'] / stats['v4']['median_seconds']
        lines += ['', f"Top-1 agreement between the two V4 backends is {agreement}/96 ({agreement/96:.2%}); the ratio of median per-prompt times is {ratio:.2f}×. "
                  'The runs use different GPU counts (4 versus 8) and quantized kernels, so this ratio is not an isolated measurement of backend speedup. '
                  'Per-prompt timing includes on-demand compilation and excludes model loading; see `reports/metrics.json` for the timing statistics.', '',
                  'Prompts with changed top-1 predictions: ' + ', '.join(f"`{r['id']}`" for r in changed) + '.']
        stats['v4_backend_comparison'] = dict(top1_agreement=agreement, changed_ids=[r['id'] for r in changed],
                                             median_seconds_ratio=ratio)
    lines += ['', 'The current V4 environment uses SM90 hardware with MARLIN MXFP4 expert kernels, FlashMLA FP8 KV storage, and an FP8 indexer. '
              'The historical official backend simulates an FP4 indexer; the MXFP4 indexer in vLLM 0.31.0 requires SM10x hardware. '
              'The numerical results differ, and these experiments do not isolate indexer precision as the cause.', '']
    for model in ['v4', 'v41']:
        figure = f'reports/{model}_code_phase.png'
        if (ROOT / figure).exists():
            lines += [f'![{model} code phase]({figure})', '']
    lines += ['### 128K key-value retrieval', '',
              'The primary metric is exactly `completion.strip().startswith(gold)`. Each binding is tested in eight source-position residue groups, '
              'with 256 prompts per group for V4 and 2,560 for V4.1. The gap across groups is reported only after all groups are complete.', '',
              '| Model | Saved / expected prompts | Status | Strict accuracy on saved prompts only |', '|---|---:|---|---:|']
    niah = {}
    for model in ['v4', 'v41']:
        rows = read_rows(REPORTS / f'{model}_niah.jsonl')
        complete = validate(rows, 'niah', model)
        expected = len(expected_ids('niah', model))
        niah[model] = rows
        stats[f'{model}_niah'] = dict(n=len(rows), expected=expected, complete=complete)
        state = 'Complete' if complete else ('Incomplete; no full retrieval conclusion' if rows else 'No observations yet')
        lines += [f'| {model} | {len(rows):,}/{expected:,} | {state} | {score(rows)} |']
    for model, rows in niah.items():
        if not stats[f'{model}_niah']['complete']:
            continue
        lines += ['', f'**{model}: complete results by residue** (95% Wilson intervals)', '',
                  '| Source position mod 8 | Correct / total | Accuracy | 95% interval | Paper accuracy |', '|---|---:|---:|---:|---:|']
        accuracies = []
        groups = []
        for phase in range(8):
            group = [r for r in rows if r['residue'] == phase]
            k, n = sum(r['correct'] for r in group), len(group)
            lo, hi = wilson(k, n)
            accuracies.append(k / n)
            groups.append(dict(phase=phase, n=n, correct=k, accuracy=k/n, wilson95=[lo, hi]))
            lines += [f'| {phase} | {k}/{n} | {k/n:.2%} | {lo:.2%}–{hi:.2%} | {PAPER[model][phase]:.2f}% |']
        gap = 100 * (max(accuracies) - min(accuracies))
        relaxed = sum(r['completion'].lstrip(':： \t\r\n').startswith(r['gold']) for r in rows)
        stats[f'{model}_niah'].update(groups=groups, gap_pp=gap, leading_colon_removed_correct=relaxed)
        lines += ['', f'The maximum gap across residue groups is {gap:.2f} percentage points. As a separate formatting diagnostic, removing leading colons and whitespace gives {relaxed}/{len(rows)} correct answers. '
                  'This diagnostic does not replace the primary metric and is not used for the primary comparison with the paper.', '']
        if (ROOT / f'reports/{model}_niah_phase.png').exists():
            lines += [f'![{model} retrieval phase](reports/{model}_niah_phase.png)', '']
    lines += ['', 'The paper reports retrieval gaps of 19.14 percentage points for V4 and 6.09 for V4.1. Prompt text, random seeds, and sampled records here reconstruct the published protocol, '
              'but cannot be assumed identical to the unpublished author inputs. No phase-effect conclusion is drawn from an incomplete run.']
    return '\n'.join(lines), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-results', type=Path, help='Export sanitized result evidence before reporting')
    parser.add_argument('--plots', action='store_true', help='Rebuild PNG/PDF/CSV; requires matplotlib')
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / '.report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.from_results:
            export_results(args.from_results)
        if args.plots:
            subprocess.run([sys.executable, str(ROOT / 'plot_results.py'), str(REPORTS)], check=True)
        report, stats = build_report()
        readme = ROOT / 'README.md'
        before, rest = readme.read_text().split(START, 1)
        _, after = rest.split(END, 1)
        atomic_text(readme, before + START + '\n\n' + report + '\n\n' + END + after)
        atomic_text(REPORTS / 'metrics.json', json.dumps(stats, indent=2) + '\n')
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(REPORTS.rglob('*')) if p.is_file() and p.suffix in ['.jsonl', '.json']
                  and p.name != 'sha256.json'}
        atomic_text(REPORTS / 'sha256.json', json.dumps(hashes, indent=2) + '\n')
        print('Updated README.md and reports/metrics.json from saved evidence.')


if __name__ == '__main__':
    main()
