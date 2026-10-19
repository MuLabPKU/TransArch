"""Plot saved observations against the paper; missing results stay missing.

Usage: bash python.sh plot_results.py results
Writes PDF and PNG figures and a CSV of observed code-probability differences.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from report import PAPER, wilson, read_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    refs=json.loads((Path(__file__).parent/'paper_code_reference.json').read_text())['deltas']
    for model in ['v4','v41']:
        source=args.results/f'{model}_all_code.jsonl'
        if source.exists() and source.stat().st_size:
            rows=read_rows(source)
            fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True,layout='constrained')
            comparisons=[]
            for ax,family in zip(axes.flat,['A','B','C','D','omarchy','sql']):
                observed=sorted([r for r in rows if r['family']==family],key=lambda r:r['filler_tokens'])
                if family in ['A','B','C']:
                    ax.plot(range(23,39),[refs[model][family][str(n)] for n in range(23,39)],
                            '--',color='0.5',label='Paper')
                elif family=='D':
                    ax.scatter([24],[refs[model]['D']['24']],marker='x',color='0.5',label='Paper (published input)')
                if observed:
                    ax.plot([r['filler_tokens'] for r in observed],[r['delta'] for r in observed],
                            'o-',markersize=3,label='Observed')
                ax.axhline(0,color='0.8',linewidth=.8)
                ax.set(title=f'{family}: {len(observed)}/16'+(' (reconstructed inputs)' if family not in 'ABC' else ''),
                       xticks=[23,26,29,32,35,38],ylim=(-1.05,1.05))
                ax.set_xlabel('Filler tokens')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7)
                for r in observed:
                    comparable=family in ['A','B','C'] or (family=='D' and r['filler_tokens']==24)
                    comparisons.append(dict(family=family,length=r['filler_tokens'],delta=r['delta'],
                        paper_delta=refs[model][family][str(r['filler_tokens'])] if comparable else '',
                        fidelity=r['fidelity']))
            for ax in axes[:,0]:
                ax.set_ylabel('P(gold) - P(competitor)')
            fig.suptitle(f'{model}: code completion ({len(rows)}/96 observed)')
            for ext in ['png','pdf']:
                fig.savefig(args.results/f'{model}_code_phase.{ext}',dpi=180)
            plt.close(fig)
            with (args.results/f'{model}_code_comparison.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['family','length','delta','paper_delta','fidelity'],lineterminator='\n')
                writer.writeheader(); writer.writerows(comparisons)
        source=args.results/f'{model}_niah.jsonl'
        if source.exists() and source.stat().st_size:
            rows=read_rows(source)
            fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
            ax.plot(range(8),PAPER[model],'x--',color='0.5',label='Paper')
            for residue in range(8):
                group=[r for r in rows if r['residue']==residue]
                if not group:
                    continue
                n=len(group); k=sum(r['correct'] for r in group)
                lo,hi=wilson(k,n); p=k/n
                ax.errorbar(residue,100*p,yerr=[[100*(p-lo)],[100*(hi-p)]],fmt='o',color='C0',capsize=3)
            expected=2048 if model=='v4' else 20480
            ax.set(title=f'{model}: 128K retrieval ({len(rows)}/{expected}; observed Wilson 95% CI)',
                   xlabel='Source-key position mod 8',ylabel='Answer-prefix accuracy (%)',xticks=range(8),ylim=(0,101))
            ax.legend()
            for ext in ['png','pdf']:
                fig.savefig(args.results/f'{model}_niah_phase.{ext}',dpi=180)
            plt.close(fig)


if __name__=='__main__':
    main()
