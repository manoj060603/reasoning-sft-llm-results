"""Render completed pilot evidence; never substitute partial or synthetic scores."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reasoning.compare import load_run
from reasoning.metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/figures'))
    args = parser.parse_args()
    names = ['filtered_random-sft', 'route-sft', 'route-dft', 'route-asft']
    labels = ['Filtered-random\nSFT', 'Route-diverse\nSFT', 'Route-diverse\nDFT', 'Route-diverse\nanchored DFT']
    dev = []
    for name in names:
        _, rows = load_run(args.results / f'pilot-{name}-dev')
        dev.append(summarize(rows['acereason_dev']))
    amc = []
    for name in ['base-amc', 'pilot-route-asft-amc']:
        _, rows = load_run(args.results / name)
        amc.append(summarize(rows['amc']))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'axes.titleweight':'bold'})
    fig, axes = plt.subplots(1,2,figsize=(12,5), gridspec_kw={'width_ratios':[1.6,1]})
    for ax, stats, ticks, title in [(axes[0],dev,labels,'Held-out development screen'),
                                   (axes[1],amc,['Base model','Selected pilot'],'Matched AMC evaluation')]:
        x=np.arange(len(stats));width=.34
        for delta,key,label,color in [(-width/2,'pass_at_1','Pass@1','#24718f'),
                                       (width/2,'pass_at_8','Pass@8','#df953b')]:
            bars=ax.bar(x+delta,[100*s[key] for s in stats],width,label=label,color=color)
            ax.bar_label(bars,fmt='%.1f%%',fontsize=9,padding=3)
        ax.set_xticks(x,ticks);ax.set_ylim(0,85);ax.set_ylabel('Accuracy (%)')
        ax.set_title(title,pad=28);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].text(.5,1.025,'32 questions · 8 samples · 8K generation limit',ha='center',transform=axes[0].transAxes,fontsize=9)
    axes[1].text(.5,1.025,'40 questions · 8 samples · native 32K context',ha='center',transform=axes[1].transAxes,fontsize=9)
    axes[0].legend(loc='upper left',frameon=False)
    fig.suptitle('Measured 512-example pilots — full 15K model pending',fontsize=15,y=.98)
    fig.text(.055,.06,'AMC: +2.50 pp pass@8 (95% paired CI −10.00 to +15.00); −3.44 pp pass@1 (−9.06 to +2.19).',fontsize=10)
    fig.text(.055,.02,'Single training seed. Development selected the recipe; AMC is a mixed result, not an established accuracy win.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.065,right=.98,bottom=.24,top=.79,wspace=.28)
    args.output.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf']:
        fig.savefig(args.output / f'pilot-measured-results.{extension}',dpi=180)
    (args.output/'plotted-values.json').write_text(json.dumps({'development':dict(zip(names,dev)),'amc':dict(zip(['base','selected_pilot'],amc))},indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
