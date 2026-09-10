"""Plot run-level S10 results; dots denote independent process runs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def label(row):
    if row["backend"]=="zero":
        return f"ZeRO-{row['stage']} / {row['world']} GPUs"
    return f"TP{row['tp']} PP{row['pp']} / {row['world']} GPUs"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[r for r in json.loads(args.input.read_text()) if r['accepted']]
    groups=defaultdict(list)
    for row in rows:groups[(row['backend'],label(row))].append(row)
    fig, axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for column,backend in enumerate(('zero','megatron')):
        names=sorted(name for b,name in groups if b==backend)
        color='#2878B5' if backend=='zero' else '#C66A24'
        for line,(metric,title,unit) in enumerate((
            ('tokens_per_second','Throughput','input tokens / second'),
            ('max_rank_peak_allocated_gib','Peak allocated memory, busiest rank','GiB'))):
            ax=axes[line,column]
            values=[[r[metric] for r in groups[(backend,name)]] for name in names]
            ax.bar(range(len(names)),[statistics.median(v) for v in values],color=color,alpha=.7)
            for x,points in enumerate(values):
                ax.scatter([x+(i-(len(points)-1)/2)*.07 for i in range(len(points))],points,
                           color='black',s=22,zorder=3)
            ax.set_xticks(range(len(names)),names,rotation=15,ha='right')
            ax.set_ylabel(unit)
            ax.set_title(('DeepSpeed: ' if backend=='zero' else 'Megatron local backend: ')+title)
            ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle('S10 · 7B-class synthetic training on one 4× A100 SXM host\n'
                 '8192 input tokens/update · 20 warmup + 100 measured updates/run · dots = runs',fontsize=14)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
