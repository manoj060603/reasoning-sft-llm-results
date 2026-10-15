"""Choose a pilot using held-out development only; never read public test scores."""
import argparse
from pathlib import Path
from .compare import load_run
from .metrics import summarize
from .data import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results');p.add_argument('--output',default='results/pilot-choice.json')
    a=p.parse_args()
    names=['filtered_random-sft','route-sft','route-dft','route-asft'];scores={}; reference=None
    for name in names:
        cfg,rows=load_run(Path(a.root)/f'pilot-{name}-dev')
        if set(rows)!={'acereason_dev'}:raise ValueError('only the held-out development set may choose a pilot')
        protocol={k:v for k,v in cfg.items() if k not in ('model','identity')}
        if reference is not None and protocol!=reference:raise ValueError('pilot protocols differ')
        reference=protocol;scores[name]=summarize(rows['acereason_dev'])
    # Tie favors the simpler control; this is a small screening experiment.
    winner=max(names,key=lambda name:(scores[name]['pass_at_8'],scores[name]['pass_at_1'],-names.index(name)))
    write_json(a.output,dict(winner=winner,development_scores=scores,
        criterion='held-out development pass@8, then pass@1, then simpler control',
        limitation='32-question pilot screening, not a statistically established best recipe'))
    print(winner)


if __name__=='__main__':main()
