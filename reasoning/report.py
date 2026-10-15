"""Summarize only complete model evaluations; list incomplete runs separately."""
import argparse
from pathlib import Path
from .compare import load_run
from .metrics import summarize
from .data import write_json
from .data import boxed
from .features import repetition


def summary_with_costs(rows):
    score=summarize(rows)
    if all('output_tokens' in row and 'finish_reasons' in row for row in rows):
        lengths=[length for row in rows for length in row['output_tokens']]
        score['generated_tokens']=sum(lengths)
        score['mean_output_tokens']=sum(lengths)/len(lengths)
        score['truncated_samples']=sum(reason=='length' for row in rows for reason in row['finish_reasons'])
    responses=[response for row in rows for response in row.get('generated_responses',[])]
    if responses:
        score['boxed_answer_samples']=sum(boxed(response) is not None for response in responses)
        score['mean_repeated_8gram_fraction']=sum(map(repetition,responses))/len(responses)
    return score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/arc-20261011/evaluations')
    p.add_argument('--output',default='results/arc-20261011/report.json')
    a=p.parse_args();report={'complete':{},'incomplete':{},'historical_leaderboard_beaten':None}
    for manifest in sorted(Path(a.root).glob('*/manifest.json')):
        name=manifest.parent.name
        try:
            cfg,data=load_run(manifest.parent)
            report['complete'][name]={b:summary_with_costs(rows) for b,rows in data.items()}
        except (ValueError,KeyError) as e:report['incomplete'][name]=str(e)
    report['note']='Development pilots and individual benchmarks are not a full historical leaderboard score.'
    write_json(a.output,report)
    import json
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
