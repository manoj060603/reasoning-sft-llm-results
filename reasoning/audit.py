"""Audit and canonicalize archived benchmarks without changing records."""
import argparse
import json
from pathlib import Path
from .data import digest, question, read_rows, write_json
from .benchmark import duplicate_audit

EXPECTED = dict(aime=30, math=500, amc=40, cn_math_2024=30, kaoyan=199,
                minerva=272, olympiadbench=675, gpqa=198)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='eval/data')
    p.add_argument('--output', default='artifacts/benchmarks')
    a = p.parse_args()
    report = {}
    for name, expected in EXPECTED.items():
        path = Path(a.input)/name/'test.jsonl'
        text = path.read_text()
        rows = read_rows(path)
        if len(rows) != expected:
            raise ValueError(f'{name}: expected {expected}, got {len(rows)}')
        for row in rows:
            question(row)
            if 'answer' not in row or row['answer'] is None:
                raise ValueError(f'{name}: missing answer')
        dest = Path(a.output)/name/'test.jsonl'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        report[name] = dict(count=len(rows), unicode_line_separators=text.count(chr(0x2028)),
                            original_sha256=digest(path), clean_sha256=digest(dest),
                            duplicate_prompt_audit=duplicate_audit(rows))
    report['limitations'] = ['AIME25 is not supplied: historical leaderboard AVG cannot be reproduced.']
    write_json(Path(a.output)/'audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
