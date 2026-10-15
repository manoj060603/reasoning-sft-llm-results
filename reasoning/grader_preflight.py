"""Check supported symbolic equivalences before any scored GPU generations."""
import sys
from pathlib import Path


def check_grader():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
    from utils.grader import check_is_correct
    cases=[('0.5',r'\frac{1}{2}',True),(r'\sqrt{4}','2',True),('x+x','2*x',True),
           ('2','2',True),('3','4',False),('A','B',False)]
    # Warm the parser before the original grader's three-second process timeout.
    for pred,gold,want in cases:
        if bool(check_is_correct(pred,gold,timeout=False))!=want:
            raise RuntimeError(f'grader preflight failed: {pred!r}, {gold!r}')
    for pred,gold,want in cases:
        if bool(check_is_correct(pred,gold))!=want:
            raise RuntimeError(f'timed grader preflight failed: {pred!r}, {gold!r}')
    return {'cases':len(cases),'timed_and_untimed':'passed'}


if __name__=='__main__':print(check_grader())
