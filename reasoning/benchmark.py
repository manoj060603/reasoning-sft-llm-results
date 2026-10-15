"""Stable benchmark record identities without dropping repeated questions."""
import hashlib
from collections import Counter, defaultdict
from .data import prompt_id, question


def record_ids(rows):
    hashes = [prompt_id(question(row)) for row in rows]
    counts = Counter(hashes)
    return [key if counts[key] == 1 else
            hashlib.sha256(f'{key}:record:{index}'.encode()).hexdigest()
            for index, key in enumerate(hashes)]


def duplicate_audit(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[prompt_id(question(row))].append(index)
    duplicates = [dict(question_hash=key, row_indices=indices,
                       answers=sorted({str(rows[i]['answer']) for i in indices}))
                  for key, indices in groups.items() if len(indices) > 1]
    return dict(records=len(rows), unique_questions=len(groups), duplicate_groups=duplicates,
                conflicting_duplicate_answers=any(len(g['answers']) > 1 for g in duplicates))
