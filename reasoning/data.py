import hashlib
import json
import re
import unicodedata
from pathlib import Path

SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'


def read_rows(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = []
        for line_no, line in enumerate(text.split('\n'), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON; refusing to skip data') from e
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: expected records')
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    # Keep digits and mathematical operators: 2+3 and 2-3 are different tasks.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def prompt_id(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def question(row):
    for key in ('question', 'problem', 'Question', 'instruction', 'input'):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    raise ValueError('record has no recognized question')


def canonical(row):
    if 'instruction' in row and 'output' in row:
        q = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
        answer = row['output']
    elif 'messages' in row or 'conversations' in row:
        msgs = row.get('messages', row.get('conversations'))
        users = [m.get('content', m.get('value')) for m in msgs if m.get('role', m.get('from')) in ('user', 'human')]
        assistants = [m.get('content', m.get('value')) for m in msgs if m.get('role', m.get('from')) in ('assistant', 'gpt')]
        if len(users) != 1 or len(assistants) != 1:
            raise ValueError('only single-turn reasoning records are supported')
        q, answer = users[0], assistants[0]
    else:
        q = question(row)
        answer = row.get('response', row.get('solution', row.get('output')))
    if not isinstance(q, str) or not isinstance(answer, str) or not q.strip() or not answer.strip():
        raise ValueError('empty or unsupported question/response')
    return dict(id=prompt_id(q), instruction=q.strip(), input='', output=answer.strip(), system=SYSTEM)


def boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return text[start:i].strip() or None
    return None


def alpaca(rows):
    return [{k: r[k] for k in ('instruction', 'input', 'output', 'system')} for r in rows]
