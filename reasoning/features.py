"""Token counts, completion checks, diversity clusters, and optional student NLL."""
import argparse
import json
import math
import importlib.metadata
from collections import Counter
from pathlib import Path
from .data import boxed, digest, read_rows, write_json


def repetition(text):
    words = text.split()
    grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
    return 0. if not grams else 1 - len(set(grams)) / len(grams)


def token_ids(tokenizer, row):
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': row['system']},
        {'role': 'user', 'content': row['instruction']}], tokenize=True, return_dict=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template([
        {'role': 'system', 'content': row['system']},
        {'role': 'user', 'content': row['instruction']},
        {'role': 'assistant', 'content': row['output']}], tokenize=True, return_dict=False, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('chat template prompt is not a prefix; response loss mask would be invalid')
    return prompt, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='artifacts/prepared/pool.json')
    p.add_argument('--output', default='artifacts/features.json')
    p.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--revision', default='main')
    p.add_argument('--clusters', type=int, default=128)
    p.add_argument('--embedding-model', help='optional sentence-transformers model; default: TF-IDF/SVD')
    p.add_argument('--score-nll', action='store_true')
    p.add_argument('--nll-tokens', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from sklearn.cluster import MiniBatchKMeans
    rows = read_rows(a.input)
    if not rows or a.clusters < 1 or a.nll_tokens < 1:
        p.error('nonempty input and positive clusters/nll-tokens required')
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    texts = [r['instruction'] for r in rows]
    if a.embedding_model:
        from sentence_transformers import SentenceTransformer
        x = SentenceTransformer(a.embedding_model).encode(texts, normalize_embeddings=True, show_progress_bar=True)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        x = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=50000, sublinear_tf=True).fit_transform(texts)
        if min(x.shape) > 2:
            x = normalize(TruncatedSVD(n_components=min(128, min(x.shape)-1), random_state=a.seed).fit_transform(x))
    labels = MiniBatchKMeans(n_clusters=min(a.clusters, len(rows)), random_state=a.seed,
                             n_init=3, batch_size=1024).fit_predict(x)
    model = None
    if a.score_nll:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            p.error('--score-nll requires CUDA; omit it for the diversity-only ablation')
        model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
                   torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    provenance = dict(
        code_sha256=digest(__file__),
        tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'),
        model_commit=getattr(model.config, '_commit_hash', None) if model else None,
        versions={k: importlib.metadata.version(k) for k in ['transformers','scikit-learn','numpy']},
        local_tokenizer_files={p.name: digest(p) for p in Path(a.model).glob('*.json')} if Path(a.model).is_dir() else {})
    run_key = dict(input_sha256=digest(a.input), arguments=vars(a), provenance=provenance)
    partial = Path(a.output+'.partial.json')
    features = []
    if partial.exists():
        cached = json.loads(partial.read_text())
        if cached['run_key'] != run_key:
            raise ValueError('partial features belong to a different input/configuration')
        features = cached['features']
        if [f['id'] for f in features] != [r['id'] for r in rows[:len(features)]]:
            raise ValueError('invalid partial feature ordering')
    for i in range(len(features), len(rows)):
        row = rows[i]
        prompt, full = token_ids(tokenizer, row)
        f = dict(id=row['id'], cluster=int(labels[i]), total_tokens=len(full),
                 response_tokens=len(full)-len(prompt), complete_box=boxed(row['output']) is not None,
                 repetition=repetition(row['output']))
        if model is not None:
            end = min(len(full), len(prompt)+a.nll_tokens)
            if end > model.config.max_position_embeddings:
                f['nll'] = None
            else:
                ids = torch.tensor([full[:end]], device='cuda')
                targets = ids.clone()
                targets[:, :len(prompt)] = -100
                with torch.inference_mode():
                    f['nll'] = float(model(input_ids=ids, labels=targets).loss)
                if not math.isfinite(f['nll']):
                    raise ValueError(f'nonfinite NLL for {row["id"]}')
        features.append(f)
        if (i+1) % 1000 == 0:
            write_json(partial, dict(run_key=run_key, features=features))
            print(f'{i+1}/{len(rows)}', flush=True)
    write_json(a.output, features)
    write_json(a.output+'.meta.json', dict(input_sha256=digest(a.input), arguments=vars(a), provenance=provenance,
                resolved_tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'),
                resolved_model_commit=getattr(model.config, '_commit_hash', None) if model else None,
                clusters=dict(Counter(map(int, labels)))))
    partial.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
