"""Resumable fixed-protocol evaluation using the assignment's answer grader."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from .data import SYSTEM, digest, prompt_id, question, read_rows, write_json
from .metrics import summarize
from .benchmark import record_ids

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_identity(model, revision):
    path = Path(model)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and
                       (p.suffix in ('.json', '.safetensors', '.bin', '.model') or p.name=='merges.txt'))
        if not files:
            raise ValueError('no checkpoint files found')
        return {str(p.relative_to(path)): digest(p) for p in files}
    from huggingface_hub import model_info
    return {'repo': model, 'commit': model_info(model, revision=revision).sha}


def valid_result(row, expected_id):
    return (row.get('problem_id') == expected_id
            and len(row.get('generated_responses', [])) == 8
            and len(row.get('answers_correctness', [])) == 8
            and all(type(x) is bool for x in row['answers_correctness']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--lora-adapter',
                   help='optional PEFT adapter applied to --model through vLLM LoRA inference')
    p.add_argument('--revision', default='main')
    p.add_argument('--data-dir', default='artifacts/benchmarks')
    p.add_argument('--benchmarks', nargs='+', default=['aime','math','cn_math_2024','kaoyan','amc','minerva','olympiadbench','gpqa'])
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--max-num-seqs', type=int, default=32,
                   help='active decoding sequence cap; changing it requires matched fresh runs')
    p.add_argument('--record-hardware', action='store_true',
                   help='bind new experiments to GPU type, capability and memory in the manifest')
    p.add_argument('--gpu-memory-utilization', type=float, default=.9)
    p.add_argument('--max-new-tokens', type=int, default=32768,
                   help='use 32768 for the assignment benchmarks; smaller budgets are explicit screening runs')
    p.add_argument('--max-model-len', type=int, default=32768,
                   help='native total context limit; generation stops at this limit including prompt')
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch size must be positive')
    if a.max_new_tokens < 1:
        p.error('max-new-tokens must be positive')
    if a.max_num_seqs < 1:
        p.error('max-num-seqs must be positive')
    # Import assignment grader unchanged; no generated programs are executed.
    sys.path.insert(0, str(ROOT/'eval'))
    from utils.parser import extract_answer
    from utils.grader import check_is_correct
    from .grader_preflight import check_grader
    check_grader()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    identity = checkpoint_identity(a.model, a.revision)
    adapter_identity = checkpoint_identity(a.lora_adapter, None) if a.lora_adapter else None
    revision = identity.get('commit') if 'repo' in identity else None
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    all_data = {b: read_rows(Path(a.data_dir)/b/'test.jsonl') for b in a.benchmarks}
    config = dict(model=a.model, identity=identity, lora_adapter=a.lora_adapter,
                  lora_identity=adapter_identity, seed=a.seed, n=8, temperature=.6, top_p=.95,
                  max_tokens=a.max_new_tokens, max_model_len=a.max_model_len, system=SYSTEM,
                  generation_config='vllm', max_num_seqs=a.max_num_seqs, top_k=-1, min_p=0., repetition_penalty=1.,
                  batch_size=a.batch_size, tensor_parallel_size=a.tensor_parallel_size,
                  gpu_memory_utilization=a.gpu_memory_utilization,
                  versions={k: importlib.metadata.version(k) for k in ['vllm','transformers','torch','sympy','antlr4-python3-runtime','latex2sympy2']},
                  grader_sha256={f: digest(ROOT/'eval'/'utils'/f) for f in ['grader.py','parser.py','math_normalization.py']},
                  problem_counts={b: len(rows) for b, rows in all_data.items()},
                  problem_ids={b: record_ids(rows) for b, rows in all_data.items()},
                  data={b: digest(Path(a.data_dir)/b/'test.jsonl') for b in a.benchmarks})
    if a.record_hardware:
        import torch
        config['hardware'] = [dict(gpu=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)),
            total_memory=torch.cuda.get_device_properties(i).total_memory)
            for i in range(a.tensor_parallel_size)]
    out = Path(a.output)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != config:
            raise ValueError('run configuration/checkpoint/data changed; use a new output directory')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('nonempty output directory has no manifest')
        write_json(out/'manifest.json', config)
    llm = None
    metrics = {}
    for name, rows in all_data.items():
        ids = record_ids(rows)
        if len(set(ids)) != len(ids) or not rows:
            raise ValueError(f'{name}: empty data or duplicate record identities')
        done = []
        for start in range(0, len(rows), a.batch_size):
            batch = rows[start:start+a.batch_size]
            batch_ids = ids[start:start+a.batch_size]
            dest = out/name/f'batch_{start:06d}.json'
            if dest.exists():
                saved = read_rows(dest)
                if len(saved) != len(batch) or not all(valid_result(r,i) for r,i in zip(saved,batch_ids)):
                    raise ValueError(f'invalid cached batch {dest}')
                done.extend(saved)
                continue
            prompts = [tokenizer.apply_chat_template([
                {'role':'system', 'content':SYSTEM}, {'role':'user','content':question(r)}],
                tokenize=False, add_generation_prompt=True) for r in batch]
            for text in prompts:
                if len(tokenizer.encode(text, add_special_tokens=False)) >= a.max_model_len:
                    raise ValueError('prompt fills context; do not silently truncate the question')
            if llm is None:
                llm = LLM(model=a.model, revision=revision, tokenizer_revision=revision,
                          dtype='bfloat16', max_model_len=a.max_model_len, seed=a.seed,
                          tensor_parallel_size=a.tensor_parallel_size,
                          gpu_memory_utilization=a.gpu_memory_utilization, trust_remote_code=False,
                          generation_config='vllm', max_num_seqs=a.max_num_seqs,
                          enable_lora=bool(a.lora_adapter), max_lora_rank=64)
            # Per-question seeds make resumed batches independent of skipped RNG draws.
            params = [SamplingParams(n=8, temperature=.6, top_p=.95, max_tokens=a.max_new_tokens,
                       seed=(a.seed+int(pid[:8],16)) % (2**31)) for pid in batch_ids]
            lora_request = None
            if a.lora_adapter:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest('evaluation-adapter', 1, a.lora_adapter)
            generated = llm.generate(prompts, params, lora_request=lora_request)
            saved = []
            for row, pid, completion in zip(batch, batch_ids, generated):
                outputs = completion.outputs
                responses = [x.text for x in outputs]
                answers = [extract_answer(x, name) for x in responses]
                flags = [bool(check_is_correct(x, str(row['answer']))) for x in answers]
                result = dict(problem_id=pid, question=question(row), generated_responses=responses,
                    generated_answers=answers, gold_answer=str(row['answer']), answers_correctness=flags,
                    finish_reasons=[x.finish_reason for x in outputs],
                    output_tokens=[len(x.token_ids) for x in outputs])
                if not valid_result(result, pid):
                    raise ValueError('generation returned incomplete samples')
                saved.append(result)
            if len(saved) != len(batch):
                raise ValueError('generation returned incomplete batch')
            write_json(dest, saved)
            done.extend(saved)
        metrics[name] = summarize(done)
        metrics[name]['truncated_samples'] = sum(x=='length' for r in done for x in r['finish_reasons'])
        write_json(out/'metrics.json', metrics)
        print(name, metrics[name], flush=True)


if __name__ == '__main__':
    main()
