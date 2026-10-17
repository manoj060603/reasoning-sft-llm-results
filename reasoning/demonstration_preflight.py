"""CPU verification of actual LLaMA-Factory curriculum, targets, and EOS labels."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .data import digest, prompt_id, read_rows, write_json


def verify(model, directory, require_audit=True):
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    directory = Path(directory)
    rows = read_rows(directory / "advanced.json")
    selection = json.loads((directory / "selection_manifest.json").read_text())
    if len(rows) != 15000 or len(set(selection["selected_ids"])) != 15000:
        raise ValueError("training dataset is not exactly 15K unique identities")
    if digest(directory / "advanced.json") != selection["data_sha256"]:
        raise ValueError("training dataset changed after export")
    if require_audit:
        audit = json.loads((directory / "audit_receipt.json").read_text())
        if not audit["passed"] or audit["selection_sha256"] != digest(directory / "selection_manifest.json"):
            raise ValueError("missing positive frozen preparation audit")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    data_args = DataArguments(template="qwen", cutoff_len=8192, train_on_prompt=False, packing=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
                                          processor=None, data_args=data_args)
    max_length, supervised = 0, 0
    for i, row in enumerate(rows):
        if prompt_id(row["instruction"]) != selection["selected_ids"][i]:
            raise ValueError("curriculum identity order changed")
        prompt = [{"role": "user", "content": row["instruction"]}]
        response = [{"role": "assistant", "content": row["output"]}]
        pairs = template.encode_multiturn(tokenizer, prompt + response, row["system"], None)
        if len(pairs) != 1:
            raise ValueError("unexpected multi-turn training record")
        source, target = pairs[0]
        expected_target = target + ([tokenizer.eos_token_id] if template.efficient_eos else [])
        ids, labels = processor._encode_data_example(prompt, response, row["system"], None, [], [], [])
        actual = [x for x in labels if x != -100]
        if actual != expected_target or len(ids) > 8192 or ids[:len(source)] != source:
            raise ValueError(f"training truncation or mask drift at row {i}")
        if labels[:len(source)] != [-100] * len(source) or actual.count(tokenizer.eos_token_id) < 1:
            raise ValueError(f"incorrect prompt mask or missing supervised EOS at row {i}")
        if not tokenizer.decode(actual, skip_special_tokens=False).startswith(row["output"]):
            raise ValueError(f"original response content lost at row {i}")
        max_length = max(max_length, len(ids)); supervised += len(actual)
    trainer = SimpleNamespace(finetuning_args=SimpleNamespace(disable_shuffling=True), train_dataset=rows)
    sampler = CustomSeq2SeqTrainer._get_train_sampler(trainer)
    if list(sampler) != list(range(15000)):
        raise ValueError("actual trainer sampler does not preserve the curriculum")
    categories = [x["category"] for x in selection["selected"]]
    if categories[:7500].count("manageable") != 6000 or categories[7500:].count("missed") != 6000:
        raise ValueError("curriculum acquisition stages differ")
    receipt = {"examples": 15000, "unique_prompts": 15000, "updates": 1875,
               "all_original_targets_complete": True, "all_eos_supervised": True,
               "prompt_tokens_masked": True, "sequential_sampler_verified": True,
               "max_lmf_tokens": max_length, "supervised_tokens": supervised,
               "selection_sha256": digest(directory / "selection_manifest.json"),
               "data_sha256": digest(directory / "advanced.json"),
               "audit_required": require_audit}
    write_json(directory / "training_preflight.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    verify(args.model, args.directory)


if __name__ == "__main__":
    main()
