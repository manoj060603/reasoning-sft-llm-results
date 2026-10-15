"""Stream linear interpolation between matched Hugging Face safetensor checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha must be in [0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        receipt = args.output / "interpolation.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("alpha") == args.alpha:
            print(f"Reusing completed interpolation {args.output}")
            return
        raise ValueError(f"nonempty output directory: {args.output}")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_index_path = args.base / "model.safetensors.index.json"
    tuned_index_path = args.tuned / "model.safetensors.index.json"
    base_index = json.loads(base_index_path.read_text())
    tuned_index = json.loads(tuned_index_path.read_text())
    base_map = base_index["weight_map"]
    tuned_map = tuned_index["weight_map"]
    if set(base_map) != set(tuned_map):
        raise ValueError("base and tuned checkpoints have different tensor keys")

    args.output.mkdir(parents=True, exist_ok=True)
    copy_names = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "merges.txt", "vocab.json", "chat_template.jinja",
    }
    for name in copy_names:
        source = args.tuned / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    output_map: dict[str, str] = {}
    total_size = 0
    for shard_name in sorted(set(tuned_map.values())):
        keys = sorted(key for key, shard in tuned_map.items() if shard == shard_name)
        base_shards = sorted({base_map[key] for key in keys})
        tensors = {}
        with ExitStack() as stack:
            tuned_file = stack.enter_context(safe_open(args.tuned / shard_name, framework="pt", device="cpu"))
            base_files = {
                name: stack.enter_context(safe_open(args.base / name, framework="pt", device="cpu"))
                for name in base_shards
            }
            for key in keys:
                base_tensor = base_files[base_map[key]].get_tensor(key)
                tuned_tensor = tuned_file.get_tensor(key)
                if base_tensor.shape != tuned_tensor.shape:
                    raise ValueError(f"shape mismatch for {key}")
                if torch.is_floating_point(base_tensor):
                    tensor = torch.lerp(base_tensor.float(), tuned_tensor.float(), args.alpha).to(torch.bfloat16)
                    if not torch.isfinite(tensor.float()).all():
                        raise ValueError(f"nonfinite interpolated tensor: {key}")
                else:
                    if not torch.equal(base_tensor, tuned_tensor):
                        raise ValueError(f"nonfloating tensor differs: {key}")
                    tensor = base_tensor
                tensors[key] = tensor.contiguous()
                total_size += tensor.numel() * tensor.element_size()
                output_map[key] = shard_name
        save_file(tensors, args.output / shard_name, metadata={"format": "pt"})
        del tensors

    (args.output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": output_map,
    }, indent=2) + "\n")
    receipt = {
        "schema": "reasoning-sft-weight-interpolation-v1",
        "alpha": args.alpha,
        "formula": "base + alpha * (full_sft - base)",
        "base_index_sha256": digest(base_index_path),
        "tuned_index_sha256": digest(tuned_index_path),
        "output_index_sha256": digest(args.output / "model.safetensors.index.json"),
        "tensor_count": len(output_map),
        "total_size": total_size,
        "output_dtype": "bfloat16",
    }
    (args.output / "interpolation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
