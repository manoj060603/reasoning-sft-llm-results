"""Stream a layerwise TIES merge of matched Hugging Face checkpoints.

The implementation keeps memory bounded by processing one tensor and then one
flat chunk at a time.  For very large tensors the magnitude cutoff is estimated
from a deterministic uniform sample; the receipt records that approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path


COPY_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "merges.txt", "vocab.json", "chat_template.jinja",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def magnitude_cutoff(delta, density: float, sample_size: int = 1_000_000) -> float:
    """Return a deterministic approximate cutoff retaining ``density`` mass entries."""
    import torch

    flat = delta.reshape(-1)
    if flat.numel() <= sample_size:
        sample = flat
    else:
        # Uniform deterministic coverage avoids RNG and makes restarts bitwise stable.
        step = max(1, flat.numel() // sample_size)
        sample = flat[::step][:sample_size]
    return float(torch.quantile(sample.float().abs(), 1.0 - density))


def ties_merge_tensor(base, experts, density: float, scale: float, chunk_size: int = 8_000_000):
    """Layerwise TIES: trim, magnitude-elect signs, disjoint-mean, then rebase."""
    import torch

    if not experts:
        raise ValueError("at least one expert is required")
    if any(t.shape != base.shape for t in experts):
        raise ValueError("shape mismatch")
    if not torch.is_floating_point(base):
        if any(not torch.equal(base, t) for t in experts):
            raise ValueError("nonfloating tensor differs")
        return base.clone()

    base_flat = base.reshape(-1)
    expert_flats = [t.reshape(-1) for t in experts]
    cutoffs = [magnitude_cutoff(t.float() - base_flat.float(), density) for t in expert_flats]
    output = torch.empty_like(base_flat, dtype=torch.bfloat16)
    for start in range(0, base_flat.numel(), chunk_size):
        stop = min(start + chunk_size, base_flat.numel())
        b = base_flat[start:stop].float()
        deltas = []
        for expert, cutoff in zip(expert_flats, cutoffs):
            d = expert[start:stop].float() - b
            d = torch.where(d.abs() >= cutoff, d, torch.zeros_like(d))
            deltas.append(d)
        stacked = torch.stack(deltas)
        elected = stacked.sum(0).sign()
        aligned = (stacked.sign() == elected.unsqueeze(0)) & (stacked != 0)
        count = aligned.sum(0)
        merged = (stacked * aligned).sum(0) / count.clamp_min(1)
        merged = torch.where(count > 0, merged, torch.zeros_like(merged))
        out = b + scale * merged
        if not torch.isfinite(out).all():
            raise ValueError("nonfinite merged tensor")
        output[start:stop] = out.to(torch.bfloat16)
    return output.reshape(base.shape).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expert", type=Path, action="append", required=True)
    parser.add_argument("--density", type=float, default=0.20)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.density <= 1.0:
        parser.error("density must be in (0, 1]")
    if args.scale < 0.0:
        parser.error("scale must be nonnegative")
    if len(args.expert) < 2:
        parser.error("TIES requires at least two expert checkpoints")
    receipt_path = args.output / "ties_merge.json"
    if args.output.exists() and any(args.output.iterdir()):
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("density") == args.density and receipt.get("scale") == args.scale:
                print(f"Reusing completed TIES merge {args.output}")
                return
        raise ValueError(f"nonempty output directory: {args.output}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    base_index_path = args.base / "model.safetensors.index.json"
    base_index = json.loads(base_index_path.read_text())
    base_map = base_index["weight_map"]
    expert_indexes = [json.loads((path / "model.safetensors.index.json").read_text()) for path in args.expert]
    expert_maps = [index["weight_map"] for index in expert_indexes]
    if any(set(weight_map) != set(base_map) for weight_map in expert_maps):
        raise ValueError("checkpoint tensor keys differ")

    args.output.mkdir(parents=True, exist_ok=True)
    for name in COPY_NAMES:
        source = args.expert[0] / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    output_map: dict[str, str] = {}
    total_size = 0
    # Keep the first expert's shard layout so vLLM sees a conventional checkpoint.
    for shard_name in sorted(set(expert_maps[0].values())):
        keys = sorted(key for key, shard in expert_maps[0].items() if shard == shard_name)
        base_shards = sorted({base_map[key] for key in keys})
        expert_shards = [sorted({weight_map[key] for key in keys}) for weight_map in expert_maps]
        tensors = {}
        with ExitStack() as stack:
            base_files = {name: stack.enter_context(safe_open(args.base / name, framework="pt", device="cpu"))
                          for name in base_shards}
            expert_files = []
            for root, names in zip(args.expert, expert_shards):
                expert_files.append({name: stack.enter_context(safe_open(root / name, framework="pt", device="cpu"))
                                     for name in names})
            for key in keys:
                base_tensor = base_files[base_map[key]].get_tensor(key)
                expert_tensors = [files[weight_map[key]].get_tensor(key)
                                  for files, weight_map in zip(expert_files, expert_maps)]
                tensor = ties_merge_tensor(base_tensor, expert_tensors, args.density, args.scale)
                tensors[key] = tensor
                total_size += tensor.numel() * tensor.element_size()
                output_map[key] = shard_name
        save_file(tensors, args.output / shard_name, metadata={"format": "pt"})
        del tensors

    output_index_path = args.output / "model.safetensors.index.json"
    output_index_path.write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": output_map,
    }, indent=2) + "\n")
    receipt = {
        "schema": "reasoning-sft-layerwise-ties-v1",
        "method": "layerwise TIES magnitude trim, mass sign election, disjoint mean",
        "density": args.density,
        "scale": args.scale,
        "large_tensor_cutoff": "deterministic uniform sample of at most 1000000 entries",
        "base_index_sha256": digest(base_index_path),
        "expert_index_sha256": [digest(path / "model.safetensors.index.json") for path in args.expert],
        "output_index_sha256": digest(output_index_path),
        "tensor_count": len(output_map),
        "total_size": total_size,
        "output_dtype": "bfloat16",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
