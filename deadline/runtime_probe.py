"""Read runtime/model availability inside an ARC CPU compute allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = ["torch", "transformers", "accelerate", "datasets", "peft", "vllm",
            "safetensors", "huggingface-hub", "scikit-learn", "numpy", "PyYAML",
            "sympy", "latex2sympy2", "antlr4-python3-runtime", "deepspeed"]


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run the probe in a compute allocation.")
    home = Path.home()
    environments = [home / ".conda/envs" / name / "bin/python" for name in
                    ("reasoning", "skydiscover-vllm", "skydiscover-runner", "result1")]
    code = ("import importlib.metadata as m,json,sys\nv={}\nfor p in " + repr(PACKAGES)
            + ":\n try:v[p]=m.version(p)\n except m.PackageNotFoundError:v[p]=None\n"
            + "print(json.dumps(dict(python=sys.version,packages=v)))\n")
    result = {"job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
              "runtimes": {}, "models": []}
    for interpreter in environments:
        if not interpreter.is_file():
            continue
        try:
            proc = subprocess.run([str(interpreter), "-c", code], text=True,
                                  capture_output=True, timeout=90)
            result["runtimes"][str(interpreter)] = (json.loads(proc.stdout) if proc.returncode == 0
                                                   else {"error": proc.stderr[-1200:]})
        except (subprocess.SubprocessError, ValueError) as exc:
            result["runtimes"][str(interpreter)] = {"error": str(exc)}
    roots = [home / ".cache/huggingface/hub",
             Path("/projects/slmreasoning/manoj/hf_cache"),
             Path("/projects/slmreasoning/manoj/hf_home/hub")]
    for root in roots:
        for snapshot in sorted(root.glob("models--Qwen--Qwen2.5-3B-Instruct/snapshots/*")):
            result["models"].append({"path": str(snapshot), "files": {
                p.name: {"bytes": p.stat().st_size} for p in snapshot.iterdir()
                if p.is_file() and (p.suffix in (".json", ".safetensors", ".bin")
                                    or p.name == "merges.txt")}})
    destination = Path(__file__).parent / "runtime_probe.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
