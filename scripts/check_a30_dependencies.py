"""CPU check of the actual A30 trainer dependency guard and offload optimizer."""
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from llamafactory.hparams import ModelArguments,FinetuningArguments
from llamafactory.hparams.parser import _check_extra_dependencies
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam

cfg=yaml.safe_load(Path('configs/lmf-full-length-15k-16k-3epoch-2a30.yaml').read_text())
flags={k:cfg[k] for k in ('use_dft_loss','use_trimsft_loss','use_prift_loss','use_gem_loss','use_asft_loss')}
assert not any(flags.values())
_check_extra_dependencies(
 ModelArguments(model_name_or_path=cfg['model_name_or_path'],flash_attn=cfg['flash_attn']),
 FinetuningArguments(finetuning_type='full',disable_shuffling=False,plot_loss=cfg['plot_loss'],**flags),
 SimpleNamespace(deepspeed=cfg['deepspeed'],predict_with_generate=False))
assert importlib.metadata.version('deepspeed')=='0.16.9'
DeepSpeedZeroConfig(**json.loads(Path('configs/ds-full-length-4a30.json').read_text())['zero_optimization'])
p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
opt=DeepSpeedCPUAdam([p],lr=5e-6,adamw_mode=True)
p.grad=torch.tensor([.1]);opt.step()
assert torch.isfinite(p).all() and p.item()<1
print('ACTUAL_TRAINER_DEPENDENCY_GUARD_ZERO3_SCHEMA_CPU_ADAM_OK',p.item())
print({k:importlib.metadata.version(k) for k in ('torch','transformers','deepspeed')})
