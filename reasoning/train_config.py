"""Generate runnable LLaMA-Factory configs from the selected datasets."""
import argparse
import json
from pathlib import Path
from .data import read_rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['random','filtered_random','advanced'], required=True)
    p.add_argument('--dataset-dir', default='artifacts/datasets')
    p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=['full','qlora'], default='full')
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--revision', default='main', help='use the same base-model commit as feature scoring')
    p.add_argument('--max-steps', type=int, default=-1, help='positive override for throughput/smoke runs')
    p.add_argument('--cutoff', type=int, default=16384)
    p.add_argument('--loss', choices=['sft','dft'], default='sft')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', help='checkpoint path; never silently overwrite runs')
    a = p.parse_args()
    import yaml
    root = Path(__file__).resolve().parents[1]
    datasets = Path(a.dataset_dir).resolve()
    manifest = json.loads((datasets/'selection_manifest.json').read_text())
    if a.cutoff != manifest['arguments']['cutoff']:
        p.error('training cutoff must match dataset selection cutoff; rerun selection for a different cutoff')
    if a.epochs <= 0 or (a.learning_rate is not None and a.learning_rate <= 0):
        p.error('epochs and learning rate must be positive')
    rows = read_rows(datasets/(a.dataset+'.json'))
    if not rows or not read_rows(datasets/'validation.json'):
        p.error('empty training or validation data')
    cfg = dict(model_name_or_path='Qwen/Qwen2.5-3B-Instruct', stage='sft', do_train=True,
        model_revision=a.revision, use_dft_loss=a.loss=='dft', max_steps=a.max_steps, do_eval=True, finetuning_type='full', flash_attn='fa2', template='qwen',
        dataset=a.dataset, eval_dataset='validation', dataset_dir=str(datasets),
        cutoff_len=a.cutoff, preprocessing_num_workers=8, overwrite_cache=False,
        train_on_prompt=False, packing=False, output_dir=str(Path(a.output).resolve()),
        overwrite_output_dir=False, logging_steps=5, save_strategy='steps', save_steps=100,
        eval_strategy='steps', eval_steps=100, save_total_limit=3, plot_loss=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        per_device_eval_batch_size=1, prediction_loss_only=True,
        learning_rate=a.learning_rate or (5e-6 if a.profile=='full' else 5e-5),
        num_train_epochs=a.epochs, lr_scheduler_type='cosine', warmup_ratio=.03,
        bf16=True, gradient_checkpointing=True, max_grad_norm=1., weight_decay=.01,
        ddp_timeout=18000, report_to='none', seed=a.seed, data_seed=a.seed,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False)
    if a.profile=='full':
        cfg['deepspeed'] = str(root/'LLaMA-Factory/examples/deepspeed/ds_z2_config.json')
    else:
        cfg.update(finetuning_type='lora', quantization_bit=4, quantization_method='bitsandbytes',
                   lora_rank=64, lora_alpha=128, lora_dropout=.05, lora_target='all')
    if a.resume:
        cfg['resume_from_checkpoint'] = str(Path(a.resume).resolve())
    elif Path(a.output).exists() and any(Path(a.output).iterdir()):
        p.error('output exists; choose another path or --resume CHECKPOINT')
    dest = root/'configs'/f'{Path(a.output).name}.yaml'
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.profile=='qlora':
        export = dict(model_name_or_path=cfg['model_name_or_path'], adapter_name_or_path=cfg['output_dir'],
            model_revision=a.revision, template='qwen', finetuning_type='lora', export_dir=cfg['output_dir']+'-merged',
            export_size=5, export_device='cpu', export_legacy_format=False)
        (dest.parent/(dest.stem+'-export.yaml')).write_text(yaml.safe_dump(export, sort_keys=False))
    print(dest)


if __name__ == '__main__':
    main()
