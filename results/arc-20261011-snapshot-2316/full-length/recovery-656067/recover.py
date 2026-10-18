import ast,copy,importlib.metadata,json,subprocess
from pathlib import Path
from reasoning.data import digest,write_json
from reasoning.a30_pair_runtime import FILES
root=Path('artifacts/datasets-full-length-15k'); saved=root/'recovery-656067'
old=json.load(open(saved/'a30_pair_execution_receipt.json'))
assert old==json.load(open(root/'a30_pair_execution_receipt.json'))
new=copy.deepcopy(old)
hashes={p:digest(p) for p in FILES}
changed=[p for p in hashes if hashes[p]!=old['execution_sha256'][p]]
assert changed==['scripts/arc_full_length_2a30_train.sh'],changed
old_driver=(saved/'arc_full_length_2a30_train.sh').read_text()
new_driver=Path(changed[0]).read_text()
assert 'deepspeed>=0.10.0,<=0.16.9' in Path('LLaMA-Factory/src/llamafactory/hparams/parser.py').read_text()
assert '_check_extra_dependencies(' in new_driver
assert '.venv-lmf-a30-ds0169/bin/python' in new_driver
assert digest(root/'preparation_receipt.json')==old['preparation_sha256']
assert digest(root/'advanced.json')==old['data_sha256']
assert digest(root/'selection_manifest.json')==old['selection_sha256']
log=(saved/'dependency_cpu_check.log').read_text()
assert 'ACTUAL_TRAINER_DEPENDENCY_GUARD_ZERO3_SCHEMA_CPU_ADAM_OK' in log
versions={k:importlib.metadata.version(k) for k in ('torch','transformers','deepspeed')}
assert versions=={'torch':'2.10.0','transformers':'4.55.0','deepspeed':'0.16.9'}
run=Path('runs/lmf-full-length-15k-16k-3epoch-2a30')
assert not list(run.glob('checkpoint-*')) and not (run/'trainer_state.json').exists()
alloc=json.load(open(run/'allocations.json'))
assert alloc['job_ids']==['656067'] and alloc==json.load(open(saved/'allocations.json'))
new['execution_sha256']=hashes;new['runtime_versions']=versions
new['operational_recovery']={'failed_job':'656067','cause':'native LLaMA-Factory requires DeepSpeed<=0.16.9; original environment had0.18.9',
 'old_runtime_versions':old['runtime_versions'],'new_runtime_versions':versions,
 'isolated_environment':'.venv-lmf-a30-ds0169','original_environments_unchanged':True,
 'failed_allocation_retained':True,'training_updates_before_failure':0,'gpu_pilot':False,
 'dependency_guard_bypassed':False}
write_json(root/'a30_pair_execution_receipt.json',new)
alloc['operational_recoveries']=[{'failed_job':'656067','before_execution_sha256':alloc['execution_sha256'],
 'before_runtime_versions':old['runtime_versions'],'after_runtime_versions':versions,
 'recovery_path':str(saved),'failed_allocation_retained':True}]
alloc['execution_sha256']=hashes
write_json(run/'allocations.json',alloc)
proof={'schema':'a30-deepspeed-native-guard-recovery-v1','failed_job':'656067','full_log_sha256':digest(saved/'full_length_2a30_656067.log'),
 'old_execution_receipt_sha256':digest(saved/'a30_pair_execution_receipt.json'),
 'new_execution_receipt_sha256':digest(root/'a30_pair_execution_receipt.json'),
 'only_bound_execution_file_changed':changed,'config_data_selection_preparation_unchanged':True,
 'native_trainer_guard_sha256':digest('LLaMA-Factory/src/llamafactory/hparams/parser.py'),
 'cpu_dependency_log_sha256':digest(saved/'dependency_cpu_check.log'),
 'cpu_tests':{'native_dependency_guard':True,'cpu_adam_step':True,'installed_zero3_schema':True,'plain_ce_loss_gradient':True,'two_rank_recovery':True},
 'overlay_initial_cpu_failure_preserved':'overlay_initial_cpu_check.log',
 'new_overlay_pth_sha256':digest('.venv-lmf-a30-ds0169/lib/python3.10/site-packages/parent_lmf_overlay.pth'),
 'new_overlay_pyvenv_sha256':digest('.venv-lmf-a30-ds0169/pyvenv.cfg'),
 'before_allocation_sha256':digest(saved/'allocations.json'),'after_allocation_sha256':digest(run/'allocations.json'),
 'training_updates':0,'model_loaded':False,'budget_charged_allocations':['656067'],'no_gpu_pilot':True,'corrected_gpu_resubmissions_allowed':1}
write_json(saved/'recovery_receipt.json',proof)
# Execute all actual staged driver CPU checks before registering any new allocation.
block=new_driver.split("<<'PYCHECK'\n",1)[1].split('\nPYCHECK',1)[0]
block=block[:block.index("run=Path(cfg['output_dir'])")]
exec(compile(block,'actual_driver_prelaunch','exec'))
print(json.dumps(proof,indent=2));print('COMPLETE_STAGED_CPU_PRELAUNCH_OK')
