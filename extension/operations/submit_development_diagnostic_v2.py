"""Metadata-only one fixed CPU diagnostic after the evidenced sealing failure."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,subprocess
from extension import submission as meta
from extension.development_submission import source_inventory,ROLE as FAILED_ROLE
ROOT=Path('/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019')
ROLE='development_level_diagnostic_20261021_v2'
REJECTED_ROLE='development_level_diagnostic_20261021_v1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':')).encode()
now=lambda:datetime.now(timezone.utc).isoformat()
root=ROOT; wrapper=root/'extension/operations/development_metadata_diagnostic_v2.slurm'
assert (meta.DEADLINE-datetime.now(timezone.utc)).total_seconds()>173400
with (root/'extension/job_ledger.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=root/'extension/job_ledger.json';ledger=json.loads(path.read_text());jobs=ledger['jobs']
 assert ledger['gpu_hour_cap'] is None
 assert sha((root/'deadline/job_ledger.json').read_bytes())==json.loads((root/'extension/source_install_receipt.json').read_text())['historical_ledger_sha256']
 failed=jobs[FAILED_ROLE];assert failed['job_id']=='1223566'
 qa_path=root/'extension/runs/development-seal-20261021-1223566/native_qa.json';qa=json.loads(qa_path.read_text())
 assert qa['success'] and qa['tests_run']==192 and qa['source_inventory']==source_inventory(root)==failed['source_sha256']
 assert all(qa[k]==0 for k in ('errors','failures','skipped','expected_failures','unexpected_successes'))
 assert not (qa_path.parent/'complete.json').exists()
 account_cmd=['sacct','--clusters=owl','-j','1223566','--parsable2','--noheader','--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,Start,End,AllocTRES']
 result=subprocess.run(account_cmd,capture_output=True,text=True,check=True,env=dict(os.environ,TZ='UTC'))
 lines=[line.split('|') for line in result.stdout.splitlines() if line]
 assert len(lines)==3 and {r[0] for r in lines}=={'1223566','1223566.batch','1223566.extern'}
 assert all(len(r)==8 and r[4]=='45' and 'gpu' not in r[7].lower() for r in lines)
 assert all(r[2:4]==(['COMPLETED','0:0'] if r[0].endswith('.extern') else ['FAILED','1:0']) for r in lines)
 failure={'schema':'extension-terminal-cpu-failure-v1','job_id':'1223566','complete':False,'terminal':True,'observed_at_utc':now(),'command':account_cmd,'stdout':result.stdout,'stderr':result.stderr,'elapsed_seconds':45,'gpu_seconds':0,'native_qa_sha256':sha(qa_path.read_bytes()),'tests_passed':192,'source_commit':'c1db50c50ef0ea228315970962843ce996cac113','exception':'ValueError: MATH subject/official level is missing','splits_chosen':False,'scientific_outcomes_viewed':False}
 fpath=root/'extension/operations/development-seal-1223566-failure.json'
 if fpath.exists():
  prior=json.loads(fpath.read_text()); assert prior['stdout']==failure['stdout'] and prior['native_qa_sha256']==failure['native_qa_sha256']
 else:fpath.write_text(json.dumps(failure,indent=2,sort_keys=True)+'\n')
 queue=meta._queue_snapshot(subprocess.run)
 assert not queue['jobs'],'Reconcile live jobs before diagnostic admission'
 expected={'cpu_feasibility_20261020_v1','labeled_feasibility_20261020_v1',FAILED_ROLE,REJECTED_ROLE}
 assert set(jobs)<=expected|{ROLE} and expected<=set(jobs)
 rejected=jobs[REJECTED_ROLE]
 expected_error='sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host\nsbatch: error: fetch_config: DNS SRV lookup failed\nsbatch: error: _establish_config_source: failed to fetch config\nsbatch: fatal: Could not establish a configuration source\n'
 assert rejected['status']=='uncertain' and rejected['scheduler_returncode']==1 and rejected['scheduler_stdout']=='' and rejected['scheduler_stderr']==expected_error and 'job_id' not in rejected
 rejection={'schema':'extension-diagnostic-pre-scheduler-rejection-v1','complete':True,'role':REJECTED_ROLE,'original_entry':rejected,'original_entry_sha256':sha(canonical(rejected)),'observed_at_utc':now(),'scheduler_snapshot':queue,'classification':'DNS configuration failure before connection to scheduler; nonzero return with empty stdout and no job identity','repair':'Use tested metadata._submission_environment preserving SLURM_CONF and SLURM_CONF_SERVER; old attempt entry/sources retained unchanged','gpu_seconds':0}
 rejection_path=root/'extension/operations/development-diagnostic-dns-rejection-20261021.json'
 if rejection_path.exists():assert json.loads(rejection_path.read_text())['original_entry']==rejected
 else:rejection_path.write_text(json.dumps(rejection,indent=2,sort_keys=True)+'\n')
 binding={'role':ROLE,'wrapper_sha256':sha(wrapper.read_bytes()),'submit_source_sha256':sha(Path(__file__).read_bytes()),'failed_job':'1223566','rejected_diagnostic_receipt_sha256':sha(rejection_path.read_bytes()),'failed_binding_sha256':failed['binding_sha256'],'failure_receipt_sha256':sha(fpath.read_bytes()),'source_sha256':source_inventory(root),'wall_seconds':600,'cpu':1,'mem_gb':4,'requested_gpu_seconds':0,'requested_gpu_hours':0,'prior_jobs_sha256':sha(canonical({k:v for k,v in jobs.items() if k!=ROLE}))}
 checksum=sha(canonical(binding))
 if ROLE in jobs:
  assert jobs[ROLE]['status']=='submitted' and jobs[ROLE]['binding_sha256']==checksum
  print(jobs[ROLE]['job_id']);raise SystemExit
 command=['sbatch','--clusters=owl','--parsable',str(wrapper)]
 entry={'status':'submitting','requested_at_utc':now(),'requested_gpu_seconds':0,'requested_gpu_hours':0,'binding':binding,'binding_sha256':checksum,'scheduler_snapshot':queue,'command':command}
 jobs[ROLE]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,sort_keys=True,separators=(',',':'))+'\n');tmp.replace(path)
 env,removed=meta._submission_environment()
 entry['removed_scheduler_environment_keys']=removed
 proc=subprocess.run(command,capture_output=True,text=True,env=env)
 entry.update(scheduler_returncode=proc.returncode,scheduler_stdout=proc.stdout,scheduler_stderr=proc.stderr)
 if proc.returncode==0 and __import__('re').fullmatch(r'[0-9]+(?:;owl)?\s*',proc.stdout):entry.update(status='submitted',job_id=proc.stdout.strip().split(';')[0],submitted_at_utc=now())
 else:entry['status']='uncertain'
 tmp.write_text(json.dumps(ledger,sort_keys=True,separators=(',',':'))+'\n');tmp.replace(path)
 assert entry['status']=='submitted','Diagnostic submission uncertain; reconcile without duplicate'
 print(entry['job_id'])
