"""One fixed CPU admission for the joint initializer repair QA, without GPU work."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import source_generation_submission as parent
from extension.cpu_feasibility import value_digest

SELF='extension/operations/submit-joint-identity-v3-qa-20261021.py'
INSTALL='extension/joint_identity_replay_v3_source_receipt.json'
INSTALL_SHA256='4d88d39475bd1309780aec6e4760447cefed68e98e8ac8f7924ab3373a638139'
PLAN='extension/joint_identity_replay_v3_plan.json'
PLAN_SHA256='88e57bc1399d180a30645b07e578d9edde7630c1dcf734e95bcc667d7f015fb8'
EXPECTED_TESTS=65
SCRIPT='extension/joint_identity_replay_v3_qa.slurm'
ROLE='joint_identity_replay_v3_qa_20261021_v1'
NAME='sft-extension-joint-identity-v3-qa'
SNAPSHOT='extension/report/joint-memory-admission-v2-failed-20261021-1227971/job_ledger_snapshot.json'
SNAPSHOT_SHA256='7de19dc305699fe87af6a5ce9f10835142608e5a0a780442abd134498c6e90a2'
PRIOR_SHA256='7e03f2b63becabd011e2e9b6cdd87a2697597ea910b8264127dc7b349608ae7c'

def guard():
    now=datetime.now(timezone.utc)
    parent.shared.require((parent.m.DEADLINE-now).total_seconds()>=174000,'Keep20min CPU plus48h reserve')
    return now

def prerequisites(root):
    s=parent.shared; m=parent.m
    s.require(s._checksum(INSTALL_SHA256) and s._checksum(PLAN_SHA256) and type(EXPECTED_TESTS) is int and EXPECTED_TESTS>0,'Root must first freeze all exact sources and test count')
    assets={n:s._record(root,n) for n in (INSTALL,PLAN,SNAPSHOT,SELF,'deadline/job_ledger.json')}
    s.require(assets[INSTALL]['sha256']==INSTALL_SHA256 and assets[PLAN]['sha256']==PLAN_SHA256 and assets[SNAPSHOT]['sha256']==SNAPSHOT_SHA256 and assets['deadline/job_ledger.json']['sha256']==s.qa.HISTORICAL_LEDGER_SHA256,'Installed plan, original ledger and exact20 snapshot changed')
    installed=m._read(root/INSTALL); snapshot=m._read(root/SNAPSHOT); s._ledger(snapshot)
    s.require(installed.get('installed_on_arc') is True and len(installed.get('source_inventory',{}))==120,'Exact120 installed sources required')
    sources={n:s._record(root,n) for n in installed['source_inventory']}
    s.require(sources==installed['source_inventory'] and len(snapshot['jobs'])==20 and value_digest(snapshot['jobs'])==PRIOR_SHA256 and ROLE not in snapshot['jobs'],'Source120 or prior20 admissions changed')
    directives=[line.strip() for line in (root/SCRIPT).read_text().splitlines() if line.lstrip().startswith('#SBATCH')]
    expected=['#SBATCH --job-name='+NAME,'#SBATCH --account=tml_2026','#SBATCH --partition=normal_q','#SBATCH --qos=owl_normal_short','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=8G','#SBATCH --time=00:20:00','#SBATCH --no-requeue','#SBATCH --output='+str(root/'extension/logs/extension-joint-identity-v3-qa-%j.out'),'#SBATCH --error='+str(root/'extension/logs/extension-joint-identity-v3-qa-%j.err')]
    s.require(directives==expected,'Exact finiteCPU2/8G20min/noGPU/no-requeue only')
    return {'sources':sources,'assets':assets,'prior':snapshot['jobs']}

def main():
    root=Path.cwd().resolve(); s=parent.shared; m=parent.m
    s.require(Path(__file__).resolve()==root/SELF,'Run exact installed adapter from root');guard()
    path,lockpath=root/'extension/job_ledger.json',root/'extension/job_ledger.lock'
    s.require(not path.is_symlink() and not lockpath.is_symlink(),'No ledger aliases')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=m._read(path);s._ledger(ledger);proof=prerequisites(root)
        before={k:v for k,v in ledger['jobs'].items() if k!=ROLE};s.require(before==proof['prior'],'Preserve exact20 prior admissions without a duplicate or concurrent role')
        command=['sbatch','--clusters=owl','--parsable',str(root/SCRIPT)]
        binding={'role':ROLE,'command':command,'source_sha256':proof['sources'],'plan_sha256':PLAN_SHA256,'initial_evidence':proof['assets'],'admission_code_sha256':proof['assets'][SELF]['sha256'],'prior_entries_sha256':PRIOR_SHA256,'requested_gpu_seconds':0,'requested_gpu_hours':0,'wall_time_seconds':1200,'closeout_reserve_seconds':172800,'expected_native_tests':EXPECTED_TESTS,'automatic_downstream':False,'automatic_retry':False}
        checksum=value_digest(binding);snapshot=s.queue_snapshot(subprocess.run);current=ledger['jobs'].get(ROLE)
        parent.known_queue(root,snapshot,{**before,**({ROLE:current} if current else {})});s.check_capacity(snapshot,already_submitted=True)
        s.require(m._read(path)==ledger and prerequisites(root)==proof,'Evidence/source/self/ledger changed during scheduler query')
        same={row['job_id'] for row in snapshot['jobs'] if row['name']==NAME}
        if current is not None:
            s._entry(current,ROLE,gpu_seconds=0,sources=proof['sources']);s.require(current.get('binding')==binding and current.get('binding_sha256')==checksum and not same-{current['job_id']},'Existing/uncertain/rejected QA requires reconciliation, never retry');print(current['job_id'],flush=True);return
        s.require(not same,'Untracked CPU role requires reconciliation');now=guard();environment,removed=m._submission_environment()
        current={'status':'submitting','binding':binding,'binding_sha256':checksum,'source_sha256':proof['sources'],'plan_sha256':PLAN_SHA256,'command':command,'requested_gpu_seconds':0,'requested_gpu_hours':0,'requested_at_utc':now.isoformat(),'scheduler_snapshot':snapshot,'removed_scheduler_environment_keys':removed};ledger['jobs'][ROLE]=current;m._write(path,ledger)
        try:result=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
        except (OSError,subprocess.SubprocessError) as error:
            current.update(status='uncertain',error_type=type(error).__name__);m._write(path,ledger);raise
        current.update(scheduler_returncode=result.returncode,scheduler_stdout=str(result.stdout)[-2000:],scheduler_stderr=str(result.stderr)[-2000:])
        match=re.fullmatch(r'([1-9][0-9]*)(?:;owl)?',result.stdout.strip()) if isinstance(result.stdout,str) else None
        if result.returncode or match is None or not isinstance(result.stderr,str) or result.stderr.strip():
            current['status']='submission_failed' if result.returncode else 'uncertain'
            if match:current['possible_job_id']=match.group(1)
            m._write(path,ledger);raise m.SubmissionError('Rejected or ambiguous QA is preserved without retry')
        job=match.group(1);current.update(status='uncertain',job_id=job);m._write(path,ledger)
        s.require(job not in s.scheduler_identities(before) and job not in {v['job_id'] for v in snapshot['jobs']},'Scheduler identity collision')
        current.update(status='submitted',submitted_at_utc=datetime.now(timezone.utc).isoformat());m._write(path,ledger);print(job,flush=True)

if __name__=='__main__':main()
