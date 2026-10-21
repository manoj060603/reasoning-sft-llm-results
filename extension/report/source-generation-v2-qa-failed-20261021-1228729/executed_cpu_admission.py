"""One fixed CPU60 admission for the sealed original-calibration collector v2.

The install hash must be sealed before use. No private cache/model work here.
"""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import re
import subprocess
from extension import source_generation_v2_submission as metadata
from extension.cpu_feasibility import value_digest

SELF = "extension/operations/submit-source-generation-v2-qa-20261021.py"
INSTALL = "extension/source_generation_v2_source_receipt.json"
INSTALL_SHA256 = "00693c4c996d3d4af2d83ccf1723577026785f4f2fb1c77358f7174824592cce"
PLAN_SHA256 = "dda716348b18d146b2c0dc0a655157c156b85ad8d22e7b03375723fc4ba13a99"
PRIOR_SHA256 = "0bdb6b0227b45433ef102ddde1137d7a5e77a62e2a9c6dc20d58576038a227d7"

def guard():
    now=datetime.now(timezone.utc)
    metadata.shared.require((metadata.m.DEADLINE-now).total_seconds()>=174000,"Keep the20-minute CPU ceiling and48-hour reserve")
    return now

def prerequisites(root):
    s,m=metadata.shared,metadata.m
    s.require(s._checksum(INSTALL_SHA256),"Root must seal exact source installation before admission")
    proof=metadata.initial_evidence(root)
    s.require(proof["plan_sha256"]==PLAN_SHA256 and len(proof["initial"])==24 and proof["initial_sha256"]==PRIOR_SHA256 and len(proof["sources"])==124 and len(proof["metadata"])==4,"Only the exact final24/core124/metadata4 CPU60 stage")
    installed=s._record(root,INSTALL);s.require(installed["sha256"]==INSTALL_SHA256,"Source installation receipt changed")
    receipt=m._read(root/INSTALL)
    s.require(receipt.get("installed_on_arc") is True and receipt.get("core_source_inventory")==proof["sources"] and receipt.get("metadata_source_inventory")==proof["metadata"] and len(receipt.get("protected_parent_sources",{}))==130,"Exact installed collector namespace and unchanged protected130 required")
    for name,record in receipt["protected_parent_sources"].items():s.require(s._record(root,name)==record,"A protected parent source changed: "+name)
    directives=[line.strip() for line in (root/metadata.QA_SCRIPT).read_text().splitlines() if line.lstrip().startswith("#SBATCH")]
    expected=["#SBATCH --job-name="+metadata.QA_NAME,"#SBATCH --account=tml_2026","#SBATCH --partition=normal_q","#SBATCH --qos=owl_normal_short","#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=2","#SBATCH --mem=8G","#SBATCH --time=00:20:00","#SBATCH --no-requeue","#SBATCH --output="+str(root/"extension/logs/extension-source-generation-v2-qa-%j.out"),"#SBATCH --error="+str(root/"extension/logs/extension-source-generation-v2-qa-%j.err")]
    s.require(directives==expected,"Exact finite2CPU/8G/20minute/noGPU wrapper required")
    assets={**proof["assets"],INSTALL:installed}
    s.require(assets[SELF]==s._record(root,SELF),"Run the actual self-bound fixed bootstrap")
    return {**proof,"assets":assets}

def main():
    root=Path.cwd().resolve();s,m=metadata.shared,metadata.m
    s.require(Path(__file__).resolve()==root/SELF,"Run the exact installed bootstrap from root");guard()
    path,lockpath=root/"extension/job_ledger.json",root/"extension/job_ledger.lock"
    s.require(not path.is_symlink() and not lockpath.is_symlink(),"No ledger aliases")
    with lockpath.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);proof=prerequisites(root);ledger=m._read(path);s._ledger(ledger)
        before={k:v for k,v in ledger["jobs"].items() if k!=metadata.QA_ROLE};s.require(before==proof["initial"],"Preserve every exact24 admission; do not bypass an uncertain or new role")
        command=["sbatch","--clusters=owl","--parsable",str(root/metadata.QA_SCRIPT)]
        binding={"role":metadata.QA_ROLE,"command":command,"source_sha256":proof["sources"],"metadata_source_inventory":proof["metadata"],"allocator_source_inventory":proof["allocator_sources"],"plan_sha256":PLAN_SHA256,"initial_evidence":proof["assets"],"admission_code_sha256":proof["assets"][SELF]["sha256"],"prior_entries_sha256":PRIOR_SHA256,"requested_gpu_seconds":0,"requested_gpu_hours":0,"wall_time_seconds":1200,"closeout_reserve_seconds":172800,"expected_native_tests":60,"automatic_downstream":False,"automatic_retry":False}
        checksum=value_digest(binding);snapshot=s.queue_snapshot(subprocess.run);current=ledger["jobs"].get(metadata.QA_ROLE)
        metadata.known_queue(root,snapshot,ledger["jobs"]);s.check_capacity(snapshot,already_submitted=True)
        s.require(m._read(path)==ledger and prerequisites(root)==proof,"Source/self/evidence/ledger changed during queue query")
        same={row["job_id"] for row in snapshot["jobs"] if row["name"]==metadata.QA_NAME}
        if current is not None:
            s._entry(current,metadata.QA_ROLE,gpu_seconds=0,sources=proof["sources"])
            s.require(current.get("binding")==binding and current.get("binding_sha256")==checksum and not same-{current["job_id"]},"Existing or uncertain CPU role requires reconciliation, never retry")
            print(current["job_id"],flush=True);return
        s.require(not same,"Untracked duplicate CPU role");when=guard();environment,removed=m._submission_environment()
        current={"status":"submitting","binding":binding,"binding_sha256":checksum,"source_sha256":proof["sources"],"command":command,"requested_gpu_seconds":0,"requested_gpu_hours":0,"requested_at_utc":when.isoformat(),"scheduler_snapshot":snapshot,"removed_scheduler_environment_keys":removed};ledger["jobs"][metadata.QA_ROLE]=current;m._write(path,ledger)
        try:response=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
        except (OSError,subprocess.SubprocessError) as error:
            current.update(status="uncertain",error_type=type(error).__name__);m._write(path,ledger);raise
        current.update(scheduler_returncode=response.returncode,scheduler_stdout=str(response.stdout)[-2000:],scheduler_stderr=str(response.stderr)[-2000:])
        match=re.fullmatch(r"([1-9][0-9]*)(?:;owl)?",response.stdout.strip()) if isinstance(response.stdout,str) else None
        if type(response.returncode) is not int or response.returncode!=0 or match is None or not isinstance(response.stderr,str) or response.stderr.strip():
            current["status"]="uncertain"
            if match:current["possible_job_id"]=match.group(1)
            m._write(path,ledger);raise m.SubmissionError("Rejected or ambiguous CPU submission is preserved without retry")
        job=match.group(1);current.update(status="uncertain",job_id=job);m._write(path,ledger)
        s.require(job not in s.scheduler_identities(before) and job not in {v["job_id"] for v in snapshot["jobs"]},"Scheduler identity collision")
        s.require(m._read(path)==ledger and prerequisites(root)==proof,"Source/self/evidence/ledger changed after acceptance; preserve uncertain identity")
        current.update(status="submitted",submitted_at_utc=datetime.now(timezone.utc).isoformat());m._write(path,ledger);print(job,flush=True)

if __name__=="__main__":main()
