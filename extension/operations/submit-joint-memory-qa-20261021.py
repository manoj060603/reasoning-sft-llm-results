"""One fixed CPU-only admission for the separate410-check joint-memory QA."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,fcntl,json,re,subprocess
from extension import submission as m
from extension import joint_memory_qa as qa
from extension import source_readiness_retry_submission as old

root=Path.cwd().resolve()
ROLE=qa.ROLE
NAME="sft-extension-joint-memory-qa"
SCRIPT="extension/joint_memory_qa.slurm"
SOURCE_COMMIT="2f56bae059e73d27cb63f9c7b6690f62521d5c91"
PLAN_SHA256="b926be22779d59af501afddd10e7129f105b68b723b9bc8386be0aa397ddfc24"
def record(p):
 b=m._small_bytes(p,root);return {"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
def guard():
 m.require((m.DEADLINE-datetime.now(timezone.utc)).total_seconds()>=180000,"Full2h work plus48h reserve required")
guard()
sources=qa.source_inventory(root)
parents=qa.initial_evidence(root,sources)
m.require(len(sources)==79 and len(parents[1])==11 and sources[qa.PLAN]["sha256"]==PLAN_SHA256,"Fixed79-source/11-prior CPU admission")
qa.joint.validate_plan(m._read(root/qa.PLAN))
diagnostic_directory="extension/report/source-grader-diagnostic-20261021-1225426"
pins={"accounting.json":"f5f1142d776cf730b31c5b264a9dfbbf1ee7217ebb416bc3432fc6afc20c78ce","job_ledger_snapshot.json":"30e4518844f8bf6c3d105b52cc88d6e863bad89acaa948d8d619a0b54177a578"}
assets=dict(parents[0])
for n,checksum in pins.items():
 relative=diagnostic_directory+"/"+n;v=record(root/relative);m.require(v["sha256"]==checksum,"Native prior terminal/history metadata changed");assets[relative]=v
snapshot=m._read(root/diagnostic_directory/"job_ledger_snapshot.json")
m.require(snapshot["jobs"]==parents[1],"All11 admissions must equal sealed native snapshot")
accounting=m._read(root/diagnostic_directory/"accounting.json")
m.require(accounting.get("complete") is True and accounting.get("job_id")=="1225426" and accounting.get("gpu_seconds")==0 and accounting.get("elapsed_seconds")==29,"Prior diagnostic must be terminal zeroGPU")
expected=["#SBATCH --job-name="+NAME,"#SBATCH --account=tml_2026","#SBATCH --partition=normal_q","#SBATCH --qos=owl_normal_short","#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=8","#SBATCH --mem=64G","#SBATCH --time=02:00:00","#SBATCH --no-requeue","#SBATCH --output="+str(root/"extension/logs/extension-joint-memory-qa-%j.out"),"#SBATCH --error="+str(root/"extension/logs/extension-joint-memory-qa-%j.err")]
m.require([line.strip() for line in (root/SCRIPT).read_text().splitlines() if line.lstrip().startswith("#SBATCH")]==expected,"Exact CPU8/64G/2h zeroGPU wrapper required")
command=["sbatch","--clusters=owl","--parsable",str(root/SCRIPT)]
binding={"role":ROLE,"command":command,"source_sha256":sources,"scientific_source_commit":SOURCE_COMMIT,"plan_sha256":PLAN_SHA256,"initial_evidence":assets,"prior_binding_sha256":{k:v["binding_sha256"] for k,v in parents[1].items()},"admission_code_sha256":record(Path(__file__).resolve())["sha256"],"requested_gpu_hours":0,"requested_gpu_seconds":0,"wall_time_seconds":7200,"closeout_reserve_seconds":172800,"deadline_utc":m.DEADLINE.isoformat(),"expected_native_tests":410,"source_grader_executed":False,"source_grader_admitted":False,"automatic_downstream":False}
checksum=hashlib.sha256(m._canonical(binding)).hexdigest()
path=root/"extension/job_ledger.json";lockpath=root/"extension/job_ledger.lock"
m.require(not path.is_symlink() and not lockpath.is_symlink(),"No ledger/lock aliases")
with lockpath.open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 ledger=m._read(path)
 m.require(ledger.get("schema")==m.SCHEMA and ledger.get("gpu_hour_cap","missing") is None and ledger.get("historical_ledger_unchanged")=="deadline/job_ledger.json" and set(ledger["jobs"])<=set(parents[1])|{ROLE} and all(ledger["jobs"].get(k)==v for k,v in parents[1].items()),"Preserve all11 earlier entries exactly")
 queue=m._queue_snapshot(subprocess.run)
 old.check_queue(queue,{k:v for k,v in parents[1].items() if k not in (old.ROLE,qa.PRIOR_DIAGNOSTIC_ROLE)})
 for row in queue["jobs"]:
  if row["job_id"]=="1225360":m.require((row["state"],row["name"])==("FAILED",old.NAME),"PriorCPUv2 state/identity changed")
  elif row["name"]==old.NAME or row["job_id"].split("_")[0]=="1225360":raise m.SubmissionError("Untracked priorCPUv2 duplicate")
  if row["job_id"]=="1225426":m.require((row["state"],row["name"])==("COMPLETED","sft-extension-source-grader-diagnostic"),"Prior timing job must remain exactterminal")
  elif row["name"]=="sft-extension-source-grader-diagnostic" or row["job_id"].split("_")[0]=="1225426":raise m.SubmissionError("Untracked timing diagnostic duplicate")
 m.require(qa.source_inventory(root)==sources and qa.initial_evidence(root,sources)==parents and all(record(root/n)==v for n,v in assets.items()) and record(Path(__file__).resolve())["sha256"]==binding["admission_code_sha256"],"Source/evidence changed during admission")
 same={v["job_id"] for v in queue["jobs"] if v["name"]==NAME}
 entry=ledger["jobs"].get(ROLE)
 if entry is not None:
  m.require(entry.get("status")=="submitted" and entry.get("binding")==binding and entry.get("binding_sha256")==checksum and not same-{entry.get("job_id")},"Existing/uncertain jointQA must not be resubmitted")
  print(entry["job_id"]);raise SystemExit
 m.require(not same,"Untracked jointQA requires reconciliation")
 guard()
 environment,removed=m._submission_environment()
 entry={"status":"submitting","binding":binding,"binding_sha256":checksum,"source_sha256":sources,"plan_sha256":PLAN_SHA256,"command":command,"requested_gpu_hours":0,"requested_gpu_seconds":0,"requested_at_utc":datetime.now(timezone.utc).isoformat(),"scheduler_snapshot":queue,"removed_scheduler_environment_keys":removed}
 ledger["jobs"][ROLE]=entry;m._write(path,ledger)
 try:r=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
 except (OSError,subprocess.SubprocessError) as error:
  entry.update(status="uncertain",error_type=type(error).__name__);m._write(path,ledger);raise
 entry.update(scheduler_returncode=r.returncode,scheduler_stdout=r.stdout[-2000:],scheduler_stderr=r.stderr[-2000:])
 match=re.fullmatch(r"([1-9][0-9]*)(?:;owl)?",r.stdout.strip())
 if r.returncode or match is None:
  entry["status"]="submission_failed" if r.returncode else "uncertain";m._write(path,ledger);raise m.SubmissionError("Preserve rejected/uncertain admission; no retry")
 entry.update(status="submitted",job_id=match.group(1),submitted_at_utc=datetime.now(timezone.utc).isoformat());m._write(path,ledger)
 print(entry["job_id"])
