"""One bounded metadata-only CPU admission for native GPU-adapter oracles."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,fcntl,re,subprocess
from extension import submission as m
from extension import joint_gpu_metadata_qa as qa
from extension import source_readiness_retry_submission as old
root=Path.cwd().resolve();ROLE=qa.ROLE;NAME="sft-extension-joint-gpu-metadata-qa";SCRIPT="extension/joint_gpu_metadata_qa.slurm"
DIR="extension/report/joint-memory-qa-20261021-1225557"
PINS={"complete.json":"66a6ba25cea01a70ee3a945d1dce9cddf474915ab6b32e506de3058d23a8f51f","native_qa.json":"00af45d78452c1d0a6d6aacf24518e9e77b749db39a2cbbd94b724a4f2cca4ee","accounting.json":"58c20b9a3c6ab0484236eda3cdf7bd06fb0bd7acca65ee096016ce7b07ed9d52","job_ledger_snapshot.json":"a3abc177a12e4ebcb687addf57b8e958738af5b985133b64dfb83df3e21c523f"}
def record(p):
 b=m._small_bytes(p,root);return {"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
def guard():m.require((m.DEADLINE-datetime.now(timezone.utc)).total_seconds()>=173400,"Keep10min work plus48h reserve")
guard();m.require(type(qa.EXPECTED_TESTS) is int and qa.EXPECTED_TESTS>0,"Freeze exact test module count before admission")
sources=qa.source_inventory(root);assets={}
for n,h in PINS.items():
 relative=DIR+"/"+n;v=record(root/relative);m.require(v["sha256"]==h,"Scientific native evidence changed");assets[relative]=v
parents=m._read(root/DIR/"job_ledger_snapshot.json")["jobs"];m.require(len(parents)==12,"Keep exact12 earlier entries")
scientific=m._read(root/DIR/"native_qa.json")["source_inventory"];m.require(len(scientific)==79,"Exact79 scientific files required")
for n,v in scientific.items():m.require(record(root/n)==v,"Scientific source changed")
m.require(record(root/"deadline/job_ledger.json")["sha256"]=="9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d","Historical ledger changed")
directives=[line.strip() for line in (root/SCRIPT).read_text().splitlines() if line.lstrip().startswith("#SBATCH")]
expected=["#SBATCH --job-name="+NAME,"#SBATCH --account=tml_2026","#SBATCH --partition=normal_q","#SBATCH --qos=owl_normal_short","#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=2","#SBATCH --mem=8G","#SBATCH --time=00:10:00","#SBATCH --no-requeue","#SBATCH --output="+str(root/"extension/logs/extension-joint-gpu-metadata-qa-%j.out"),"#SBATCH --error="+str(root/"extension/logs/extension-joint-gpu-metadata-qa-%j.err")]
m.require(directives==expected,"ExactCPU2/8G/10min noGPU wrapper required")
command=["sbatch","--clusters=owl","--parsable",str(root/SCRIPT)]
binding={"role":ROLE,"command":command,"source_sha256":sources,"scientific_source_sha256":scientific,"initial_evidence":assets,"prior_entries_sha256":hashlib.sha256(m._canonical(parents)).hexdigest(),"admission_code_sha256":record(Path(__file__).resolve())["sha256"],"expected_native_tests":qa.EXPECTED_TESTS,"requested_gpu_hours":0,"requested_gpu_seconds":0,"wall_time_seconds":600,"closeout_reserve_seconds":172800,"deadline_utc":m.DEADLINE.isoformat(),"automatic_downstream":False}
checksum=hashlib.sha256(m._canonical(binding)).hexdigest();path=root/"extension/job_ledger.json";lockpath=root/"extension/job_ledger.lock"
m.require(not path.is_symlink() and not lockpath.is_symlink(),"No ledger/lock aliases")
with lockpath.open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=m._read(path)
 m.require(ledger.get("schema")==m.SCHEMA and ledger.get("gpu_hour_cap","missing") is None and ledger.get("historical_ledger_unchanged")=="deadline/job_ledger.json" and set(ledger["jobs"])<=set(parents)|{ROLE} and all(ledger["jobs"].get(k)==v for k,v in parents.items()),"Preserve exact12 prior entries")
 q=m._queue_snapshot(subprocess.run)
 old.check_queue(q,{k:v for k,v in parents.items() if k not in (old.ROLE,"source_grader_diagnostic_20261021_v1","joint_memory_qa_20261021_v1")})
 known={"1225360":("FAILED",old.NAME),"1225426":("COMPLETED","sft-extension-source-grader-diagnostic"),"1225557":("COMPLETED","sft-extension-joint-memory-qa")}
 for row in q["jobs"]:
  if row["job_id"] in known:m.require((row["state"],row["name"])==known[row["job_id"]],"Known scientific prior identity/state changed")
  elif row["name"] in {x[1] for x in known.values()} or row["job_id"].split("_")[0] in known:raise m.SubmissionError("Untracked prior duplicate")
 m.require(qa.source_inventory(root)==sources and all(record(root/n)==v for n,v in assets.items()) and all(record(root/n)==v for n,v in scientific.items()) and record(Path(__file__).resolve())["sha256"]==binding["admission_code_sha256"],"Source/evidence changed during admission")
 same={v["job_id"] for v in q["jobs"] if v["name"]==NAME};entry=ledger["jobs"].get(ROLE)
 if entry is not None:
  m.require(entry.get("status")=="submitted" and entry.get("binding")==binding and entry.get("binding_sha256")==checksum and not same-{entry.get("job_id")},"Never resubmit existing/uncertain metadata QA")
  print(entry["job_id"]);raise SystemExit
 m.require(not same,"Untracked metadata QA requires reconciliation");guard()
 environment,removed=m._submission_environment()
 entry={"status":"submitting","binding":binding,"binding_sha256":checksum,"source_sha256":sources,"command":command,"requested_gpu_hours":0,"requested_gpu_seconds":0,"requested_at_utc":datetime.now(timezone.utc).isoformat(),"scheduler_snapshot":q,"removed_scheduler_environment_keys":removed}
 ledger["jobs"][ROLE]=entry;m._write(path,ledger)
 try:r=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
 except (OSError,subprocess.SubprocessError) as e:
  entry.update(status="uncertain",error_type=type(e).__name__);m._write(path,ledger);raise
 entry.update(scheduler_returncode=r.returncode,scheduler_stdout=r.stdout[-2000:],scheduler_stderr=r.stderr[-2000:]);match=re.fullmatch(r"([1-9][0-9]*)(?:;owl)?",r.stdout.strip())
 if r.returncode or match is None:
  entry["status"]="submission_failed" if r.returncode else "uncertain";m._write(path,ledger);raise m.SubmissionError("Preserve rejected/ambiguous metadata QA; no retry")
 entry.update(status="submitted",job_id=match.group(1),submitted_at_utc=datetime.now(timezone.utc).isoformat());m._write(path,ledger);print(entry["job_id"])
