"""One fixed metadata-only admission for a synthetic grading timing diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,re,subprocess
from extension import submission as m
from extension import source_readiness_retry_submission as prior
ROOT=Path.cwd().resolve()
ROLE="source_grader_diagnostic_20261021_v1"
NAME="sft-extension-source-grader-diagnostic"
SCRIPT="extension/source_grader_diagnostic.slurm"
def sha(b):return hashlib.sha256(b).hexdigest()
def record(p):
 b=m._small_bytes(p,ROOT);return {"sha256":sha(b),"bytes":len(b)}
def guard():
 m.require((m.DEADLINE-datetime.now(timezone.utc)).total_seconds()>=173400,"Keep full10min plus48hour reserve")
guard()
directory=ROOT/"extension/report/source-readiness-v2-failed-20261021-1225360"
pins={"accounting.json":"1e725f1fd3bf40e29fff1004b36228695cf219f66ac50ed158b060f2c6bf7d2b","unit_qa_failure.json":"8cdb5a256023a64623616f43c99f19f5deaf20cad8b540e1ae0edaf8b078d8b2","job_ledger_snapshot.json":"8ddea4b35c0b8f06b77632378c418feb891b0281192c1b8386e83d2eec9c99bc","source_install_receipt.json":"fcfbdfcfade0a4889db061c02f8c0ccc91b953f6ad7e86965dff1eac65b55717"}
assets={}
for n,d in pins.items():
 r=record(directory/n);m.require(r["sha256"]==d,"Failure evidence changed")
 assets[str((directory/n).relative_to(ROOT))]=r
failed=m._read(directory/"accounting.json")
m.require(failed.get("terminal") is True and failed.get("complete") is False and failed.get("job_id")=="1225360" and failed.get("elapsed_seconds")==65 and failed.get("gpu_seconds")==0 and failed.get("tests_run")==465 and failed.get("failures")==1 and failed.get("calibration_prepared") is False and failed.get("model_or_benchmark_outcomes_viewed") is False,"Need exact terminal pre-model failure")
archived=m._read(directory/"job_ledger_snapshot.json")
m.require(archived.get("schema")==m.SCHEMA and len(archived["jobs"])==10,"Keep exact10 earlier entries")
old=m._read(directory/"source_install_receipt.json")["source_inventory"]
sources=prior.source_inventory(ROOT)
m.require(sources==old and len(sources)==71,"All71 failed-attempt source bytes must remain unchanged")
for n in ("extension/source_grader_diagnostic.py",SCRIPT):sources[n]=record(ROOT/n)
for n in ("job_ledger_snapshot.json","source_install_receipt.json"):
 assets[str((directory/n).relative_to(ROOT))]=record(directory/n)
m.require(record(ROOT/"deadline/job_ledger.json")["sha256"]=="9fc4dfa2506ccd33580bab3e5f4c4e91d0731e221c4f88ac9778e54800dc141d","Historical ledger changed")
directives=[line.strip() for line in (ROOT/SCRIPT).read_text().splitlines() if line.lstrip().startswith("#SBATCH")]
expected=["#SBATCH --job-name="+NAME,"#SBATCH --account=tml_2026","#SBATCH --partition=normal_q","#SBATCH --qos=owl_normal_short","#SBATCH --nodes=1","#SBATCH --ntasks=1","#SBATCH --cpus-per-task=1","#SBATCH --mem=8G","#SBATCH --time=00:10:00","#SBATCH --no-requeue","#SBATCH --output="+str(ROOT/"extension/logs/extension-source-grader-diagnostic-%j.out"),"#SBATCH --error="+str(ROOT/"extension/logs/extension-source-grader-diagnostic-%j.err")]
m.require(directives==expected,"Exact bounded zero-GPU wrapper required")
command=["sbatch","--clusters=owl","--parsable",str(ROOT/SCRIPT)]
binding={"role":ROLE,"command":command,"source_sha256":sources,"failure_evidence":assets,"prior_entries_sha256":sha(m._canonical(archived["jobs"])),"admission_code_sha256":record(Path(__file__).resolve())["sha256"],"requested_gpu_hours":0,"requested_gpu_seconds":0,"wall_time_seconds":600,"closeout_reserve_seconds":172800,"deadline_utc":m.DEADLINE.isoformat(),"automatic_downstream":False,"scope":"fixed_public_synthetic_fixtures_only_no_source_labels_models_or_outputs","scorer_changes":False,"timeout_seconds":5,"repetitions":2}
checksum=sha(m._canonical(binding))
with (ROOT/"extension/job_ledger.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=ROOT/"extension/job_ledger.json"; ledger=m._read(path)
 m.require(ledger.get("gpu_hour_cap","missing") is None and ledger.get("historical_ledger_unchanged")=="deadline/job_ledger.json" and set(ledger["jobs"])<=set(archived["jobs"])|{ROLE} and all(ledger["jobs"].get(k)==v for k,v in archived["jobs"].items()),"Preserve all10 earlier jobs")
 q=m._queue_snapshot(subprocess.run)
 prior.check_queue(q,{k:v for k,v in archived["jobs"].items() if k!=prior.ROLE})
 for row in q["jobs"]:
  if row["job_id"]=="1225360":m.require(row["name"]==prior.NAME and row["state"]=="FAILED","Prior CPU retry must be exact terminal failure")
  elif row["name"]==prior.NAME or row["job_id"].split("_")[0]=="1225360":raise m.SubmissionError("Untracked prior retry job")
 same={r["job_id"] for r in q["jobs"] if r["name"]==NAME}
 observed=prior.source_inventory(ROOT)
 for n in ("extension/source_grader_diagnostic.py",SCRIPT):observed[n]=record(ROOT/n)
 m.require(observed==sources and all(record(ROOT/n)==v for n,v in assets.items()) and record(Path(__file__).resolve())["sha256"]==binding["admission_code_sha256"],"Source/evidence changed during diagnostic admission")
 existing=ledger["jobs"].get(ROLE)
 if existing is not None:
  m.require(existing.get("status")=="submitted" and existing.get("binding")==binding and existing.get("binding_sha256")==checksum and not same-{existing.get("job_id")},"Existing/uncertain diagnostic must not be resubmitted")
  print(existing["job_id"]);raise SystemExit
 m.require(not same,"Untracked diagnostic admission")
 guard()
 environment,removed=m._submission_environment()
 entry={"status":"submitting","binding":binding,"binding_sha256":checksum,"source_sha256":sources,"command":command,"requested_gpu_hours":0,"requested_gpu_seconds":0,"requested_at_utc":datetime.now(timezone.utc).isoformat(),"scheduler_snapshot":q,"removed_scheduler_environment_keys":removed}
 ledger["jobs"][ROLE]=entry;m._write(path,ledger)
 try:r=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
 except (OSError,subprocess.SubprocessError) as error:
  entry.update(status="uncertain",error_type=type(error).__name__);m._write(path,ledger);raise
 entry.update(scheduler_returncode=r.returncode,scheduler_stdout=r.stdout[-2000:],scheduler_stderr=r.stderr[-2000:])
 match=re.fullmatch(r"([1-9][0-9]*)(?:;owl)?",r.stdout.strip())
 if r.returncode or match is None:
  entry["status"]="submission_failed" if r.returncode else "uncertain";m._write(path,ledger);raise m.SubmissionError("Preserve failed or ambiguous diagnostic; no retry")
 entry.update(status="submitted",job_id=match.group(1),submitted_at_utc=datetime.now(timezone.utc).isoformat());m._write(path,ledger)
 print(entry["job_id"])
