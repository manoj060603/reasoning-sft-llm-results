"""Native CPU-only QA of the joint GPU metadata adapter; no model/data work."""
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,unittest
from pathlib import Path
from extension.cpu_feasibility import require_cpu_allocation,deadline_guard,read_json,write_new_json,require,value_digest,digest
from extension import submission as metadata
ROLE="joint_gpu_metadata_qa_20261021_v1"
EXPECTED_TESTS=46
SOURCES=("extension/joint_gpu_admission.py","tests/test_joint_gpu_admission.py","extension/joint_gpu_metadata_qa.py","extension/joint_gpu_metadata_qa.slurm")
def source_inventory(root):
 root=Path(root).resolve();result={}
 for n in SOURCES:
  raw=metadata._small_bytes(root/n,root);result[n]={"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
 return result
def main():
 compute=require_cpu_allocation();deadline_guard(173400)
 require(compute["allocated_cpus"]==2,"Exact CPU2 metadata allocation required")
 root=Path.cwd().resolve();out=Path(sys.argv[1]).resolve()
 require(root==Path(__file__).resolve().parents[1] and out.parent==root/"extension/runs" and out.name=="joint-gpu-metadata-qa-"+compute["job_id"] and not out.exists(),"Fresh current CPU output required")
 before=source_inventory(root);ledger=read_json(root/"extension/job_ledger.json");entry=ledger["jobs"].get(ROLE,{})
 require(type(EXPECTED_TESTS) is int and EXPECTED_TESTS>0,"Root must first freeze exact test count")
 require(entry.get("status")=="submitted" and entry.get("job_id")==compute["job_id"] and entry.get("source_sha256")==before and entry.get("requested_gpu_seconds")==0 and entry.get("binding",{}).get("source_sha256")==before and entry["binding"].get("role")==ROLE and entry["binding"].get("expected_native_tests")==EXPECTED_TESTS and entry["binding"].get("wall_time_seconds")==600 and entry["binding"].get("automatic_downstream") is False and entry.get("binding_sha256")==value_digest(entry["binding"]),"Exact current metadata QA admission missing")
 priors={k:v for k,v in ledger["jobs"].items() if k!=ROLE}
 require(len(priors)==12 and entry["binding"].get("prior_entries_sha256")==value_digest(priors),"Preserve exact12 scientific prior entries")
 parent=root/"extension/runs/joint-memory-qa-1225557/native_qa.json"
 require(digest(parent)=="00af45d78452c1d0a6d6aacf24518e9e77b749db39a2cbbd94b724a4f2cca4ee","Exact prior410 native QA required")
 scientific=read_json(parent)["source_inventory"]
 for n,v in scientific.items():
  raw=metadata._small_bytes(root/n,root);require({"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}==v,"Scientific source changed: "+n)
 for n in ("extension/joint_gpu_metadata_qa.slurm","extension/joint_memory_admission.slurm"):subprocess.run(["bash","-n",str(root/n)],check=True)
 out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
 write_new_json(out/"started.json",{"schema":"extension-joint-gpu-metadata-qa-start-v1","job_id":compute["job_id"],"source_inventory":before,"expected_tests":EXPECTED_TESTS})
 suite=unittest.defaultTestLoader.discover(str(root/"tests"),pattern="test_joint_gpu_admission.py",top_level_dir=str(root/"tests"))
 require(suite.countTestCases()==EXPECTED_TESTS,"Exact new metadata test module count changed")
 begun=datetime.now(timezone.utc).isoformat();result=unittest.TextTestRunner(verbosity=2).run(suite)
 if result.testsRun!=EXPECTED_TESTS or not result.wasSuccessful() or result.skipped or result.expectedFailures or result.unexpectedSuccesses:
  write_new_json(out/"unit_qa_failure.json",{"schema":"extension-joint-gpu-admission-native-qa-v1","success":False,"job_id":compute["job_id"],"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"expected_failures":len(result.expectedFailures),"unexpected_successes":len(result.unexpectedSuccesses),"source_inventory":before,"compute_proof":compute,"automatic_retry":False})
  raise RuntimeError("Metadata QA failed; preserve without resubmission")
 require(source_inventory(root)==before and read_json(root/"extension/job_ledger.json")==ledger,"Source/ledger changed during metadata QA")
 for n,v in scientific.items():
  raw=metadata._small_bytes(root/n,root);require({"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}==v,"Scientific source changed during QA")
 ending=require_cpu_allocation();require(ending["job_id"]==compute["job_id"] and ending["node"]==compute["node"],"CPU allocation changed")
 receipt={"schema":"extension-joint-gpu-admission-native-qa-v1","success":True,"job_id":compute["job_id"],"node":compute["node"],"compute_proof":compute,"compute_end":ending,"gpu_disabled":True,"wrapper_syntax_passed":True,"tests_run":result.testsRun,"failures":0,"errors":0,"skipped":0,"expected_failures":0,"unexpected_successes":0,"source_inventory":before,"code_sha256":{n:v["sha256"] for n,v in before.items()},"admission_binding_sha256":entry["binding_sha256"],"prior_admissions":priors,"scientific_parent":{"path":str(parent.relative_to(root)),"sha256":digest(parent)},"source_grader_executed":False,"original_model_or_private_cache_loaded":False,"actual_model_or_memory_admission_established":False,"automatic_downstream":False,"started_at_utc":begun,"created_at_utc":datetime.now(timezone.utc).isoformat()}
 write_new_json(out/"native_qa.json",receipt)
 write_new_json(out/"complete.json",{"schema":"extension-joint-gpu-metadata-qa-complete-v1","complete":True,"job_id":compute["job_id"],"cpu_only":True,"gpu_seconds":0,"tests_run":result.testsRun,"native_qa_sha256":digest(out/"native_qa.json"),"automatic_downstream":False,"actual_model_or_memory_admission_established":False})
 print(json.dumps({"metadata_qa_complete":True,"job_id":compute["job_id"],"tests_run":result.testsRun,"gpu_seconds":0}),flush=True)
if __name__=="__main__":main()
