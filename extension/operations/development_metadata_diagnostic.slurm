#!/bin/bash
#SBATCH --job-name=sft-extension-level-diagnostic
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-level-diagnostic-%j.out
#SBATCH --error=/projects/slmreasoning/manoj/reasoning-sft-deadline-20261019/extension/logs/extension-level-diagnostic-%j.err
set -euo pipefail
cd /projects/slmreasoning/manoj/reasoning-sft-deadline-20261019
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
"$PWD/.venv/bin/python" - <<'PY'
from collections import Counter
from pathlib import Path
import hashlib,json,os,re,subprocess
from datetime import datetime,timezone
from extension.cpu_feasibility import require_cpu_allocation
root=Path.cwd(); proof=require_cpu_allocation(); job=os.environ['SLURM_JOB_ID']
ledger=json.loads((root/'extension/job_ledger.json').read_text())
entry=ledger['jobs']['development_level_diagnostic_20261021_v1']
assert entry['job_id']==job and entry['status']=='submitted'
source=root/'extension/operations/development_metadata_diagnostic.slurm'
assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['binding']['wrapper_sha256']
subprocess.run(['bash','-n',str(source)],check=True)
pop=root/'extension/runs/labeled-feasibility-20261020-1222926/surviving.jsonl'
assert hashlib.sha256(pop.read_bytes()).hexdigest()=='f1304725d60087666fa2c3995fe7b4b344067ad28d1f2af35e8dbc5dab4e0551'
rows=[json.loads(line) for line in pop.read_text().splitlines()]; math=[r for r in rows if r['source_kind']=='math']
assert len(rows)==7132 and len(math)==6013
counts=Counter((r['source']['file'].split('/')[0],r['original_record']['level']) for r in math)
invalid=Counter(r['original_record']['level'] for r in math if re.fullmatch(r'Level [1-5]',r['original_record']['level']) is None)
receipt={'schema':'extension-development-level-diagnostic-v1','complete':True,'compute':proof,'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':hashlib.sha256(pop.read_bytes()).hexdigest(),'rows':len(rows),'math_rows':len(math),'subject_level_counts':[{'subject':s,'official_level':level,'rows':n} for (s,level),n in sorted(counts.items())],'non_1_to_5_levels':dict(sorted(invalid.items())),'no_labels_or_solutions_inspected':True,'no_split_assignment_or_model_outcomes':True,'gpu_seconds':0,'admission_binding_sha256':entry['binding_sha256']}
out=root/('extension/runs/development-level-diagnostic-20261021-'+job);out.mkdir(exist_ok=False)
with (out/'complete.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'complete':True,'job_id':job,'non_1_to_5_levels':receipt['non_1_to_5_levels'],'math_rows':len(math)}))
PY
