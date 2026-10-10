"""Exact explicit source pins only. No manifests, banks, global scans or GPU calls."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import torch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/anchor_compact_qualification_runner_v1_independent'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OUT/'source_snapshot_manifest.json').read_text())
checks={}
for name,entry in manifest.items():
    current=ROOT/name;snapshot=OUT/'sealed_source'/name
    checks[name]=dict(expected=entry['sha256'],current_sha256=sha(current),snapshot_sha256=sha(snapshot),bytes=current.stat().st_size)
    assert current.read_bytes()==snapshot.read_bytes()
config=json.loads((ROOT/'configs/anchor_compact_actual_bank_v1.json').read_text())
pins={name:dict(expected=digest,actual=sha(ROOT/name)) for name,digest in config['source_sha256'].items()}
assert all(x['expected']==x['actual'] for x in pins.values())
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
metrics=json.loads((OUT/'independent_metrics_attempt02.json').read_text())
numerical=[]
for cell in metrics:
    numerical+=list(cell['result']['checks'].values()) if 'result' in cell else cell['checks']
assert all(x['passed'] for x in numerical)
report=dict(all_snapshot_source_bytes_unchanged=True,all_frozen_adapter_source_map_pins_match=True,
    exact_sources=checks,frozen_adapter_source_map=pins,
    numerical_summary=dict(recorded_cells=len(metrics),comparisons=len(numerical),elements=sum(x['elements'] for x in numerical),all_passed=True,max_absolute_error=max(x['max_absolute_error'] for x in numerical),max_acceptance_units=max(x['max_acceptance_units'] for x in numerical),
        default_performer_cells=3,stage_math_cells=4,explicit_AdamW_oracle_cells=1,
        dtype_scope=['CPU_float64','CPU_float32_stage_math'],Q=184,R=96,graphs=2,readouts=7,anchors=32,scales=3),
    numerical_attempt01_attempt02_bytes_identical=(OUT/'independent_metrics_attempt01.json').read_bytes()==(OUT/'independent_metrics_attempt02.json').read_bytes(),
    runtime=dict(python=platform.python_version(),torch=torch.__version__,python_executable=sys.executable,python_executable_sha256=sha(Path(sys.executable)),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_context_initialized=torch.cuda.is_initialized()))
p=OUT/'source_and_numerical_audit_v1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('exact_sources','frozen_adapter_source_map')},indent=2,sort_keys=True))
