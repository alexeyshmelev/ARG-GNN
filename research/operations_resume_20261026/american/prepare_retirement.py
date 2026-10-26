"""Prepare a parent-specific readable retirement adapter without deleting data."""
import ast
import copy
import difflib
import json
from pathlib import Path
import sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from arggnn.benchmarking.paper_american import protocol as p
BASE=ROOT/'research/operations_resume_20261026/american'
SOURCE=ROOT/'src/arggnn/benchmarking/paper_american/retirement.py'
source=SOURCE.read_text()
original=source
source=source.replace('from . import protocol as p','from arggnn.benchmarking.paper_american import protocol as p',1)
marker='PROTECTED_REUSED = {"argformer_american_test_000", "argformer_american_test_001"}\n'
addition='''
RESUMED_PARENT = "argformer_american_validation_001"
RESUME_LEAF = "resume_20261026_batch119500"
PRESERVED_CHECKPOINT_BYTES = 2014910662

def producer_output(base, stage):
    native = base / stage / "native"
    return native / RESUME_LEAF if stage == "mlm" else native

def retained_nonmodel_bound():
    return 128 * 2**20 + PRESERVED_CHECKPOINT_BYTES

def preserved_checkpoint(request):
    need(request["parent_id"] == RESUMED_PARENT,
         "Only the stopped validation001 parent uses this retirement adapter")
    evidence = p.read(request["resume_evidence_ref"])
    reference = evidence["checkpoint_ref"]
    expected = CAMPAIGN / RESUMED_PARENT / "mlm/native/checkpoints/ep2-ba119500-rank0.pt"
    need(evidence["schema"] == "american_argformer_full_state_resume_20261026"
         and p.path(reference["path"]) == expected
         and expected.stat().st_size == PRESERVED_CHECKPOINT_BYTES,
         "Exact preserved interrupted checkpoint and storage charge required")
    return reference
'''
assert source.count(marker)==1
source=source.replace(marker,marker+addition,1)
changes=[
 ('and p.path(executed["output_directory"]) == base/stage/"native",',
  'and p.path(executed["output_directory"]) == producer_output(base, stage),'),
 ('    base, prep_ref, prep, records = _chain(request)\n',
  '    preserved = preserved_checkpoint(request)\n    base, prep_ref, prep, records = _chain(request)\n'),
 ('    retained = [prep_ref, prep["vocab_ref"], prep["observation_ref"], prep["focals_ref"],',
  '    retained = [preserved, request["resume_evidence_ref"], prep_ref, prep["vocab_ref"], prep["observation_ref"], prep["focals_ref"],'),
 ('        directory = native/"checkpoints"',
  '        directory = producer_output(base, stage)/"checkpoints"'),
 ('    need(remaining_nonmodel_bytes <= 128*2**20-8*2**20,',
  '    need(remaining_nonmodel_bytes <= retained_nonmodel_bound()-8*2**20,'),
 ('"Surviving source/vocab/statistics/logs must fit128MiB before any deletion")',
  '"Surviving metadata plus preserved interrupted checkpoint exceed explicit retention bound")'),
 ('retired_file_count=len(inventory), retained_nonmodel_cap_bytes=128*2**20,',
  'retired_file_count=len(inventory), retained_nonmodel_cap_bytes=retained_nonmodel_bound(),\n                  preserved_interrupted_checkpoint_ref=preserved,\n                  preserved_interrupted_checkpoint_bytes=PRESERVED_CHECKPOINT_BYTES,'),
]
for before,after in changes:
 assert source.count(before)==1,before
 source=source.replace(before,after,1)
ast.parse(source)
destination=BASE/'argformer_retirement.py'
with destination.open('x') as stream:stream.write(source)
with (BASE/'argformer_retirement.diff').open('x') as stream:
 stream.write(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
  fromfile=str(SOURCE.relative_to(ROOT)),tofile=str(destination.relative_to(ROOT)))))
plan=json.loads((ROOT/'configs/benchmarks/paper_campaign_20261024_v2.json').read_text())
task=copy.deepcopy(next(t for t in plan['tasks'] if t['id']=='paper_american_argformer_american_validation_001_retire'))
task['call']['module']='research.operations_resume_20261026.american.argformer_retirement'
task['call']['spec']['resume_evidence_ref']=p.ref(BASE/'argformer_evidence.json')
task['source_refs'] += [p.ref(BASE/'__init__.py'),p.ref(destination)]
task_ref=p.save(task,BASE/'argformer_retirement.replacement_task.json')
p.save(dict(schema='american_resume_retirement_preparation_v1',
 status='prepared_not_executed',original_source_ref=p.ref(SOURCE),source_ref=p.ref(destination),task_ref=task_ref,
 preserved_checkpoint_bytes=2014910662,retained_nonmodel_cap_bytes=128*2**20+2014910662,
 original_task_resources_unchanged=True,production_files_deleted=0),BASE/'argformer_retirement.prepared.json')
print(json.dumps(task_ref))
