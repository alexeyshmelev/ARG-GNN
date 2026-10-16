"""Focused source-only metadata checks. No genuine authority, guards or payloads."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy, gzip, hashlib, importlib, json, math
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/reviews/FLARE_original_VAL001004_completion_independent_review_20261017_v1'
SOURCE=ROOT/'research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1'
module=importlib.import_module('research.reviews.FLARE_original_VAL001004_completion_source_20261017_v1.completion_v1')
PINS={'completion_v1.py':'522cacd5e580d72c2723beee12022de5efff06b14003b6c59e8be57d5027bd38','source_manifest.v1.json':'3592b0fae199d43b01bd1c3cb87480423bb2045e6b87cdbed438009ad09bed0c','source_handoff.v1.json':'cf8908d12a7cd02cab506184930b635f06274c6d46714a6948cc5aecd3805153'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in PINS.items(): assert sha(SOURCE/name)==digest
manifest=json.loads((SOURCE/'source_manifest.v1.json').read_text())
pins=json.loads((SOURCE/'source_pins.v1.json').read_text())
for row in manifest['files']+pins['refs']:
 p=ROOT/row['path'];assert p.suffix in {'.py','.json'} and sha(p)==row['sha256'],row['path']
module._sources()
actual_tools=module._tools
read,bound,ref=actual_tools()
checks={}
class Spy:
 def __init__(self): self.writes=[];self.guards=[];self.truth=[]
 def json(self,*args): self.writes.append(args);raise AssertionError('No writer permitted')
 def guard(self,*args): self.guards.append(args);raise AssertionError('No runtime guard permitted')
 def resolve(self,*args): self.truth.append(args);raise AssertionError('No truth resolver permitted')
def rejected(action,contains):
 try: action()
 except ValueError as error:
  assert contains in str(error),str(error)
  return str(error)
 raise AssertionError('Expected rejection')
spy=Spy()
rows={parent:{} for parent in module.MISSING[:-1]}
checks['public_freeze_missing_VAL004']=rejected(lambda:module.freeze_remaining_predictions_v1(None,rows,max_npz_uncompressed_bytes=1,writer=spy,runtime_guard=spy.guard),'Allfour')
rows={parent:{} for parent in module.MISSING};rows[module.MISSING[-1]]=None
checks['public_freeze_null_VAL004']=rejected(lambda:module.freeze_remaining_predictions_v1(None,rows,max_npz_uncompressed_bytes=1,writer=spy,runtime_guard=spy.guard),'Allfour')
base=json.loads((SOURCE/'context.INACTIVE.v1.json').read_text())
base.update(root_authorizer='root_agent',remaining_VAL_truth_not_previously_opened=True)
# In-memory negative specimen, never saved or used as genuine authority.
for missing in (False,True):
 ctx=copy.deepcopy(base)
 if missing:del ctx['targets'][module.MISSING[-1]]
 reads=[]
 def controlled_read(record):
  reads.append(record);assert record=='SOURCE_ONLY_NEGATIVE_CONTEXT'
  return ctx
 module._tools=lambda:(controlled_read,bound,ref)
 try:
  checks['context_'+('missing' if missing else 'null')+'_VAL004']=rejected(lambda:module._context('SOURCE_ONLY_NEGATIVE_CONTEXT',module.MISSING[-1]),'Allfour' if missing else 'Selected original')
  assert reads==['SOURCE_ONLY_NEGATIVE_CONTEXT']
 finally:module._tools=actual_tools
checks['null_missing_context_no_downstream_metadata_or_payload_reads']=True
# Actual scorer rejects a four-parent barrier before sources, cells or truth.
now=datetime.now(timezone.utc)
barrier={'status':'sealed_FLARE_original_remaining4_before_their_VAL_truth_v1','context_ref':'SOURCE_ONLY_CONTEXT','source_manifest_ref':module._local_ref('source_manifest.v1.json'),'fixed_VAL000_basis_ref':module._local_ref('VAL000_basis.v1.json'),'all4_remaining_sealed_before_their_truth':True,'VAL000_truth_previously_exposed':True,'VAL000_exploratory_retained':True,'VAL000_native_rerun':False,'cells':[{'parent_id':p}for p in module.VAL[:-1]],'max_npz_uncompressed_bytes':1}
authority={'created_at_utc':(now-timedelta(minutes=1)).isoformat(),'expires_at_utc':(now+timedelta(minutes=1)).isoformat(),'schema':'FLARE_original_remaining_VAL_ROOT_score_authority_v1','root_authorizer':'root_agent','active':True,'barrier_ref':'SOURCE_ONLY_BARRIER','source_manifest_ref':module._local_ref('source_manifest.v1.json'),'context_ref':'SOURCE_ONLY_CONTEXT','all4remaining_seals_before_truth':True,'max_npz_uncompressed_bytes':1,'max_truth_uncompressed_bytes':1,'VAL001004_truth_authorized':True,'VAL000_supplemental_raw_NLL_authorized':True,'native_rerun_authorized':False,'TEST_authorized':False}
reads=[]
def controlled_read(record):
 reads.append(record)
 return {'SOURCE_ONLY_BARRIER':barrier,'SOURCE_ONLY_AUTHORITY':authority}[record]
module._tools=lambda:(controlled_read,bound,ref)
try:
 checks['public_score_incomplete_barrier']=rejected(lambda:module.score_five_original_VAL_v1('SOURCE_ONLY_BARRIER',authority_ref='SOURCE_ONLY_AUTHORITY',max_npz_uncompressed_bytes=1,max_truth_uncompressed_bytes=1,writer=spy,resolve_val_truth=spy.resolve,runtime_guard=spy.guard),'Exact descriptive')
 assert reads==['SOURCE_ONLY_BARRIER','SOURCE_ONLY_AUTHORITY']
finally:module._tools=actual_tools
assert not spy.writes and not spy.guards and not spy.truth
checks['public_negative_gate_callbacks']={'writes':0,'runtime_guards':0,'truth_resolvers':0}
# Tiny synthetic header only; no VCF data record or scientific payload.
header=REVIEW/'synthetic_header_metadata_only.v1.vcf.gz'
header.write_bytes(gzip.compress(b'##fileformat=VCFv4.2\n##ANCESTRY=<AFR=0,EUR=1,EAS=2>\n#CHROM\tPOS\n',mtime=0))
native={'native_output_refs':{'.anc.vcf.gz':ref(str(header.relative_to(ROOT)))}}
assert module._header_ancestry(native,['AFR','EUR','EAS'])=={'AFR':0,'EUR':1,'EAS':2}
checks['actual_header_permutation_pass']=True
checks['actual_header_serialized_order_mismatch_rejected']=rejected(lambda:module._header_ancestry(native,['EUR','EAS','AFR']),'Raw posterior ancestryorder')
summary=module._summary([1,2,3,4,5])
assert summary['equal_parent_mean']==3 and math.isclose(summary['sample_SD'],math.sqrt(2.5)) and summary['SD_ddof']==1 and list(summary['per_parent'])==list(module.VAL)
checks['actual_five_parent_summary']=summary
result={'schema':'FLARE_original_remaining_independent_focused_metadata_checks_v1','status':'PASS','source_pins':PINS,'manifest_entries_authenticated':len(manifest['files']),'source_pin_entries_authenticated':len(pins['refs']),'unique_metadata_or_source_paths_authenticated':len({r['path']for r in manifest['files']+pins['refs']}),'actual_source_pin_callable_pass':True,'checks':checks,'limits':'Controlled in-memory metadata stubs exercise negative public gates and selected context validation. Actual sources imported. No genuine authority, guard, payload, truth, binary, supervisor, resource probe, admission, scientific process, download or Git. Only synthetic header and five scalar values used.'}
out=REVIEW/'focused_metadata_checks.result.v1.json'
out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
