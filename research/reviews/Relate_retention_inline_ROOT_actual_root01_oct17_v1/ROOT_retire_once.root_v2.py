from pathlib import Path
import json,hashlib,importlib,fcntl,os,stat,time,datetime,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'research/reviews/Relate_retention_inline_ROOT_actual_root01_oct17_v1'
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def put(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
phase='construct_exact_reviewed_helper'
try:
 instruction_ref=ref(OUT/'ROOT_instruction.root_active_v1.json')
 assert instruction_ref['sha256']=='caabe15aefa80c95597a0b181f3ec0f1a2403c6454acd5a14151164fd35c868a'
 spec=json.loads((OUT/'ROOT_instruction.root_active_v1.json').read_text())
 assert all(not (ROOT/path).exists() for path in spec['fresh_metadata_paths'].values())
 selector={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
 assert ref(ROOT/selector['path'])==selector
 module=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1')
 g=module.load_selected_guard_v1(selector)
 source=ROOT/spec['helper_ref']['path'];assert ref(source)==spec['helper_ref']
 h=importlib.import_module('research.reviews.Relate_retirement_inline_ROOT_evidence_source_20261017_v1.inline_ROOT_retire_v1')
 helper=h.InlineRootRetireStoppedMemo(g,root_instruction_ref=instruction_ref)
 phase='first_actual_original_blocking_mutex_call'
 assert all(not(ROOT/path).exists() for path in spec['fresh_metadata_paths'].values())
 policy=json.loads((ROOT/'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json').read_text())
 phase='actual_one_inline_ROOT_retirement';t=time.monotonic()
 put(OUT/'ROOT_actual_retirement_invocation.root_v2.json',{'pid':os.getpid(),'instruction_ref':instruction_ref,'observed_utc':g._utc(),'original_operation_cap_seconds':300,'original_fresh_proof_seconds':120,'automatic_retry':False})
 print(json.dumps({'phase':phase,'pid':os.getpid(),'time_utc':g._utc()}),flush=True)
 receipt=helper.apply_retire_stopped_v1(expected_source_sha256=policy['implementation_ref']['sha256'],expected_launcher_sha256=policy['launcher_ref']['sha256'])
 put(OUT/'actual_retirement_receipt.root_v1.json',receipt)
 result={'actual_operation_elapsed_seconds':time.monotonic()-t,'receipt_ref':ref(OUT/'actual_retirement_receipt.root_v1.json'),'receipt_kind':receipt.get('kind'),'retained_bytes':receipt.get('retained_bytes'),'credit_bytes':receipt.get('credit_bytes'),'projected_bytes':receipt.get('projected_bytes'),'memo_counts':helper.counts,'ROOT_outer0_and_final_journal_acceptance_pending':True,'no_scientific_success_or_file_deletion_asserted':True}
 put(OUT/'actual_retirement_result.root_v1.json',result);print(json.dumps(result),flush=True)
except BaseException:
 text=traceback.format_exc();put(OUT/'actual_retirement_failure.root_v2.json',{'phase':phase,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'traceback':text,'automatic_retry':False,'credit_not_accepted':True});print(text,flush=True);raise SystemExit(1)
