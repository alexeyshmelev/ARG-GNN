from pathlib import Path
import json,hashlib,datetime,importlib,os,traceback,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');os.chdir(ROOT)
OUT=ROOT/'research/reviews/Relate_retention_ROOT_actual_memo2_20261017_v1'
def ref(p):
 p=Path(p);return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
try:
 src=ROOT/'research/reviews/Relate_retirement_transaction_memo_source_20261017_v1/retire_transaction_memo_v1.py'
 assert ref(src)['sha256']=='1c8a2038980e05d8d2bdea865907c8e10c51b8ccb00fd766999c3631e5662d12'
 instruction=json.loads((OUT/'ROOT_instruction.root_active_v1.json').read_text())
 m=importlib.import_module('research.reviews.Relate_failed64GiB_retention_successor_source_20261017_v1.relate_failed64_retention_v1')
 selector={'path':'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}
 g=m.load_selected_guard_v1(selector)
 h=importlib.import_module('research.reviews.Relate_retirement_transaction_memo_source_20261017_v1.retire_transaction_memo_v1')
 helper=h.RetireStoppedMemo(g,integration_manifest_ref=instruction['source_manifest_ref'],helper_ref=instruction['helper_ref'],review_ref=instruction['independent_review_ref'],memo_review_ref=instruction['memo_review_ref'],root_instruction_ref=ref(OUT/'ROOT_instruction.root_active_v1.json'),cache_refs=instruction['cache_refs'],uncached_current_refs=[])
 policy=json.loads((ROOT/'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json').read_text())
 t0=time.monotonic();state=helper.read_state_v1(expected_source_sha256=policy['implementation_ref']['sha256'],expected_launcher_sha256=policy['launcher_ref']['sha256'])
 key=instruction['key'];assert key in state[0] and key not in state[1]
 own=state[0][key];assert own['reserved_bytes']==68719476736 and g.budget.digest(own)==g._target(key)['original_reservation_sha256']
 rec={'schema':'ROOT_Relate64_actual_memo2_preoperation_state_v1','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t0,'selector_ref':selector,'ROOT_instruction_ref':ref(OUT/'ROOT_instruction.root_active_v1.json'),'actual_view':state[2],'expected_before':g._pins(state[2]),'actual_bindings':state[3],'actual_original_reservation':own,'actual_original_reservation_sha256':g.budget.digest(own),'target_key':key,'memo_counts':helper.core.counts,'no_retirement_operation_run':True,'no_credit_claimed':True}
 with (OUT/'actual_preoperation_state.root_v1.json').open('x') as f:json.dump(rec,f,indent=2)
 print(json.dumps({'state_ref':ref(OUT/'actual_preoperation_state.root_v1.json'),'elapsed_seconds':rec['elapsed_seconds'],'projected_bytes':state[2]['projected_bytes'],'actual_guard_role_count':len(state[3]),'memo_counts':helper.core.counts,'no_credit_claimed':True}),flush=True)
except BaseException:
 text=traceback.format_exc()
 with (OUT/'actual_preoperation_failure.root_v1.txt').open('x') as f:f.write(text)
 print(text,flush=True);raise
