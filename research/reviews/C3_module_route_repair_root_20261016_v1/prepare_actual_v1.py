"""Root execution recipe for the reviewed bootstrap routing correction."""
from pathlib import Path
import copy,datetime,hashlib,importlib,json,os,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
FAMILY='research/reviews/C3_native_worker_module_route_fix_source_20261016_v3'
HANDOFF={'path':FAMILY+'/source_handoff.v3.json','sha256':'5199bcfd129a09a3e79a4c9646cc25a76062f778d6a5b90e6bae9c042e4c115c'}
BASE='research/reviews/C3_module_route_repair_root_20261016_v1'
OLD='research/reviews/C3_actual_full_state_handover_root_20261016_v1'
C3_INPUT={'path':'research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/input.json','sha256':'b420377394b1246a55dd2ce2a731f523692ef557d8ea074d2fd3137231008a25'}
PRIOR={'path':OLD+'/prior_controller_closed_transport_unknown.proof_v1.json','sha256':'e545b8c331237fcb9da085125d7fd45b76826203c836799fe2af076d4eff54a6'}
def ref(name):
 p=ROOT/name
 assert p.resolve()==p and p.is_file() and not p.is_symlink()
 return {'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(name,obj):
 path=ROOT/BASE/name
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(path.relative_to(ROOT)))
def run():
 assert Path.cwd()==ROOT and ref(HANDOFF['path'])==HANDOFF
 module=FAMILY.replace('/','.')
 e=importlib.import_module(module+'.finite_envelopes_v1')
 p=importlib.import_module(module+'.prepare_resume_v1')
 pool=e.pool_v1(); c=pool.contract
 proof=copy.deepcopy(p.read(PRIOR))
 proof['bootstrap_failed_controller_case_v1']=c.transport_helper_v1().bootstrap_case_metadata_v1(c,c.TRANSPORT_HELPER_REF)
 prior=save('bootstrap_handover.proof.root_active_v1.json',proof)
 c.validate_previous_controller_v1(prior)
 methods={'C3_RESUME_TRAIN':p.c3_method_v1(C3_INPUT),'NATIVE_MLM':p.native_method_unchanged_v1()}
 assert methods['NATIVE_MLM']['input_ref']=={'path':'research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/input.json','sha256':'52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2'}
 oldcap=p.read(proof['prior_capability_ref'])
 pool.legacy.install_marker_renewal(oldcap['marker_repair_renewal_ref'])
 guard=pool.legacy.current_guard()
 oldstorage=p.read(p.read(oldcap['controller_storage_ref'])['reservation']['root_admission'])
 components=p.read(oldcap['controller_storage_ref'])['budget_components']
 source_sha=components['implementation']['sha256']; launcher_sha=components['launcher']['sha256']
 def inspect():
  return guard.inspect_v1(expected_source_sha256=source_sha,expected_launcher_sha256=launcher_sha)
 view=inspect();bindings=list(view['bindings'].values())
 # Pins are refreshed for every actual reserve through the unchanged mutex guard.
 def reserve(tag,request,job_id,expiry):
  state=inspect()
  auth=guard.operation_draft_v1('reserve',request,expected_before=guard._pins(state),bindings=state['bindings'])
  auth.update(active=True,expires_at_utc=expiry)
  authority_ref=save(tag+'/reserve_authority.json',auth)
  receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=source_sha,expected_launcher_sha256=launcher_sha)
  receipt_ref=save(tag+'/reserve_receipt.json',receipt)
  storage=e.storage_from_actual_reserve_v1(request,receipt_ref,job_id=job_id,expires_at_utc=expiry,actual_budget_components=components,actual_policy_ref=pool.legacy.POLICY)
  value=save(tag+'/storage.json',storage)
  print(json.dumps({'admitted':job_id,'direct_bindings':len(request['consumer_bindings']),'bytes':request['reserved_bytes'],'projected_bytes':receipt['projected_bytes'],'storage_ref':value}),flush=True)
  return value
 reuse=save('native_never_entered_scientific_reuse.root_v1.json',e.native_reuse_proof_v1(prior))
 slots={}
 for slot,job_id,out in [('C3_RESUME_TRAIN','C3-routefixed-outer-GPU0-v1','experiments/results/C3_routefixed_outer_GPU0_v1'),('NATIVE_MLM','native-MLM-routefixed-outer-GPU0-v1','experiments/results/native_MLM_routefixed_outer_GPU0_v1')]:
  req=save(slot+'/operational_request.json',e.operational_request_v1(slot,methods[slot],prior,actual_native_reuse_ref=reuse if slot=='NATIVE_MLM' else None))
  reserve_request=e.outer_reserve_request_v1(req,job_id=job_id,output_directory=out,actual_guard_bindings=bindings)
  expiry=p.read(methods[slot]['input_ref'])['expires_at_utc']
  storage=reserve(slot+'/admission',reserve_request,job_id,expiry)
  slots[slot]=save(slot+'/envelope.json',e.operational_envelope_v1(req,storage,actual_guard_bindings=bindings))
 jobmap=save('job_map.root_active_v1.json',e.finite_job_map_v1(slots['C3_RESUME_TRAIN'],slots['NATIVE_MLM']))
 ends=[datetime.datetime.fromisoformat(p.read(m['input_ref'])['expires_at_utc'].replace('Z','+00:00')) for m in methods.values()]
 end=max(ends);drain=end+datetime.timedelta(minutes=10)
 plan=save('controller_plan.root_active_v1.json',e.controller_plan_v1(jobmap,prior,pool='research0-routefix-oct16-v1',controller_job_id='GPU0-routefixed-controller-v1',expires_at_utc=end.isoformat(),drain_expires_at_utc=drain.isoformat(),actual_guard_bindings=bindings))
 reserve_request=e.controller_reserve_request_v1(plan,output_directory='experiments/results/GPU0_routefixed_C3_native_controller_v1',actual_guard_bindings=bindings)
 storage=reserve('controller/admission',reserve_request,'GPU0-routefixed-controller-v1',drain.isoformat())
 capability=save('controller/capability.root_active_v1.json',e.controller_capability_v1(plan,storage,actual_guard_bindings=bindings))
 summary={'schema':'root_routefixed_existing_science_actual_admission_v1','actual_capability_ref':capability,'source_handoff_ref':HANDOFF,'prior_controller_proof_ref':prior,'C3_input_ref_unchanged':C3_INPUT,'native_input_ref_unchanged':methods['NATIVE_MLM']['input_ref'],'new_bytes':40*2**20,'scientific_inputs_reused':True,'new_scientific_reservations':0,'migration_replayed':False,'new_optimizer_updates':0}
 summary_ref=save('actual_admission_summary_v1.json',summary)
 print(json.dumps({'summary_ref':summary_ref,**summary}),flush=True)
 return 0
if __name__=='__main__':
 result=1
 try: result=run()
 except Exception:
  traceback.print_exc(file=sys.stdout);sys.stdout.flush()
  raise
 sys.exit(result)
