"""ROOT prepares one fresh full H058 v5 admission after genuine source review."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
def ref(p):
 b=(ROOT/p).read_bytes();return {'path':p,'sha256':hashlib.sha256(b).hexdigest()}
def read(r):
 b=(ROOT/r['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256'];return json.loads(b)
def save(name,v):
 p=P/name
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 return ref(str(p.relative_to(ROOT)))
a=argparse.ArgumentParser();a.add_argument('--review-ref',required=True);args=a.parse_args()
review_ref=json.loads(args.review_ref);review=read(review_ref)
ns='research.reviews.H058_GPU0_full_training_successor_source_20261017_v5'
c=importlib.import_module(ns+'.common_v1');d=importlib.import_module(ns+'.direct_gpu0_v1')
assert review['independent'] is True and review['verdict']=='SOURCE_PASS' and review['blockers']==[]
assert review['source_ref']==ref(c.DRIVER) and review['source_manifest_ref']==ref(c.MANIFEST)
q=read(ref(c.HERE+'/ROOT_request.INACTIVE.template.v1.json'))
P=ROOT/q['preparation_directory'];assert not P.exists()
assert not (ROOT/q['packet_directory']).exists() and not (ROOT/q['output_directory']).exists()
old=read(c.CLOCK_ROOT_REQUEST_V4);oldcfg=read(old['config_ref'])
P.mkdir()
created=q['created_at_utc'];expires=q['expires_at_utc']
assert (created,expires)==(old['created_at_utc'],old['expires_at_utc'])
handoff=read(q['science_handoff_ref'])
cfg=copy.deepcopy(oldcfg)
cfg['producer_ref']=handoff['producer_ref'];cfg['source_manifest_ref']=handoff['source_manifest_ref']
for k,name in [('runtime_equivalence_ref','runtime_equivalence.root_active_v1.json'),('layout_resource_proof_ref','layout_resource_proof.root_active_v1.json')]:
 cfg[k]=save(name,read(oldcfg[k]))
directive=read(oldcfg['full_training_directive_ref']);directive['GPU_entry_ref']=cfg['producer_ref']
cfg['full_training_directive_ref']=save('full_training_directive.root_active_v1.json',directive)
q['config_ref']=save('scientific_config.root_active_v1.json',cfg)
prior=read(old['prior_H058_CPU_disposition_ref'])
prior.update(new_gpu_request_job_id=q['job_id'],new_full_GPU_budget_explicitly_authorized=True,old_launch_resume_write_routes_revoked=True,old_files_charges_clocks_and_counters_preserved=True,prior_GPU_v4_actual_wait_ref=c.PRIOR_V4_ACTUAL_WAIT_REF,prior_GPU_v4_actual_outer_terminal_ref=c.PRIOR_V4_ACTUAL_OUTER_REF,prior_GPU_v4_files_charges_clocks_and_failure_preserved=True,created_at_utc=created,expires_at_utc=expires,repeat_reason='V4 genuine worker1/outer1/drain and zero updates before context due inherited24hour clock rule. New separately reviewed v5 full unfinished cell keeps original authentic finite ROOT clocks, all32parents/1920updates, original warmstart and fresh optimizers. Original CPU final counters remain UNKNOWN.')
q['prior_H058_CPU_disposition_ref']=save('prior_CPU_and_GPU_disposition.root_active_v1.json',prior)
base=d.base_v1()
native=read(old['native_resource_disposition_ref'])
oldabsence=read(native['fresh_targeted_native_absence_ref'])
rows=[]
for expected in native['native_process_identities']:
 try:observed=base.proc_identity(expected['pid'])
 except (FileNotFoundError,ProcessLookupError):observed=None
 live=bool(observed and observed['uid']==expected['uid'] and observed['start_ticks']==expected['start_ticks'] and observed['state'] not in ('Z','X'))
 assert not live
 rows.append(dict(expected=expected,observed=observed,exact_old_identity_live=live))
groups=[]
for pgid in native['native_process_groups']:
 try:os.killpg(pgid,0)
 except ProcessLookupError:pass
 else:raise ValueError('Original native process group remains present')
 groups.append(dict(pgid=pgid,absent=True,method='targeted os.killpg(pgid,0) ESRCH',checked_utc=datetime.now(timezone.utc).isoformat()))
gpu=[r for r in base.real_inventory() if r['index']==0];assert len(gpu)==1 and gpu[0]['uuid']==d.GPU0_UUID
assert not any(x['pid'] in {r['pid'] for r in native['native_process_identities']} for x in gpu[0]['compute'])
observed_at=datetime.now(timezone.utc).isoformat()
absence=copy.deepcopy(oldabsence)
absence.update(actual_identity_observations=rows,actual_process_group_observations=groups,actual_GPU0_inventory=gpu[0],observed_at_utc=observed_at)
absence_ref=save('native_targeted_absence.root_active_v1.json',absence)
native.update(adapter_manifest_ref=ref(c.MANIFEST),H058_job_id=q['job_id'],fresh_targeted_native_absence_ref=absence_ref,observed_at_utc=observed_at,created_at_utc=created,expires_at_utc=expires)
q['native_resource_disposition_ref']=save('native_resource_disposition.root_active_v1.json',native)
cap=read(d.CURRENT_CAP);controller_storage=read(cap['controller_storage_ref'])
registry=json.loads((ROOT/controller_storage['output_directory']/'registry.json').read_bytes())
registry_ref=save('controller_registry.actual_snapshot.v1.json',registry)
terminal=read(old['controller_terminal_qualification_ref']);terminal['registry_snapshot_ref']=registry_ref
q['controller_terminal_qualification_ref']=save('controller_terminal.root_active_v1.json',terminal)
authority=read(old['direct_attachment_authority_ref'])
authority.update(H058_job_id=q['job_id'],native_resource_disposition_ref=q['native_resource_disposition_ref'])
q['direct_attachment_authority_ref']=save('direct_attachment_authority.root_active_v1.json',authority)
q.update(status='active',execution_authorized=True,source_review_passed=True,source_review_ref=review_ref)
q['fit_clock_compatibility_v5']['accepted']=True
request_ref=save('ROOT_request.root_active_v1.json',q)
c.validate_request_v1(request_ref)
capacity=d.capacity_v1(q,base)
summary_ref=save('ROOT_preparation_summary.root_v1.json',dict(schema='ROOT_H058_GPU0_v5_actual_preparation_v1',root_authorizer='root_agent',actual_prepared_utc=datetime.now(timezone.utc).isoformat(),request_ref=request_ref,source_review_ref=review_ref,actual_native_absence_ref=absence_ref,actual_GPU0_capacity=capacity,metadata_preflight='PASS',no_reserve_or_science=True,created_at_utc=created,expires_at_utc=expires,original_files_charges_clocks_and_failures_preserved=True,new_full_scope_budget_authorized=True,clock_compatibility_accepted_only_for_exact_new_v5=True))
print(json.dumps({'request_ref':request_ref,'summary_ref':summary_ref,'no_reserve_or_science':True,'admit_argv':[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',ns+'.root_driver_v1','--admit','--request-ref',json.dumps(request_ref,separators=(',',':'))]}),flush=True)
