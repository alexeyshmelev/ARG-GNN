"""Authenticate final recovery plan wiring and current idle controller identities."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
from research.joint_evaluation_20261029.root import recover_argmix_controllers_20261030 as handoff
from arggnn.benchmarking.paper_campaign import runtime as r

ROOT=r.ROOT
BASE=ROOT/'research/joint_evaluation_20261029'
REVIEW=BASE/'review_finish/recovery_20261030'
checks=[]
def read(path):return r.read(r.ref(path))
acceptance=read(REVIEW/'worker_controller_acceptance_v1.json')
assert acceptance['passed'] is True
manifest=r.read(acceptance['manifest_ref']);task_plan=r.read(acceptance['plan_ref'])
old=read(BASE/'root/finish_plan_v4.json');new=read(BASE/'root/finish_plan_v5.json')
changed={'argmix_completed_chain_path','argmix_task_plan_ref','model_chains','publisher_argv','source_refs'}
assert {k:v for k,v in old.items() if k not in changed}=={k:v for k,v in new.items() if k not in changed}
checks.append('finish_plan_changes_confined_to_ARGMix_operational_bindings_and_publisher')
assert new['argmix_task_plan_ref']==acceptance['plan_ref']
assert new['argmix_completed_chain_path']==str((Path(manifest['records_directory'])/'completed.json').relative_to(ROOT))
assert set(new['model_chains'])==set(old['model_chains'])
for name,spec in old['model_chains'].items():
    if name!='argmix_joint':assert new['model_chains'][name]==spec
checks.append('American_and_European_chains_and_existing_scoring_paths_preserved')
chain=new['model_chains']['argmix_joint'];oldchain=old['model_chains']['argmix_joint']
assert set(chain)==set(oldchain) and chain['GPU_stage_count']==11 and chain['binding_key']=='manifest_ref'
assert chain['binding_ref']==acceptance['manifest_ref']
assert chain['wait_path']==str((Path(manifest['records_directory'])/'gpu_chain_completed.json').relative_to(ROOT))
gpu=[task for task in task_plan['tasks'] if task['kind']=='gpu']
assert len(gpu)==len(chain['stages'])==11
for expected,spec in zip(gpu,chain['stages']):
    assert spec==dict(command=expected['argv_by_physical_gpu']['3'],plan_ref=acceptance['plan_ref'],task_id=expected['id'],wait_path=str((Path(manifest['records_directory'])/expected['id']/'actual_wait.json').relative_to(ROOT)))
checks.append('all11_GPU_wait_paths_commands_order_and_ownership_match_exact_reviewed_recovery')
publisher_ref=r.ref(BASE/'root/publish_results_recovery_v1.py')
assert new['source_refs']==old['source_refs']+[publisher_ref]
assert new['publisher_argv']==old['publisher_argv'][:-1]+['research.joint_evaluation_20261029.root.publish_results_recovery_v1','--manifest',acceptance['manifest_ref']['path']]
publisher_tests=read(REVIEW/'publisher_checks_result_v1.json')
assert publisher_tests['passed'] and publisher_tests['source_ref']['sha256']==publisher_ref['sha256']
checks.append('publisher_argv_exact_final_tested_wrapper_and_all_old_source_pins_retained')
old_final=read(BASE/'root/final_review_plan_v1.json');final=read(BASE/'root/final_review_plan_v2.json')
assert {k:v for k,v in old_final.items() if k!='finish_plan_ref'}=={k:v for k,v in final.items() if k!='finish_plan_ref'}
assert final['finish_plan_ref']==r.ref(BASE/'root/finish_plan_v5.json')
assert final['trigger_path']=='research/joint_evaluation_20261029/root/finish_records_v1/ready_for_final_review.json'
assert final['audit_source_ref']['sha256']=='d14fef02fbd5fdf4a627a89e90b9135ad5c2f3c9ee6be2fbccf74393aec8dc05'
checks.append('final_audit_changes_only_finish_plan_ref_retains_trigger_frozen_audit_and_truth_gates')
new_plan_refs={spec['name']:r.ref(spec['new_plan']) for spec in handoff.SPECS}
idle=[]
for spec in handoff.SPECS:
    assert handoff.source_check(spec['new_plan'],new_plan_refs[spec['name']])==new_plan_refs[spec['name']]
    old_launch=r.read(r.ref(spec['old']))
    idle.append(dict(name=spec['name'],original_launch_ref=r.ref(spec['old']),observed_identity=handoff.verify_idle(old_launch)))
checks.append('both_new_plans_and_all_sources_authenticate_current_two_old_controllers_are_verified_idle')
mock=read(REVIEW/'handoff_checks_result_v1.json')
assert mock['passed'] and mock['handoff_source_ref']['sha256']==r.ref(handoff.__file__)['sha256'] and len(mock['checks'])==14
checks.append('final_handoff_source_matches14passed_mocked_identity_signal_and_replacement_checks')
receipt=dict(schema='independent_argmix_recovery_handoff_acceptance_v1',passed=True,reviewer='benchmark_audit_oct29',recorded_utc=datetime.now(timezone.utc).isoformat(),
    handoff_source_ref=r.ref(handoff.__file__),new_plan_refs=new_plan_refs,test_source_ref=r.ref(__file__),
    worker_controller_acceptance_ref=r.ref(REVIEW/'worker_controller_acceptance_v1.json'),publisher_test_ref=r.ref(REVIEW/'publisher_checks_result_v1.json'),
    handoff_test_ref=r.ref(REVIEW/'handoff_checks_result_v1.json'),launcher_source_ref=r.ref(BASE/'root/launch_argmix_recovery_20261030.py'),
    checks=checks,current_idle_controller_observations=idle,findings=[],actual_signals_sent=0,production_launched=False,truth_opened=False,
    interpretation='Source and exact-plan acceptance only. Root must recheck both controller identities and childless state during the pidfd handoff. Only real future child waits and authenticated completion products authorize downstream release, scoring, numerical audit and publication; no success has been manufactured.')
out=REVIEW/'handoff_acceptance_v1.json'
with out.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,checks=len(checks),receipt_ref=r.ref(out),new_plan_refs=new_plan_refs),sort_keys=True))
