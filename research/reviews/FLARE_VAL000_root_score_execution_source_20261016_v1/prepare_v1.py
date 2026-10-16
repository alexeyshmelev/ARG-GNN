"""Root-only fresh FLARE scoring admission after actual H020 v3 activation."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/FLARE_VAL000_root_score_execution_source_20261016_v1/prepare_v1.py'
PACKET=ROOT/'research/reviews/FLARE_VAL000_fresh_score_source_20261016_v1'
OUT=ROOT/'research/reviews/FLARE_VAL000_score_H020980_root_20261016_v1'
def ref(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def load(record,name):
    path=ROOT/record['path'];assert ref(path)==record
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def save(name,value):
    path=OUT/name
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    return ref(path)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--guard-path',required=True);parser.add_argument('--guard-sha256',required=True)
    parser.add_argument('--policy-path',required=True);parser.add_argument('--policy-sha256',required=True)
    args=parser.parse_args()
    guard_ref=dict(path=args.guard_path,sha256=args.guard_sha256)
    policy_ref=dict(path=args.policy_path,sha256=args.policy_sha256)
    assert ref(ROOT/policy_ref['path'])==policy_ref
    policy=json.loads((ROOT/policy_ref['path']).read_text())
    assert policy['active'] is True and policy['source_ref']==guard_ref
    assert policy['journal']=='.runtime/storage/prospective_980gb_H020_v3'
    assert not OUT.exists()
    handoff=json.loads((PACKET/'source_handoff.v1.json').read_text())
    renewal=load(handoff['marker_reader_renewal_refs'][0],'_FLARE_root_approved_renewal')
    renewal.install_renewal_v1(handoff['marker_reader_renewal_refs'][1])
    guard=load(guard_ref,'_FLARE_root_actual_capacity_guard')
    guard.configure_v2(handoff['repair_authority_ref'],policy_ref=policy_ref)
    output='experiments/results/FLARE_C3_VAL000_common_score_H020980_20261016_v1'
    auxiliary=str(OUT/'worker_aux')
    assert not (ROOT/output).exists() and not Path(auxiliary).exists()
    OUT.mkdir(exist_ok=False)
    config=json.loads((PACKET/'score_config.inactive.json').read_text())
    config['execution_enabled']=True
    config_ref=save('config.json',config)
    expiry=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=8)).isoformat()
    impl=policy['implementation_ref'];launcher=policy['launcher_ref']
    with guard.budget.v4.mutex(ROOT):
        state=guard._state_locked(impl['sha256'],launcher['sha256'],fresh=True)
        fixed=json.loads((PACKET/'fixed_consumer_basis.v1.json').read_text())['fixed_refs']
        root_refs=[ref(PACKET/'source_handoff.v1.json'),ref(PACKET/'fixed_consumer_basis.v1.json'),
            ref(SOURCE),config_ref,guard_ref,policy_ref,policy['root_activation_ref'],policy['empty_v2_supersession_ref']]
        bindings=guard.consumer_bindings_v1(list(state[3].values()),fixed,root_refs)
        assert len(bindings)<=64
        assert not any(any(part in ('truth','oracle') for part in Path(row['path']).parts) for row in bindings)
        request=dict(key='FLARE/VAL000-common-score-H020980-20261016-v1',output=str(ROOT/output),
            auxiliary_scopes=[auxiliary],reserved_bytes=8388608,consumer_bindings=bindings)
        authority=guard.operation_draft_v1('reserve',request,expected_before=guard._pins(state[2]),bindings=state[3])
        authority.update(active=True,expires_at_utc=expiry)
        authority_ref=save('reserve_authority.root_active_v1.json',authority)
    receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=impl['sha256'],expected_launcher_sha256=launcher['sha256'])
    save('reserve_receipt.root_actual_v1.json',receipt)
    storage=dict(kind='project_budget_v9_storage_record',active=True,consumer_uses_v9_guard=True,
        accounting_mode=guard.MODE,job_id=request['key'],output_directory=output,reserved_new_bytes=request['reserved_bytes'],
        reservation=receipt['value'],initial_ledger_pins=receipt['initial_ledger_pins'],expires_at_utc=expiry,
        completed_overlay_policy_ref=policy_ref,
        budget_components=dict(implementation=impl,launcher=launcher,consumer_adapter=guard_ref))
    admission=json.loads((PACKET/'admission.inactive.template.json').read_text())
    admission.update(execution_authorized=True,config_ref=config_ref,storage_adapter_ref=guard_ref,
        completed_overlay_policy_ref=policy_ref,accounting_mode=guard.MODE,expires_at_utc=expiry,
        storage=storage,additional_required_refs=bindings)
    admission_ref=save('admission.root_active_v1.json',admission)
    result=dict(status='admitted_not_scored',admission_ref=admission_ref,output=output,guard_ref=guard_ref,
        policy_ref=policy_ref,unique_direct_binding_count=len(bindings),reserved_bytes=request['reserved_bytes'],
        projected_bytes=receipt['projected_bytes'],truth_opened=False,GPU_authorized=False,
        root_fixed_C3_training_schedule_unchanged=True)
    save('admitted.root_v1.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
