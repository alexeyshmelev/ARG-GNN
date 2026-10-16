"""Root-only actual activation and exact H020 retirement, after independent review."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys,traceback
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE={'path':'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py','sha256':'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}
OUT=ROOT/'research/reviews/prospective980_H020_root_activation_20261016_v3'
def ref(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read_bound(r):
    p=ROOT/r['path'];raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==r['sha256'],r
    return raw
def save(name,value):
    p=OUT/name
    with p.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush()
    return ref(p)
def load(r,name):
    read_bound(r)
    spec=importlib.util.spec_from_file_location(name,ROOT/r['path'])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-path',required=True)
    parser.add_argument('--review-sha256',required=True)
    args=parser.parse_args()
    review={'path':args.review_path,'sha256':args.review_sha256}
    read_bound(review)
    assert not OUT.exists(),'Never blindly replay activation or credit'
    renewal=load({'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py','sha256':'491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'},'_root_existing_marker_renewal')
    renewal.install_renewal_v1({'path':'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json','sha256':'66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'})
    module=load(SOURCE,'_root_actual_H020_v2')
    original_policy=json.loads(read_bound(module.BASE_POLICY_REF))
    repair_ref=original_policy['marker_repair_authority_ref']
    impl_ref=original_policy['implementation_ref'];launcher_ref=original_policy['launcher_ref']
    parent=load(module.BASE_REF,'_root_original_prospective_v1')
    parent.configure_v2(repair_ref,policy_ref=module.BASE_POLICY_REF)
    print('Authenticating original open accounting state under original mutex',flush=True)
    with module.budget.v4.mutex(ROOT):
        parent_state=parent._state_locked(impl_ref['sha256'],launcher_ref['sha256'],fresh=True)
        view=parent_state[2];pins=parent._pins(view)
        assert view['accounting_mode']==module.BASE_MODE
        assert module.LEGACY_KEY in parent_state[0] and module.LEGACY_KEY not in parent_state[1]
        assert module.budget.digest(parent_state[0][module.LEGACY_KEY])==module.LEGACY_RESERVATION_SHA
        assert view['projected_bytes']+module.BOOTSTRAP_CHARGE<module.CEILING_BYTES
        assert not (ROOT/module.V2_JOURNAL).exists(),'Superseded empty v2 journal must remain absent'
        OUT.mkdir(exist_ok=False)
        save('started.root_v1.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_ref=SOURCE,independent_review_ref=review,physical_GPU_ids=[0],
            old_live_C3_and_controller_unchanged=True,no_original_file_deleted=True))
        activation=module.root_activation_draft_v1(base_pins=pins,base_projected_bytes=view['projected_bytes'])
        activation['active']=True
        activation_ref=save('frozen_parent_activation.root_active_v1.json',activation)
        supersession=module.empty_v2_supersession_draft_v1();supersession['active']=True
        supersession_ref=save('empty_v2_supersession.root_active_v1.json',supersession)
        policy=module.activation_policy_draft_v1(repair_authority_ref=repair_ref,base_pins=pins,
            base_projected_bytes=view['projected_bytes'],implementation_ref=impl_ref,
            launcher_ref=launcher_ref,root_activation_ref=activation_ref,empty_v2_supersession_ref=supersession_ref)
        policy['active']=True
        policy_ref=save('activation_policy.root_active_v1.json',policy)
        module.configure_v2(repair_ref,policy_ref=policy_ref)
        current=module._state_locked(impl_ref['sha256'],launcher_ref['sha256'],fresh=True)
        assert current[2]['new_reserved_bytes']==0 and current[2]['completed_credit_bytes']==0
        assert current[2]['projected_bytes']==view['projected_bytes']+module.BOOTSTRAP_CHARGE
        save('activated_before_credit.root_v1.json',dict(base_projected_bytes=view['projected_bytes'],
            projected_bytes=current[2]['projected_bytes'],base_pins=pins,new_pins=module._pins(current[2]),
            source_ref=SOURCE,policy_ref=policy_ref,independent_review_ref=review,
            future_parent_accounting_mutations_permanently_cooperatively_revoked=True,
            old_live_admissions_keep_original_caps=True,credit_bytes=0))
        observation=module.legacy_observation_draft_v1()
        observation['PID_observations']=[module._legacy_pid_absent(row) for row in module.LEGACY_WRITERS]
        observation['group_observations']=[module._legacy_group_absent(pgid) for pgid in module.LEGACY_GROUPS]
        observation['active']=True
        observation_ref=save('H020_actual_absence_and_permanent_revocation.root_active_v1.json',observation)
        request=module.legacy_request_draft_v1(observation_ref)
        authority=module.operation_draft_v1('retire_H020_seed202_v1',request,
            expected_before=module._pins(current[2]),bindings=current[3])
        authority['active']=True
        authority['expires_at_utc']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=6)).isoformat()
        authority_ref=save('retire_H020_operation.root_active_v1.json',authority)
    print('Performing one exact owned-scope retained-byte measurement and retirement',flush=True)
    receipt=module.apply_retire_H020_seed202_v1(authority_ref,
        expected_source_sha256=impl_ref['sha256'],expected_launcher_sha256=launcher_ref['sha256'])
    receipt_ref=save('retirement_receipt.root_actual_v1.json',receipt)
    final=module.inspect_v1(expected_source_sha256=impl_ref['sha256'],expected_launcher_sha256=launcher_ref['sha256'])
    save('post_retirement_current_view.root_v1.json',final)
    result=dict(status='completed_one_exact_legacy_retirement',source_ref=SOURCE,policy_ref=policy_ref,
        receipt_ref=receipt_ref,original_cap_bytes=module.LEGACY_CAP,
        retained_bytes=receipt['retained_bytes'],credit_bytes=receipt['credit_bytes'],
        final_projected_bytes=final['projected_bytes'],remaining_to_admission_ceiling_bytes=module.CEILING_BYTES-final['projected_bytes'],
        original_outer_tool_status='UNKNOWN',historical_whole_descendant_drain='UNKNOWN',
        files_deleted=0,live_C3_or_controller_changed=False,
        new_consumer_direct_binding_limit=64,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save('completed.root_v1.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
