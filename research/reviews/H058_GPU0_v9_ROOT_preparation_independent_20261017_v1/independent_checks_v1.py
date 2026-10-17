"""Independent bounded source/RAM review; no production guard/resource operations.
Author27/9 reproduction is separate. Additional cases use the disclosed author
RAM filesystem/fixture scaffold, with independently chosen faults and assertions.
"""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,importlib,io,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/H058_GPU0_v9_ROOT_preparation_source_20261017_v1'
OUT='research/reviews/H058_GPU0_v9_ROOT_preparation_independent_20261017_v1'
def run():
    reports=[];negatives=[]
    def ck(name,value=True):
        if not value:raise AssertionError(name)
        reports.append(dict(name=name,status='PASS'))
    def reject(name,call):
        try:call()
        except (AssertionError,ValueError,PermissionError,KeyError):
            negatives.append(name);ck(name)
        else:raise AssertionError('Unexpected acceptance: '+name)
    def source(name):return (ROOT/HERE/name).read_text()
    text=source('prepare_ROOT_v1.py');tree=ast.parse(text)
    expected={'source_handoff.v1.json':'07340a1b6030c7c815bed347e16766142d6f82f2a79992ed2280e70ccea78ad2',
        'source_manifest.v1.json':'11b402faa4f32c157c9fba6d7f88c293270ae54d22464afb7a477617f231b423',
        'prepare_ROOT_v1.py':'fc234535b447710b8408b24ac3735eb591f595449dd95ec608d8305669619dfc'}
    for name,digest in expected.items():
        ck('exact_requested_'+name,hashlib.sha256((ROOT/HERE/name).read_bytes()).hexdigest()==digest)
    manifest=json.loads(source('source_manifest.v1.json'));audit=json.loads(source('MANIFEST.v1.json'))
    refs={r['path']:r['sha256']for r in manifest['refs']}
    refs.update({r['path']:r['sha256']for r in audit['files']})
    for path,digest in refs.items():
        p=ROOT/path
        if not(p.is_file() and not p.is_symlink() and p.stat().st_size<=2**20
            and hashlib.sha256(p.read_bytes()).hexdigest()==digest):raise AssertionError(path)
    ck('all25_package_manifest_and_final_audit_refs_exact',len(refs)==25)
    # No author result writer is enabled.
    authored={'__name__':'H058v9_review_author_RAM_reproduction'}
    saved_argv=sys.argv;sys.argv=['checks_source_only_v1.py']
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source('checks_source_only_v1.py'),str(ROOT/HERE/'checks_source_only_v1.py'),'exec'),authored)
    finally:sys.argv=saved_argv
    author_result=authored['result']
    ck('author_reproduction_separately_27_assertions',author_result['assertions']==27
        and author_result['operational_events_all_RAM'] and not author_result['independent'])
    s=authored['scope'];cr=authored['cr'];memory=authored['memory'];dirs=authored['dirs']
    ref=authored['putram'];read=s['read'];directive=authored['directive'];review=authored['review']
    review_ref=authored['review_ref'];directive_ref=authored['directive_ref']
    def authority_fault(name,review_change=None,directive_change=None,registration_change=None):
        rev=copy.deepcopy(review);di=copy.deepcopy(directive)
        if review_change:review_change(rev)
        rr=ref('RAM_ONLY/independent_'+name+'_review.json',rev);di['preparation_review_ref']=rr
        if registration_change:
            reg=copy.deepcopy(authored['registration']);registration_change(reg)
            di['unfinished_cell_registration_ref']=ref('RAM_ONLY/independent_'+name+'_registration.json',reg)
        if directive_change:directive_change(di)
        dr=ref('RAM_ONLY/independent_'+name+'_directive.json',di)
        reject(name,lambda:s['authority_v1'](rr,dr,cr))
    authority_fault('nonindependent_helper_review_blocks_before_observation',review_change=lambda v:v.update(independent=False))
    authority_fault('different_helper_source_review_blocks',review_change=lambda v:v.update(source_ref={}))
    authority_fault('different_candidate_manifest_blocks',review_change=lambda v:v.update(candidate_source_manifest_ref={}))
    authority_fault('alternate_fixed_namespace_blocks',directive_change=lambda v:v.update(instance_id='fallback'))
    authority_fault('false_full_history_claim_blocks',directive_change=lambda v:v.update(historical_complete_v7_roster_claimed=True))
    authority_fault('non27h_interval_blocks',directive_change=lambda v:v.update(expires_at_utc='2026-10-18T15:00:01+00:00'))
    authority_fault('optimizer_transfer_registration_blocks',registration_change=lambda v:v.update(no_old_CPU_or_GPU_optimizer_state_transfer=False))
    # Restore actual context function, and supply inert import results + fixed RAM clock.
    commonmod=importlib.import_module(s['CANDIDATE'].replace('/','.')+'.common_v1')
    directmod=importlib.import_module(s['CANDIDATE'].replace('/','.')+'.direct_gpu0_v1')
    fake_d=SimpleNamespace(HERE=cr.HERE,SOURCE=cr.DIRECT,
        validate_native_resource_disposition_v1=cr.validate_native_resource_disposition_v1)
    imported=[]
    s['importlib']=SimpleNamespace(import_module=lambda name:imported.append(name)or(cr if name.endswith('.common_v1')else fake_d))
    dirs.discard(str(authored['ramroot']/s['SCOPES']['preparation_directory']))
    real_context=authored['original_context']
    answer=real_context(review_ref,directive_ref)
    ck('actual_context_uses_directive_clocks_fixed_new_scopes_and_preserved_Census_source',
        answer[3]['created_at_utc']==directive['created_at_utc']
        and answer[3]['expires_at_utc']==directive['expires_at_utc']
        and answer[3]['storage_policy_selector_ref']==cr.SELECTOR_REF
        and cr.GUARD['sha256']=='72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb'
        and answer[3]['fresh_scope_lineage_ref']==directive_ref)
    ck('context_exact_direct_common_function_identity',fake_d.validate_native_resource_disposition_v1 is cr.validate_native_resource_disposition_v1)
    # Check actual wrapper ordering using a bad active directive and observation tripwire.
    bad=copy.deepcopy(directive);bad['remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope']=False
    badref=ref('RAM_ONLY/independent_wrapper_bad_directive.json',bad)
    s['context_v1']=real_context;observations=[]
    s['observe_v1']=lambda *args:observations.append(True)or(_ for _ in ()).throw(AssertionError('observation reached'))
    reject('bad_ROOT_authority_stops_actual_wrapper_before_any_observation',
        lambda:authored['original_prepare'](review_ref,badref))
    ck('bad_authority_observation_tripwire_not_reached',not observations)
    # Independently chosen target-read denial and native live/context failures.
    def denied(pid):raise PermissionError('RAM denied known identity')
    reject('v7_unreadable_identity_is_not_skipped',
        lambda:s['observe_v7_v1'](cr,SimpleNamespace(proc_identity=denied)))
    base_native=authored['observation']['native'];first=base_native['native_process_identities'][0]
    fakebase=SimpleNamespace(proc_identity=lambda pid:dict(pid=pid,uid=first['uid'],start_ticks=first['start_ticks'],state='S'),
        real_inventory=lambda:[])
    probe_calls=[]
    fakeobserve_d=SimpleNamespace(base_v1=lambda:fakebase,CURRENT_CAP=authored['cap_ref'],
        GPU0_UUID=authored['d'].GPU0_UUID,held_writable_probe_v1=lambda p:probe_calls.append(True)or True)
    reject('native_exact_live_identity_stops_before_GPU_probe',
        lambda:authored['original_observe'](cr,fakeobserve_d,authored['old']))
    ck('live_native_no_writable_probe',not probe_calls)
    def absent(pid):raise ProcessLookupError(pid)
    fakebase.proc_identity=absent
    fakebase.real_inventory=lambda:[dict(index=0,uuid=fakeobserve_d.GPU0_UUID,compute=[{'pid':first['pid']}])]
    reject('native_numeric_GPU_context_blocks_even_without_old_process_identity',
        lambda:authored['original_observe'](cr,fakeobserve_d,authored['old']))
    ck('native_GPU_context_no_writable_probe',not probe_calls)
    # Original controller/expiry and capacity bodies, independently supplied RAM metadata.
    direct_source=(ROOT/s['CANDIDATE']/'direct_gpu0_v1.py').read_text()
    dtree=ast.parse(direct_source)
    selected=ast.Module(body=[n for n in dtree.body if isinstance(n,ast.FunctionDef)
        and n.name in ('controller_v1','capacity_v1')],type_ignores=[])
    q=copy.deepcopy(authored['q_out'])
    authority=read(q['direct_attachment_authority_ref']);terminal=read(q['controller_terminal_qualification_ref'])
    declared=dict(cuda_cap_bytes=24*2**30,margin_mib=2048,peak_mib=26*1024)
    c3row=dict(logical_job=directmod.C3_JOB,state='failed',exit_code=1,
        leader_wait_status_known=True,owned_drain_observed=True,requested=declared,
        process_identities=[dict(pid=600,uid=1000,start_ticks=123)])
    registry=dict(controller=dict(pid=directmod.CONTROLLER_PID,start_ticks=directmod.CONTROLLER_START),
        physical_gpu=0,lifecycle='running',safety_error=None,ambiguous_descendants_or_lease_holders=False,
        capability_ref=directmod.CURRENT_CAP,jobs=[c3row],updated_unix=authored['FIXED'].timestamp())
    storage_ref={'path':'RAM_ONLY/independent_controller_storage.json','sha256':'a'*64}
    cap=dict(expires_unix=authored['FrozenDateTime'].fromisoformat(q['expires_at_utc']).timestamp(),
        gpu_uuid=directmod.GPU0_UUID,physical_gpu=0,jobs={directmod.C3_JOB:declared},
        controller_storage_ref=storage_ref)
    doc={q['direct_attachment_authority_ref']['path']:authority,directmod.CURRENT_CAP['path']:cap,
        storage_ref['path']:dict(output_directory='RAM_ONLY/controller_boundary'),
        q['controller_terminal_qualification_ref']['path']:terminal,
        terminal['registry_snapshot_ref']['path']:registry}
    class RAMRegistryPath(PurePosixPath):
        def is_file(self):return True
        def is_symlink(self):return False
        def stat(self):return SimpleNamespace(st_size=1024)
        def read_bytes(self):return json.dumps(registry).encode()
    def proc(pid):
        if pid==directmod.CONTROLLER_PID:
            return dict(pid=pid,uid=1000,start_ticks=directmod.CONTROLLER_START,state='S')
        if pid==600:return dict(pid=600,uid=1000,start_ticks=123,state='S')
        raise ProcessLookupError(pid)
    def killpg(*args):raise ProcessLookupError()
    inventory=dict(index=0,uuid=directmod.GPU0_UUID,total_mib=80*1024,used_mib=3*1024,
        compute=[dict(pid=600,name='AI Auto Research Pipeline Test',used_mib=2*1024),
                 dict(pid=601,name='RAM external',used_mib=1024)])
    b=SimpleNamespace(proc_identity=proc,real_inventory=lambda:[copy.deepcopy(inventory)])
    class RAMCapacityError(ValueError):pass
    dn=dict(require=lambda ok,msg:None if ok else(_ for _ in ()).throw(ValueError(msg)),
        read=lambda record:copy.deepcopy(doc[record['path']]),
        active=cr.active,validate_native_resource_disposition_v1=lambda q,fresh:copy.deepcopy(base_native),
        datetime=authored['FrozenDateTime'],time=SimpleNamespace(time=lambda:authored['FIXED'].timestamp()),
        os=SimpleNamespace(getuid=lambda:1000,killpg=killpg),json=json,
        local=lambda p:RAMRegistryPath(str(ROOT/p)),
        CONTROLLER_PID=directmod.CONTROLLER_PID,CONTROLLER_START=directmod.CONTROLLER_START,
        CURRENT_CAP=directmod.CURRENT_CAP,C3_JOB=directmod.C3_JOB,GPU0_UUID=directmod.GPU0_UUID,
        GPUCapacityFailureV3=RAMCapacityError,bounded_capacity_failure_v3=lambda *args:{'RAM':True})
    exec(compile(selected,'<actual-controller-capacity-independent-RAM>','exec'),dn)
    ck('actual_preserved_controller_accepts_expiry_exactly_at_cap_boundary',
        dn['controller_v1'](q,b)[0]['expires_unix']==cap['expires_unix'])
    badq=copy.deepcopy(q)
    badq['expires_at_utc']='2026-10-18T15:00:01+00:00'
    reject('actual_controller_rejects_v9_expiry_one_second_beyond_unchanged_cap',
        lambda:dn['controller_v1'](badq,b))
    cap['expires_unix']=authored['FIXED'].timestamp()-1
    reject('actual_controller_rejects_expired_cap_without_renewal',lambda:dn['controller_v1'](q,b))
    cap['expires_unix']=authored['FrozenDateTime'].fromisoformat(q['expires_at_utc']).timestamp()
    sample=dn['capacity_v1'](q,b)
    ck('actual_capacity_keeps_C3_26_H058_30_external_and_4GiB_headroom',
        sample['C3_fixed_envelope_mib']==26*1024 and sample['H058_cap_plus_margin_mib']==30*1024
        and sample['headroom_mib']==4096 and sample['external_and_driver_mib']==1024
        and sample['projected_mib']==61*1024)
    inventory.update(used_mib=32*1024)
    inventory['compute'][1]['used_mib']=30*1024
    reject('actual_capacity_rejects_external_occupancy_over_physical_fit',lambda:dn['capacity_v1'](q,b))
    inventory.update(used_mib=3*1024);inventory['compute'][1]['used_mib']=1024
    inventory['compute'][0]['name']='wrong'
    reject('actual_capacity_requires_prescribed_known_C3_title',lambda:dn['capacity_v1'](q,b))
    # Source/scientific relationships not inferred from writer assertions.
    cfg=read(q['config_ref']);oldcfg=authored['oldcfg'];prior=read(q['prior_H058_CPU_disposition_ref'])
    ck('scientific_config_values_and_caps_exact_except_five_required_refs',
        {k for k in cfg if cfg[k]!=oldcfg.get(k)}=={'producer_ref','source_manifest_ref','runtime_equivalence_ref',
            'layout_resource_proof_ref','full_training_directive_ref'}
        and cfg['scientific_protocol']==oldcfg['scientific_protocol'] and q['caps']==authored['old']['caps']==cr.CAPS)
    ck('science_storage_and_runtime_caps_original_unchanged',
        cr.CAPS['reservation_bytes']==256*2**20 and cr.CAPS['scientific_output_bytes']==192*2**20
        and cr.CAPS['auxiliary_bytes']==60*2**20 and cr.CAPS['directory_and_accounting_allowance_bytes']==4*2**20
        and cr.CAPS['max_seconds']==86400 and cr.CAPS['work_cutoff_margin_seconds']==300
        and cr.CAPS['terminal_evidence_margin_seconds']==150 and cr.CAPS['cleanup_seconds']==30
        and cr.CAPS['max_cpu_rss_bytes']==128*2**30 and cr.CAPS['max_cuda_reserved_bytes']==28*2**30)
    ck('v7_unknown_totals_truncated_roster_and_cooperative_history_acceptance_preserved',
        prior['prior_GPU_v7_actual_optimizer_steps']=={'same_marker_aligned':None,'same_marker_alignment_null':None}
        and prior['prior_GPU_v7_final_optimizer_totals_status']=='UNKNOWN_no_new_update_verified'
        and prior['prior_GPU_v7_verified_optimizer_steps']==0
        and read(prior['fresh_targeted_v7_absence_ref'])['historical_complete_roster_claimed'] is False
        and read(prior['fresh_targeted_v7_absence_ref'])['remaining_historical_ownership_uncertainty_ROOT_accepted_for_fresh_scope'] is True)
    ck('direct_source_is_exact_reviewed_candidate_manifest_entry',
        next(r for r in manifest['refs']if r['path']==cr.DIRECT)['sha256']
        ==hashlib.sha256((ROOT/cr.DIRECT).read_bytes()).hexdigest())
    called={n.func.attr if isinstance(n.func,ast.Attribute)else n.func.id for n in ast.walk(tree)
        if isinstance(n,ast.Call)and isinstance(n.func,(ast.Name,ast.Attribute))}
    ck('no_reserve_admission_launch_process_walk_delete_or_renewal_calls',
        not called.intersection({'reserve_v1','apply_reserve_v1','admit','admit_v1','execute_v1','Popen','run',
            'unlink','rmtree','iterdir','glob','walk','renew'}))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    wrapper=functions['prepare_ROOT_v1']
    order=[ast.unparse(n.value) for n in wrapper.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)]
    ck('actual_wrapper_readiness_metadata_then_one_original_capacity_call',
        order[0].startswith('context_v1(') and order[1].startswith('observe_v1(')
        and order[2].startswith('metadata_v1(') and order[3]=='d.capacity_v1(q, base)')
    ck('no_scientific_framework_imports',
        not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
    return dict(schema='H058_GPU0_v9_ROOT_preparation_independent_source_RAM_checks_v1',
        status='PASS',independent=True,source_only=True,independent_assertions=len(reports),
        independent_negative_fixtures=len(negatives),reports=reports,negative_cases=negatives,
        author_reproduction=dict(assertions=27,negative_fixtures=9,independent=False,
            writer_enabled=False,source_or_bounded_metadata_refs=author_result['read_source_or_bounded_metadata_files']),
        scaffold_disclosure='Author in-memory filesystem/fixture scaffold reproduced once; additional faults/assertions independently selected. Original context/authority/observation/controller/capacity function bodies exercised with fixed clock and explicit RAM process/GPU/registry/probe doubles.',
        actual_clock_resource_process_probe_state_guard_accounting_admission_GPU_science_or_Git_operations=False,
        current_readiness_or_real_capacity_deadline_fit_established=False,
        source_uses_assert_requires_nonoptimized_ROOT_interpreter=True)
if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
