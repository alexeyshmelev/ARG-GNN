"""Fresh finite native MLM + original C3 EVAL contracts. Source-only until ROOT admission.
No TRAIN resume, mutable slot, numerical import, probe, reservation or launch at import.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, math, os, sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REL='research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2/controller_v1'
SOURCE=REL+'/finite_jobs_v1.py'
POOL=REL+'/gpu0_declared_pool_v1.py'
LAUNCHER=REL+'/launch_declared_gpu0_v1.py'
DISPATCHER=POOL
CATALOGUE=REL+'/finite_catalogue.v1.json'
MIB=2**20
import re
NAME=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
CURRENT_CONTROLLER_CAP_REF={'path':'research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json','sha256':'4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870'}

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    require(isinstance(path,str) and not Path(path).is_absolute() and ".." not in Path(path).parts,"Contained relative source/metadata path")
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),
        "Canonical bound source/metadata")
    require(p.stat().st_size<=32*2**20,"Bounded source/metadata, no scientific payload")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and isinstance(record["sha256"],str)
        and len(record["sha256"])==64 and all(c in "0123456789abcdef" for c in record["sha256"]),"Exact path/SHA256 reference")
    require(ref(record["path"])==record,"Exact authenticated source/metadata bytes")
    return json.loads((ROOT/record["path"]).read_bytes())

def expiry(value):
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    require(parsed.tzinfo is not None,"Absolute timezone-aware authority")
    return parsed.timestamp()

def direct_bindings(storage,required):
    rows=storage["reservation"]["consumer_bindings"]
    require(isinstance(rows,list) and len(rows)<=64 and all(r in rows for r in required),
        "Original64 direct-binding ceiling; every actual facade/runtime/science ref bound")
    for row in rows:ref(row["path"]);require(ref(row["path"])==row,"Exact direct binding")
    return rows


FAMILY='research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v2'
FAMILY_HANDOFF=FAMILY+'/source_handoff.v2.json'
CORE_MANIFEST=FAMILY+'/core_source_manifest.v2.json'
NATIVE_BRIDGE=FAMILY+'/native_v1/root_prepare_native_concurrent_v1.py'
NATIVE_APPROVAL='research/reviews/native_VAL000_GPU0_fresh_successor_source_20261016_v3/GPU0_successor_execution_approval.root_active_v3.json'

def family_authority_refs_v2():
    """Actual ROOT approval -> final seal -> immutable full core closure.

    The final seal and genuine later approval are deliberately outside the
    immutable core manifest. Their actual hashes are read only after ROOT has
    created them, never fabricated in a source template. Every call verifies
    the full core before returning its four mandatory direct authority refs.
    """
    approval_ref=ref(NATIVE_APPROVAL)
    approval=read(approval_ref)
    require(approval.get('schema')=='native_VAL000_GPU0_execution_approval_v1'
        and approval.get('active') is True and approval.get('root_authorizer')=='root_agent'
        and expiry(approval['created_at_utc'])<=datetime.now(timezone.utc).timestamp()<expiry(approval['expires_at_utc']),
        'Genuine current fixed-path native ROOT execution approval')
    handoff_ref=approval.get('concurrent_execution_handoff_ref')
    require(isinstance(handoff_ref,dict) and handoff_ref.get('path')==FAMILY_HANDOFF,
        'Actual approval binds this exact final reviewed concurrent source seal')
    handoff=read(handoff_ref)
    require(handoff.get('controller_contract_ref')==ref(SOURCE)
        and handoff.get('controller_pool_ref')==ref(POOL)
        and handoff.get('controller_bootstrap_ref')==ref(LAUNCHER)
        and handoff.get('native_bridge_ref')==ref(NATIVE_BRIDGE)
        and approval.get('actual_concurrent_pool_source_ref')==ref(POOL)
        and approval.get('actual_concurrent_launcher_source_ref')==ref(LAUNCHER),
        'Exact actual concurrent contract/pool/launcher/admission producer')
    manifest_ref=handoff.get('core_source_manifest_ref')
    require(isinstance(manifest_ref,dict) and manifest_ref.get('path')==CORE_MANIFEST,
        'Final seal binds the immutable full family core closure anchor')
    manifest=read(manifest_ref)
    require(manifest.get('schema')=='C3_native_EVAL_immutable_core_source_manifest_v2'
        and isinstance(manifest.get('refs'),list) and manifest['refs']
        and len(manifest['refs'])==manifest['source_ref_count'],
        'Exact complete immutable source/metadata core')
    seen={}
    for record in manifest['refs']:
        require(record.get('path') not in (FAMILY_HANDOFF,NATIVE_APPROVAL,CORE_MANIFEST),
            'No circular self/future authority SHA in immutable core')
        require(ref(record['path'])==record
            and (record['path'] not in seen or seen[record['path']]==record),
            'Every full core source/metadata ref authenticated before actual admission/runtime')
        seen[record['path']]=record
    require(len(seen)==manifest['source_ref_count']
        and all(r['path'] in seen and seen[r['path']]==r for r in
            (ref(SOURCE),ref(POOL),ref(LAUNCHER),ref(NATIVE_BRIDGE))),
        'No duplicate/drop of the actual contract/controller/bridge core roles')
    return [ref(NATIVE_BRIDGE),handoff_ref,manifest_ref,approval_ref]

def validate_native_direct_authority_v2(spec):
    rows=family_authority_refs_v2()
    approval_ref=rows[-1]
    require(spec['source_sha256'].get(approval_ref['path'])==approval_ref['sha256'],
        'Actual original native source map binds the same genuine fixed-path approval')
    direct_bindings(read(spec['refs']['storage']),rows)
    return rows

def catalogue_v1():return json.loads((ROOT/CATALOGUE).read_bytes())

def envelope_parts_v1(envelope_ref):
    envelope=read(envelope_ref)
    require(envelope.get("schema")=="GPU0_declared_operational_envelope_v1"
        and envelope.get("status")=="active" and envelope.get("root_authorizer")=="root_agent",
        "Fresh separately bound operational envelope")
    request=read(envelope["request_ref"])
    require(request.get("schema")=="GPU0_declared_operational_request_v1"
        and request.get("root_authorizer")=="root_agent" and request.get("status")=="active","Explicit finite root request")
    return envelope,request,request["method_job"]

def job_from_envelope_v1(envelope_ref):
    envelope,request,method=envelope_parts_v1(envelope_ref)
    limits=request["declared_limits"]
    margin=3072 if request["slot"]=="NATIVE_MLM" else 2048
    return dict(worker_kind=method["worker_kind"],logical_slot=request["slot"],job_id=method["job_id"],
        module=method["module"],module_ref=method["module_ref"],runtime_entry_ref=ref(DISPATCHER),
        input_ref=envelope_ref,storage_ref=envelope["operational_storage_ref"],
        cuda_caps_ref=method["cuda_caps_ref"],cuda_cap_bytes=limits["cuda_cap_bytes"],
        cpu_rss_cap_bytes=limits["cpu_rss_cap_bytes"],operational_deadline_seconds=limits["operational_deadline_seconds"],
        margin_mib=margin,peak_mib=math.ceil(limits["cuda_cap_bytes"]/MIB)+margin,
        log_limit_bytes=limits["log_limit_bytes"],env=dict(method["env"]),
        args=["--execute","--input",envelope_ref["path"],"--expected-input-sha256",envelope_ref["sha256"]])

def job_expiry_v2(outer_job):
    """Authenticated existing per-method bound; no authority or clock is changed."""
    envelope,request,method=envelope_parts_v1(outer_job["input_ref"])
    require(job_from_envelope_v1(outer_job["input_ref"])==outer_job
        and envelope["operational_storage_ref"]==outer_job["storage_ref"],
        "Exact predeclared outer job for per-method expiry")
    spec=read(method["input_ref"])
    envelope_end=expiry(envelope["expires_at_utc"]);method_end=expiry(spec["expires_at_utc"])
    require(envelope_end<=method_end,"Operational envelope cannot extend scientific expiry")
    return envelope_end

def recheck_closed_controller_absence_v2(record):
    proof,_=validate_previous_controller_v1(record)
    absence=read(proof['owned_roots_groups_absence_certificate_ref'])
    require(len(absence['PID_observations'])<=256 and len(absence['group_observations'])<=256,'Bounded exact old-owned roots only')
    for row in absence['PID_observations']:
        try:os.kill(row['pid'],0)
        except ProcessLookupError:pass
        else:raise ValueError('Original owned PID present or reused, no new controller')
    for row in absence['group_observations']:
        try:os.killpg(row['pgid'],0)
        except ProcessLookupError:pass
        else:raise ValueError('Original owned group present or reused, no new controller')
    return True

def _original_genuine_closed_handover_v3(record):
    proof=read(record)
    require(proof.get("schema")=="GPU0_current_controller_genuine_closed_native_successor_v3"
        and proof.get("root_authorizer")=="root_agent" and proof.get("active") is True
        and proof.get("prior_capability_ref")==CURRENT_CONTROLLER_CAP_REF
        and proof.get("controller_pid")==3349405 and proof.get("controller_start_ticks")==481281638
        and proof.get("actual_direct_controller_wait_consumed") is True
        and proof.get("all_owned_workers_and_groups_drained") is True
        and proof.get("canonical_lease_released") is True
        and proof.get("all_old_controller_and_failed_native_write_resume_relaunch_capabilities_revoked") is True
        and proof.get("failed_native_artifacts_and_7GiB_charge_preserved") is True
        and proof.get("C3_not_interrupted_for_this_transition") is True,
        "Genuine terminal current controller handover only after C3 finishes, no reuse of old UNKNOWN evidence")
    old=read(CURRENT_CONTROLLER_CAP_REF);terminal=read(proof["actual_controller_wait_ref"]);registry=read(proof["final_registry_ref"])
    require(terminal.get("controller_pid")==3349405 and terminal.get("controller_start_ticks")==481281638
        and terminal.get("actual_direct_controller_wait") is True
        and type(terminal.get("returncode")) is int and proof.get("actual_controller_returncode")==terminal["returncode"],
        "Actual authenticated ROOT-observed terminal, preserve its real code")
    require(registry.get("pool")==old["pool"] and registry.get("lifecycle")=="closed",
        "Actual final registry/lease release, never a live snapshot")
    rows=registry.get("jobs");rows={r["logical_job"]:r for r in rows} if isinstance(rows,list) else rows
    require(isinstance(rows,dict) and set(rows)==set(old["jobs"]),"Exact original finite job rows")
    for name,row in rows.items():
        require(row.get("state") in ("succeeded","failed","cancelled")
            and row.get("requested")==old["jobs"][name],"Every original finite job genuinely terminal")
        if row.get("pid") is not None:
            require(row.get("leader_wait_status_known") is True and row.get("owned_drain_observed") is True
                and type(row.get("exit_code")) is int,"Actual started-job wait and owned drain")
    succeeded={name:row for name,row in rows.items() if row['state']=='succeeded'}
    completion_refs=proof.get('succeeded_job_completion_refs')
    require(isinstance(completion_refs,dict) and set(completion_refs)==set(succeeded),
        'Actual producer completion ref for each genuinely succeeded job, no relabelled completed state')
    for name,row in succeeded.items():
        require(row.get('exit_code')==0 and row.get('leader_wait_status_known') is True
            and row.get('owned_drain_observed') is True and not row.get('resource_abort')
            and not row.get('cancel_requested')
            and all(r.get('returncode')==0 and r.get('wait_status_known') is True for r in row.get('descendant_waits',[])),
            'Authentic successful lifecycle row requires actual wait0 and complete owned drain')
        outer=read(row['requested']['input_ref']);request=read(outer['request_ref']);method=request['method_job']
        original_spec=read(method['input_ref']);completion_ref=completion_refs[name];completion=read(completion_ref)
        require(completion_ref['path']==original_spec['output_directory']+'/completed.json'
            and isinstance(completion.get('status'),str) and completion['status'].startswith('completed_'),
            'Actual source-bound original producer completion, not a rewritten registry terminal')
    require(proof.get("owned_roots_groups_absence_certificate_ref") is not None,"Actual fresh ROOT owned-absence certificate required")
    absence=read(proof["owned_roots_groups_absence_certificate_ref"])
    require(absence.get("root_authorizer")=="root_agent" and absence.get("prior_capability_ref")==CURRENT_CONTROLLER_CAP_REF
        and absence.get("all_original_known_roots_and_groups_absent") is True
        and absence.get("all_original_future_writer_capabilities_permanently_revoked") is True,
        "Fresh exact owned-root/group absence and permanent revocation certificate")
    identities=[dict(pid=3349405,start_ticks=481281638)]
    for row in rows.values():
        identities.extend(dict(pid=r['pid'],start_ticks=r['start_ticks']) for r in row.get('process_identities',[]))
    unique={r['pid']:r for r in identities}
    require(all(unique[r['pid']]==r for r in identities),'No conflicting old owned PID/start identity')
    actual=absence.get('PID_observations');groups=absence.get('group_observations')
    require(isinstance(actual,list) and len(actual)==len(unique) and {r.get('pid') for r in actual}==set(unique),
        'Complete actual original controller/worker identity absence roster')
    for row in actual:
        require(row.get('start_ticks')==unique[row['pid']]['start_ticks'] and row.get('absent') is True
            and row.get('method')=='targeted os.kill(pid,0) ESRCH','Conservative original PID/start ESRCH facts')
        stamp=datetime.fromisoformat(row['checked_utc'].replace('Z','+00:00'))
        require(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'Actual nonfuture old-root observation')
    expected_groups={r['pgid'] for row in rows.values() for r in row.get('process_identities',[])}
    controller=registry.get("controller",{})
    require(controller.get("pid")==3349405 and controller.get("start_ticks")==481281638,"Actual original controller identity")
    expected_groups.add(controller["pgid"])
    require(isinstance(groups,list) and len(groups)==len(expected_groups)
        and {r.get('pgid') for r in groups}==expected_groups,'Complete exact old owned-group roster')
    for row in groups:
        require(row.get('absent') is True and row.get('method')=='targeted os.killpg(pgid,0) ESRCH',
            'Actual original group ESRCH')
    return proof,rows


def validate_previous_controller_v1(record):
    proof,rows=_original_genuine_closed_handover_v3(record)
    c3=[(name,row) for name,row in rows.items() if row['requested']['worker_kind']=='C3']
    require(len(c3)==1 and c3[0][1]['state']=='succeeded','Original C3 TRAIN must genuinely finish before this EVAL successor')
    name,row=c3[0]
    completion=proof['succeeded_job_completion_refs'][name]
    outer_train=read(completion)
    require(outer_train.get('phase',outer_train.get('stage'))=='train'
        and str(outer_train.get('status','')).startswith('completed'),
        'Handover TRAIN completion is the actual outer train envelope, not only an inner checkpoint record')
    require(proof.get('original_C3_training_completion_ref')==completion,
        'ROOT binds the actual original successful TRAIN producer completion, never a future SHA')
    return proof,rows

def resume_transition_refs_v1(record):
    return []  # No C3 TRAIN or resume slot exists in this family.

def transport_transition_refs_v1(record):
    validate_previous_controller_v1(record)
    return [record]

def eval_contract_v1():
    record=catalogue_v1()['eval_contract_ref']
    require(ref(record['path'])==record,'Exact reviewed original-EVAL adapter contract')
    sp=importlib.util.spec_from_file_location('_C3_full_EVAL_exact_contract',ROOT/record['path'])
    module=importlib.util.module_from_spec(sp)
    sys.modules[sp.name]=module
    sp.loader.exec_module(module)
    return module

def validate_jobs_v1(cap,common_required,admin,env_keys):
    cat=catalogue_v1()
    jobmap=read(cap['finite_job_map_ref'])
    proof,final_rows=validate_previous_controller_v1(cap['prior_controller_completion_ref'])
    require(jobmap.get('schema')=='GPU0_root_bound_finite_job_map_v1' and jobmap.get('status')=='active'
        and jobmap.get('root_authorizer')=='root_agent' and isinstance(jobmap.get('jobs'),dict)
        and 2<=len(jobmap['jobs'])<=6 and set(jobmap['jobs'])==set(cap['jobs']),
        'Native MLM plus1..5 genuinely admitted original EVAL parents only')
    selected={row['slot'] for row in jobmap['jobs'].values()}
    require('NATIVE_MLM' in selected and len(selected)==len(jobmap['jobs'])
        and selected<=set(cat['slots']) and cap.get('C3_EVAL_8GiB_operational_ceiling_authorized') is True,
        'One native and distinct ready EVAL parent slots from fixed000..004 catalogue')
    sources=list(cap['source_refs'].values())
    direct_bindings(admin,common_required+[cap['plan_ref'],cap['finite_job_map_ref'],cap['prior_controller_completion_ref']])
    ids={admin['job_id']}
    outputs={admin['output_directory']}
    for name,row in jobmap['jobs'].items():
        require(type(name) is str and NAME.fullmatch(name),'Exact safe declared logical identity')
        envelope,request,method=envelope_parts_v1(row['operational_input_ref'])
        fixed=cat['slots'][row['slot']]
        require(request['slot']==row['slot'] and method['job_id']==name
            and job_from_envelope_v1(row['operational_input_ref'])==cap['jobs'][name],
            'Actual envelope and frozen launch fields')
        require(all(method[k]==fixed[k] for k in ['worker_kind','module','module_ref'])
            and method['runtime_entry_ref']==fixed['known_runtime_ref'],
            'Exact frozen scientific module and per-method runtime')
        for record in [method['runtime_entry_ref'],method['module_ref']]:
            require(ref(record['path'])==record and record in sources,'Actual called source in controller closure')
        spec=read(method['input_ref'])
        caps=read(method['cuda_caps_ref'])
        science=read(method['storage_ref'])
        require(spec.get('status')=='active' and spec.get('root_authorizer')=='root_agent'
            and spec['job_id']==science['job_id']==name and spec['phase']==fixed['phase']
            and spec['physical_gpu']==0 and spec['refs']['storage']==method['storage_ref']
            and spec['caps_ref']==method['cuda_caps_ref']
            and spec['output_directory']==science['output_directory'],
            'Exact CPU-prepared scientific identity, input/caps and original separate scope')
        require(method['args']==['--execute','--input',method['input_ref']['path'],
            '--expected-input-sha256',method['input_ref']['sha256']],
            'Literal source-bound scientific arguments')
        require(science['budget_components']['consumer_adapter']==json.loads(os.environ['ARGGNN_NATIVE_CAPACITY_GUARD_REF'])
            and science['completed_overlay_policy_ref']==json.loads(os.environ['ARGGNN_COMPLETED_CREDITS_POLICY_REF']),
            'Actual H020v3 science guard/policy, never frozen or revoked future authority')
        for path,sha in spec['source_sha256'].items():
            require(dict(path=path,sha256=sha) in sources,'Entire real scientific source closure retained')
        limits=request['declared_limits']
        for key,capkey in [('cuda_cap_bytes','max_cuda_reserved_bytes'),('cpu_rss_cap_bytes','max_cpu_rss_bytes')]:
            require(type(limits[key]) is int and limits[key]>0
                and limits[key]==method[key]==caps[capkey]==spec[capkey],
                'Exact explicit science/runtime/outer cap agreement: '+key)
        require(type(limits['operational_deadline_seconds']) is int
            and limits['operational_deadline_seconds']==method['operational_deadline_seconds']==caps['operational_deadline_seconds']
            and type(limits['log_limit_bytes']) is int and limits['log_limit_bytes']>0,
            'Explicit finite deadline and original small log cap')
        require(set(method['env'])<=set(env_keys) and all(type(v) is str for v in method['env'].values()),
            'Only original allowed environment fields')
        if row['slot']=='NATIVE_MLM':
            authority_rows=validate_native_direct_authority_v2(spec)
            require(all(r in sources for r in authority_rows),'All native concurrent authorities in actual controller source closure')
            require(name=='American-C3-native-val000-MLM-monitor-corrected-GPU0-mlm-v3'
                and spec['output_directory']=='experiments/results/American_C3_native_val000_full30mb_MLM_monitor_corrected_GPU0_v3'
                and spec['expires_at_utc']=='2026-10-18T23:41:32.476768+00:00'
                and science['reserved_new_bytes']==7*2**30
                and limits['cuda_cap_bytes']==40*2**30 and limits['cpu_rss_cap_bytes']==384*2**30
                and limits['operational_deadline_seconds']==259200
                and cap.get('native_40GiB_operational_ceiling_authorized') is True
                and cap.get('native_72h_operational_deadline_authorized') is True,
                'Original native full budget, fresh7GiB, unchanged job/output/expiry/science')
            require(spec['operational_pool_source_ref']==ref(POOL)
                and spec['operational_launcher_ref']==ref(LAUNCHER),
                'Admission bridge truthfully names the actual concurrent outer controller')
        else:
            require(spec['parent_id']==fixed['parent_id'] and spec['split']=='val'
                and row['slot']=='C3_EVAL_'+spec['parent_id'][-3:]
                and limits['cuda_cap_bytes']==8*2**30 and limits['cpu_rss_cap_bytes']==128*2**30
                and limits['operational_deadline_seconds']==86400
                and science['reserved_new_bytes']==32*2**20,
                'Fixed original full30Mb parent, actual8GiB EVAL wrapper and separate32MiB science')
            e=eval_contract_v1()
            e.validate_method_v1(method)
            eval_request=e.read(spec['root_stage_request_ref'])
            require(eval_request['training']['completion_ref']==proof['original_C3_training_completion_ref']
                and eval_request['prior_controller_completion_ref']==cap['prior_controller_completion_ref'],
                'Every EVAL uses same actual fully completed original TRAIN product and genuine closed handover')
        require(request.get('actual_pool_ref')==ref(POOL) and request.get('actual_launcher_ref')==ref(LAUNCHER)
            and request.get('actual_dispatcher_ref')==ref(DISPATCHER)
            and request.get('unchanged_scientific_input_ref')==method['input_ref']
            and request.get('prior_controller_completion_ref')==cap['prior_controller_completion_ref'],
            'Honest actual outer source and genuine post-TRAIN handover')
        require(expiry(envelope['expires_at_utc'])<=expiry(spec['expires_at_utc'])<=cap['expires_unix'],
            'Individual original expiries retained within finite controller')
        outer=read(envelope['operational_storage_ref'])
        require(outer['reserved_new_bytes']==4*2**20 and outer['reservation']['auxiliary_scopes']==[]
            and outer['job_id'] not in ids and outer['job_id']!=name
            and outer['output_directory'] not in outputs and outer['output_directory']!=science['output_directory'],
            'Separate explicit4MiB outer scope, not a repeated science charge')
        ids.add(outer['job_id'])
        outputs.add(outer['output_directory'])
        require(name not in ids and science['output_directory'] not in outputs,'Distinct fixed logical/scientific identities')
        ids.add(name)
        outputs.add(science['output_directory'])
        runtime_refs=request['runtime_source_refs']
        required=[method['runtime_entry_ref'],method['module_ref'],*fixed.get('required_runtime_source_refs',[])]
        require(all(r in runtime_refs and r in sources for r in required)
            and all(ref(r['path'])==r and r in sources for r in runtime_refs),
            'Every executed entry/runtime/IPC/contract/source role exactly authenticated')
        direct_bindings(outer,common_required+runtime_refs+[envelope['request_ref'],method['input_ref'],
            method['storage_ref'],method['cuda_caps_ref'],cap['prior_controller_completion_ref']])
        require(len(science['reservation']['consumer_bindings'])<=64,'Original science64direct-ref ceiling retained')
    return jobmap
