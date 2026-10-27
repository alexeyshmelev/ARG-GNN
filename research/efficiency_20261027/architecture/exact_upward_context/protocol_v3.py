"""Metadata-only scientific and resource gates for the combined two-arm protocol."""
import json
import math
import os
from pathlib import Path
from research.efficiency_20261027.architecture.european_cohort_train.qualification_gate_v1 import (
    ROOT,RUNS,GPU_UUID,need,contained,reference,read,predecessor_release as original_full_release)
from research.efficiency_20261027.architecture.exact_upward_residency.full_release_v2 import require_comparison
from research.efficiency_20261027.execution.deferred_gpu_gate import owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity

HERE=Path(__file__).parent
MODULE='research.efficiency_20261027.architecture.exact_upward_context.compare_v3'
KINDS=('upward_direct_resident_CPUcontext512','released_upward_GPUcontext512')
EUROPEAN=RUNS/'european_cohort8_full_gpu5_v1'
EUROPEAN_REQUEST_SHA='9a9a961079ad478320e0361449a554139197025d9e72c95f7fab63169f192de2'
EUROPEAN_LAUNCH_SHA='5d5f87f6f6c5869a265091559f35ae1617b91eeffd618cc9f046e3795465f849'
ORIGINAL_FULL_COMPLETED_SHA='8529b65d7eaa591ff12bc244a742a92a94b4ae5f59e4ccd38e644f1c4029701c'
CHECKPOINT_REF=dict(path='research/preprocessing_optimization_20261026/integration/migration/update241.pt',sha256='12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24')
COMPOSITION_REF=dict(path='research/efficiency_20261027/architecture/exact_upward_context/source_pins_v1.json',sha256='4f95e88a4482f9275f53794f366e25237f587413eb4d4391c069bed40532592d')
COMPOSITION_REVIEW_REF=dict(path='research/efficiency_20261027/fidelity/UPWARD_CONTEXT_V1_SOURCE_REVIEW.json',sha256='748a68499c8306a9059c6609f7faff5a69f556a408a2d31b131f7c8705675955')
GIB=2**30


def bounds(mode):
    need(mode=='prefix','Exactly one diagnostic prefix pair only; full is forbidden')
    return dict(worker_RSS_bytes=(32 if mode=='prefix' else 256)*GIB,
        device_soft_bytes=(24 if mode=='prefix' else 74)*GIB,
        expected_holds=1153 if mode=='prefix' else 77472,
        cache_bytes=(256*2**20 if mode=='prefix' else 8*GIB),output_bytes=64*2**20,
        recommended_family_bytes=(48 if mode=='prefix' else 288)*GIB)


def verify_sources(digest,review_path,review_sha):
    manifest,ref=read(HERE/'runner_pins_v3.json',digest)
    need(manifest.get('schema')=='TreeLoom_upward_context_runner_sources_v3','Exact combined runner manifest')
    refs=manifest['source_refs']
    need(len({r['path'] for r in refs})==len(refs) and COMPOSITION_REF in refs and COMPOSITION_REVIEW_REF in refs,
         'Unique complete reviewed composition bindings')
    for row in refs:need(reference(row['path'])==row,'Reviewed source changed: '+row['path'])
    review,review_ref=read(review_path,review_sha)
    need(review.get('schema')=='TreeLoom_upward_context_runner_review_v3' and review.get('passed') is True
         and review.get('source_manifest_ref')==ref,'Exact passed independent combined runner review')
    return dict(source_manifest_ref=ref,source_review_ref=review_ref,source_refs=refs)


def terminal_wait(launch,wait,current,members,expected_returncode):
    need(type(expected_returncode) is int and type(wait.get('actual_returncode')) is int,
         'Explicit ROOT-selected terminal return code and actual recorded integer exit')
    need(wait.get('actual_wait') is True and wait.get('owned_group_absent') is True
         and wait.get('pid')==launch['pid'] and wait.get('process_start_ticks')==launch['process_start_ticks']
         and current!=launch['process_start_ticks'] and not members,'Exact terminal wait and absent original process family')
    need(wait['actual_returncode']==expected_returncode,
         'Unexpected predecessor terminal status: ROOT must independently select observed nonzero exit in a fresh command')
    return wait['actual_returncode']


def european_resource_release(expected_returncode):
    request,request_ref=read(EUROPEAN/'request.json',EUROPEAN_REQUEST_SHA)
    launch,launch_ref=read(EUROPEAN/'launch.json',EUROPEAN_LAUNCH_SHA)
    need(request['command']==launch['actual_argv'] and request['physical_gpu']==5 and request['gpu_uuid']==GPU_UUID,
         'Exact European resource predecessor identity')
    if not (EUROPEAN/'actual_wait.json').exists():return None
    wait,wait_ref=read(EUROPEAN/'actual_wait.json')
    code=terminal_wait(launch,wait,identity(launch['pid']),owned_group_members(launch['pid']),expected_returncode)
    failure_ref=reference(EUROPEAN/'failure.json') if (EUROPEAN/'failure.json').is_file() else None
    return dict(kind='resource_predecessor_only',request_ref=request_ref,launch_ref=launch_ref,actual_wait_ref=wait_ref,
        actual_returncode=code,actual_wait=True,owned_group_absent=True,failure_ref=failure_ref,
        European_numerical_success_required=False,ROOT_selected_expected_terminal_returncode=expected_returncode,
        no_predecessor_process_signaled=True)


def scientific_predecessor():
    result=original_full_release(ORIGINAL_FULL_COMPLETED_SHA)
    need(result is not None,'Successful exact full-residentCPU scientific qualification and actual reaping required')
    return result


def compare_norm(reference_value,candidate_value):
    need(type(reference_value) in (float,int) and type(candidate_value) in (float,int)
         and math.isfinite(reference_value) and math.isfinite(candidate_value),'Finite original scalar gradient norms')
    difference=abs(candidate_value-reference_value)
    return dict(passed=difference<=1e-6+1e-4*abs(reference_value),reference_value=reference_value,
        candidate_value=candidate_value,absolute_error=difference,atol=1e-6,rtol=1e-4)


def require_completed(value,mode,manifest_ref):
    need(value.get('schema')=='TreeLoom_upward_context_prefix_diagnostic_v1' and value.get('mode')==mode
         and value.get('status')=='diagnostic_completed' and value.get('source_manifest_ref')==manifest_ref
         and value.get('checkpoint_ref')==CHECKPOINT_REF and value.get('strict_downstream_passed') is True
         and value.get('eligible_for_full_qualification') is False,
         'Exact source-bound completed combined comparison')
    scope=value.get('scope',{})
    need(scope.get('parent_id')=='argformer_american_train_000' and scope.get('haplotypes')==1600
         and scope.get('native_holds')==bounds(mode)['expected_holds']
         and scope.get('heldout_truth_opened') is False and scope.get('full_parent_claim') is (mode=='full'),
         'Exact original prefix/full support and no held-out truth')
    if mode=='full':
        need(scope.get('native_nodes')==205369272 and scope.get('sites')==227474
             and scope.get('supervised_original_focals')==10000 and scope.get('physical_support_bp')==[0,30000000],
             'All native full-parent nodes/sites/targets/support')
    rows=value.get('records',[])
    need([r.get('kind') for r in rows]==list(KINDS),'Exactly one reference and one combined call')
    for index,row in enumerate(rows):
        need(row.get('parameter_count')==49 and row.get('optimizer_step')==242
             and row.get('CPU_context') is (index==0) and row.get('hotpack') is False,
             'Original one-step Adam242 and exact selected context interface')
        require_comparison(row.get('comparison'))
        expected_device='cpu' if index==0 else 'cuda:0'
        expected_bytes=bounds(mode)['expected_holds']*1600*32*4
        need(row.get('context_boundary')==dict(device=expected_device,bytes=expected_bytes)
             and row.get('VJP_boundary')==dict(device=expected_device,bytes=expected_bytes)
             and type(row.get('torch_peak_reserved_bytes')) is int
             and 0<row['torch_peak_reserved_bytes']<=bounds(mode)['device_soft_bytes'],
             'Actual complete context/VJP devices and finite measured reserved peak')
    norm=compare_norm(rows[0]['gradient_norm'],rows[1]['gradient_norm'])
    need(norm['passed'] and value.get('gradient_norm_comparison')==norm,'Exact original norm comparison')
    context=rows[1].get('context_comparison',{})
    need(type(context.get('strict_mismatched_elements')) is int and 0<=context['strict_mismatched_elements']<=context.get('elements',0)
         and context.get('passed') is (context['strict_mismatched_elements']==0)
         and context.get('strict_context_passed') is context['passed']
         and value.get('strict_context_passed') is context['passed']
         and value.get('all_strict_comparisons_passed') is context['passed']
         and context.get('practical_context_passed') is True and value.get('practical_context_passed') is True
         and context.get('practical_mismatched_elements')==0 and context.get('practical_atol')==5e-6 and context.get('practical_rtol')==1e-5
         and context.get('shape')==[bounds(mode)['expected_holds'],1600,32]
         and context.get('dtype')=='torch.float32' and context.get('elements')==bounds(mode)['expected_holds']*1600*32
         and context.get('maximum_chunk_holds')==128 and context.get('atol')==1e-6 and context.get('rtol')==1e-5
         and context.get('all_values_finite') is True and context.get('reference_CPU_storage_retained_without_clone') is True
         and context.get('reference_released_before_decoder') is True and context.get('context_values_serialized') is False,
         'Complete bounded untimed context comparison before candidate decoder')
    for row in rows:
        raw=row.get('update_elapsed_including_diagnostic_seconds');diagnostic=row.get('diagnostic_context_seconds')
        adjusted=row.get('update_seconds')
        need(all(type(v) in (float,int) and math.isfinite(v) for v in (raw,diagnostic,adjusted))
             and 0<=diagnostic<raw and adjusted==raw-diagnostic,'Raw update time and exact excluded diagnostic duration')
    need(rows[0]['diagnostic_context_seconds']==0 and context.get('diagnostic_seconds')==rows[1]['diagnostic_context_seconds'],
         'Only candidate context comparison duration is excluded')
    life=rows[1].get('explicit_vjp',{}).get('residency_lifetime',{})
    need(life.get('phase')=='finished' and life.get('live_bank_bytes')==0 and life.get('future_bank_bytes')==0
         and life.get('upward_blocks_released')==life.get('direct_blocks_released')==rows[1]['resident_bank_plan']['blocks'],
         'Actual complete released-upward/direct lifetime required')


def prefix_release(*args,**kwargs):
    raise ValueError('Diagnostic prefix is never a scientific/full-comparison release')


def validate_guardian(request,launch,args,argv,pid,ticks,parent,parent_ticks):
    b=bounds(args.mode);command=request['command']
    need(command==launch['actual_argv'] and command==['.runtime/env/bin/python3.11','-B','-m',MODULE,*argv],
         'Exact visible ROOT command for standalone combined pair')
    need(launch['pid']==pid and launch['process_start_ticks']==ticks and request['supervisor_pid']==parent
         and request['supervisor_start_ticks']==parent_ticks,'Exact actual child and ROOT guardian PID generations')
    need(request.get('physical_gpu')==5 and request.get('gpu_uuid')==GPU_UUID
         and request.get('host_cap_bytes')==args.family_cap_bytes and args.family_cap_bytes>=b['worker_RSS_bytes']
         and request.get('device_budget_bytes')==b['device_soft_bytes'] and request.get('output_cap_bytes')==b['output_bytes'],
         'ROOT-selected whole-family cap and fixed protocol GPU/output bounds')
    peers=args.required_peer_directory
    need(request.get('peer_directories')==peers and len(set(peers))==len(peers)
         and str(EUROPEAN.relative_to(ROOT)) in peers,'Exact complete ROOT-selected peers including European resource predecessor')
    for peer in peers:need(contained(peer).parent==RUNS,'Exact ROOT peer directory')
    need(0<args.gpu_wait_seconds<=args.max_seconds<=86400,'Finite ROOT-selected wait and whole-worker wall bounds')
