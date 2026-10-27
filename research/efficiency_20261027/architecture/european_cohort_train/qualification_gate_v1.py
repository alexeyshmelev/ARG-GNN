"""Read-only exact ROOT predecessor release; no scheduler, CUDA or process signals."""
import hashlib
import json
import stat
from pathlib import Path

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
RUNS=ROOT/'research/efficiency_20261027/root/runs'
PREDECESSOR=RUNS/'upward_residency_full_gpu5_v2'
PREDECESSOR_REQUEST_SHA='237f9e0671c685bd9f107f004f2d79d7e01cebb79e28a1b108dd5db3ea1bd127'
PREDECESSOR_LAUNCH_SHA='dada2292c04bae6834d85895becc930a4bd51ede9107a12f5071311b8c17173b'
PREDECESSOR_SOURCE_REF=dict(path='research/efficiency_20261027/architecture/exact_upward_residency/source_pins_full_v2.json',sha256='0650cf70c438b16e6a364da636bb250892f6d5d712c979244814157b2ae159bd')
GPU_UUID='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'


def need(value,message):
    if not value:raise ValueError(message)


def contained(path):
    path=ROOT/Path(path)
    need(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts,
         'Lexically contained source/metadata path')
    for part in (path,*path.parents):
        if part==ROOT:break
        need(not part.is_symlink(),'No symlink in source/metadata path')
    return path


def reference(path):
    path=contained(path)
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode),'Ordinary contained source/metadata file')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())


def read(path,digest=None):
    path=contained(path)
    need(path.is_file() and path.stat().st_size<=8*2**20,'Bounded metadata file')
    data=path.read_bytes()
    need(len(data)<=8*2**20,'Bounded metadata bytes')
    ref=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
    need(digest is None or ref['sha256']==digest,'Exact selected metadata hash')
    value=json.loads(data)
    need(reference(path)==ref,'Metadata changed during read')
    return value,ref


def predecessor_identity():
    request,request_ref=read(PREDECESSOR/'request.json',PREDECESSOR_REQUEST_SHA)
    launch,launch_ref=read(PREDECESSOR/'launch.json',PREDECESSOR_LAUNCH_SHA)
    command=request.get('command')
    need(command==launch.get('actual_argv') and request.get('physical_gpu')==5 and request.get('gpu_uuid')==GPU_UUID
         and type(launch.get('pid')) is int and launch['pid']>1 and type(launch.get('process_start_ticks')) is str,
         'Exact admitted GPU5 predecessor request/launch identity')
    need(command[:4]==['.runtime/env/bin/python3.11','-B','-m',
         'research.efficiency_20261027.architecture.exact_upward_residency.compare_full_v2']
         and command.count('--output')==1 and command.count('--source-pins-sha256')==1,
         'Exact visible predecessor module and arguments')
    need((ROOT/command[command.index('--output')+1]).resolve()==PREDECESSOR/'data'
         and command[command.index('--source-pins-sha256')+1]==PREDECESSOR_SOURCE_REF['sha256'],
         'Exact predecessor output and frozen numerical sources')
    return request,launch,request_ref,launch_ref


def require_full_numerical(value):
    from research.efficiency_20261027.architecture.exact_upward_residency.full_release_v2 import require_comparison
    need(value.get('schema')=='TreeLoom_upward_direct_residency_full_v1' and value.get('status')=='completed'
         and value.get('source_manifest_ref')==PREDECESSOR_SOURCE_REF
         and value.get('all_strict_comparisons_passed') is True
         and value.get('all49_gradient_and_parameter_tensors_checked') is True
         and value.get('all147_Adam_tensors_checked') is True and value.get('physical_gpu')=='5'
         and value.get('checkpoint_ref')==dict(path='research/preprocessing_optimization_20261026/integration/migration/update241.pt',
             sha256='12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24'),
         'Successful exact full-residency predecessor numerical receipt')
    scope=value.get('scope',{})
    need(scope.get('mode')=='full' and scope.get('full_parent_claim') is True
         and scope.get('parent_id')=='argformer_american_train_000' and scope.get('haplotypes')==1600
         and scope.get('native_holds')==77472 and scope.get('native_nodes')==205369272
         and scope.get('sites')==227474 and scope.get('physical_support_bp')==[0,30000000]
         and scope.get('supervised_original_focals')==10000 and scope.get('heldout_truth_opened') is False,
         'Complete authenticated predecessor scope; these dimensions do not configure European execution')
    records=value.get('records',[])
    need([row.get('kind') for row in records]==['boundary_fused_CPU512','upward_direct_resident_CPUcontext512'],
         'Exactly the selected full predecessor pair')
    for row in records:
        need(row.get('optimizer_step')==242 and row.get('parameter_count')==49
             and row.get('CPU_context') is True and row.get('hotpack') is False,'Complete predecessor CPU-context Adam update')
        require_comparison(row.get('comparison'))


def predecessor_release(expected_completed_sha=None):
    from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion,owned_group_members
    from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity
    request,launch,request_ref,launch_ref=predecessor_identity()
    if not (PREDECESSOR/'actual_wait.json').is_file():return None
    wait,wait_ref=read(PREDECESSOR/'actual_wait.json')
    validate_completion(launch,wait,identity(launch['pid']),owned_group_members(launch['pid']))
    completed,completed_ref=read(PREDECESSOR/'data/completed.json',expected_completed_sha)
    require_full_numerical(completed)
    return dict(request_ref=request_ref,launch_ref=launch_ref,actual_wait_ref=wait_ref,completed_ref=completed_ref,
        actual_wait=True,actual_returncode=0,owned_group_absent=True,
        completion_hash_selected_after_exact_pinned_process_exited=expected_completed_sha is None)
