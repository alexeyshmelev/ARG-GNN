"""Unchanged qualification/reaping rules, allowing reviewed FIT worker v2 or v3."""
from .fit_gate_v1 import ROOT,RUNS,GPU_UUID,need,contained,reference,read,qualification_release
from .fit_gate_v1 import SOURCE_MANIFEST_REF,SOURCE_REVIEW_REF
from research.efficiency_20261027.execution.deferred_gpu_gate import owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity


def resume_release(path,digest):
    if path is None:
        need(digest is None,'Resume checkpoint path/hash must be supplied together')
        return None
    need(digest is not None,'Explicit exact resume checkpoint hash')
    checkpoint=contained(path);directory=checkpoint.parent.parent.parent
    need(directory.parent==RUNS and checkpoint.parent.name=='resume' and checkpoint.parent.parent.name=='data'
         and checkpoint.name.startswith('update_') and checkpoint.suffix=='.pt'
         and checkpoint.is_file() and checkpoint.stat().st_size<=8*2**20,'Bounded original ROOT FIT resume artifact')
    checkpoint_ref=reference(checkpoint);need(checkpoint_ref['sha256']==digest,'Exact selected resume bytes')
    request,request_ref=read(directory/'request.json');launch,launch_ref=read(directory/'launch.json')
    wait,wait_ref=read(directory/'actual_wait.json')
    command=request.get('command')
    need(type(command) is list and command==launch.get('actual_argv')
         and command[:3]==['.runtime/env/bin/python3.11','-B','-m']
         and len(command)>=4 and command[3] in (
             'research.efficiency_20261027.execution.european_cohort_fit.fit_cli_v2',
             'research.efficiency_20261027.execution.european_cohort_fit.fit_cli_v3')
         and type(launch.get('pid')) is int and launch['pid']>1 and type(launch.get('process_start_ticks')) is str,
         'Exact previously admitted resume-capable FIT worker')
    need(command.count('--output')==1 and command.index('--output')+1<len(command)
         and contained(command[command.index('--output')+1])==directory/'data','Resume checkpoint belongs to exact previous worker output')
    need(wait.get('actual_wait') is True and wait.get('owned_group_absent') is True
         and type(wait.get('actual_returncode')) is int and wait.get('pid')==launch['pid']
         and wait.get('process_start_ticks')==launch['process_start_ticks']
         and identity(launch['pid'])!=launch['process_start_ticks'] and not owned_group_members(launch['pid']),
         'Actual previous FIT wait and absent original family; interrupted exit is permitted')
    return dict(resume_checkpoint_ref=checkpoint_ref,request_ref=request_ref,launch_ref=launch_ref,actual_wait_ref=wait_ref,
        prior_actual_returncode=wait['actual_returncode'],owned_group_absent=True)
