"""Read-only live replacement assessment. Never credit an exit in a launch check."""
import time
from arggnn.benchmarking.paper_campaign import processes
from research.efficiency_20261027.root import run_guarded_profile_with_peer as guard

GIB=2**30
PID=1166415
TICKS='574375145'
TASK='paper_american_treeloom_fit'
REQUIRED_PEERS={
    'research/efficiency_20261027/root/runs/two_parent_preparation_v1',
    'research/efficiency_20261027/root/runs/two_parent_preparation_v2',
    'research/efficiency_20261027/root/runs/european_cohort8_full_gpu5_v1',
}


def capacity(*,available,old_used,old_growth,campaign_growth,peer_growth,new_cap,before_stop):
    if any(type(v) is not int or v<0 for v in (available,old_used,old_growth,campaign_growth,peer_growth,new_cap)):
        raise ValueError('Nonnegative integer byte accounting required')
    if old_growth>campaign_growth or new_cap<=0:
        raise ValueError('Consistent original growth and positive successor cap required')
    if not before_stop and (old_used or old_growth):
        raise ValueError('No predecessor capacity credit at actual submission')
    prospective=available+(old_used if before_stop else 0)
    required=new_cap+campaign_growth-(old_growth if before_stop else 0)+peer_growth+64*GIB
    return dict(available_bytes=available,hypothetical_exit_credit_bytes=old_used if before_stop else 0,
        excluded_predecessor_growth_bytes=old_growth if before_stop else 0,
        required_bytes=required,margin_bytes=prospective-required,admitted=prospective>=required)


def assess(prior,peer_directories,*,before_stop):
    if len(peer_directories)!=len(set(peer_directories)) or not REQUIRED_PEERS<=set(peer_directories):
        raise ValueError('Include every active known ROOT peer exactly once')
    state,growth,workers=guard.campaign_growth()
    if before_stop:
        selected=[r for r in workers if r.get('task')==TASK and r.get('pid')==PID]
        if len(selected)!=1 or guard.identity(PID)!=TICKS:
            raise ValueError('Expected running predecessor required before hypothetical replacement')
        used=selected[0]['observed_pss_bytes']
        old_growth=selected[0]['remaining_growth_bytes']
    else:
        if guard.identity(PID)==TICKS or processes.session_members(PID):
            raise ValueError('Actual predecessor family must be absent before submission')
        if any(r.get('task')==TASK for r in workers):
            raise ValueError('Current task already running, inspect rather than resubmit')
        used=old_growth=0
    peers=[]
    total_peer_growth=0
    for name in peer_directories:
        amount,record=guard.peer_growth(name)
        total_peer_growth+=amount
        peers.append(record)
    result=capacity(available=guard.current_memory(),old_used=used,old_growth=old_growth,
        campaign_growth=growth,peer_growth=total_peer_growth,new_cap=768*GIB,before_stop=before_stop)
    free=guard.gpu_free(3)
    own=guard.child_gpu_bytes(PID,3) if before_stop else 0
    required=max(prior['minimum_free_gpu_bytes'],76*GIB)
    gpu_admitted=free+own>=required
    storage_admitted=state['project_reserved_bytes']+256*2**20<=995_000_000_000
    result.update(observed_unix=time.time(),before_stop_hypothesis_only=before_stop,
        actual_launch_admission=not before_stop,gpu_free_bytes=free,
        hypothetical_GPU_exit_credit_bytes=own,required_GPU_free_bytes=required,
        GPU_admitted=gpu_admitted,storage_admitted=storage_admitted,
        workers=workers,peers=peers,project_reserved_bytes=state['project_reserved_bytes'])
    result['admitted']=result['admitted'] and gpu_admitted and storage_admitted
    result['reason']='Capacity available' if result['admitted'] else 'Wait for live host, GPU or storage capacity without stopping predecessor'
    return result
