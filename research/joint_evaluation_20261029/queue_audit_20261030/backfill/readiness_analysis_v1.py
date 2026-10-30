"""Read current exact TEST009 capacity and ownership; no execution or signals."""
import datetime
import json
from pathlib import Path
import subprocess
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import storage_credit_v1 as credit

HERE=Path(__file__).resolve().parent
BASE=r.ROOT/'research/joint_evaluation_20261029'

def main():
    plan_ref=r.ref(BASE/'queue/sidecar_plan.json');plan=r.read(plan_ref)
    state=json.loads((BASE/'queue/records_v1/state.json').read_text())
    finish=r.read(r.ref(BASE/'root/reporting_20261030_v1/finish_plan_candidate_v1.json'))
    config=r.read(finish['successor_config_ref'])
    template=dict(r.read(config['successor_template_ref']),handoff_config_ref=finish['successor_config_ref'])
    accounting=credit.commitment(template,finish['storage_credit_ref'],argmix_override_bytes=25*2**30)
    selected=[task for task in plan['tasks'] if 'argformer_american_test_009_' in task['id']]
    assert len(selected)==5
    running=[task for task in plan['tasks'] if state['tasks'][task['id']]['status']=='running']
    remaining=sum(max(0,task['max_rss_bytes']-state['tasks'][task['id']].get('observed_pss_bytes',0)) for task in running)
    free=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus={}
    for line in gpu_lines.splitlines():
        number,uuid,available,used,utilization=[item.strip() for item in line.split(',')]
        if int(number) in (3,4,5):gpus[number]=dict(uuid=uuid,free_bytes=int(available)*2**20,used_bytes=int(used)*2**20,utilization=int(utilization))
    tasks=[]
    for task in selected:
        output=r.path(task['output_directory'])
        tasks.append(dict(id=task['id'],dependencies=task['dependencies'],status=state['tasks'][task['id']]['status'],
            output_bytes=r.scope_bytes(output),output_cap_bytes=task['max_output_bytes'],ram_cap_bytes=task['max_rss_bytes'],
            gpu_minimum_bytes=task.get('minimum_free_gpu_bytes'),stage_ref_kind='unchanged original task',
            current_files=[entry.name for entry in output.iterdir()] if output.is_dir() else []))
    prepare=selected[0]
    demand=prepare['max_rss_bytes']+remaining+plan['foreign_ram_reserve_bytes']+64*2**30
    value=dict(schema='argformer_test009_backfill_readiness_analysis_v1',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_ref=r.ref(__file__),plan_ref=plan_ref,controller_pid=state['controller_pid'],
        predecessor_start_ticks=(Path('/proc')/str(state['controller_pid'])/'stat').read_text().rsplit(')',1)[1].split()[19],
        state_counts=state['counts'],tasks=tasks,closed_writer_accounting=accounting,
        storage_headroom_bytes=accounting['ceiling_bytes']-accounting['total_bytes'],all_five_stage_caps_already_charged=True,
        test009_total_original_stage_caps_bytes=sum(task['max_output_bytes'] for task in selected),
        storage_capacity_missing_bytes=max(0,accounting['total_bytes']-accounting['ceiling_bytes']),
        memory_available_bytes=free,original_sidecar_prepare_admission_required_bytes=demand,
        original_sidecar_prepare_admission_margin_bytes=free-demand,physical_gpus=gpus,
        working_set_estimate=dict(serialization_bound_bytes=prepare['call']['spec']['full_corpus_serialization_bound_bytes'],
            preparation_full_output_cap_bytes=prepare['max_output_bytes'],checkpoint_caps_bytes=[task['max_output_bytes'] for task in selected if task['id'].endswith(('_mlm','_contrastive'))]),
        ownership_blocker='Live sidecar owns TEST009 and holds the sole sidecar.lock. It has no pending-row transfer protocol. Failed/blocked-only repair requests cannot transfer pending work; plan/state file edits do not alter its in-memory plan.',
        prohibited_shortcuts='Do not alter claimed preparation marker or add files: permit_claimed_output raises on non-marker content outside a protective try, causing the live controller finally block to drain its workers. Do not launch a duplicate owner in shadow output directories.',
        safe_transfer='A new root-authorized watcher may take a natural childless boundary, freeze only the exact controller, require genuine actual waits for all historical running rows and absent sessions, seal original and reconciled state, then replace only the childless owner under the same lifetime lock.',
        proposed_changes=['Remove only the operational TEST009.prepare dependency on TEST007.retire; preserve all scientific input refs and full settings.',
            'Permit only TEST009 GPU stages on GPU3 through shared exclusive mutex; retain GPUs4/5 and their one-job limit.',
            'Keep original public plan_ref and add explicit operational plan/continuation refs for unchanged consumer aliases.',
            'Reuse exact genuine completed waits and preserve pending rows; never replay started or partial stages.',
            'Recheck complete shared storage and host memory accounting before each admission; unknown observations pause admissions.'],
        live_original_workers={task['id']:{key:state['tasks'][task['id']].get(key) for key in ('pid','process_start_ticks','physical_gpu','observed_pss_bytes')} for task in running},
        production_launches=0,signals_sent=0,git_commands=0)
    target=HERE/'readiness_analysis_v1.json'
    with target.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(report_ref=r.ref(target),storage_headroom_bytes=value['storage_headroom_bytes'],ram_margin_bytes=free-demand)))

if __name__=='__main__':main()
