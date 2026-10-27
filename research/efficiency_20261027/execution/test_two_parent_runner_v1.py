"""CPU metadata, own-process sampler and visible no-launch argv checks."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .compare_two_parent_preparation_v1 import BASE,SPEC,FamilyMonitor,identity,ARMS
from .launch_two_parent_comparison_v1 import argv_for,LIVE_PEER


def main():
    output=BASE/'two_parent_runner_CPU_v1';output.mkdir(exist_ok=False)
    monitor=FamilyMonitor(output,4*2**30,30)
    monitor.start_sampling();result=monitor.finish()
    assert result['peaks']['encoder_children']==0 and result['peaks']['main_RSS_bytes']>0
    assert result['peaks']['family_PSS_or_RSS_bytes']>0 and result['signals_sent']==0
    assert any(row['pid']==os.getpid() and row['start_ticks']==identity(os.getpid()) for row in result['tracked_processes'])
    spec=json.loads(SPEC.read_text())
    assert spec['parents']==['argformer_american_train_001','argformer_american_train_007']
    assert all(set(item)=={'observation_ref','native_preparation_ref','focal_registry_ref'} for item in spec['observed_inputs'].values())
    args=SimpleNamespace(directory='research/efficiency_20261027/root/runs/never_created_CPU_plan',gpu=5,
        peer_directory=[LIVE_PEER],source_pins_sha256='review_placeholder')
    argv=argv_for(args)
    assert argv[argv.index('--device-gib')+1]=='0' and argv[argv.index('--host-gib')+1]=='288'
    assert not Path(args.directory).exists()
    for name in ('compare_two_parent_preparation_v1.py','launch_two_parent_comparison_v1.py'):
        ast.parse((BASE/name).read_text())
    receipt=dict(status='passed',own_process_PSS_RSS_PID_start_sampler=True,signals_sent=0,
        actual_qualification_processes_started=0,visible_ROOT_guardian_argv=argv,
        only_observation_inputs_in_frozen_spec=True,serial_then_concurrent_fresh_arm_order=list(ARMS),
        limitations='No complete real observations or encoder workers were executed by this CPU preparation check.')
    with (BASE/'TWO_PARENT_RUNNER_CPU_v1.json').open('x') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(dict(status='passed',sampler_processes=len(result['tracked_processes']),launches=0)))


if __name__=='__main__':main()
