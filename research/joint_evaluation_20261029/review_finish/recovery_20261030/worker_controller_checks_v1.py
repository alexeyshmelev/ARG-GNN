"""Independent CPU-only recovery checks; no production stage or signals execute."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
from types import SimpleNamespace
import contextlib
import copy
import hashlib
import io
import json
import sys
from research.joint_evaluation_20261029.argmix.repair_20261030_memory_v1 import worker as w
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue import controller_v1 as c

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_evaluation_20261029'
REVIEW=BASE/'review_finish/recovery_20261030'
checks=[]
def rejects(action):
    try:action()
    except ValueError:return
    raise AssertionError('Unsafe input accepted')

def main():
    manifest_path=Path(sys.argv[1])
    manifest_ref=c.ref(manifest_path);manifest=c.read(manifest_ref)
    plan_ref=manifest['plan_ref'];plan=c.read(plan_ref)
    binding,host=c.validate_recovery(manifest,manifest_ref,plan)
    assert binding['effective_owned_RAM_max_bytes']==384*2**30 and host==448*2**30
    checks.append('real_final_manifest_plan_amendment_failure_and_original_source_bindings_authenticate')
    amendment,protocol,freeze_ref=w.validate_amendment(manifest['resource_amendment_ref'],manifest['freeze_ref']['path'],BASE/'root/argmix_joint_production_admission_v1.json')
    assert protocol['workers']==16 and protocol['epochs']==5 and protocol['training_targets']==500000 and protocol['training_target_presentations']==2500000
    assert amendment['cache_max_bytes']==24*2**30 and amendment['results_max_bytes']==2**30 and amendment['owner_storage_ceiling_bytes']==1200000000000
    checks.append('frozen_five_epoch_500000_target_16worker_science_and_storage_limits_unchanged')
    profile=c.read(amendment['memory_profile_ref'])
    peak,bound=w.measured_bound(profile)
    assert peak==17368395776 and bound==295262728192 and 384*2**30-bound==117054132224
    assert hashlib.sha256((ROOT/profile['public_bindings']['published_recon_train_002']['observation_ref']['path']).read_bytes()).hexdigest()==profile['public_bindings']['published_recon_train_002']['observation_ref']['sha256']
    assert hashlib.sha256((ROOT/amendment['supporting_source_refs'][0]['path']).read_bytes()).hexdigest()==profile['source_script_sha256']
    metadata=[]
    for parent,row in protocol['parents'].items():
        if row['split']!='train':continue
        observation=c.read(c.ref(ROOT/'experiments/paper_argmix_1611_rebuild_20261024_v1'/parent/'treeloom_observation/completed.json'))
        assert observation['private_truth_opened'] is False
        bridge=c.read(observation['bridge_ref'])
        tree=ROOT/bridge['tree_ref']['path']
        assert tree.stat().st_size==bridge['tree_ref']['bytes']
        metadata.append(dict(parent=parent,bytes=tree.stat().st_size,tree_ref=bridge['tree_ref']))
    assert max(metadata,key=lambda row:row['bytes'])['parent']=='published_recon_train_002'
    checks.append('TRAIN002_largest_original_TRAIN_tree_by_metadata_and_measured_replacement_peak_margin_verified')
    for ref in amendment['retained_cache_files']:
        path=ROOT/ref['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    assert sum(ref['bytes'] for ref in amendment['retained_cache_files'])==8804156
    checks.append('four_original_technical_cache_shards_preserved_byte_for_byte')
    original_functions={name:function for name,function in w.STAGES.items()}
    assert original_functions=={'prepare-cache':w.pipeline.prepare_cache,'train':w.pipeline.train,'predict':w.pipeline.predict,'barrier':w.pipeline.prediction_barrier,'score':w.pipeline.score,'validation-score':w.pipeline.validation_score}
    original_guard=w.pipeline.check_ram
    with patch.object(w.pipeline,'RAM_CAP',64*2**30), patch.object(w.pipeline,'check_ram',original_guard):
        w.apply_ram_authority()
        assert w.pipeline.RAM_CAP==384*2**30 and w.pipeline.check_ram is w.effective_check_ram
        assert w.runtime.RAM_CAP==64*2**30 and w.STAGES==original_functions
        rejects(w.apply_ram_authority)
    checks.append('only_pipeline_resource_guard_replaced_original_functions_and_storage_admission_constant_retained')
    class Process:
        def __init__(self,rss,children=(),vanished=False):self.rss,self.family,self.vanished=rss,children,vanished
        def children(self,recursive=False):assert recursive;return list(self.family)
        def memory_info(self):
            if self.vanished:raise w.psutil.NoSuchProcess(2147483600)
            return SimpleNamespace(rss=self.rss)
    for total,success in [(384*2**30,True),(384*2**30+1,False)]:
        family=[Process(total-1000),Process(123,vanished=True)]
        with patch.object(w.psutil,'Process',return_value=Process(1000,family)):
            if success:assert w.effective_check_ram()==total
            else:rejects(w.effective_check_ram)
    checks.append('aggregate_RSS_includes_parent_and_descendants_allows_exact_cap_rejects_one_byte_above')
    for field,value in [('max_rss_bytes',64*2**30),('max_seconds',1),('dependencies',[]),('output_directory','/wrong')]:
        bad=copy.deepcopy(plan);bad['tasks'][0][field]=value
        rejects(lambda bad=bad:c.validate_recovery(manifest,manifest_ref,bad))
    checks.append('changed_cap_walltime_dependencies_or_output_rejected')
    bad=copy.deepcopy(plan);bad['host_admission_minimum_available_bytes']=448*2**30-1
    rejects(lambda:c.validate_recovery(manifest,manifest_ref,bad))
    bad=copy.deepcopy(plan);bad['shared_resource_limits']['project_ceiling_bytes']+=1
    rejects(lambda:c.validate_recovery(manifest,manifest_ref,bad))
    checks.append('weakened_host_gate_or_changed_original_storage_ceiling_rejected')
    bad=copy.deepcopy(plan);bad['tasks'][0]['argv']+=['--unregistered']
    rejects(lambda:c.validate_recovery(manifest,manifest_ref,bad))
    checks.append('unregistered_stage_command_rejected')
    chosen={task['stage']:task for task in plan['tasks']}
    with patch.object(w,'dispatch',side_effect=AssertionError('No production dispatch permitted')):
        for stage,task in chosen.items():
            command=task['argv']
            with patch('sys.argv',['worker']+command[4:]+['--check-only']),contextlib.redirect_stdout(io.StringIO()) as capture:
                w.main()
            response=json.loads(capture.getvalue())
            assert response['valid'] is True and response['stage']==stage and response['production_stage_executed'] is False
    checks.append('all_seven_real_stage_command_shapes_parse_and_authenticate_without_dispatch')
    with patch.object(c,'stage',side_effect=AssertionError('No controller stage permitted')),patch('sys.argv',['controller','--manifest',str(manifest_path),'--check-only']),contextlib.redirect_stdout(io.StringIO()) as capture:
        c.main()
    assert json.loads(capture.getvalue())['production_launched'] is False
    checks.append('real_controller_check_only_validates_all15_tasks_without_spawn')
    baseline=c.read(c.ref(REVIEW/'failure_baseline_v1.json'))
    for ref in baseline['original_failed_refs']:
        path=ROOT/ref['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    checks.append('original_failure_start_and_worker_log_unchanged_after_review')
    receipt=dict(schema='independent_argmix_recovery_worker_controller_checks_v1',passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),
        controller_source_ref=c.ref(c.__file__),worker_source_ref=c.ref(w.__file__),manifest_ref=manifest_ref,plan_ref=plan_ref,
        resource_amendment_ref=manifest['resource_amendment_ref'],test_source_ref=c.ref(__file__),checks=checks,
        original_TRAIN_observed_tree_metadata=metadata,aggregate_measured_estimate_bytes=bound,guard_bytes=384*2**30,
        host_admission_bytes=host,headroom_above_measured_estimate_bytes=384*2**30-bound,
        future_heldout_peak_not_yet_measured=True,live_RSS_failstop_preserved=True,
        production_launched=False,signals_sent=0,heldout_payloads_or_truth_opened=False)
    output=REVIEW/'worker_controller_checks_result_v1.json'
    with output.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,checks=len(checks),receipt_ref=c.ref(output)),sort_keys=True))

if __name__=='__main__':main()
