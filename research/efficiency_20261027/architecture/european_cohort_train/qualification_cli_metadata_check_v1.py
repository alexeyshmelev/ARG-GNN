"""Bounded metadata-only CLI contract checks; never call main or acquire a lease."""
import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
from . import qualification_gate_v1 as gate
from . import qualify_cli_v1 as cli
from research.efficiency_20261027.execution.deferred_gpu_gate import validate_completion


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=gate.contained(args.output)
    gate.need(output.parent==cli.HERE and not output.exists(),'Fresh architecture metadata result only')
    before_env=dict(os.environ)
    cases=[]
    def reject(name,call):
        try:call()
        except (ValueError,FileNotFoundError,KeyError,TypeError):cases.append(name)
        else:raise AssertionError('Did not reject '+name)
    peers=[str(gate.PREDECESSOR.relative_to(gate.ROOT)),
           'research/efficiency_20261027/root/runs/synthetic_adaptive_peer_for_metadata_only']
    resource=SimpleNamespace(family_cap_bytes=512*cli.GIB,device_soft_bytes=64*cli.GIB,
        output_cap_bytes=64*2**20,max_seconds=86400,gpu_wait_seconds=7200,required_peer_directory=peers)
    argv=['--output','research/efficiency_20261027/root/runs/synthetic_not_created/data']
    command=['.runtime/env/bin/python3.11','-B','-m',cli.MODULE,*argv]
    request=dict(command=command,supervisor_pid=101,supervisor_start_ticks='1001',physical_gpu=5,
        gpu_uuid=gate.GPU_UUID,host_cap_bytes=resource.family_cap_bytes,device_budget_bytes=resource.device_soft_bytes,
        output_cap_bytes=resource.output_cap_bytes,peer_directories=peers)
    launch=dict(actual_argv=command,pid=102,process_start_ticks='1002')
    def validate(req=request,run=launch,res=resource):
        cli.validate_guardian_records(req,run,res,argv,102,'1002',101,'1001')
    validate();cases.append('exact_guardian_identity_caps_and_all_selected_peers_accepted')
    for key,value in dict(supervisor_pid=100,supervisor_start_ticks='bad',physical_gpu=3,gpu_uuid='GPU-wrong',
        host_cap_bytes=resource.family_cap_bytes+1,device_budget_bytes=resource.device_soft_bytes+1,
        output_cap_bytes=resource.output_cap_bytes+1,peer_directories=peers[:1]).items():
        modified=copy.deepcopy(request);modified[key]=value
        reject('guardian_'+key+'_tamper',lambda:validate(req=modified))
    modified=copy.deepcopy(launch);modified['pid']=103
    reject('child_PID_tamper',lambda:validate(run=modified))
    modified=copy.deepcopy(launch);modified['actual_argv']=[*command,'--unreviewed']
    reject('actual_argv_tamper',lambda:validate(run=modified))
    modified=copy.deepcopy(resource);modified.family_cap_bytes=511*cli.GIB
    req=copy.deepcopy(request);req['host_cap_bytes']=modified.family_cap_bytes
    reject('family_below_original_worker',lambda:validate(req=req,res=modified))
    modified=copy.deepcopy(resource);modified.gpu_wait_seconds=modified.max_seconds+1
    reject('wait_exceeds_wall',lambda:validate(res=modified))
    req=copy.deepcopy(request);req['peer_directories']=peers[1:]
    modified=copy.deepcopy(resource);modified.required_peer_directory=peers[1:]
    reject('missing_exact_predecessor_peer',lambda:validate(req=req,res=modified))
    comparison=dict(passed=True,argmax_equal=True,argmax_disagreements=0,categories={})
    for name,count in dict(logits=1,loss=1,gradients=49,parameters=49,moments=147).items():
        comparison['categories'][name]=dict(passed=True,tensors=count,mismatched_elements=0,mismatched_paths=[],
            structural_errors=[],atol=1e-6,rtol=1e-5 if name in ('logits','loss') else 1e-4)
    full=dict(schema='TreeLoom_upward_direct_residency_full_v1',status='completed',
        source_manifest_ref=gate.PREDECESSOR_SOURCE_REF,all_strict_comparisons_passed=True,
        all49_gradient_and_parameter_tensors_checked=True,all147_Adam_tensors_checked=True,physical_gpu='5',
        checkpoint_ref=dict(path='research/preprocessing_optimization_20261026/integration/migration/update241.pt',
            sha256='12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24'),
        scope=dict(mode='full',full_parent_claim=True,parent_id='argformer_american_train_000',haplotypes=1600,
            native_holds=77472,native_nodes=205369272,sites=227474,physical_support_bp=[0,30000000],
            supervised_original_focals=10000,heldout_truth_opened=False),
        records=[dict(kind=kind,optimizer_step=242,parameter_count=49,CPU_context=True,hotpack=False,
            comparison=copy.deepcopy(comparison)) for kind in ('boundary_fused_CPU512','upward_direct_resident_CPUcontext512')])
    gate.require_full_numerical(full);cases.append('complete_strict_predecessor_metadata_accepted')
    for name in comparison['categories']:
        modified=copy.deepcopy(full);modified['records'][1]['comparison']['categories'][name]['rtol']*=10
        reject('loosened_'+name+'_tolerance',lambda:gate.require_full_numerical(modified))
    modified=copy.deepcopy(full);del modified['records'][1]['comparison']['categories']['moments']
    reject('missing_Adam_category',lambda:gate.require_full_numerical(modified))
    modified=copy.deepcopy(full);modified['records'][1]['comparison']['argmax_disagreements']=1
    reject('changed_argmax',lambda:gate.require_full_numerical(modified))
    modified=copy.deepcopy(full);modified['scope']['native_nodes']-=1
    reject('incomplete_predecessor_parent',lambda:gate.require_full_numerical(modified))
    wait=dict(actual_wait=True,owned_group_absent=True,actual_returncode=0,pid=102,process_start_ticks='1002')
    validate_completion(launch,wait,None,[]);cases.append('actual_wait_and_absent_process_family_accepted')
    reject('live_predecessor_identity',lambda:validate_completion(launch,wait,'1002',[]))
    reject('surviving_predecessor_descendant',lambda:validate_completion(launch,wait,None,[103]))
    for key,value in dict(actual_wait=False,owned_group_absent=False,actual_returncode=1,pid=101,process_start_ticks='wrong').items():
        modified=copy.deepcopy(wait);modified[key]=value
        reject('predecessor_'+key+'_tamper',lambda:validate_completion(launch,modified,None,[]))
    manifest,manifest_ref=gate.read(cli.HERE/'qualification_cli_pins_v1.json')
    for entry in manifest['source_refs']:gate.need(gate.reference(entry['path'])==entry,'Exact wrapper closure')
    cases.append('all_wrapper_and_immutable_driver_dependencies_match')
    reject('missing_independent_wrapper_review',lambda:cli.verify_wrapper(manifest_ref['sha256'],
        cli.HERE/'ABSENT_wrapper_review_for_negative_check.json','0'*64))
    real_read=cli.read
    fake_path=cli.HERE/'in_memory_review_only.json'
    def fake_read(path,digest=None):
        if path==fake_path:return dict(schema='European_cohort8_qualification_CLI_review_v1',passed=False,
            wrapper_manifest_ref=manifest_ref),dict(path='in_memory_only',sha256='0'*64)
        return real_read(path,digest)
    with patch.object(cli,'read',fake_read):
        reject('failed_independent_wrapper_review',lambda:cli.verify_wrapper(manifest_ref['sha256'],fake_path,'0'*64))
    reject('outside_project_path',lambda:gate.contained('/tmp/not_allowed'))
    reject('parent_traversal_path',lambda:gate.contained('research/../not_allowed'))
    exact_request,exact_launch,request_ref,launch_ref=gate.predecessor_identity()
    cases.append('actual_pinned_predecessor_request_and_launch_match')
    gate.need(dict(os.environ)==before_env and 'torch' not in sys.modules,'No CUDA/torch import or environment mutation')
    gate.need(not (gate.ROOT/argv[1]).exists(),'No ROOT output created')
    result=dict(schema='European_cohort8_qualification_CLI_metadata_check_v1',passed=True,cases=cases,
        number_of_cases=len(cases),wrapper_manifest_ref=manifest_ref,actual_predecessor_request_ref=request_ref,
        actual_predecessor_launch_ref=launch_ref,real_locks_acquired=False,ROOT_outputs_created=False,
        scientific_arrays_opened=False,torch_imported=False,CUDA_initialized=False,simulations_launched=False,
        production_files_modified=False,source_check=gate.reference(__file__))
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,number_of_cases=len(cases),result_ref=gate.reference(output))))

if __name__=='__main__':main()
