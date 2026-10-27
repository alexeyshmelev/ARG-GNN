"""Small metadata tests only: no main, observations, Torch, CUDA, or lock acquisition."""
import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
from . import protocol_v3 as p


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=p.contained(args.output);p.need(output.parent==p.HERE and not output.exists(),'Fresh own metadata test output')
    cases=[];environment=dict(os.environ)
    def reject(name,call):
        try:call()
        except (ValueError,FileNotFoundError,KeyError,TypeError,AssertionError):cases.append(name)
        else:raise AssertionError('Did not reject '+name)
    launch=dict(pid=102,process_start_ticks='1002')
    wait=dict(actual_wait=True,owned_group_absent=True,actual_returncode=-15,pid=102,process_start_ticks='1002')
    p.terminal_wait(launch,wait,None,[],-15);cases.append('explicit_selected_nonzero_resource_exit_accepted')
    reject('unselected_nonzero_exit',lambda:p.terminal_wait(launch,wait,None,[],0))
    reject('live_predecessor',lambda:p.terminal_wait(launch,wait,'1002',[],-15))
    reject('surviving_descendant',lambda:p.terminal_wait(launch,wait,None,[104],-15))
    for key,value in dict(actual_wait=False,owned_group_absent=False,pid=103,process_start_ticks='other',actual_returncode=True).items():
        modified=dict(wait,**{key:value})
        reject('wait_'+key,lambda:p.terminal_wait(launch,modified,None,[],-15))
    peers=[str(p.EUROPEAN.relative_to(p.ROOT)),'research/efficiency_20261027/root/runs/synthetic_metadata_peer']
    for mode in ('prefix',):
        b=p.bounds(mode)
        res=SimpleNamespace(mode=mode,family_cap_bytes=b['recommended_family_bytes'],max_seconds=7200,
            gpu_wait_seconds=60,required_peer_directory=peers)
        argv=['--mode',mode,'--output','research/efficiency_20261027/root/runs/synthetic_not_created/data']
        command=['.runtime/env/bin/python3.11','-B','-m',p.MODULE,*argv]
        req=dict(command=command,supervisor_pid=101,supervisor_start_ticks='1001',physical_gpu=5,gpu_uuid=p.GPU_UUID,
            host_cap_bytes=res.family_cap_bytes,device_budget_bytes=b['device_soft_bytes'],output_cap_bytes=b['output_bytes'],peer_directories=peers)
        launched=dict(launch,actual_argv=command)
        def validate(request=req,run=launched,resource=res):p.validate_guardian(request,run,resource,argv,102,'1002',101,'1001')
        validate();cases.append(mode+'_exact_guardian_accepted')
        for key,value in dict(supervisor_pid=103,supervisor_start_ticks='bad',physical_gpu=3,gpu_uuid='wrong',
            host_cap_bytes=res.family_cap_bytes+1,device_budget_bytes=b['device_soft_bytes']+1,
            output_cap_bytes=b['output_bytes']+1,peer_directories=peers[:1]).items():
            modified=dict(req,**{key:value});reject(mode+'_guardian_'+key,lambda:validate(request=modified))
        reject(mode+'_actual_command',lambda:validate(run=dict(launched,actual_argv=command+['--extra'])))
        resource=copy.deepcopy(res);resource.family_cap_bytes=b['worker_RSS_bytes']-1
        reject(mode+'_family_below_worker',lambda:validate(request=dict(req,host_cap_bytes=resource.family_cap_bytes),resource=resource))
        resource=copy.deepcopy(res);resource.gpu_wait_seconds=resource.max_seconds+1
        reject(mode+'_wait_above_wall',lambda:validate(resource=resource))
        resource=copy.deepcopy(res);resource.required_peer_directory=peers[1:]
        reject(mode+'_missing_resource_predecessor_peer',lambda:validate(request=dict(req,peer_directories=peers[1:]),resource=resource))
    comparison=dict(passed=True,argmax_equal=True,argmax_disagreements=0,categories={})
    for key,count in dict(logits=1,loss=1,gradients=49,parameters=49,moments=147).items():
        comparison['categories'][key]=dict(passed=True,tensors=count,mismatched_elements=0,mismatched_paths=[],structural_errors=[],
            atol=1e-6,rtol=1e-5 if key in ('logits','loss') else 1e-4)
    manifest,manifest_ref=p.read(p.HERE/'runner_pins_v3.json')
    for row in manifest['source_refs']:p.need(p.reference(row['path'])==row,'Exact runner source closure')
    cases.append('all_runner_dependencies_match')
    for mode in ('prefix',):
        b=p.bounds(mode);amount=b['expected_holds']*1600*32*4
        full=dict(schema='TreeLoom_upward_context_prefix_diagnostic_v1',mode=mode,status='diagnostic_completed',source_manifest_ref=manifest_ref,
            checkpoint_ref=p.CHECKPOINT_REF,all_strict_comparisons_passed=True,strict_context_passed=True,
            strict_downstream_passed=True,practical_context_passed=True,eligible_for_full_qualification=False,
            scope=dict(parent_id='argformer_american_train_000',haplotypes=1600,native_holds=b['expected_holds'],
                heldout_truth_opened=False,full_parent_claim=mode=='full',native_nodes=205369272,sites=227474,
                supervised_original_focals=10000,physical_support_bp=[0,30000000]),records=[])
        for index,kind in enumerate(p.KINDS):
            device='cpu' if index==0 else 'cuda:0'
            full['records'].append(dict(kind=kind,parameter_count=49,optimizer_step=242,CPU_context=index==0,hotpack=False,
                comparison=copy.deepcopy(comparison),context_boundary=dict(device=device,bytes=amount),VJP_boundary=dict(device=device,bytes=amount),
                gradient_norm=1.,update_elapsed_including_diagnostic_seconds=2.,diagnostic_context_seconds=0.,update_seconds=2.,
                torch_peak_reserved_bytes=1,resident_bank_plan=dict(blocks=3),explicit_vjp=dict(residency_lifetime=dict(
                    phase='finished',live_bank_bytes=0,future_bank_bytes=0,upward_blocks_released=3,direct_blocks_released=3))))
        full['gradient_norm_comparison']=p.compare_norm(1.,1.)
        full['records'][1]['context_comparison']=dict(passed=True,strict_context_passed=True,strict_mismatched_elements=0,practical_context_passed=True,
            practical_mismatched_elements=0,practical_atol=5e-6,practical_rtol=1e-5,shape=[b['expected_holds'],1600,32],dtype='torch.float32',
            elements=b['expected_holds']*1600*32,maximum_chunk_holds=128,atol=1e-6,rtol=1e-5,all_values_finite=True,
            reference_CPU_storage_retained_without_clone=True,reference_released_before_decoder=True,context_values_serialized=False,
            diagnostic_seconds=0.)
        p.require_completed(full,mode,manifest_ref);cases.append(mode+'_strict_complete_pair_accepted')
        for category in comparison['categories']:
            modified=copy.deepcopy(full);modified['records'][1]['comparison']['categories'][category]['rtol']*=10
            reject(mode+'_loosened_'+category,lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['comparison']['argmax_disagreements']=1
        reject(mode+'_changed_argmax',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['explicit_vjp']['residency_lifetime']['live_bank_bytes']=1
        reject(mode+'_unreleased_bank',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['context_boundary']['device']='cpu'
        reject(mode+'_wrong_context_device',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['torch_peak_reserved_bytes']=b['device_soft_bytes']+1
        reject(mode+'_actual_peak_over_limit',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['scope']['native_holds']-=1
        reject(mode+'_incomplete_support',lambda:p.require_completed(modified,mode,manifest_ref))
        for key,value in dict(elements=1,maximum_chunk_holds=512,atol=1e-5,rtol=1e-4,all_values_finite=False,
            reference_released_before_decoder=False,context_values_serialized=True,dtype='torch.float64').items():
            modified=copy.deepcopy(full);modified['records'][1]['context_comparison'][key]=value
            reject(mode+'_context_'+key,lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['gradient_norm']=2.
        reject(mode+'_norm_changed',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full);modified['records'][1]['update_seconds']=1.
        reject(mode+'_diagnostic_clock_mismatch',lambda:p.require_completed(modified,mode,manifest_ref))
        modified=copy.deepcopy(full)
        modified['records'][1]['context_comparison'].update(passed=False,strict_context_passed=False,strict_mismatched_elements=6)
        modified.update(strict_context_passed=False,all_strict_comparisons_passed=False)
        p.require_completed(modified,mode,manifest_ref);cases.append('honest_strict_context_failure_with_practical_pass_accepted_as_diagnostic')
        modified['eligible_for_full_qualification']=True
        reject('diagnostic_cannot_authorize_full',lambda:p.require_completed(modified,mode,manifest_ref))
    reject('full_mode_forbidden',lambda:p.bounds('full'))
    reject('diagnostic_cannot_release_full',p.prefix_release)
    reject('nonfinite_norm',lambda:p.compare_norm(1.,float('nan')))
    reject('missing_review',lambda:p.verify_sources(manifest_ref['sha256'],p.HERE/'ABSENT_review_metadata_test.json','0'*64))
    with patch.object(p,'original_full_release',return_value=None):
        reject('original_full_scientific_success_mandatory',p.scientific_predecessor)
    original=p.scientific_predecessor();cases.append('actual_successful_full_scientific_predecessor')
    resource=p.european_resource_release(-15);p.need(resource is not None and resource['actual_returncode']==-15,'Exact observed resource terminal status')
    cases.append('actual_European_resource_exit_minus15_accepted_explicitly')
    reject('actual_European_nonzero_not_silently_accepted',lambda:p.european_resource_release(0))
    reject('outside_project',lambda:p.contained('/tmp/no'))
    reject('parent_traversal',lambda:p.contained('research/../no'))
    p.need('torch' not in sys.modules and dict(os.environ)==environment,'No Torch or environment changes')
    p.need(not (p.ROOT/argv[3]).exists(),'No ROOT output created')
    result=dict(schema='TreeLoom_upward_context_runner_metadata_v3',passed=True,number_of_cases=len(cases),cases=cases,
        source_manifest_ref=manifest_ref,scientific_predecessor=original,European_resource_predecessor=resource,
        real_locks_acquired=False,scientific_arrays_opened=False,torch_imported=False,CUDA_initialized=False,
        ROOT_outputs_created=False,production_files_modified=False,source_check=p.reference(__file__))
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,cases=len(cases),result_ref=p.reference(output))))


if __name__=='__main__':main()
