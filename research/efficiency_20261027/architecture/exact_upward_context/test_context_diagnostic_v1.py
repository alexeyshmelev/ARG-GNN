"""Tiny CPU-only diagnostic reduction tests, without models or scientific data."""
import argparse
import json
import sys
import torch
from .context_diagnostic_v1 import compare_context
from .protocol_v3 import HERE,contained,need,reference


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=contained(args.output);need(output.parent==HERE and not output.exists(),'Fresh own CPU-only test report')
    torch.set_num_threads(1)
    cases=[]
    def reject(name,call):
        try:call()
        except ValueError:cases.append(name)
        else:raise AssertionError('Did not reject '+name)
    ref=torch.zeros((129,2,3),dtype=torch.float32);actual=ref.clone();checks=[]
    exact=compare_context(torch,actual,ref,checks.append)
    need(exact['passed'] and exact['practical_context_passed'] and exact['elements']==774
         and len(exact['chunks'])==2 and exact['delta_RMS']==0 and len(checks)==2,'Complete exact two-chunk reduction')
    cases.append('exact_values_complete_two_chunks')
    actual[0,0,0]=1.4e-6;actual[128,1,2]=4e-6;actual[1,0,1]=.5e-6
    changed=compare_context(torch,actual,ref,lambda stage:None)
    need(not changed['passed'] and changed['strict_mismatched_elements']==2 and changed['practical_context_passed']
         and changed['practical_mismatched_elements']==0 and changed['unequal_FP32_values']==3
         and changed['worst_absolute_error_examples'][0]['coordinate']==[128,1,2]
         and changed['reference_magnitude_bins'][0]['strict_mismatches']==2
         and changed['maximum_strict_normalized_error']>1 and changed['maximum_practical_normalized_error']<=1
         and changed['delta_RMS']>0,'Separate strict failure and practical pass with correct cross-chunk coordinates')
    cases.append('strict_mismatches_preserved_practical_pass_and_worst_global_coordinate')
    actual[2,0,0]=1e-4
    failed=compare_context(torch,actual,ref,lambda stage:None)
    need(not failed['practical_context_passed'] and failed['practical_mismatched_elements']==1,'Fixed practical envelope fails')
    cases.append('fixed_practical_envelope_failure')
    relative_ref=torch.ones((1,2,3),dtype=torch.float32)
    relative=compare_context(torch,relative_ref+5e-6,relative_ref,lambda stage:None)
    need(relative['passed'] and relative['reference_RMS']==1. and relative['relative_L2_error']>0,'Original relative tolerance')
    cases.append('original_relative_tolerance_and_RMS')
    nonfinite=actual.clone();nonfinite[0,0,0]=float('nan')
    reject('nonfinite_rejected',lambda:compare_context(torch,nonfinite,ref,lambda stage:None))
    reject('shape_rejected',lambda:compare_context(torch,actual[:1],ref,lambda stage:None))
    reject('dtype_rejected',lambda:compare_context(torch,actual.double(),ref,lambda stage:None))
    reject('empty_rejected',lambda:compare_context(torch,actual[:0],ref[:0],lambda stage:None))
    need(not torch.cuda.is_initialized(),'CPU-only noCUDA')
    result=dict(schema='TreeLoom_context_diagnostic_CPU_v1',passed=True,cases=cases,synthetic_elements=774,
        no_model_constructed=True,no_scientific_arrays_opened=True,CUDA_initialized=False,
        source_refs=[reference(__file__),reference(HERE/'context_diagnostic_v1.py')])
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,cases=len(cases),result_ref=reference(output))))


if __name__=='__main__':main()
