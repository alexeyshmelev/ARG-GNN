"""Bounded complete-context FP32 diagnostics; no model call or array persistence."""
import math
from research.efficiency_20261027.architecture.european_cohort_train.qualification_gate_v1 import need

STRICT_ATOL=1e-6
PRACTICAL_ATOL=5e-6
RTOL=1e-5
CHUNK_HOLDS=128
MAX_EXAMPLES=16


def compare_context(torch,actual_context,reference_context,check):
    need(reference_context.device.type=='cpu' and actual_context.dtype==reference_context.dtype==torch.float32
         and actual_context.ndim==3 and actual_context.shape==reference_context.shape and actual_context.numel()>0,
         'Exact nonempty complete FP32 context shape and CPU reference')
    count=strict_count=practical_count=unequal=0
    maximum_error=maximum_strict_ratio=maximum_practical_ratio=0.
    reference_square_sum=delta_square_sum=0.
    examples=[];chunks=[]
    bins=[dict(reference_abs_lower=lo,reference_abs_upper=hi,elements=0,strict_mismatches=0,practical_mismatches=0)
          for lo,hi in [(0.,1e-6),(1e-6,1e-4),(1e-4,1e-2),(1e-2,1.),(1.,None)]]
    shape=list(actual_context.shape)
    for first in range(0,len(actual_context),CHUNK_HOLDS):
        check('untimed_complete_context_diagnostic')
        expected=reference_context[first:first+CHUNK_HOLDS]
        actual=actual_context[first:first+CHUNK_HOLDS].detach().cpu()
        need(bool(torch.isfinite(expected).all()) and bool(torch.isfinite(actual).all()),
             'Finite full reference and candidate contexts')
        reference_abs=expected.abs();difference=(actual-expected).abs()
        strict_limit=STRICT_ATOL+RTOL*reference_abs;practical_limit=PRACTICAL_ATOL+RTOL*reference_abs
        strict=difference>strict_limit;practical=difference>practical_limit
        elements=actual.numel();bad=int(strict.sum());practical_bad=int(practical.sum())
        local_max=float(difference.max());strict_ratio=float((difference/strict_limit).max())
        practical_ratio=float((difference/practical_limit).max())
        count+=elements;strict_count+=bad;practical_count+=practical_bad;unequal+=int((actual!=expected).sum())
        maximum_error=max(maximum_error,local_max);maximum_strict_ratio=max(maximum_strict_ratio,strict_ratio)
        maximum_practical_ratio=max(maximum_practical_ratio,practical_ratio)
        # Double accumulation is only scalar diagnostic reduction. All model,
        # context, VJP and optimizer arithmetic remains the original FP32.
        reference_square_sum+=float(expected.square().sum(dtype=torch.float64))
        delta_square_sum+=float(difference.square().sum(dtype=torch.float64))
        chunks.append(dict(first_hold=first,last_hold=first+len(actual),elements=elements,
            strict_mismatches=bad,practical_mismatches=practical_bad,maximum_absolute_error=local_max,
            maximum_strict_normalized_error=strict_ratio,maximum_practical_normalized_error=practical_ratio))
        for row in bins:
            selected=reference_abs>=row['reference_abs_lower']
            if row['reference_abs_upper'] is not None:selected&=reference_abs<row['reference_abs_upper']
            row['elements']+=int(selected.sum());row['strict_mismatches']+=int((selected&strict).sum())
            row['practical_mismatches']+=int((selected&practical).sum())
        values,indices=torch.topk(difference.reshape(-1),min(MAX_EXAMPLES,elements))
        flat_reference=expected.reshape(-1);flat_actual=actual.reshape(-1)
        for error,index in zip(values.tolist(),indices.tolist()):
            hold,remainder=divmod(index,shape[1]*shape[2]);sample,feature=divmod(remainder,shape[2])
            reference_value=float(flat_reference[index]);actual_value=float(flat_actual[index])
            examples.append(dict(coordinate=[first+hold,sample,feature],reference=reference_value,candidate=actual_value,
                absolute_error=error,strict_limit=STRICT_ATOL+RTOL*abs(reference_value),
                practical_limit=PRACTICAL_ATOL+RTOL*abs(reference_value)))
        examples=sorted(examples,key=lambda row:(-row['absolute_error'],row['coordinate']))[:MAX_EXAMPLES]
        del expected,actual,reference_abs,difference,strict_limit,practical_limit,strict,practical
        del selected,values,indices,flat_reference,flat_actual
    need(count==actual_context.numel() and sum(row['elements'] for row in bins)==count,'Every complete context value counted once')
    return dict(passed=strict_count==0,strict_context_passed=strict_count==0,
        practical_context_passed=practical_count==0,shape=shape,dtype='torch.float32',elements=count,
        maximum_chunk_holds=CHUNK_HOLDS,atol=STRICT_ATOL,rtol=RTOL,
        strict_mismatched_elements=strict_count,practical_atol=PRACTICAL_ATOL,practical_rtol=RTOL,
        practical_mismatched_elements=practical_count,unequal_FP32_values=unequal,
        maximum_absolute_error=maximum_error,reference_RMS=math.sqrt(reference_square_sum/count),
        delta_RMS=math.sqrt(delta_square_sum/count),
        relative_L2_error=None if reference_square_sum==0 else math.sqrt(delta_square_sum/reference_square_sum),
        maximum_strict_normalized_error=maximum_strict_ratio,maximum_practical_normalized_error=maximum_practical_ratio,
        worst_absolute_error_examples=examples,reference_magnitude_bins=bins,chunks=chunks,all_values_finite=True,
        reference_CPU_storage_retained_without_clone=True,reference_context_strides=list(reference_context.stride()),
        candidate_context_strides=list(actual_context.stride()),candidate_context_contiguous=actual_context.is_contiguous(),
        context_values_serialized=False,model_precision='original_FP32',full_model_calls_added=0,
        interpretation='Original strict status is preserved separately. Fixed prospective practical envelope is diagnostic only; no automatic full qualification.')
