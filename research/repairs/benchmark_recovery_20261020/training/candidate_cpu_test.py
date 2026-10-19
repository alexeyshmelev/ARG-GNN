"""CPU proof that recurrent chunk boundaries preserve full sequence gradients."""
import copy
import json
from pathlib import Path
import torch
from research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_frozen_v1 import AmericanTreeLoomC3FullV1
from .candidate_gru import install_chunked_gru


def main():
    torch.set_num_threads(4)
    torch.manual_seed(70019)
    reference=AmericanTreeLoomC3FullV1()
    candidate=install_chunked_gru(copy.deepcopy(reference),chunk_holds=7)
    geometry=torch.rand(31,3)
    source=torch.randn(31,9,32)
    weights=torch.randn(31,9,32)
    expected_input=source.clone().requires_grad_(True)
    actual_input=source.clone().requires_grad_(True)
    outputs=[]
    for model,value in ((reference,expected_input),(candidate,actual_input)):
        forward,forward_state=model.scan_context(value,geometry,direction='forward')
        reverse,reverse_state=model.scan_context(value,geometry,direction='reverse')
        loss=(forward*weights).sum()+(reverse*weights).sum()+forward_state.square().sum()+reverse_state.square().sum()
        loss.backward()
        outputs.append((forward.detach(),reverse.detach(),forward_state.detach(),reverse_state.detach()))
    errors=[float((a-b).abs().max()) for a,b in zip(outputs[0],outputs[1])]
    assert all(torch.allclose(a,b,rtol=5e-5,atol=2e-6) for a,b in zip(outputs[0],outputs[1]))
    input_error=float((expected_input.grad-actual_input.grad).abs().max())
    assert torch.allclose(expected_input.grad,actual_input.grad,rtol=5e-5,atol=3e-6)
    parameter_errors={}
    for (name,a),(other,b) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other
        if a.grad is not None:
            assert b.grad is not None and torch.allclose(a.grad,b.grad,rtol=5e-5,atol=3e-5),name
            parameter_errors[name]=float((a.grad-b.grad).abs().max())
        else:
            assert b.grad is None
    assert not torch.cuda.is_initialized()
    result=dict(schema='treeloom_chunked_GRU_CPU_v1',passed=True,CUDA_initialized=False,
        sequence_holds=31,chunk_holds=7,haplotypes=9,full_BPTT=True,hidden_detached=False,
        max_output_absolute_error=max(errors),max_input_gradient_absolute_error=input_error,
        max_parameter_gradient_absolute_error=max(parameter_errors.values()),
        parameter_count=sum(p.numel() for p in candidate.parameters()))
    (Path(__file__).parent/'candidate_cpu_test.result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
