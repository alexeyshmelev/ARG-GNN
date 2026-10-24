"""Small FP32 full-depth cavity DAG proof with duplicated and changed subtrees."""
from pathlib import Path
import copy,json,time
import torch
from .cavity_dag import CavityDAG,CavityOperator

def main():
    torch.set_num_threads(1);torch.manual_seed(9024);started=time.monotonic()
    trees=[[4,4,5,5,6,6,-1]]*3+[[4,5,4,5,6,6,-1]]
    parent=[p+7*t if p>=0 else -1 for t,tree in enumerate(trees) for p in tree]
    base_ids=list(range(7))*4;base_ids[21]=7
    plan=CavityDAG(parent,base_ids,list(range(7))*4,[[7*t+i for i in range(4)] for t in range(4)])
    dense=CavityOperator();compact=copy.deepcopy(dense)
    a=torch.randn(8,8,requires_grad=True);g=torch.randn(7,9,requires_grad=True)
    b=a.detach().clone().requires_grad_();h=g.detach().clone().requires_grad_()
    expected=dense.dense_reference(plan,a,g);actual=compact(plan,b,h)
    weights=torch.linspace(.1,1.7,expected.numel()).reshape_as(expected)
    (expected*weights).square().mean().backward();(actual*weights).square().mean().backward()
    output_error=float((expected-actual).abs().max())
    parameter_error=max(float((x.grad-y.grad).abs().max()) for x,y in zip(dense.parameters(),compact.parameters()))
    input_error=max(float((a.grad-b.grad).abs().max()),float((g.grad-h.grad).abs().max()))
    assert output_error<1e-5 and parameter_error<1e-4 and input_error<1e-4
    assert plan.up[2]==plan.up[23] and plan.out[2]!=plan.out[23]
    assert torch.equal(actual[0],actual[1]) and not torch.equal(actual[0],actual[3])
    result=dict(schema="hash_consed_cavity_DAG_FP32_CPU_proof_v1",passed=True,CPU_only=True,
        logit_max_abs_error=output_error,parameter_gradient_max_abs_error=parameter_error,
        source_input_gradient_max_abs_error=input_error,relative_sample_routes_preserved=True,
        unchanged_upward_subtree_changed_outside_context_test=True,full_depth_recursions=True,
        **plan.receipt(),seconds=time.monotonic()-started,
        limitations="Small standalone first-sweep operator proof only; no full TreeLoom/site/history/GRU equivalence, real-data speed or ancestry score")
    path=Path("research/scalability_20261024/model/dag_cpu_proof.json")
    path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
