"""Reproduce full-tree/GNN tiny-fixture dependence and gradient receipts."""
from pathlib import Path
from dataclasses import replace
import datetime, hashlib, importlib.util, json, os
import torch
from torch.func import functional_call
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[3]
assert Path.cwd()==root
dest=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("h021_tree_tests",root/"tests/test_census_tree_encoder_v1.py")
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
rows=[]
for dtype in (torch.float64,torch.float32):
    pairs=[t.prepared(which,dtype,True) for which in (0,1)]
    inputs=[x[0] for x in pairs];b=t.bank(dtype);gnn=t.model(dtype)
    for control in ("proposed","eta0","permuted_groups","raw_diagonal","local"):
        gnn.zero_grad(set_to_none=True)
        for _,value in pairs:value.grad=None
        cb=t.permuted_group_control_v1(b,torch.tensor([2,3,1,0])) if control=="permuted_groups" else b
        out=gnn(inputs,cb,control=control,readout_chunk_size=2)
        t.loss(out.logits).backward()
        active={name:float(parameter.grad.abs().sum()) for name,parameter in gnn.named_parameters() if parameter.grad is not None}
        inactive=[name for name,parameter in gnn.named_parameters() if parameter.grad is None]
        rows.append({"dtype":str(dtype),"control":control,"active_parameter_tensors":len(active),"inactive_parameter_names":inactive,"minimum_active_parameter_gradient_L1":min(active.values()),"minimum_retained_node_feature_gradient_L1":min(float(value.grad.abs().sum(1).min()) for _,value in pairs),"logits_finite":bool(torch.isfinite(out.logits).all()),"decoded_actual_ids":out.scored_query_ids.tolist()})

inputs=[t.prepared(which)[0] for which in (0,1)];b=t.bank();gnn=t.model()
changes={}
base=gnn.encoder(inputs[0])
for node in (8,9,5):
    value=inputs[0].features.clone();value[node,3]+=2.
    altered=gnn.encoder(replace(inputs[0],features=value))
    changes[str(node)]=float((altered[:4]-base[:4]).abs().detach().max())
changed=inputs[1].features.clone();changed[4,2]+=4.
other=[inputs[0],replace(inputs[1],features=changed)]
mask=b.target_graph==0
local=gnn(inputs,b,control="local").logits;changed_local=gnn(other,b,control="local").logits
finite=gnn(inputs,b).logits;changed_finite=gnn(other,b).logits

pairs=[t.prepared(which,gradients=True) for which in (0,1)]
inputs=[x[0] for x in pairs];values=[x[1] for x in pairs];gnn=t.model();named=list(gnn.named_parameters())
grad=torch.autograd.grad(t.loss(gnn(inputs,b).logits),(*values,*(p for _,p in named)))
generator=torch.Generator().manual_seed(551)
directions=[torch.randn(v.shape,dtype=torch.float64,generator=generator) for v in (*values,*(p for _,p in named))]
for d in directions[:2]:d[:,6]=0.
exact=sum((g*d).sum() for g,d in zip(grad,directions));step=2e-4;f={}
for k in (-2,-1,1,2):
    shifted=[replace(item,features=item.features.detach()+k*step*d) for item,d in zip(inputs,directions[:2])]
    params={name:p.detach()+k*step*d for (name,p),d in zip(named,directions[2:])}
    f[k]=t.loss(functional_call(gnn,params,(shifted,b)).logits)
difference=(f[-2]-8*f[-1]+8*f[1]-f[2])/(12*step)
receipt={"schema":"h021_census_tree_encoder_v1_tiny_fixture_numbers","created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"Handmade CPU whole-tree encoder/two-depth GNN unit forwards and gradients only. Two10-node/9-edge trees,Q4/R2,A3,S3,H3,D7,P8,scoredQ2,C4. No native data, inference, fitting, CUDA, performance benchmark or new backbone novelty.","torch_version":torch.__version__,"source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ("src/arggnn/census_tree_encoder_v1.py","tests/test_census_tree_encoder_v1.py")},"controls":rows,"dependence_fixture":{"query_context_change_max_abs_by_native_node_perturbation":changes,"node_meanings":{"8":"branching interior","9":"root","5":"age600 pure reference"},"other_hold_older_reference_change":{"local_graph0_max_abs":float((local[mask]-changed_local[mask]).abs().detach().max()),"local_graph1_max_abs":float((local[~mask]-changed_local[~mask]).abs().detach().max()),"finite_graph0_max_abs":float((finite[mask]-changed_finite[mask]).abs().detach().max())}},"full_GNN_five_point_directional_derivative":{"step":step,"autograd":float(exact.detach()),"finite_difference":float(difference.detach()),"absolute_error":float((exact-difference).abs().detach()),"rtol":2e-8,"atol":1e-9,"perturbation_scope":"All active GNN parameters and both observed F47 inputs; native encoded-age column fixed."},"invariance_qualification":"Dedicated suite verifies actual sample/native-node permutation, unary insertion after suppression, unrelated isolated-node exclusion, sparse provenance, identities absent from features and parameter sharing across all controls."}
(dest/"tiny_fixture_numbers.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"control_gradient_rows":rows,"dependence_fixture":receipt["dependence_fixture"],"full_GNN_five_point_directional_derivative":receipt["full_GNN_five_point_directional_derivative"]},indent=2))
