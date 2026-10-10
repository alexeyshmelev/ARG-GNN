"""Reproduce independent handmade fixture summary; never loads project arrays."""
import json,runpy,math,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(1)
x=runpy.run_path('research/reviews/census_biological_bank_v1_independent/test_independent_join.py')
ts,roster=x['handcrafted']();cert=x['prepare'](ts,roster)
q=[r.native_sample_node for r in roster if r.role!='pure_source_reference'];r=[r.native_sample_node for r in roster if r.role=='pure_source_reference']
errors=[]
for graph,tree in enumerate(ts.trees()):
    tmrca=np.array([[tree.time(tree.mrca(a,b)) for b in r] for a in q]);logits=-np.log1p(tmrca)[None]/math.log1p(10000.)/np.array([.05,.1,.2])[:,None,None];oracle=np.exp(logits-logits.max(-1,keepdims=True));oracle/=oracle.sum(-1,keepdims=True)
    errors.append(float(np.abs(oracle-cert.bank.donor_weights[graph].numpy()).max()))
support=x['collect_census_biological_support_v1']([cert],[cert.certificate_sha256],expected_tiles=1)
model=x['model']();outputs={};active={}
for control in ('proposed','eta0','permuted_groups','raw_diagonal','local'):
    kwargs={'group_permutation':torch.tensor([5,2,0,1,3,4])} if control=='permuted_groups' else {}
    out=x['forward_certified_census_biological_v1'](model,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,**kwargs)
    grads=torch.autograd.grad(out.logits.sin().sum(),tuple(model.parameters()),allow_unused=True)
    active[control]={'active_gradient_tensors':sum(g is not None for g in grads),'finite':all(g is None or torch.isfinite(g).all().item() for g in grads)}
    outputs[control]={'shape':list(out.logits.shape),'finite':torch.isfinite(out.logits).all().item(),'max_abs_logits':float(out.logits.detach().abs().max())}
metrics={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Independent handmade CPU observations only. No actual biological panel distribution or performance claim.','native_ts_nodes':ts.num_nodes,'native_ts_samples':ts.num_samples,'native_ts_trees':ts.num_trees,'stored_x48_shape':list(cert.stored_sequence.x.shape),'connected_node_counts':[len(t.parent) for t in cert.native_trees],'retained_node_counts':[len(i.parent) for i in cert.inputs],'excluded_nonancestor_rows_by_tree':cert.provenance['excluded_unrelated_nodes_by_tree'],'donor_oracle_max_abs_errors':errors,'group_members':[[sorted(g) for g in p.group_members] for p in cert.bank.partitions],'target_graph':cert.bank.target_graph.tolist(),'anchor_graph':cert.bank.layout.anchor_graph.tolist(),'sample_join_actual_ids':cert.native_trees[0].sample_ids.tolist(),'scored_ids':[r.actual_id for r in roster if r.role=='query'],'older_R_proxy_count':len(cert.bank.partitions[0].older_reference_ids),'all_five_control_outputs':outputs,'all_five_control_gradient_checks':active,'support':support}
p=Path('research/reviews/census_biological_bank_v1_independent/handmade_metrics.json');p.write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({'donor_oracle_max_abs_errors':errors,'handmade_certificate_sha256':cert.certificate_sha256,'metrics_path':str(p)}))
