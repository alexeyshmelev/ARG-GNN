from pathlib import Path
from dataclasses import replace
import json, torch
ns={'__name__':'independent_fixture_definitions'}
src=Path('research/reviews/census_cut_encoder_v1_independent/independent_checks.py').read_text()
exec(compile(src.split('METRICS={}',1)[0],'<independent_fixture_definitions>','exec'),ns)
Q,R,SCORED,TREES,INT,POS=map(ns.get,['Q','R','SCORED','TREES','INT','POS'])
features,close,error=map(ns.get,['features','close','error'])
prepare,preparebank,Observed=map(ns.get,['prepare_whole_tree_encoder_inputs_v1','prepare_census_exchange_bank_v1','ObservedDatedTreeV1'])
GNN=ns['CensusTreeGNNV1'];sigmas=ns['SIGMAS'];torch.set_num_threads(1)
fs=[features(t,grad=True) for t in TREES];inputs=[prepare(t,f) for t,f in zip(TREES,fs)]
b=preparebank(TREES,INT,POS,Q,R,SCORED,num_anchors=5,sigmas=sigmas)
torch.manual_seed(316);gnn=GNN(6,2,4,2,2,initial_exchange=(.8,1.4),initial_damping=.5).double()
base=gnn(inputs,b);qp=torch.tensor([5,2,0,4,1,3]);rp=torch.tensor([2,0,1]);graph_order=torch.tensor([2,0,1]);sample_order=torch.tensor([8,2,4,6,0,7,1,5,3])
changed_trees=[];changed_inputs=[]
for graph in graph_order.tolist():
 t=TREES[graph];n=len(t.parent);order=torch.arange(n).roll(5).flip(0);inverse=torch.argsort(order)
 old_parent=t.parent[order];parent=torch.where(old_parent<0,old_parent,inverse[old_parent.clamp_min(0)])
 changed=Observed(parent,t.node_times[order],t.sample_ids[sample_order],inverse[t.sample_nodes[sample_order]])
 changed_trees.append(changed);changed_inputs.append(prepare(changed,fs[graph][order]))
changed_bank=preparebank(changed_trees,INT[graph_order],POS,Q[qp],R[rp],SCORED,num_anchors=5,sigmas=sigmas)
z=gnn(changed_inputs,changed_bank)
score_lookup={q:i for i,q in enumerate(base.scored_query_ids.tolist())};score_index=torch.tensor([score_lookup[q] for q in z.scored_query_ids.tolist()])
close(z.logits,base.logits[:,score_index]);close(z.anchor_hidden,base.anchor_hidden[:,qp]);close(z.target_hidden,base.target_hidden[:,qp]);close(z.encoder_query_context,base.encoder_query_context[graph_order][:,qp]);close(z.encoder_reference_context,base.encoder_reference_context[graph_order][:,rp])
gradients=[torch.autograd.grad(out.logits.square().sum(),(*fs,*gnn.parameters()),retain_graph=True) for out in (base,z)]
for a,c in zip(*gradients):close(a,c,grad=True)
rename=95677;renamed_trees=[replace(t,sample_ids=t.sample_ids+rename) for t in TREES];renamed_inputs=[replace(i,sample_ids=i.sample_ids+rename) for i in inputs]
renamed_bank=preparebank(renamed_trees,INT,POS,Q+rename,R+rename,SCORED+rename,num_anchors=5,sigmas=sigmas)
zr=gnn(renamed_inputs,renamed_bank);torch.testing.assert_close(zr.logits,base.logits,rtol=0,atol=0)
report={'schema':'h021_independent_native_sample_forest_permutation','passed':True,'full_GNN_logit_max_error':error(z.logits,base.logits[:,score_index]),'anchor_hidden_max_error':error(z.anchor_hidden,base.anchor_hidden[:,qp]),'max_parameter_feature_VJP_error':max(error(a,c) for a,c in zip(*gradients)),'sample_ID_rename_exact':True,'serialized_graph_order':graph_order.tolist(),'new_scored_ID_order':z.scored_query_ids.tolist(),'status':'handmade_CPU_only'}
p=Path('research/reviews/census_cut_encoder_v1_independent/permutation_metrics.json');assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
