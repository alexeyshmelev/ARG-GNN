"""CPU full-panel parity test of the plaintext European executor."""
from pathlib import Path
import json
import time
import numpy as np
import torch
import tskit
from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g
from arggnn.benchmarking.paper_argmix.paper_argmix_treeloom import legacy_ref,tensor_bytes
from arggnn.benchmarking.paper_argmix.paper_argmix_treeloom_executor import PaperEuropeanBoundedExecutionV1
from arggnn.benchmarking import european_treeloom as european

base=g.ROOT/'research/benchmark_rebuild_20261024/argmix/smoke_treeloom_cpu_v1'
g.need(not base.exists(),'Fresh CPU parity output required')
base.mkdir()
torch.set_num_threads(1);torch.manual_seed(71)
started=time.monotonic()
config=g.metadata(g.CONFIG)
rows=[{k:v for k,v in row.items() if k!='sampling_population'} for row in g.roster(config)]
tables=tskit.TableCollection(sequence_length=100)
current=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=row['age_generations']) for row in rows]
clock=500
while len(current)>1:
 parents=[]
 for start in range(0,len(current),2):
  parent=tables.nodes.add_row(time=clock)
  for child in current[start:start+2]: tables.edges.add_row(0,100,parent,child)
  parents.append(parent)
 current=parents;clock*=2
positions=np.asarray([10,30,60,90],dtype=np.float64)
for position in positions: tables.sites.add_row(position,'A')
tables.sort();ts=tables.tree_sequence()
nodes=np.asarray(ts.samples(),dtype=np.int64)
for row,node in zip(rows,nodes): row['inferred_tree_node_id']=int(node)
rng=np.random.Generator(np.random.PCG64(71))
genotype=rng.binomial(1,.2,size=(4,3222)).astype(np.int8)
query=np.asarray([r['sample_index'] for r in rows if r['query_eligible']],dtype=np.int64)
focal=query[[0,1,300,1000,2000,2971]]
registry=dict(record_ids=np.arange(6,dtype=np.int64),sample_nodes=nodes[focal],sample_indices=focal,
 positions_bp0=np.asarray([10,20,30,40,50,60],dtype=np.float64),parent_id=np.asarray('synthetic_fullpanel_cpu_smoke'),class_order=np.asarray(g.CLASSES))
np.savez(base/'registry.npz',**registry)
(base/'observed_state.json').write_text(json.dumps(dict(synthetic_cpu_smoke=True,truth_read=False))+'\n')
refs=dict(inference_ref=legacy_ref(base/'observed_state.json'))
labels=np.asarray([-1 if r['observed_reference_class'] is None else r['observed_reference_class'] for r in rows],dtype=np.int64)
state=dict(manifest=dict(parent_id='synthetic_fullpanel_cpu_smoke',split='train',sequence_length_bp=35178458,truth_or_oracle_read=False),
 ts=ts,rows=rows,nodes=nodes,genotype=genotype,positions=positions,allele_count=np.full(4,2,dtype=np.int8),
 labels=labels,ages=np.asarray([r['age_generations'] for r in rows],dtype=np.float64),query_columns=query,coordinate_offset_bp=0.,source_refs=refs)
def check(label): pass
bundle,metadata=european.prepare_observation(state,protocol_id='european_c4_prospective1611',parent_id='synthetic_fullpanel_cpu_smoke',split='train',source_refs=refs,
 sampled_registry_ref=legacy_ref(base/'registry.npz'),max_nodes=100000,max_holds=10,max_episode_payload_bytes=4*1024**2,resource_check=check)
model=european.build_model('european_c4_prospective1611')
model.eval()
with torch.no_grad():
 reference=model.forward_holds(bundle.bank,episodes=bundle.original)[bundle.focal_hold,bundle.focal_sample_column]
 actual=PaperEuropeanBoundedExecutionV1(model,bundle,metadata,resource_check=check).forward()
error=float((reference-actual).abs().max())
g.need(torch.allclose(reference,actual,rtol=1e-4,atol=1e-5),'Static executor differs from frozen European full forward')
model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
optimizer.zero_grad(set_to_none=True)
logits=PaperEuropeanBoundedExecutionV1(model,bundle,metadata,resource_check=check).forward()
loss=torch.nn.functional.cross_entropy(logits,torch.arange(6)%4)
loss.backward()
parameters=list(model.parameters())
g.need(len(parameters)==49 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),'All49 complete-parent finite gradients required')
torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
g.need({int(optimizer.state[p]['step'].item()) for p in parameters}=={1},'All49 real Adam counters must equal1')
result=dict(schema='paper_treeloom1611_CPU_parity_smoke_v1',status='pass',samples=3222,queries=2972,native_holds=1,native_nodes=bundle.bank.parent.numel(),
 sampled_rows=6,model_parameters=sum(p.numel() for p in parameters),trainable_tensors=len(parameters),
 maximum_absolute_logit_error=error,loss=float(loss.detach()),all49_gradient_tensors_finite=True,all49_Adam_counters=1,
 prepared_tensor_bytes=tensor_bytes(bundle),seconds=time.monotonic()-started,cuda_allocated=False,
 source_ref=g.file_ref(__file__),executor_ref=g.file_ref(g.ROOT/'src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py'),
 limits='Synthetic single native tree only; full35Mb parent RSS and timing remain unmeasured')
(base/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
