"""Bounded handmade native source -> original cached bank -> producer smoke.

The metadata gate is separately exercised by pure handmade metadata fixtures.
Here its return value is injected, with only author-owned handmade artifacts.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import tskit
from arggnn import biological_candidate as frozen
from arggnn import anchor_parent_producer_v1 as p

HERE=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
frozen.configure_cpu()
# The original CPU policy has been applied before fixture construction. Avoid
# repeating Torch's once-only interop setting inside the source API under test.
frozen.configure_cpu=lambda:None
checks=[]
def check(name,value):
 checks.append({'check':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
fixture=HERE/'handmade_native_fixture'
fixture.mkdir(exist_ok=True)
used=0
cap=256000000
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
def ref(path,**kw):return dict(path=str(path.relative_to(p.ROOT)),sha256=sha(path),**kw)
def account(path):
 global used
 used+=path.stat().st_size
 check('bounded handmade artifact '+path.name,used<=cap)
tables=tskit.TableCollection(sequence_length=2000000.)
for i in range(280):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
root=tables.nodes.add_row(time=100.)
for child in range(280):tables.edges.add_row(0.,2000000.,parent=root,child=child)
tables.sort();ts=tables.tree_sequence();native=fixture/'handmade_native.trees';ts.dump(native);account(native)
rows=[]
classes=('Ana','CHG','WHG','EHG')
for i in range(280):
 role='query' if i<120 else 'mixed_reference_only' if i<184 else 'pure_source_reference'
 c=(i-184)//24 if i>=184 else None
 rows.append(dict(sample_index=i,sample_id=f'fixture_{i//2}',haplotype=i%2,role=role,observed_cohort='present_day' if i<64 else 'ancient' if i<120 else 'mixed' if i<184 else classes[c],observed_reference_class=c,age_generations=0.,observed_node_id=i))
positions=np.arange(50.,2000000.,100.);edges=np.arange(0.,2000001.,100.)
grid=fixture/'handmade_grid.npz';np.savez(grid,positions_bp=positions,bin_edges_bp=edges);account(grid)
commands=fixture/'handmade_commands.json';commands.write_text('{}\n');account(commands)
parent={'parent_id':'bio_paths_pilot_train_000','split':'train','samples':rows,'query_slots':np.arange(120),'dynamics_slots':np.arange(184),'reference_slots':np.arange(184,280),'positions':positions,'edges':edges,'observation':{'nodes':np.arange(280),'offset':0.}}
record={'parent_id':parent['parent_id'],'split':'train','original_trees_preserved':1,'tiles':[]}
for i in range(8):
 left,right=i*250000.,(i+1)*250000.
 sequence,indexes,tree_ids,landmarks=frozen.pack_observed_tile(ts,parent,left,right)
 age_bank=frozen.precompute_age_interfaces(sequence,time_column=6,parent_branch_column=7,time_scale=10000.,resistance_scale=.0001,age_scales=(350.,1400.),strength=3.,reference_ids=torch.arange(184,280),max_condition=1e12)
 prepared=frozen.prepare_anchor_kernel_bank(age_bank.plain_bank,age_bank.operator_cache,num_anchors=32,num_scales=3)
 bundle={'schema':'biological_candidate_full_age_bank_v1','age_bank':age_bank.as_dict(),'fixed_prepared':prepared}
 graph=fixture/f'handmade_tile_{i:02d}.graph.pt';bank=fixture/f'handmade_tile_{i:02d}.bank.pt'
 torch.save({'sequence':sequence.as_dict(),'labels_in_input':False},graph);torch.save(bundle,bank);account(graph);account(bank)
 record['tiles'].append({'tile_id':i,'physical_interval_bp0':[left,right],'position_indexes':indexes.tolist(),'original_tree_ids':tree_ids,'landmarks_observed_node_ids':landmarks,'whole_observed_graph':ref(graph),'bank':ref(bank),'full_bank_field_fingerprint_sha256':frozen.fingerprint_sha(bundle),'all_original_age_cache_fields_retained':True,'exact_serialization_roundtrip_checked':True})
 del bundle,prepared,age_bank,sequence
binding={'parent_id':parent['parent_id'],'native_tree':ref(native),'native_tree_count':1,'observed_only_authorized':True,'cpu_headroom_bytes':8000000000,'base_config':ref(p.ROOT/'configs/biological_candidate_pilot_v1.json'),'prepared_manifest':ref(commands),'observed_manifest':ref(commands)}
entry={'coordinate_contract':{'fixed_grid':ref(grid)},'commands_provenance':ref(commands),'original_observed_sample_manifest':ref(commands),'observed_sample_manifest':ref(commands),'completion_provenance':ref(commands)}
metadata={'nodes':list(range(280)),'query_slots':list(range(120)),'dynamics_slots':list(range(184)),'reference_slots':list(range(184,280)),'positions':positions.tolist(),'edges':edges.tolist(),'coverage':[0.,2000000.],'retained':list(range(20000)),'source_rows':rows}
cfg=json.loads((p.ROOT/'configs/biological_candidate_pilot_v1.json').read_text())
p.read_metadata_v1=lambda _: (binding,cfg,record,entry,metadata)
assemblies,report=p.qualify_observed_v1(ref(commands))
check('original observed qualification path completes on handmade source',report['status']=='observed_inputs_bound_no_numerical_profile_or_fit')
check('full original eight source reconstructions',len(report['tile_source_checks'])==8 and all(row['serialized_graph_reproduced'] and row['plain_bank_inputs_reproduced'] for row in report['tile_source_checks']))
check('original repeated native tree retained all eight occurrences',report['original_tree_count']==1 and report['cache_graph_occurrences']==8 and report['unique_original_tree_ids']==[0] and report['inverse_cache_graph_indexes']==[0]*8)
check('all original20000 fixed readouts preserved',len(assemblies['age'].parent.layout.target_output)==20000)
check('genuine handmade source Q184/R96 and all280 IDs',len(report['query_ids'])==184 and len(report['reference_ids'])==96 and report['all_leaf_ids']==list(range(280)))
check('age/plain exact256 union',all(len(a.parent.layout.anchors)==256 for a in assemblies.values()) and len(report['exact_weight_union'])==256)
check('qualification instantiates no model/fit/GPU/targets',report['models_instantiated'] is False and report['fit_performed'] is False and report['GPU_used'] is False and report['target_arrays_opened'] is False)
check('qualification explicit memory lower bound',report['memory_is_measured_peak'] is False and report['input_lower_bound_bytes']<=binding['cpu_headroom_bytes'])
(HERE/'observed_path_checks.json').write_text(json.dumps({'kind':'anchor_parent_producer_v1_handmade_observed_path_checks','status':'passed','check_count':len(checks),'checks':checks,'handmade_artifact_bytes':used,'handmade_artifact_cap_bytes':cap,'metadata_gate_injected_handmade_only':True,'handmade_native_source':True,'actual_dataset_opened':False,'truth_opened':False,'GPU_used':False,'models_instantiated':False,'fit_performed':False,'qualification_report':report},indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'handmade_artifact_bytes':used,'handmade_native_tree_count':1,'readouts':20000},sort_keys=True))
