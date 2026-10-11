"""Handmade pure metadata fixtures, no dataset/array readers."""
import copy
import json
from pathlib import Path
from arggnn import anchor_parent_producer_v1 as p

HERE = Path(__file__).resolve().parent
cfg = json.loads((p.ROOT/'configs/biological_candidate_pilot_v1.json').read_text())
pid = p.PARENTS[0]; root = 'research/reviews/anchor_parent_producer_v1_author/'
def ref(name, **kw):return dict(path=root+name, sha256='1'*64, **kw)
binding = {'kind':'anchor_parent_producer_binding_v1','parent_id':pid,'base_config':{'path':'configs/biological_candidate_pilot_v1.json','sha256':p._sha(p.ROOT/'configs/biological_candidate_pilot_v1.json')},'observed_manifest':ref('handmade_observed.json'),'prepared_manifest':ref('handmade_prepared.json'),'native_tree':ref('handmade.trees',condition='post_five_size_iterations_primary'),'native_tree_count':8,'observed_only_authorized':False,'cpu_headroom_bytes':128000000000}
rows=[]
for i in range(280):
 role='query' if i<120 else 'mixed_reference_only' if i<184 else 'pure_source_reference'
 rows.append({'sample_index':i,'sample_id':f'fixture_{i//2:03d}','haplotype':i%2,'observed_node_id':i,'observed_cohort':'present_day' if i<64 else 'ancient' if i<120 else 'mixed' if i<184 else p.PARENTS[0], 'role':role,'age_generations':0.,'observed_reference_class':(i-184)//24 if i>=184 else None})
nodes=[i+1000 for i in range(280)];derived=[]
for row,node in zip(rows,nodes):
 d=dict(row);d['source_observed_node_id']=d.pop('observed_node_id');d.update(inferred_tree_node_id=node,observed_node_id=node);derived.append(d)
source_ref={'path':f'data/biological_paths_execution_pilot_v1/{pid}/observed/sample_manifest.json','sha256':cfg['observed_roster_sha256'][pid]}
derived_ref=ref('handmade_derived.json',rows=280,source_manifest=source_ref)
coordinate={'original_parent_interval_bp0':[0,2000000],'normalization_length_bp':2000000,'physical_coordinate_transform':'physical_original = local_tree_position + coverage_left','uncovered_tails':'excluded; no extrapolation','coverage_original_bp0':[0,2000000],'local_to_original_offset_bp':0,'retained_fixed100bp_indexes':list(range(20000)),'fixed_grid':{'path':f'data/biological_paths_execution_pilot_v1/{pid}/observed/scoring_coordinates.npz','sha256':'1'*64,'total':20000}}
mapping=[{'sample_index':i,'inferred_tree_node_id':node} for i,node in enumerate(nodes)]
entry={'parent_id':pid,'split':'train','status':'inferred_both_conditions','truth_or_oracle_read':False,'inferred_tree':binding['native_tree'],'observed_sample_manifest':derived_ref,'original_observed_sample_manifest':source_ref,'sample_index_to_inferred_tree_node_id':mapping,'completion_provenance':ref('handmade_completion.json'),'commands_provenance':ref('handmade_commands.json'),'coordinate_contract':coordinate,'post_five_iterations':{'trees':8,'clean_tree':binding['native_tree']['path'],'clean_tree_sha256':'1'*64,'sample_index_to_inferred_tree_node_id':mapping,'observed_sample_manifest':derived_ref,'coverage_original_bp0':[0,2000000],'local_to_original_offset_bp':0,'sequence_length_local_bp':2000000,'retained_fixed100bp_indexes':list(range(20000))}}
record={'parent_id':pid,'split':'train','relate_entry':entry,'original_trees_preserved':8,'coverage_original_bp0':[0,2000000],'retained_fixed100bp_indexes':list(range(20000)),'positions_bp':list(range(50,2000000,100)),'bin_edges_bp':list(range(0,2000001,100)),'dynamics_sample_indexes':list(range(184)),'actual_query_sample_indexes':list(range(120)),'reference_sample_indexes':list(range(184,280)),'actual_query_output_indexes':list(range(120)),'query_node_ids':nodes[:120],'dynamics_query_node_ids':nodes[:184],'tiles':[]}
for i in range(8):
 indexes=list(range(i*2500,(i+1)*2500));record['tiles'].append({'tile_id':i,'physical_interval_bp0':[i*250000,(i+1)*250000],'normalized_interval':[i/8,(i+1)/8],'position_indexes':indexes,'original_grid_indexes':indexes,'original_tree_ids':[i],'all_tree_nodes_retained':True,'all280_leaves_retained':True,'all_original_age_cache_fields_retained':True,'exact_serialization_roundtrip_checked':True,'query_dynamics':184,'pure_reference_sources':96,'supervised_outputs':120,'mixed_latent_outputs':64,'mixed_query_flag':0,'bank':ref(f'handmade_{i}.bank.pt'),'whole_observed_graph':ref(f'handmade_{i}.graph.pt')})
observed={'kind':'biological_paths_relate_v1','status':'four_development_parents_inferred_pre_and_post_five_iterations','primary_condition':'post_five_size_iterations_primary','truth_or_oracle_read':False,'implementation_sha256':cfg['relate_producer']['implementation_sha256'],'config_sha256':cfg['relate_producer']['config_sha256'],'replicates':[dict(entry,parent_id=name,split='train' if '_train_' in name else 'val') for name in p.PARENTS]}
prepared={'kind':'biological_candidate_prepared_v1','status':'complete_observed_only_Q184_R96_actualQ120','truth_opened':False,'labels_in_features_or_banks':False,'config_sha256':binding['base_config']['sha256'],'observed_manifest_sha256':'1'*64,'observed_manifest_path':binding['observed_manifest']['path'],'source_sha256':cfg['source_sha256'],'geometry':cfg['geometry'],'parents':[dict(record,parent_id=name,split='train' if '_train_' in name else 'val') for name in p.PARENTS]}
completion={k:v for k,v in entry.items() if k!='completion_provenance'}
checks=[]
def call(b=binding,pr=prepared,o=observed,c=cfg,s=rows,d=derived,co=completion):return p.validate_metadata_records_v1(b,pr,o,c,s,d,co)
def ok(name,value):
 checks.append({'check':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
def reject(name,fn):
 try:fn()
 except ValueError:ok(name,True)
 else:ok(name,False)
rec,ent,meta=call();ok('complete handmade metadata joins accepted',len(rec['tiles'])==8 and len(meta['nodes'])==280 and len(meta['positions'])==20000)
for name,mut in [
 ('truth preparation rejected',lambda x:x.update(truth_opened=True)),
 ('input label flag rejected',lambda x:x.update(labels_in_features_or_banks=True)),
 ('source producer config drift rejected',lambda x:x.update(config_sha256='2'*64)),
 ('prepared original tree ID omitted rejected',lambda x:x['parents'][0]['tiles'][3].update(original_tree_ids=[])),
 ('mixed latent flag mutation rejected',lambda x:x['parents'][0]['tiles'][0].update(mixed_query_flag=1)),
 ('incomplete eight-tile parent rejected',lambda x:x['parents'][0]['tiles'].pop()),
 ('tile support mutation rejected',lambda x:x['parents'][0]['tiles'][0].update(physical_interval_bp0=[0,249999])),
 ('readout center omission rejected',lambda x:x['parents'][0]['tiles'][0]['position_indexes'].pop()),
 ('280 role join permutation rejected',lambda x:x['parents'][0]['dynamics_sample_indexes'].reverse()),
 ('original47 source closure drift rejected',lambda x:x['source_sha256'].update({'src/arggnn/biological_candidate.py':'2'*64})),
 ]:
 bad=copy.deepcopy(prepared);mut(bad);reject(name,lambda bad=bad:call(pr=bad))
bad=copy.deepcopy(completion);bad['status']='other';reject('completion provenance drift rejected',lambda:call(co=bad))
bad=copy.deepcopy(derived);bad[0]['observed_node_id']+=1;reject('derived sampled ID change rejected',lambda:call(d=bad))
bad=copy.deepcopy(rows);bad[120]['observed_reference_class']=0;reject('class in mixed input rejected',lambda:call(s=bad))
bad=copy.deepcopy(binding);bad['native_tree_count']=7;reject('native inventory metadata count drift rejected',lambda:call(b=bad))
bad=copy.deepcopy(binding);bad['parent_id']='bio_paths_pilot_test_000';reject('TEST parent rejected by metadata',lambda:call(b=bad))
result={'kind':'anchor_parent_producer_v1_handmade_metadata_checks','status':'passed','check_count':len(checks),'checks':checks,'actual_dataset_opened':False,'array_files_opened':False,'truth_opened':False,'GPU_used':False,'model_fitting':False}
(HERE/'metadata_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','checks':len(checks)},sort_keys=True))
