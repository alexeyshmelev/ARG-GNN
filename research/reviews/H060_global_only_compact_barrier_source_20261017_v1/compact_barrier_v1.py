"""Inactive durable commitment only; full admitted in-memory barrier remains authoritative to original gates."""
import copy,hashlib,json
SOURCE='research/reviews/H060_global_only_compact_barrier_source_20261017_v1/compact_barrier_v1.py'
SCHEMA='H060_all20_global_observation_durable_commitment_v1'
MAX_COMPACT_JSON_BYTES=262144
LIST_PATHS=(('observation_metadata','isolated_zero_site_holds'),
 ('episode_metadata','proper_clades_per_tree'),('episode_metadata','native_nodes_per_tree'),
 ('episode_metadata','unary_equivalences_per_tree'),('cut_metadata','native_hold_cut_group_counts'),
 ('cut_metadata','episode_starts'),('cut_metadata','conditional_information_nats_per_leaf'),('isolated_zero_site_holds',))
def need(ok,message):
 if not ok:raise ValueError(message)
def canonical_commitment_v1(value):
 """Exact existing sorted compact JSON encoding, streamed only to a digest/count."""
 digest=hashlib.sha256();count=0
 encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False)
 for chunk in encoder.iterencode(value):
  data=chunk.encode();digest.update(data);count+=len(data)
 return dict(sha256=digest.hexdigest(),canonical_JSON_bytes=count)
def make_durable_barrier_v1(barrier,*,full_barrier_sha256,validate_full_barrier):
 """Caller passes the unchanged original validator, before every original target gate."""
 need(callable(validate_full_barrier)and validate_full_barrier(barrier)is barrier,'Original complete in-memory barrier validation required')
 full=canonical_commitment_v1(barrier)
 need(full['sha256']==full_barrier_sha256,'Exact original full observation commitment unchanged')
 parents=barrier['parent_ids'];need(len(parents)==20 and len(set(parents))==20,'All20 original independent TRAIN parents')
 common=barrier['parents'][parents[0]]['source_refs'];sources=canonical_commitment_v1(common);rows={}
 for parent in parents:
  seal=barrier['parents'][parent];metadata=seal['preparation_metadata']
  need(seal['parent_id']==parent and seal['source_refs']==common and seal['global_identity']==barrier['validated_cache_identities'][parent],'Every original parent/source/current global identity retained')
  lists={}
  for path in LIST_PATHS:
   item=metadata
   for key in path:
    if type(item)is not dict or key not in item:break
    item=item[key]
   else:
    need(type(item)is list,'Original structural metadata list');lists['/'.join(path)]=dict(length=len(item),**canonical_commitment_v1(item))
  rows[parent]=dict(parent_id=parent,split=seal['split'],plan_ref=copy.deepcopy(seal['plan_ref']),
   observation_ref=copy.deepcopy(seal['observation_ref']),focal_registry_ref=copy.deepcopy(seal['focal_registry_ref']),
   native_preparation_ref=copy.deepcopy(seal['native_preparation_ref']),
   sequence_length_bp=seal['sequence_length_bp'],haplotypes=seal['haplotypes'],reference_haplotypes=seal['reference_haplotypes'],
   query_haplotypes=seal['query_haplotypes'],class_order=copy.deepcopy(seal['class_order']),trees=seal['trees'],sites=seal['sites'],
   focal_records=seal['focal_records'],focal_identity_sha256=seal['focal_identity_sha256'],
   global_identity=copy.deepcopy(seal['global_identity']),plan_support=copy.deepcopy(seal['plan_support']),
   original_full_seal=canonical_commitment_v1(seal),preparation_metadata=canonical_commitment_v1(metadata),
   structural_list_commitments=lists,source_refs_sha256=sources['sha256'],source_ref_count=len(common),
   all_native_trees_nodes_sites_samples_retained=seal['all_native_trees_nodes_sites_samples_retained'])
 result=dict(schema=SCHEMA,original_in_memory_barrier_schema=barrier['schema'],full_original_barrier=full,
  registry_ref=copy.deepcopy(barrier['registry_ref']),parent_ids=list(parents),parents=rows,
  source_refs=copy.deepcopy(common),source_refs_commitment=sources,
  all20_ready=barrier['all20_ready'],observation_only=barrier['observation_only'],
  all20_global_cache_before_TRAIN_truth=barrier['all20_global_cache_before_TRAIN_truth'],
  truth_resolver_calls_before_barrier=barrier['truth_resolver_calls_before_barrier'],
  legacy_block_seal_claimed=barrier['legacy_block_seal_claimed'],protocol=barrier['protocol'],
  representation='canonical_full_barrier_and_per_parent_commitments_not_a_replacement_for_in_memory_target_gates',
  full_structural_arrays_retained_in_original_owned_memory=True,
  numerical_preparation_target_equations_native_support_and_update_schedule_unchanged=True)
 data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 need(len(data)<=MAX_COMPACT_JSON_BYTES,'Compact commitment byte bound within unchanged aggregate writer cap')
 return result
if __name__=='__main__':
 print(json.dumps(dict(status='INACTIVE_SOURCE_ONLY',actual_operations=0,actual_adoption=False)))
