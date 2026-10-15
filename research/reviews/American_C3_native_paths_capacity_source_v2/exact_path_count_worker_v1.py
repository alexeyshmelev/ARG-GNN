"""Exact full C3 native path counts only. Root-admitted CPU execution only."""
import json
from arggnn.american_c3_common_v1 import ROOT,require,read,bound,ref
from arggnn import american_c3_paths_v1 as paths
SOURCE='research/reviews/American_C3_native_paths_capacity_source_v2/exact_path_count_worker_v1.py'
DEFAULTS=dict(schema='american_c3_exact_native_path_count_v1',execution_enabled=False,
    observation_ref=None,root_counts_ref=None,max_path_occurrences=122980800,
    max_unique_paths_per_sample=76863,max_vocabulary_nodes=106769,max_raw_path_nodes=106769,
    count_json_cap_bytes=1048576,completed_json_cap_bytes=65536,output_cap_bytes=100000000)
COMPLETED_STATUS='completed_native_C3_exact_path_count'

def execute(cfg,w):
    import tskit
    counts=read(cfg['root_counts_ref'])
    require(counts['root_authorizer']=='root_agent' and counts['truth_opened'] is False
        and counts['source_observation_ref']==cfg['observation_ref'],
        'Exact approved observation count/tree binding required')
    ts=tskit.load(str(bound(counts['tree_ref'])))
    require((ts.num_samples,ts.num_nodes,ts.num_edges,ts.num_trees,ts.num_sites,ts.sequence_length)==
        (counts['num_samples'],counts['num_nodes'],counts['num_edges'],counts['num_trees'],
         counts['num_sites'],counts['sequence_length']), 'Approved full tree metadata changed')
    occurrences=ts.num_samples*ts.num_trees
    require(occurrences==counts['full_path_occurrence_upper_count']==cfg['max_path_occurrences']
        and ts.num_trees==cfg['max_unique_paths_per_sample']
        and ts.num_nodes==cfg['max_vocabulary_nodes']==cfg['max_raw_path_nodes'],
        'Exact full metadata bounds; no scientific subset fallback')
    f=paths.author_functions()  # SHA-frozen original definitions, not reimplementations.
    total_unique=total_tokens=total_occurrences=max_raw=0
    vocabulary_nodes=set();per_sample=[]
    for sample_index,sample in enumerate(ts.samples()):
        sample=int(sample);seen=set();sample_tokens=sample_occurrences=sample_max=0
        for tree in ts.trees():
            key=f['get_path_key'](tree,sample)
            require(type(key) is tuple and key and key[0]==sample,
                'Original full tuple must begin with its sample: exact cross-sample disjointness proof')
            length=len(key)
            require(length<=cfg['max_raw_path_nodes'], 'Acyclic raw path exceeds observed node bound')
            sample_max=max(sample_max,length);sample_occurrences+=1
            if key not in seen:
                require(len(seen)<cfg['max_unique_paths_per_sample'], 'More unique keys than marginal trees')
                seen.add(key);sample_tokens+=length;vocabulary_nodes.update(key)
        require(sample_occurrences==ts.num_trees, 'Every marginal tree must be visited for every sample')
        count=len(seen)
        total_unique+=count;total_tokens+=sample_tokens;total_occurrences+=sample_occurrences
        max_raw=max(max_raw,sample_max)
        per_sample.append(dict(sample_index=sample_index,sample_node=sample,unique_paths=count,
            path_tokens=sample_tokens,path_occurrences=sample_occurrences,max_raw_path_nodes=sample_max))
        # Exact keys are discarded only after counting this sample. Keys from distinct samples
        # cannot collide because their asserted first tuple entries are distinct.
        del seen
        if (sample_index+1)%100==0:
            print(json.dumps(dict(stage='C3_exact_path_count_progress',samples_completed=sample_index+1,
                unique_paths=total_unique,path_tokens=total_tokens,path_occurrences=total_occurrences)),flush=True)
    require(len(per_sample)==ts.num_samples and total_occurrences==occurrences,
        'Complete original all-sample/all-tree scan required')
    # The original vocabulary function depends only on the exact union of raw node tokens.
    # Reusing it on that union gives the exact native node-token count without a corpus/view.
    vocabulary=f['build_vocab_from_sequences']([vocabulary_nodes])
    vocab_count=len(vocabulary['node_id_tokens'])
    require(vocab_count==len(vocabulary_nodes)<=cfg['max_vocabulary_nodes'], 'Exact native vocabulary count')
    cap_fields=dict(max_unique_paths=total_unique,max_path_tokens=total_tokens,
        max_path_occurrences=occurrences,max_vocabulary_nodes=vocab_count)
    native_bound=paths.storage_bound(cap_fields)  # Unchanged original serialization formula.
    detail=dict(schema='american_c3_exact_native_path_per_sample_counts_v1',
        root_counts_ref=cfg['root_counts_ref'],parent_id='american_c3_train_000',
        sample_count=len(per_sample),per_sample_counts=per_sample,
        full_corpus_scanned=True,query_ancestry_truth_opened=False,corpus_written=False)
    detail_bytes=(json.dumps(detail,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(detail_bytes)<=cfg['count_json_cap_bytes'], 'Bounded per-sample count metadata only')
    detail_ref=w.json('per_sample_counts.json',detail)
    value=dict(status=COMPLETED_STATUS,schema='american_c3_exact_native_path_count_result_v1',
        source_ref=ref(SOURCE),native_paths_source_ref=ref(paths.SOURCE),
        source_author_function_ref=ref(paths.AUTHOR),root_counts_ref=cfg['root_counts_ref'],
        observation_ref=cfg['observation_ref'],tree_ref=counts['tree_ref'],
        parent_id='american_c3_train_000',split='train',full_corpus_scanned=True,
        unique_paths=total_unique,path_tokens=total_tokens,path_occurrences=total_occurrences,
        vocabulary_nodes=vocab_count,max_raw_path_nodes=max_raw,per_sample_counts_ref=detail_ref,
        native_storage_cap_fields=cap_fields,native_storage_bound_bytes=native_bound,
        original_raw_path_cap_1024_sufficient=max_raw<=1024,
        original_unique_path_cap_10000000_sufficient=total_unique<=10000000,
        original_token_cap_512000000_sufficient=total_tokens<=512000000,
        exact_get_path_key_tuples=True,per_sample_dedup_equals_original_global_dedup=True,
        native_split_not_called_count_only=True,original_split_seed42_preserved_for_future_full_prep=True,
        query_ancestry_truth_opened=False,labels_or_roster_opened=False,corpus_written=False,
        path_keys_or_vocab_written=False,scientific_subsampling=False,
        metadata_only_output=True,author_Table1_exact_reproduction=False)
    encoded=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(encoded)<=cfg['completed_json_cap_bytes'], 'Compact completed metadata must fit original 64KiB root reader')
    return w.json('completed.json',value)
