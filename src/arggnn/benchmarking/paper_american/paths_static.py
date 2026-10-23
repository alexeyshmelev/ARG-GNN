"""Native-release C3 full-path preparation and label-blind common focal registry."""
import hashlib,json,random
from pathlib import Path
from arggnn.american_c3_common_v1 import ROOT,CLASSES,require,ref,bound,read,cli
SOURCE='src/arggnn/benchmarking/paper_american/paths_static.py'
AUTHOR=ROOT/'.runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py'
AUTHOR_SHA='e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c'
DEFAULTS=dict(schema='american_c3_native_path_preparation_v1',execution_enabled=False,observation_ref=None,
    genealogy_kind='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG',
    max_unique_paths=10000000,max_path_tokens=512000000,max_path_occurrences=500000000,max_vocabulary_nodes=2000000,
    max_raw_path_nodes=1024,query_focal_count=10000,path_sampling_seed=92001,output_cap_bytes=16000000000)


def author_functions():
    require(hashlib.sha256(AUTHOR.read_bytes()).hexdigest()==AUTHOR_SHA,
            "Pinned original author path function source required")
    from .author_path_functions import get_path_key,build_vocab_from_sequences,split_unique_sequences
    return dict(get_path_key=get_path_key,build_vocab_from_sequences=build_vocab_from_sequences,
                split_unique_sequences=split_unique_sequences)

def storage_bound(cfg):
    p=cfg['max_unique_paths'];t=cfg['max_path_tokens'];o=cfg['max_path_occurrences'];v=cfg['max_vocabulary_nodes']
    # <=32bit raw nodes: pickle integer <=5B; bounded sequence/set/dict overhead.
    return 3*5*t+256*p+5*o+7*64*v+16*16777216


def execute(cfg,w):
    import numpy as np,tskit
    m=read(cfg['observation_ref']);roster=read(m['roster_ref']);rows=roster['samples'];ts=tskit.load(str(bound(m['tree_ref'])))
    require(roster['parent_id']==m['parent_id'] and roster['class_order']==list(CLASSES)
        and [r['sample_index'] for r in rows]==list(range(len(rows))),'Stable original genotype-column/parent identity required')
    require(m['schema']=='american_c3_observed_v1' and m['class_order']==list(CLASSES)
        and m['genealogy_kind']==cfg['genealogy_kind'] and m['query_ancestry_truth_in_inputs'] is False,'Declared C3 observation/genealogy kind required')
    require(ts.num_migrations==0 and (ts.tables.nodes.population==-1).all() and ts.num_populations==0,
            'No migration/internal source labels in model genealogy')
    require(list(map(int,ts.samples()))==[r['observed_node_id'] for r in rows],'Original ordered sampled identity required')
    require(ts.num_nodes<2**31 and ts.num_samples*ts.num_trees<=cfg['max_path_occurrences']
        and storage_bound(cfg)<=cfg['output_cap_bytes'],'Full corpus occurrence/32bit/production bound exceeded')
    f=author_functions();keys={};seqs=[];mapping={};tokens=0;first=[]
    for sample in ts.samples():
        for tree in ts.trees():
            key=f['get_path_key'](tree,int(sample))
            require(len(key)<=cfg['max_raw_path_nodes'],'Complete raw path length cap exceeded; no truncated deduplication')
            if key not in keys:
                require(len(seqs)<cfg['max_unique_paths'],'Unique-path cap exceeded; no sampling/subset fallback')
                tokens+=len(key);require(tokens<=cfg['max_path_tokens'],'Complete path-token cap exceeded')
                keys[key]=len(seqs);seqs.append(list(key));mapping[key]=set();first.append(tree.index)
            mapping[key].add(tree.index)
    vocabulary=f['build_vocab_from_sequences'](seqs)
    v=len(vocabulary['node_id_tokens']);require(v<=cfg['max_vocabulary_nodes'],'Observed vocabulary cap exceeded')
    # Exact release: shuffle ALL unique raw full paths with seed42 and90/10.
    train,val=f['split_unique_sequences'](seqs.copy(),train_frac=.9)
    sample_meta={r['observed_node_id']:r for r in rows};trees=list(ts.breakpoints())
    query_ids=[keys[tuple(s)] for s in seqs if sample_meta[s[0]]['role']=='query']
    require(len(query_ids)>=cfg['query_focal_count'],'Not enough complete unique query paths')
    chosen=np.random.default_rng(cfg['path_sampling_seed']).choice(np.asarray(query_ids,dtype=np.int64),
        size=cfg['query_focal_count'],replace=False);chosen.sort()
    samples=np.asarray([seqs[int(i)][0] for i in chosen],dtype=np.int64)
    left=np.asarray([trees[first[int(i)]] for i in chosen],dtype=np.float64)
    right=np.asarray([trees[first[int(i)]+1] for i in chosen],dtype=np.float64)
    positions=(left+right)/2
    chosen_set=set(map(int,chosen))
    focal_ref=w.npz('focals.npz',record_ids=chosen,sample_nodes=samples,
        sample_indices=np.asarray([sample_meta[int(u)]['sample_index'] for u in samples],dtype=np.int64),
        parent_id=np.asarray(m['parent_id']),class_order=np.asarray(CLASSES),positions_bp0=positions,
        first_occurrence_intervals_bp=np.column_stack((left,right)))
    view_refs={};dataset_refs={}
    pickle_margin=16777216
    for split,all_seqs in [('train',train),('val',val)]:
        views={'mlm':all_seqs,'contrastive':[s for s in all_seqs if sample_meta[s[0]]['role']=='pure_source_reference'],
               'query_paths':[s for s in all_seqs if keys[tuple(s)] in chosen_set]}
        for view,selected in views.items():
            require(selected,'Every registered native view/split must be nonempty')
            ids=[keys[tuple(s)] for s in selected]
            planned=5*sum(map(len,selected))+32*len(selected)+64*v+pickle_margin
            dataset_refs[f'{view}/{split}']=w.pickle(f'{view}/{split}/raw_{split}_sequences_and_vocab.pkl',
                dict(raw_sequences=selected,vocab=vocabulary),planned)
            meta=dict(record_ids=np.asarray(ids,dtype=np.int64),sample_nodes=np.asarray([s[0] for s in selected],dtype=np.int64),
                sample_indices=np.asarray([sample_meta[s[0]]['sample_index'] for s in selected],dtype=np.int64),
                parent_id=np.asarray(m['parent_id']),class_order=np.asarray(CLASSES))
            if view=='contrastive':meta['reference_classes_0based']=np.asarray([sample_meta[s[0]]['observed_reference_class'] for s in selected],dtype=np.int64)
            view_refs[f'{view}/{split}']=w.npz(f'{view}/{split}/row_order_metadata.npz',**meta)
            if view=='contrastive':
                labels=[sample_meta[s[0]]['observed_reference_class'] for s in selected]
                require(set(labels)<=set(range(3)),'Only known reference classes enter native contrastive labels')
                w.pickle(f'{view}/{split}/labels.pkl',labels,5*len(labels)+pickle_margin)
    vocab_ref=w.pickle('vocab.pkl',vocabulary,64*v+pickle_margin)
    map_ref=w.pickle('sequence_to_tree_indices.pkl',mapping,5*tokens+64*len(seqs)+5*ts.num_samples*ts.num_trees+pickle_margin)
    return w.json('completed.json',dict(status='completed_native_C3_path_preparation',source_author_function_ref=ref(AUTHOR),
        observation_ref=cfg['observation_ref'],parent_id=m['parent_id'],genealogy_kind=m['genealogy_kind'],class_order=list(CLASSES),
        native_dataset_refs=dataset_refs,full_path_id_order_refs=view_refs,vocab_ref=vocab_ref,sequence_to_tree_indices_ref=map_ref,
        focals_ref=focal_ref,focal_coordinate_rule='label-blind first-occurrence midpoint: reconstruction assumption, not recovered Table1 rule',
        unique_paths=len(seqs),path_tokens=tokens,path_occurrences=ts.num_samples*ts.num_trees,vocabulary_nodes=v,
        native_split_seed=42,native_train_fraction=.9,query_focal_count=len(chosen),
        contrastive_labels_reference_only=True,query_embedding_view_prespecified_focals_only=True,query_ancestry_truth_opened=False,author_Table1_exact_reproduction=False))


if __name__=='__main__':cli(SOURCE,DEFAULTS,execute,dependencies=(AUTHOR.relative_to(ROOT).as_posix(),))
