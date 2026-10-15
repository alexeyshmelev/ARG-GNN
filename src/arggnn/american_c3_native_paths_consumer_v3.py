"""Complete original C3 corpus with narrower prospective serialization budgets.

The original path extraction, author functions, full deduplication/order,
seed42 split, views, focal identities, native object schemas and filenames
are preserved. Only budget plumbing, evidence assertions and budget metadata
change. Default mode imports neither science nor a guard.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
SOURCE='src/arggnn/american_c3_native_paths_consumer_v3.py'
AUTHOR=ROOT/'.runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py'
ORIGINAL_PATHS_REF=dict(path='src/arggnn/american_c3_paths_v1.py',sha256='3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c')
PER_SAMPLE_REF=dict(path='research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/per_sample_counts.json',sha256='74505550902e3bc0c990f956868bfe8149bfa68173b05e1217cab2b9d099254f')
EXACT_COUNT_REF=dict(path='research/reviews/American_C3_native_paths_capacity_source_v2/actual_counts_v1/completed.json',sha256='d743c880b5c7773221aaa87da23a26307b0b7fbed6ce4b03217be3c256d2c86a')
DEFAULTS=dict(schema='american_c3_native_path_preparation_v1',execution_enabled=False,observation_ref=None,
    genealogy_kind='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG',
    max_unique_paths=10000000,max_path_tokens=512000000,max_path_occurrences=500000000,max_vocabulary_nodes=2000000,
    max_raw_path_nodes=1024,query_focal_count=10000,path_sampling_seed=92001,output_cap_bytes=6000000000)
MARGIN=16777216


def require(ok,message):
    if not ok:raise ValueError(message)


def author_functions():
    # SHA verification and all three original author definitions are unchanged.
    from . import american_c3_paths_v1 as original
    return original.author_functions()


def storage_bound(cfg):
    expected=dict(max_unique_paths=25574587,max_path_tokens=316672003,
        max_path_occurrences=122980800,max_vocabulary_nodes=106769,
        max_raw_path_nodes=1024,query_focal_count=10000,path_sampling_seed=92001,
        reference_unique_paths=9583977,reference_path_tokens=116080787,exact_max_raw_path_nodes=29,
        exact_native_path_count_ref=EXACT_COUNT_REF,exact_per_sample_counts_ref=PER_SAMPLE_REF)
    require(all(cfg.get(k)==v for k,v in expected.items()),'Exact original full-count/reference/focal controls required for narrow bound')
    p=cfg['max_unique_paths'];t=cfg['max_path_tokens'];v=cfg['max_vocabulary_nodes']
    pr=cfg['reference_unique_paths'];tr=cfg['reference_path_tokens'];f=cfg['query_focal_count'];n=p+pr+f
    return 5*(2*t+tr+29*f)+8*n+8*p+404779200+448*v+24*n+8*pr+5*pr+16*MARGIN


def native_dataset_bound_v3(n,t,v):
    require(all(type(x) is int and x>=0 for x in (n,t,v)) and 0<n<=25574587
        and n<=t<=29*n and v==106769,'Exact native builtin full-path row/token/vocabulary dimensions required')
    return 5*t+8*n+64*v+MARGIN


def mapping_bound_v3(tokens,paths,samples,trees):
    require((tokens,paths,samples,trees)==(316672003,25574587,1600,76863),'Exact complete original mapping dimensions required')
    return 5*tokens+8*paths+404779200+MARGIN


def npz_save_v3(w,name,**arrays):
    # Same direct standard NPZ serialization and arrays as common.Writer.npz.
    # 1% plus1MiB dominates zlib expansion, short NPY headers and ZIP64 records
    # for these <=8 ordinary <=2D numerical/Unicode members; no scratch output.
    import numpy as np
    require(1<=len(arrays)<=8 and all(not a.dtype.hasobject and a.dtype.kind in ('i','f','U')
        and a.dtype.itemsize<=128 and a.ndim<=2 and all(type(d) is int and 0<=d<2**31 for d in a.shape)
        for a in arrays.values()),'Bounded original numerical/Unicode NPZ member schemas required')
    raw=sum(a.nbytes for a in arrays.values());planned=raw+(raw+99)//100+1048576
    p=w.prepare(name,planned)
    with p.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return w.finish(p,planned)


def execute(cfg,w):
    import numpy as np,tskit
    from .american_c3_common_v1 import require,ref,bound,read,CLASSES
    m=read(cfg['observation_ref']);roster=read(m['roster_ref']);rows=roster['samples'];ts=tskit.load(str(bound(m['tree_ref'])))
    require(roster['parent_id']==m['parent_id'] and roster['class_order']==list(CLASSES)
        and [r['sample_index'] for r in rows]==list(range(len(rows))),'Stable original genotype-column/parent identity required')
    require(m['schema']=='american_c3_observed_v1' and m['class_order']==list(CLASSES)
        and m['genealogy_kind']==cfg['genealogy_kind'] and m['query_ancestry_truth_in_inputs'] is False,'Declared C3 observation/genealogy kind required')
    require(len(rows)==1600 and [r['observed_node_id'] for r in rows]==list(range(1600))
        and [r['role'] for r in rows]==['pure_source_reference']*600+['query']*1000,
        'Exact original first600 reference / last1000 query role partition required; no fallback')
    require((ts.num_samples,ts.num_trees,ts.num_nodes)==(1600,76863,106769),
        'Exact full original tree dimensions for occurrence serialization required')
    require(ts.num_migrations==0 and (ts.tables.nodes.population==-1).all() and ts.num_populations==0,
            'No migration/internal source labels in model genealogy')
    require(list(map(int,ts.samples()))==[r['observed_node_id'] for r in rows],'Original ordered sampled identity required')
    require(ts.num_nodes<2**31 and ts.num_samples*ts.num_trees<=cfg['max_path_occurrences']
        and storage_bound(cfg)<=cfg['output_cap_bytes'],'Full corpus occurrence/32bit/production bound exceeded')
    f=author_functions();keys={};seqs=[];mapping={};tokens=0;first=[]
    reference_paths=reference_tokens=max_raw=0
    for sample in ts.samples():
        for tree in ts.trees():
            key=f['get_path_key'](tree,int(sample))
            require(len(key)<=cfg['max_raw_path_nodes'],'Complete raw path length cap exceeded; no truncated deduplication')
            max_raw=max(max_raw,len(key))
            require(len(key)<=29,'Exact full-count maximum29 required for serialization proof')
            if key not in keys:
                require(len(seqs)<cfg['max_unique_paths'],'Unique-path cap exceeded; no sampling/subset fallback')
                tokens+=len(key);require(tokens<=cfg['max_path_tokens'],'Complete path-token cap exceeded')
                keys[key]=len(seqs);seqs.append(list(key));mapping[key]=set();first.append(tree.index)
                if key[0]<600:
                    reference_paths+=1;reference_tokens+=len(key)
            mapping[key].add(tree.index)
    require((len(seqs),tokens,reference_paths,reference_tokens,max_raw)==
        (25574587,316672003,9583977,116080787,29),
        'Exact complete corpus/reference counts required before durable products; no subset fallback')
    vocabulary=f['build_vocab_from_sequences'](seqs)
    v=len(vocabulary['node_id_tokens']);require(v<=cfg['max_vocabulary_nodes'],'Observed vocabulary cap exceeded')
    require(v==106769,'Exact original complete vocabulary required')
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
    focal_ref=npz_save_v3(w,'focals.npz',record_ids=chosen,sample_nodes=samples,
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
            planned=native_dataset_bound_v3(len(selected),sum(map(len,selected)),v)
            dataset_refs[f'{view}/{split}']=w.pickle(f'{view}/{split}/raw_{split}_sequences_and_vocab.pkl',
                dict(raw_sequences=selected,vocab=vocabulary),planned)
            meta=dict(record_ids=np.asarray(ids,dtype=np.int64),sample_nodes=np.asarray([s[0] for s in selected],dtype=np.int64),
                sample_indices=np.asarray([sample_meta[s[0]]['sample_index'] for s in selected],dtype=np.int64),
                parent_id=np.asarray(m['parent_id']),class_order=np.asarray(CLASSES))
            if view=='contrastive':meta['reference_classes_0based']=np.asarray([sample_meta[s[0]]['observed_reference_class'] for s in selected],dtype=np.int64)
            view_refs[f'{view}/{split}']=npz_save_v3(w,f'{view}/{split}/row_order_metadata.npz',**meta)
            if view=='contrastive':
                labels=[sample_meta[s[0]]['observed_reference_class'] for s in selected]
                require(set(labels)<=set(range(3)),'Only known reference classes enter native contrastive labels')
                w.pickle(f'{view}/{split}/labels.pkl',labels,5*len(labels)+pickle_margin)
    vocab_ref=w.pickle('vocab.pkl',vocabulary,64*v+pickle_margin)
    map_ref=w.pickle('sequence_to_tree_indices.pkl',mapping,mapping_bound_v3(tokens,len(seqs),ts.num_samples,ts.num_trees))
    return w.json('completed.json',dict(status='completed_native_C3_path_preparation',source_author_function_ref=ref(AUTHOR),
        observation_ref=cfg['observation_ref'],parent_id=m['parent_id'],genealogy_kind=m['genealogy_kind'],class_order=list(CLASSES),
        native_dataset_refs=dataset_refs,full_path_id_order_refs=view_refs,vocab_ref=vocab_ref,sequence_to_tree_indices_ref=map_ref,
        focals_ref=focal_ref,focal_coordinate_rule='label-blind first-occurrence midpoint: reconstruction assumption, not recovered Table1 rule',
        unique_paths=len(seqs),path_tokens=tokens,path_occurrences=ts.num_samples*ts.num_trees,vocabulary_nodes=v,
        native_split_seed=42,native_train_fraction=.9,query_focal_count=len(chosen),
        contrastive_labels_reference_only=True,query_embedding_view_prespecified_focals_only=True,query_ancestry_truth_opened=False,author_Table1_exact_reproduction=False,
        reference_unique_paths=reference_paths,reference_path_tokens=reference_tokens,max_raw_path_nodes=max_raw,
        budget_successor_source_ref=ref(SOURCE),scientific_preparation_original_source_ref=ORIGINAL_PATHS_REF,
        exact_per_sample_counts_ref=PER_SAMPLE_REF,native_serialization_bound_bytes=storage_bound(cfg)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--admission");parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="source_only_inactive_native_paths_consumer",CPU_only=True,
            science_imported=False,guard_imported=False)));return 0
    from research.reviews.American_C3_native_paths_root_v3 import native_paths_contract_v3 as c
    c.require(args.admission and args.expected_admission_sha256,"Exact active root CPU admission required")
    spec=c.read(dict(path=args.admission,sha256=args.expected_admission_sha256))
    c.require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False and spec["parent_id"]=="american_c3_train_000"
        and spec["split"]=="train" and spec["consumer_wrapper_ref"]==c.ref(SOURCE)
        and spec["exact_native_path_count_ref"]==c.EXACT_COUNT_REF
        and spec["exact_per_sample_counts_ref"]==c.PER_SAMPLE_REF
        and spec["source_ref"]==c.PATHS_REF and spec["source_closure_refs"]==c.NATIVE_CLOSURE
        and spec["integration_source_closure_refs"]==c.closure_v1(),"Exact fresh TRAIN-only CPU producer/wrapper closure")
    c.require(json.loads(os.environ["ARGGNN_COMPLETED_CREDITS_POLICY_REF"])==c.POLICY_REF,
              "Same exact current policy must reach common CLI adapter")
    now=datetime.now(timezone.utc)
    c.require(datetime.fromisoformat(spec["created_at_utc"])<=now<datetime.fromisoformat(spec["expires_at_utc"]),
              "Unexpired root CPU capability")
    cfg=c.read(spec["config_ref"]);expected,bound=c.configuration_v1(active=True)
    c.require(cfg==expected and storage_bound(cfg)==bound,"Frozen complete native corpus and narrow budget; no cap fallback")
    import pickle
    c.require(sys.implementation.name=="cpython" and sys.version_info[:3]==(3,11,17)
        and pickle.HIGHEST_PROTOCOL==5,"Exact qualified CPython3.11.17/protocol5 serialization required")
    counts=c.validate_counts_v1();c.validate_exact_count_v3(counts)
    for record in [*c.closure_v1(),c.COUNTS_REF,c.EXACT_COUNT_REF,c.PER_SAMPLE_REF,c.SERIALIZER_CONTRACT_REF,c.RUNTIME_REF,c.POLICY_REF,c.REPAIR_REF]:
        c.require(c.ref(record["path"])==record,"Frozen source/count/runtime/authority changed")
    c.validate_close_receipt_v1(spec["completed_inference_close_receipt_ref"])
    bindings=spec["storage"]["reservation"]["consumer_bindings"]
    c.require(all(r in bindings for r in [*c.closure_v1(),c.COUNTS_REF,c.EXACT_COUNT_REF,c.PER_SAMPLE_REF,c.SERIALIZER_CONTRACT_REF,c.RUNTIME_REF,c.POLICY_REF,
        c.REPAIR_REF,spec["config_ref"],spec["completed_inference_close_receipt_ref"]]),
        "Actual wrapper/integration/count/runtime/config/close receipt must be reserved")
    from . import american_c3_common_v1 as common
    observed=common.read(c.OBSERVATION_REF)
    c.require(observed["parent_id"]=="american_c3_train_000" and observed["split"]=="train"
        and observed["sequence_length_bp"]==30000000. and observed["class_order"]==["EUR","EAS","AFR"]
        and observed["query_ancestry_truth_in_inputs"] is False
        and observed["tree_ref"]==counts["tree_ref"],"Exact current TRAIN inferred observation; no truth")
    c.require(common.runtime_report()==common.read(c.RUNTIME_REF),"Exact qualified shared CPU runtime metadata")
    # Install only into this fresh consumer before common.cli loads its original adapter.
    from . import project_budget_v4 as v4
    c.require(c.ref(c.V4_REF["path"])==c.V4_REF,"Originalv4 source required")
    ms=importlib.util.spec_from_file_location("_C3_native_paths_consumer_memo2",ROOT/c.MEMO_REF["path"])
    memo=importlib.util.module_from_spec(ms);ms.loader.exec_module(memo)
    adapter=memo.install(v4,expected_source_path=ROOT/c.V4_REF["path"],
                         expected_source_sha256=c.V4_REF["sha256"])
    common.cli(SOURCE,DEFAULTS,execute,
        dependencies=(AUTHOR.relative_to(ROOT).as_posix(),ORIGINAL_PATHS_REF["path"]))
    return 0
if __name__=="__main__":raise SystemExit(main())
