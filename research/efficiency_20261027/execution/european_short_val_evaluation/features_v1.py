"""Exact native feature arithmetic with an isolated truthful10Mb boundary."""
from . import contracts_v1 as c


def iter_native_sampled_features(recipe,observed_ts,panel,query_node_ids,sampled_rows,*,
                                 guard,execution_check,write_headroom_bytes):
    c.short.validate_recipe(recipe)
    c.io._ready(guard,execution_check,write_headroom_bytes)
    import numpy as np
    from arggnn import ancestralpaths_native_bridge as bridge
    # Scientific loop is the original european_execution.iter_native_sampled_features.
    queries = set(query_node_ids)
    if len(queries) != 686 or len(panel.haplotype_node_ids) != 952:
        raise ValueError("Exact complete AP panel and query identity required")
    iterator = iter(sampled_rows)
    pending = next(iterator, None)
    previous = -1.
    seen = set()
    for tree in observed_ts.trees():
        selected = []
        while pending is not None:
            position = pending["position_local_bp"]
            if (not isinstance(position, (int, float)) or not 0 <= position < observed_ts.sequence_length
                    or position < previous or pending["query_node_id"] not in queries
                    or type(pending["example_index"]) is not int or pending["example_index"] < 0
                    or pending["example_index"] in seen):
                raise ValueError("Frozen sorted sampled registry has invalid position/query/example identity")
            if position >= tree.interval.right:
                break
            if position < tree.interval.left:
                raise ValueError("Sampled position lacks native observed tree coverage")
            selected.append(pending)
            seen.add(pending["example_index"])
            previous = position
            pending = next(iterator, None)
        if selected:
            guard(write_headroom_bytes)
            nodes = list(dict.fromkeys(row["query_node_id"] for row in selected))
            features = bridge.extract_native_features(tree, nodes, panel,
                                                       age_mode="training_int", dtype=np.float64)
            if features.shape != (len(nodes), 5, 9):
                raise ValueError("Native source feature shape changed")
            index = {node: i for i, node in enumerate(nodes)}
            for row in selected:
                yield row["example_index"], features[index[row["query_node_id"]]].astype(np.float32)
    if pending is not None:
        raise ValueError("Sampled position lacks native observed tree coverage")


def prepare_features(registry_ref,inference_ref,output_directory,*,source_review_ref,
                     guard,execution_check,write_headroom_bytes,max_feature_bytes,max_metadata_bytes):
    import numpy as np
    import tskit
    from arggnn import ancestralpaths_native_bridge as bridge
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    registry,_=c.registry(registry_ref,inference_ref)
    inference=c.io._read(inference_ref,observed=True,cap=max_metadata_bytes)
    c.need(inference.get('schema')=='AP_short_development_native_inference_v1'
        and inference.get('short_development_plan_ref')==c.short.PLAN_REF
        and inference.get('parent_id')==registry['parent_id'] and inference.get('split')=='validation'
        and inference.get('sequence_length_bp')==10000000 and inference.get('truth_or_oracle_read') is False,
        'Actual observed10Mb inference only')
    c.need(type(max_feature_bytes) is int and 10000*45*4<=max_feature_bytes,'Explicit full feature byte allowance')
    rows=c.io._read(inference['observed_sample_manifest'],observed=True,cap=max_metadata_bytes)
    panel=bridge.ObservedPanel(tuple(row['sample_id'] for row in rows[::2]),
        tuple(row['observed_node_id'] for row in rows),
        {row['observed_node_id']:row['observed_cohort'] for row in rows})
    tree=tskit.load(c.io._checked(inference['inferred_tree'],observed=True))
    queries=[row['observed_node_id'] for row in rows if row['role']=='query']
    x=np.empty((10000,5,9),dtype=np.float32);seen=np.zeros(10000,dtype=bool)
    for index,feature in iter_native_sampled_features(c.short.validate_plan()['recipe'],tree,panel,queries,registry['records'],
            guard=guard,execution_check=execution_check,write_headroom_bytes=write_headroom_bytes):
        c.need(type(index) is int and 0<=index<10000 and not seen[index] and feature.shape==(5,9)
            and feature.dtype==np.float32 and np.isfinite(feature).all(),'Exact complete original feature rows')
        x[index],seen[index]=feature,True
    c.need(bool(seen.all()),'All10000 shared observed examples')
    output=c.io._new_output(output_directory)
    arrays_ref=c.io._npz(output/'features.npz',dict(features=x,example_indices=np.arange(10000,dtype=np.int64)),guard)
    return c.io._json(output/'features.json',dict(schema='European_short_VAL_native_features_v1',
        parent_id=registry['parent_id'],split='validation',registry_ref=registry_ref,inference_ref=inference_ref,
        arrays_ref=arrays_ref,rows=10000,feature_shape=[5,9],short_plan_ref=c.short.PLAN_REF,
        source_manifest_ref=source,original_native_feature_arithmetic=True,
        source_repeated_add_float64_final_input_float32=True,features_contain_no_query_truth=True,
        all_sampled_rows_retained=True),guard)


def load_features(features_ref,*,source_manifest_ref):
    import numpy as np
    feature=c.io._read(features_ref,observed=True)
    registry,_=c.registry(feature['registry_ref'],feature['inference_ref'])
    c.need(feature.get('source_manifest_ref')==source_manifest_ref
        and feature.get('schema')=='European_short_VAL_native_features_v1'
        and feature.get('parent_id')==registry['parent_id'] and feature.get('split')=='validation'
        and feature.get('short_plan_ref')==c.short.PLAN_REF and feature.get('rows')==10000
        and feature.get('feature_shape')==[5,9] and feature.get('original_native_feature_arithmetic') is True
        and feature.get('source_repeated_add_float64_final_input_float32') is True
        and feature.get('features_contain_no_query_truth') is True and feature.get('all_sampled_rows_retained') is True,
        'Exact shared short observed feature product')
    arrays=c.io._arrays(feature['arrays_ref'],cap=32*2**20,observed=True);x=arrays['features']
    c.need(x.shape==(10000,5,9) and x.dtype==np.float32 and np.isfinite(x).all()
        and np.array_equal(arrays['example_indices'],np.arange(10000)),'Complete canonical feature identity')
    return feature,registry,x
