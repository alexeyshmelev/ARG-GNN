"""All-five prediction seal before the isolated10Mb external truth join."""
import numpy as np
from . import contracts_v1 as c, metrics_v1

LENGTH=c.short.LENGTH
_need=c.need


def validated_predictions(prediction_refs,registry_ref,inference_ref,source):
    from .predictions_v1 import validate_fit_metadata
    from .prior_v1 import load_prior
    c.need(type(prediction_refs) is dict and set(prediction_refs)==set(c.METHODS),
        'All five predeclared conditions required before truth access')
    registry,rows=c.registry(registry_ref,inference_ref);probabilities={}
    definition=c.io._read(c.definition_ref(),observed=True)
    for method in c.METHODS:
        product=c.io._read(prediction_refs[method],observed=True)
        c.need(product.get('schema')=='European_short_VAL_prediction_v1' and product.get('method')==method
            and product.get('parent_id')==registry['parent_id'] and product.get('split')=='validation'
            and product.get('rows')==10000 and product.get('registry_ref')==registry_ref
            and product.get('inference_ref')==inference_ref and product.get('definition_ref')==c.definition_ref()
            and product.get('source_manifest_ref')==source and product.get('class_order')==list(c.CLASSES)
            and product.get('VAL_truth_opened') is False and product.get('TEST_accessed') is False,
            'Exact source/input/checkpoint bound predictions on identical10000 observed rows')
        provenance=product['provenance']
        if method=='TRAIN_cohort_prior':load_prior(provenance['fitted_prior_ref'],source)
        else:
            fitted=validate_fit_metadata(method,c.io._read(provenance['fit_completion_ref'],observed=True),definition)
            c.need(fitted['checkpoint_ref']==provenance['checkpoint_ref'],'Exact accepted fixed-final weights')
            c.acceptance(method,provenance['fit_completion_ref'],provenance['checkpoint_ref'],
                provenance['checkpoint_acceptance_ref'],source)
        arrays=c.io._arrays(product['predictions_ref'],cap=16*2**20,observed=True)
        c.need(np.array_equal(arrays['example_indices'],np.arange(10000)),
            'No missing,reordered or changed canonical example identities')
        probabilities[method]=c.probability_array(arrays['probabilities'],10000)
    return registry,rows,probabilities


def seal_barrier(prediction_refs,registry_ref,inference_ref,output_directory,*,source_review_ref,
                 guard,execution_check,write_headroom_bytes):
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    validated_predictions(prediction_refs,registry_ref,inference_ref,source)
    output=c.io._new_output(output_directory)
    return c.io._json(output/'prediction_barrier.json',dict(schema='European_short_VAL_all_predictions_barrier_v1',
        methods=list(c.METHODS),prediction_refs=prediction_refs,registry_ref=registry_ref,inference_ref=inference_ref,
        source_manifest_ref=source,definition_ref=c.definition_ref(),source_review_ref=source_review_ref,
        all_predictions_and_TRAIN_prior_sealed_before_VAL_truth=True,VAL_truth_opened=False,TEST_accessed=False),guard)


def score(barrier_ref,truth_refs,output_directory,*,source_review_ref,guard,execution_check,
          write_headroom_bytes,max_truth_uncompressed_bytes):
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    barrier=c.io._read(barrier_ref,observed=True)
    c.need(barrier.get('schema')=='European_short_VAL_all_predictions_barrier_v1'
        and barrier.get('methods')==list(c.METHODS) and barrier.get('source_manifest_ref')==source
        and barrier.get('definition_ref')==c.definition_ref() and barrier.get('source_review_ref')==source_review_ref
        and barrier.get('all_predictions_and_TRAIN_prior_sealed_before_VAL_truth') is True
        and barrier.get('VAL_truth_opened') is False and barrier.get('TEST_accessed') is False,
        'Complete immutable all-method barrier before any VAL target metadata')
    registry,rows,probabilities=validated_predictions(barrier['prediction_refs'],barrier['registry_ref'],barrier['inference_ref'],source)
    execution_check();guard(0)
    # First access to the truth reference mapping occurs only after every seal.
    c.need(set(truth_refs)=={'generation_ref','target_schema_ref','truth_ref'},'Explicit three external truth references')
    generated=c.io._read(truth_refs['generation_ref']);schema=c.io._read(truth_refs['target_schema_ref'])
    frozen=c.short.frozen_parent();inferred=c.io._read(barrier['inference_ref'],observed=True)
    c.need(generated.get('parent_id')==schema.get('parent_id')==registry['parent_id']==frozen['parent_id']
        and generated.get('split')==schema.get('split')=='validation' and generated.get('sequence_length_bp')==LENGTH
        and generated.get('seeds')=={key:frozen['seeds'][key] for key in ('ancestry','mutations')}
        and schema.get('native_to_C4')==list(c.NATIVE_TO_C4)
        and generated['files']['truth/full_truth_intervals.npz']['sha256']==truth_refs['truth_ref']['sha256']
        and generated['files']['truth/target_schema.json']['sha256']==truth_refs['target_schema_ref']['sha256'],
        'Same independently registered10Mb producer and exact frozen derived ancestry mapping')
    for name,digest in inferred['input_observed_sha256'].items():
        c.need(generated['files']['observed/'+name]['sha256']==digest,'External truth producer matches every inferred observed input')
    base=c.io._checked(truth_refs['generation_ref']).parent
    c.need(c.io._path(truth_refs['truth_ref']['path'])==base/'truth/full_truth_intervals.npz'
        and c.io._path(truth_refs['target_schema_ref']['path'])==base/'truth/target_schema.json',
        'Exact same-parent truth namespace')
    truth=c.io._arrays(truth_refs['truth_ref'],cap=max_truth_uncompressed_bytes)
    labels=native_truth_labels(registry,schema,truth);del truth
    result=metrics_v1.report(labels,probabilities,np.asarray([row['cohort'] for row in rows]))
    output=c.io._new_output(output_directory)
    return c.io._json(output/'evaluation.json',dict(schema='European_short_VAL_common_evaluation_v1',
        prediction_barrier_ref=barrier_ref,definition_ref=c.definition_ref(),source_manifest_ref=source,
        registry_ref=barrier['registry_ref'],truth_refs=truth_refs,rows=10000,TEST_accessed=False,
        all_method_prediction_barrier_before_VAL_truth=True,original_TRAIN_TEST_unchanged=True,**result),guard)


# Exact original native_truth_labels body; only this module's LENGTH is10Mb.
def native_truth_labels(registry, target_schema, truth):
    """Pure same-site native-six join. No file I/O or sampling changes.

    A TRAIN caller may pass its separately authenticated supervised targets.
    An external evaluator may call this only after its all-method prediction
    barrier. This pure join provides no capability to resolve private files.
    """
    import numpy as np
    _need(registry.get("schema") == "ap_paper_sampled_registry_v1"
          and target_schema["parent_id"] == registry["parent_id"]
          and target_schema["split"] == registry["split"]
          and target_schema["native_paths_1based"] == [1, 2, 3, 4, 5, 6],
          "Same-parent native-six target schema required")
    intervals = np.asarray(truth["intervals_bp"])
    paths = np.asarray(truth["native_paths_1based"])
    raw = np.asarray(truth["query_raw_node_ids"])
    _need(intervals.ndim == 2 and intervals.shape[1] == 2 and len(intervals) > 0
          and np.isfinite(intervals).all() and intervals[0, 0] == 0
          and intervals[-1, 1] == LENGTH and (intervals[:, 1] > intervals[:, 0]).all()
          and np.array_equal(intervals[:-1, 1], intervals[1:, 0])
          and raw.shape == (686,) and raw.dtype.kind in "iu"
          and len(set(raw.tolist())) == 686
          and paths.shape == (len(intervals), 686) and paths.dtype.kind in "iu"
          and ((paths >= 1) & (paths <= 6)).all(),
          "Complete native-six truth domain and actual query columns required")
    _need(np.array_equal(raw, target_schema["query_sample_indices"])
          and len(target_schema["query_keys"]) == 686,
          "Explicit raw sample and semantic query-key target mapping required")
    by_key = {tuple(key): (int(sample), column)
              for column, (key, sample) in enumerate(zip(target_schema["query_keys"], raw))}
    _need(len(by_key) == 686 and len(registry["query_identities"]) == 686
          and all(by_key.get((r["sample_id"], r["haplotype"]), (None,))[0] == r["sample_index"]
                  for r in registry["query_identities"]),
          "Actual biological query identities must agree across observed/truth namespaces")
    result = np.empty(registry["examples"], dtype=np.int64)
    for record in registry["records"]:
        position = record["position_original_bp"]
        _need(0 <= position < LENGTH, "No same-site truth extrapolation")
        column = by_key[(record["sample_id"], record["haplotype"])][1]
        row = int(np.searchsorted(intervals[:, 1], position, side="right"))
        path = int(paths[row, column])
        _need(path in ((5, 6) if record["cohort"] == "BAA" else (1, 2, 3, 4)),
              "Native BAA and non-BAA path meanings must remain distinct")
        result[record["example_index"]] = path - 1
    return result
