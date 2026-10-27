"""Separately admitted CPU-only scoring of paired complete original VAL support.

Reuses original truth resolution, half-open focal join and posterior metrics.
This new registered research barrier is both arms x all5VAL x all fitted seeds;
it is not the original all15-parent/three-method final-paper score.
"""
import json,os,time
from pathlib import Path
from .paired_training import ROOT,OWNED,ARMS,need,reference,read


def run(completion_ref,output_directory,*,resource_check,max_seconds=3600,max_RSS_bytes=16*2**30):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and callable(resource_check),'Separately admitted CPU-only scorer required')
    import numpy as np
    from arggnn.benchmarking import american_scoring as original
    from arggnn.american_c3_evaluate_v1 import verify_prediction
    completed=read(completion_ref);config=completed['config']
    def verify_sources():
        for source_ref in config['source_refs']:
            need(reference(source_ref['path'])==source_ref,'Registered source drift before or during separate CPU scoring')
    verify_sources()
    need(completed['status']=='completed_training_and_all_VAL_prediction_seals' and
         completed['TEST_opened'] is False and completed['VAL_truth_opened'] is False,
         'Complete paired prediction artifact must exist before any VAL truth access')
    registry=original._registry(completed['parent_registry_ref']);parents=registry.ids('validation')
    need(set(config['validation_inputs'])==set(config['validation_generation_ready_refs'])==set(parents),
         'Exactly original5VAL parents; noTEST resolver permitted')
    out=(ROOT/output_directory).resolve()
    need(out.is_relative_to(OWNED/'runs') and not out.exists(),'Fresh owned CPU evaluation output required')
    out.mkdir(parents=True);deadline=time.monotonic()+max_seconds
    def check(stage):
        resource_check(stage)
        rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        need(time.monotonic()<deadline and rss<=max_RSS_bytes,'Scoring wall/RSS budget exceeded')
    def write(name,payload):
        with (out/name).open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
        return reference(out/name)
    # Authenticate ALL actual prediction arrays before creating a truth resolver.
    records={};predictions={};barrier={};resolution=dict(generation_ready_refs_by_parent={},generation_completion_paths_by_parent={})
    need([r['seed'] for r in completed['seeds']]==config['seeds'],'Complete predeclared seed set required')
    for parent in parents:
        item=config['validation_inputs'][parent];fp=read(item['native_preparation_ref'])
        genref=config['validation_generation_ready_refs'][parent];gen=read(genref)
        inferred=read(fp['input_ref']);observed=read(item['observation_ref'])
        need(fp['parent_id']==gen['parent_id']==inferred['parent_id']==observed['parent_id']==parent and
             fp['split']==gen['split']==inferred['split']==observed['split']=='val' and
             fp['focals_ref']==item['focal_registry_ref'] and fp['observation_ref']==item['observation_ref'] and
             inferred['input_ref']==genref and inferred['observation_ref']==item['observation_ref'] and
             observed['source_observation_ref']==gen['observation_ref'] and observed['sequence_length_bp']==30000000 and
             gen['query_truth_in_training_inputs'] is False and gen['truth_scoring_authorized'] is False and
             fp['query_ancestry_truth_opened'] is False and observed['query_ancestry_truth_in_inputs'] is False,
             'Exact frozen generation-to-inferred-observation-to-focal lineage required')
        records[parent]=original._load_npz(item['focal_registry_ref'],8*2**20,keys=(*original.IDENTITY,'first_occurrence_intervals_bp'))
        r=records[parent];intervals=r['first_occurrence_intervals_bp']
        need(len(r['record_ids'])==10000 and r['parent_id'].item()==parent and
             len(set(r['record_ids'].tolist()))==10000 and intervals.shape==(10000,2) and
             np.array_equal(intervals.mean(1),r['positions_bp0']),'All original10000 unchanged focal identities required')
        barrier[parent]={}
        for seed in completed['seeds']:
            need(seed['all10_VAL_predictions_sealed_before_truth'] is True,'Paired full prediction seal required')
            for arm in ARMS:
                ref=seed['predictions'][arm][parent]
                p=original._load_npz(ref,8*2**20,keys=(*original.IDENTITY,'probabilities'))
                verify_prediction(p,r)
                need((p['probabilities']<=1).all(),'Unmodified valid softmax probabilities required')
                predictions[(seed['seed'],arm,parent)]=p
                barrier[parent][f"{seed['seed']}/{arm}"]=ref
        resolution['generation_ready_refs_by_parent'][parent]=genref
        resolution['generation_completion_paths_by_parent'][parent]=str((Path(genref['path']).parent/'private/generation_completed.json'))
        check('all_actual_prediction_payloads_before_VAL_truth')
    barrier_ref=write('all_paired_VAL_predictions_barrier.json',dict(schema='TreeLoom_Transport_all_paired_VAL_barrier_v1',
        completion_ref=completion_ref,parents=barrier,seeds=config['seeds'],records_per_parent=10000,
        TEST_access=False,VAL_truth_payloads_opened=0))
    verify_sources()
    results={};truth_provenance={}
    for parent in parents:
        need(registry.split_for(parent)=='validation','Refuse anyTEST parent before resolution')
        truth_ref,generation_ref=original._resolve_truth(resolution,parent)
        truth=original._load_npz(truth_ref,8*2**30,keys=original.TRUTH_KEYS,public=False)
        target=original.join_truth(records[parent],truth,resource_check=check);del truth
        truth_provenance[parent]=dict(truth_ref=truth_ref,generation_completion_ref=generation_ref)
        for seed in config['seeds']:
            for arm in ARMS:
                results.setdefault(str(seed),{}).setdefault(arm,{})[parent]=original.score_arrays(
                    target,predictions[(seed,arm,parent)],genuine_posterior=True)
        check('complete_original_VAL_parent_scored')
    # Conditional-on-trained-seeds inference: average paired seed deltas for
    # each independent parent first, then resample parents as intact clusters.
    endpoints=['accuracy','macro_F1','EUR_recall','EAS_recall','AFR_recall','EUR_F1','EAS_F1','AFR_F1','Brier','NLL']
    family_size=len(endpoints)+1  # Boundary co-endpoint remains required/unavailable.
    rng=np.random.default_rng(20261027);draws=rng.integers(0,len(parents),size=(100000,len(parents)))
    differences={}
    for metric in endpoints:
        seed_rows=[];finite=True
        for seed in config['seeds']:
            values=[]
            for parent in parents:
                a=original._flat(results[str(seed)]['transport'][parent])[metric]
                b=original._flat(results[str(seed)]['frozen_current'][parent])[metric]
                if a is None or b is None:finite=False;break
                values.append((b-a) if metric in ('Brier','NLL') else (a-b))
            if not finite:break
            seed_rows.append(values)
        if not finite:
            differences[metric]=dict(status='inconclusive_nonfinite_original_metric',zero_margin_pass=False);continue
        matrix=np.asarray(seed_rows);per_parent=matrix.mean(0)
        lower=float(np.quantile(per_parent[draws].mean(1),.05/family_size,method='linear'))
        differences[metric]=dict(mean_improvement=float(per_parent.mean()),per_parent_seed_averaged_improvement=dict(zip(parents,per_parent.tolist())),
            per_seed_parent_mean_improvement=dict(zip(map(str,config['seeds']),matrix.mean(1).tolist())),
            simultaneous_one_sided_lower_bound=lower,zero_margin_pass=lower>=0,
            direction='positive_favors_transport',confidence_scope='conditional_on_predeclared_trained_seeds; independent_parent_resampling',
            nominal_family_confidence=.95,family_size=family_size,bootstrap_replicates=100000)
    verify_sources()
    return write('completed.json',dict(schema='TreeLoom_Transport_original_VAL_comparison_v1',status='completed_original_focal_metrics',
        prediction_barrier_ref=barrier_ref,per_seed=results,paired_endpoints=differences,truth_provenance=truth_provenance,
        equal_independent_parent_weighting=True,TEST_opened=False,original_truth_join_and_metrics_reused=True,
        boundary_endpoint=dict(status='unavailable_on_original_sparse_focal_support',zero_margin_pass=False),
        accuracy_preservation_established=False,
        limitations=['Only5 independent VAL parents; bootstrap bounds are approximate and may be inconclusive.',
                     'Seed variation is reported separately, not independently resampled inside each parent.',
                     'Original sparse focal support cannot establish dense boundary preservation.',
                     'GPU activity and full-step efficiency require separately measured profile evidence.']))
