"""ARGMix sealed-probability external H038 score. No action or data read at import."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import numpy as np

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AP_HELPER='research/reviews/expanded_native_AP_score_root_assistance_v1/root_score_helper.py'
AP_SHA='735ce1ea8f5c14ee260b642472434f425284edf367b99144c18d412aa7d8fb05'
COMPLETED='experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/completed.json'
COMPLETED_SHA='9e27f2c188d74ed652ecdf10f5fdf51d23e6ffd20742f35d9832bdefcf3c698a'
VAL_IDS=tuple(f'ap_expanded_val_{i:03d}' for i in range(8))
ESTIMAND='source arithmetic average of overlapping-chunk softmax probabilities'
CAVEAT=('Current author-release ARGMix with full952 input/slot metadata adaptations, source seed50 '
        'and unchanged native inner VAL checkpoint selection; not an exact paper-original reproduction.')
FIELDS={'probabilities','supported','grid_indexes','positions_bp','local_positions_bp','bin_edges_bp',
        'local_site_positions_bp','original_site_positions_bp','query_nodes'}
_AP=None


def require(ok,message):
    if not ok:raise ValueError(message)


def ap():
    global _AP
    if _AP is None:
        p=ROOT/AP_HELPER
        require(hashlib.sha256(p.read_bytes()).hexdigest()==AP_SHA,'Frozen AP scoring helper changed')
        spec=importlib.util.spec_from_file_location('ARGMix_shared_AP_scoring',p)
        _AP=importlib.util.module_from_spec(spec);spec.loader.exec_module(_AP)
    return _AP


def read_h038_scoring_only_v1(index,parent_id,reader):
    return ap().read_h038_scoring_only_v1(index,parent_id,reader)


def grid_v1(seal,parent,reader):
    """Read raw sealed probabilities; identity join never transforms their values."""
    from arggnn.full_panel_native_baseline_contracts_v1 import validate_observed_roster
    frame=seal['frame']
    # The same-support target binding is checked independently by score_root_v1.
    rows=json.loads(reader.read(frame['roster']))
    roster=validate_observed_roster(rows,require_argmix_native_ids=True)
    queries=[r for r in rows if r['role']=='query']
    q=np.asarray(parent['query_slots'])
    require(np.array_equal([r['sample_index'] for r in queries],q)
        and [[r['sample_id'],r['haplotype']] for r in queries]==
            [[parent['samples'][int(i)]['sample_id'],parent['samples'][int(i)]['haplotype']] for i in q]
        and np.array_equal([r['observed_cohort'] for r in queries],parent['cohorts']),
        'Original686 sampled identity/order/cohorts differ')
    with np.load(io.BytesIO(reader.read(seal['grid_prediction'])),allow_pickle=False) as z:
        require(set(z.files)==FIELDS,'Original ARGMix sealed grid schema differs')
        arrays={k:z[k].copy() for k in z.files}
    for key,target in [('grid_indexes','retained_indexes'),('positions_bp','positions'),('bin_edges_bp','edges')]:
        require(np.array_equal(arrays[key],parent[target]),'ARGMix/H038 physical support differs: '+key)
    require(frame['parent_id']==parent['parent_id'] and frame['split']=='val'
        and frame['queries']==686 and frame['pure_refs']==266 and frame['truth_opened'] is False
        and frame['stock_native_ID_correspondence_verified'] is True
        and np.array_equal(frame['coverage_original_bp0'],parent['coverage_original'])
        and frame['local_offset_bp']==float(parent['coverage_original'][0])
        and np.array_equal(arrays['query_nodes'],roster.query_nodes)
        and np.array_equal(arrays['local_positions_bp']+frame['local_offset_bp'],arrays['positions_bp']),
        'Exact native observed frame/query/coordinate identity required')
    p=arrays['probabilities'];support=arrays['supported']
    require(p.dtype==np.float32 and p.shape==(len(parent['positions']),686,4)
        and support.dtype==bool and support.shape==p.shape[:2] and support.all()
        and np.isfinite(p).all() and (p>=0).all()
        and np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5),
        'Complete unmodified finite FP32 arithmetic-average probabilities required')
    return p


def score_root_v1(*,completed_ref,parents,bindings,reader,helpers,on_parent=None,on_stage=None):
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1,PRIMARY,means_v1
    from arggnn.mutation_routed_fast_primary_v1 import mass_cache_v1,score_primary_v1
    h=ap()
    require(completed_ref==dict(path=COMPLETED,sha256=COMPLETED_SHA),'Exact completed native ARGMix run required')
    completed=json.loads(reader.read(completed_ref));fit=json.loads(reader.read(completed['fit_ref']))
    require(completed['schema']=='expanded_native_argmix_GPU_fit_root_v1_completed'
        and completed['status']=='completed_native_fit_all8_VAL'
        and completed['TRAIN_parents']==32 and completed['VAL_parents']==8 and completed['outer_parents']==0
        and completed['source_seed']==50 and completed['original_paper_reproduction'] is False
        and completed['truth_opened_for_external_score'] is False
        and fit['kind']=='expanded_native_argmix_stock_fit_v1','Complete current native release lineage required')
    seals={s['parent_id']:s for s in completed['VAL_seals']}
    require(len(completed['VAL_seals'])==8 and set(seals)==set(parents)==set(bindings)==set(VAL_IDS),
        'Exactly all eight registered internal VAL seals/parents required')
    # This small SHA-bound completion contains all eight prediction-seal refs.
    # Dense products are all verified below; targets rebind this complete manifest.
    records=[completed_ref]
    # Verify EVERY sealed product before constructing any external target.
    for pid in VAL_IDS:
        seal=seals[pid]
        h._check_parent(parents[pid],bindings[pid],reader)
        require(seal['grid_prediction']['path']==
            f'experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/VAL/{pid}/sealed_C4_grid_probabilities.npz'
            and seal['checkpoint']==fit['checkpoint'] and seal['frame']['config_sha256']==fit['config_sha256']
            and seal['full_support_complete'] is True and seal['missing_grid_query_cells']==0
            and seal['native_final_estimand']==ESTIMAND and seal['truth_opened_for_external_score'] is False
            and seal['inference_only_edge_option'] is False
            and 0<=seal['native_replay_max_probability_error']<=1e-6,
            'Original native selected checkpoint/complete arithmetic-average seal required')
        p=grid_v1(seal,parents[pid],reader);del p
        if on_stage:on_stage('ARGMix_seal_preflight_parent_verified',pid)
    if on_stage:on_stage('all_eight_ARGMix_seals_verified_before_truth')
    cells=[]
    for pid in VAL_IDS:
        p=grid_v1(seals[pid],parents[pid],reader)
        target=FullPanelExternalTargetV1(parents[pid],bindings[pid],'VAL_after_sealed_predictions',reader,sealed_records=records)
        cache=mass_cache_v1(parents[pid],target.target_mass(parents[pid],helpers),
                           target_provenance=target.provenance,expected_present_queries=182)
        cell=h.jsonable_v1(dict(parent_id=pid,split='val',prediction_seal=seals[pid]['grid_prediction'],
            target_provenance=target.provenance,groups={PRIMARY:score_primary_v1(p,cache,backend='numpy')},
            native_checkpoint_selection=completed['native_checkpoint_selection'],
            development_only=True,outer_parents_accessed=0))
        cells.append(cell)
        if on_parent:on_parent(pid,cell)
        del p,target,cache
    return h.jsonable_v1(dict(schema='expanded_native_ARGMix_H038_score_summary_v1',completed_ref=completed_ref,
        VAL_parents=8,TEST_or_outer_parents=0,C4_primary_equal_parent=means_v1(cells),per_parent=cells,
        all_eight_seals_verified_before_truth=True,C4_arithmetic='same_H038_FP32_primary_scorer',
        prediction_scope='original_sealed_arithmetic_mean_of_overlapping_chunk_softmax_probabilities',
        native_checkpoint_selection=completed['native_checkpoint_selection'],author_release_caveat=CAVEAT,
        retraining_or_new_prediction=False,calibration=False,probability_floor=False,posthoc_renormalization=False))
