"""Versioned real TRAIN probe after independent interval-guard qualification."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from research.redesign_20261028.data.window_data import TrainWindowDataset,uniform_coordinate_context
HERE=Path('research/redesign_20261028/data')


def main():
    for name in ['interval_revision_equivalence_v2.json','train000_uniform_coordinate_window_v2.npz','train000_uniform_coordinate_window_v2.json']:
        if (HERE/name).exists():raise FileExistsError('Preserve versioned probe output: '+name)
    job=json.loads(Path('.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json').read_text())
    parent='ancestralpaths_european_train_000'
    ds=TrainWindowDataset(parent,job['call']['spec']['scientific_spec']['train_inputs'][parent],halo=2)
    start=time.perf_counter()
    old_report=json.loads((HERE/'train000_first_window.json').read_text())
    current=ds.batch(old_report['selected_example_indices'])
    with np.load(HERE/'train000_first_window.npz',allow_pickle=False) as old:
        for key in old.files:np.testing.assert_array_equal(current[key],old[key],err_msg=key)
        matched_keys=list(old.files)
    equivalence=dict(prior_packet_sha256=old_report['output_sha256'],matched_arrays=matched_keys,
        matched_array_count=len(matched_keys),all_original_arrays_identical=True,
        elapsed_seconds=time.perf_counter()-start,
        source_sha256=hashlib.sha256((HERE/'window_data.py').read_bytes()).hexdigest(),
        reason='Independent review correction authenticates both native interval boundaries')
    (HERE/'interval_revision_equivalence_v2.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    rng=np.random.default_rng(20261028)
    index=int(rng.integers(0,len(ds)))
    focal=ds.records[index]['position_local_bp']
    spec=uniform_coordinate_context(rng,focal,50000,5,ds._trees.sequence_length)
    before_vcf=ds._vcf.bytes_read;start=time.perf_counter()
    packet=ds.batch([index],context_specs={index:spec})
    out=HERE/'train000_uniform_coordinate_window_v2.npz'
    if out.exists():raise FileExistsError('Never overwrite the versioned real packet')
    np.savez(out,**packet)
    report=dict(schema='bounded_observed_TRAIN_coordinate_window_v2',parent_id=parent,
        seed=20261028,selected_example_index=index,target_probability_conditional_parent=1/len(ds),
        selected_cohort=ds.records[index]['cohort'],radius_bp=50000,total_coordinates=5,
        random_coordinates=4,forced_focal_coordinates=1,source_evidence=ds.source_evidence,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in packet.items()},
        sampled_coordinates_local_bp=packet['context_coordinate_local_bp'].tolist(),
        sampled_coordinates_original_bp=(packet['context_coordinate_local_bp']+ds.offset).tolist(),
        coordinate_tree_index=packet['context_tree_index'].tolist(),
        native_intervals_local_bp=packet['tree_interval_local_bp'].tolist(),
        native_intervals_original_bp=packet['tree_interval_original_bp'].tolist(),
        context_density_per_bp=packet['context_proposal_density_per_bp'].tolist(),
        context_quadrature_weight=packet['context_quadrature_weight'].tolist(),
        forced_focal_mask=packet['context_forced_focal'].tolist(),
        elapsed_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        VCF_bytes_read=ds._vcf.bytes_read-before_vcf,output_bytes=out.stat().st_size,
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((HERE/'window_data.py').read_bytes()).hexdigest(),
        cold_cache_status='uncontrolled; prior real equivalence probe in same process',
        no_training=True,no_TEST=True,no_simulation=True,not_a_learning_or_accuracy_diagnostic=True)
    (HERE/'train000_uniform_coordinate_window_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(equivalence_arrays=len(matched_keys),selected_example_index=index,
        selected_cohort=report['selected_cohort'],trees=len(packet['tree_core_mask']),
        nodes=len(packet['parent']),sites=len(packet['site_tree_index']),
        elapsed_seconds=report['elapsed_seconds'],peak_RSS_KiB=report['peak_RSS_KiB'],
        VCF_bytes_read=report['VCF_bytes_read'],output_bytes=report['output_bytes'])))

if __name__=='__main__':main()
