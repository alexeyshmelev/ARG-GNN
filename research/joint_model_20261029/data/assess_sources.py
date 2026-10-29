"""Bounded source/registry assessment; no label-array or TEST payload reads."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from research.redesign_20261028.data.window_data import small_json, path_for, prefix_value, KastoreArrays

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = ROOT / 'research/joint_model_20261029/data'
JOB = ROOT / '.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json'


def main():
    started = time.perf_counter()
    job_raw = JOB.read_bytes()
    inputs = json.loads(job_raw)['call']['spec']['scientific_spec']['train_inputs']
    report = dict(schema='joint_model_original_source_assessment_v1', parents={},
                  authority_job=dict(path=str(JOB.relative_to(ROOT)), sha256=hashlib.sha256(job_raw).hexdigest()),
                  labels_arrays_opened=False, TEST_payloads_opened=False)
    for parent, refs in inputs.items():
        assert parent in [f'ancestralpaths_european_train_{i:03d}' for i in range(3)]
        registry = small_json(refs['sampled_registry_ref'])
        targets = small_json(refs['training_targets_ref'])
        with path_for(refs['inference_ref']).open('rb') as stream:
            prefix = stream.read(16384).decode()
        tree_ref = prefix_value(prefix, 'inferred_tree')
        panel_ref = prefix_value(prefix, 'observed_sample_manifest')
        panel = small_json(panel_ref)
        columns = KastoreArrays(path_for(tree_ref))
        records = registry['records']
        sites = Counter(row['native_site_id'] for row in records)
        positions = Counter(row['position_local_bp'] for row in records)
        identities = Counter((row['native_site_id'], row['sample_index']) for row in records)
        report['parents'][parent] = dict(refs=refs, tree_ref=tree_ref, panel_ref=panel_ref,
            target_array_ref=targets['arrays_ref'], canonical_examples=len(records),
            unique_native_sites=len(sites), unique_positions=len(positions),
            unique_site_sample_pairs=len(identities), canonical_duplicate_rows=sum(identities.values())-len(identities),
            targets_per_position_histogram=dict(sorted(Counter(positions.values()).items())),
            cohort_counts=dict(Counter(row['cohort'] for row in records)),
            classes=targets['native_path_names'], registry_split=registry['split'],
            haplotypes=len(panel), observed_query_haplotypes=sum(row['role']=='query' for row in panel),
            source_sample_identity_example=[{key:panel[0][key] for key in ('sample_id','haplotype','sample_index','observed_node_id')}],
            sequence_length_bp=float(columns['sequence_length'][0]),
            coordinate_offset_bp=float(prefix_value(prefix, 'local_to_original_offset_bp')),
            native_sites=len(columns['sites/position']), native_nodes=len(columns['nodes/time']),
            native_edges=len(columns['edges/left']),
            source_bytes=dict(tree=path_for(tree_ref).stat().st_size,
                              VCF=path_for(refs['observed_vcf_ref']).stat().st_size,
                              registry=path_for(refs['sampled_registry_ref']).stat().st_size,
                              targets=path_for(targets['arrays_ref']).stat().st_size),
            dense_int8_genotype_disk_bytes=len(columns['sites/position'])*len(panel),
            source_checksums='Small metadata authenticated; large source hashes inherited, no full rescans',
            registry_record_keys=sorted(records[0]))
    report['wall_seconds']=time.perf_counter()-started
    output=HERE/'SOURCE_ASSESSMENT.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
