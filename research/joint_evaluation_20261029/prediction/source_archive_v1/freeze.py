"""Seal runtime and scientific provenance before any TEST observation access."""
import argparse
import json
from pathlib import Path
from research.redesign_20261028.data.window_data import ROOT, path_for, require
from .common import BASE, MODEL_ID, TEST_PARENTS, now, ref, verify_ref, write_json

PROTOCOL = dict(path='research/joint_evaluation_20261029/root/assessment_protocol_v1.json',
                sha256='22e939bcb5334f2ed86e76f26740523bd515d0c9e7b2b27b5ced4777759b9cc2')
READINESS = dict(path='research/joint_model_20261029/data/TEST_EVALUATION_READINESS.json',
                 sha256='3722e898c36babd0736cdc2e9d15dc61ac8cd17195d668a60997fd18e3cc7f39')
DEPENDENCIES = {
 'research/joint_model_20261029/model/joint_model_v1.py': 'ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4',
 'research/joint_model_20261029/data/observations.py': '56acdd6a4aa48dac2b1b676e2c2a3e45a24b2a24bd58a5a745b4f305d06ab727',
 'research/redesign_20261028/data/window_data.py': 'a1788ea0a33335450bfb43c689aa3c598070f07b28c40930be3fb086f42aa309'}
RUNTIME_FILES = ('__init__.py', 'common.py', 'observations.py', 'workers.py', 'predict.py', 'freeze.py', 'smoke_test.py')
GPU_UUIDS = {4: 'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71', 5: 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


def stat_ref(reference):
    info = path_for(reference).stat()
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def check_inputs(frozen, parent_id):
    item = frozen['parents'][parent_id]
    for key, reference in item['observation_refs'].items():
        require(stat_ref(reference) == item['source_stats'][key], 'Frozen observation source stat changed: ' + key)


def validate_freeze(path):
    frozen = json.loads(Path(path).read_text())
    require(frozen['schema'] == 'joint_model_native6_prediction_freeze_v1' and frozen['model_id'] == MODEL_ID,
            'Exact prediction freeze schema/model')
    require(frozen['parents_order'] == list(TEST_PARENTS) and frozen['mode'] == 'structure' and
            frozen['model'] == dict(width=64, layers=3, classes=6), 'Frozen scientific configuration')
    require(frozen['context'] == dict(seed=20261029, parent_ordinals=[3, 4, 5, 6, 7], radius_bp=50000.,
                                    coordinate_count=5, weights=[0., .25, .25, .25, .25]), 'Frozen context recipe')
    for reference in frozen['runtime_source_refs'] + frozen['dependency_refs'] + [frozen['checkpoint_ref'],
            frozen['assessment_protocol_ref'], frozen['training_protocol_ref'], frozen['smoke_test_ref']]:
        verify_ref(reference)
    smoke = json.loads(path_for(frozen['smoke_test_ref']).read_text())
    require(smoke['passed'] is True and smoke['TEST_payloads_opened'] is False, 'Passing TRAIN-only parity gate')
    require(smoke['runtime_source_refs'] == frozen['runtime_source_refs'], 'Tested source is frozen source')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', default=str(BASE / 'smoke_result.json'))
    parser.add_argument('--output', default=str(BASE / 'freeze.json'))
    args = parser.parse_args()
    verify_ref(PROTOCOL)
    verify_ref(READINESS)
    protocol = json.loads(path_for(PROTOCOL).read_text())
    readiness = json.loads(path_for(READINESS).read_text())
    require(protocol['method_id'] == MODEL_ID and protocol['parents'] == list(TEST_PARENTS), 'Assessment protocol identity')
    verify_ref(protocol['checkpoint'])
    verify_ref(protocol['model_source'])
    verify_ref(protocol['original_training_protocol'])
    require(protocol['checkpoint']['sha256'] == '651afd90ef9ccb51436c5b5816569b77d883109c474db7c81a84ce8faf068a9b',
            'Selected final structure checkpoint')
    dependencies = [verify_ref(dict(path=path, sha256=sha)) for path, sha in DEPENDENCIES.items()]
    sources = [ref(BASE / name) for name in RUNTIME_FILES]
    smoke_ref = ref(args.smoke)
    smoke = json.loads(Path(args.smoke).read_text())
    require(smoke['passed'] is True and smoke['TEST_payloads_opened'] is False, 'Passing TRAIN-only parity gate')
    require(smoke['runtime_source_refs'] == sources, 'Freeze only the exact source that passed smoke')
    parents = {}
    for ordinal, parent_id in enumerate(TEST_PARENTS, start=3):
        item = readiness['parents'][parent_id]
        refs = item['observation_bindings']
        require(set(refs) == {'sampled_registry_ref', 'inference_ref', 'tree_ref', 'panel_ref', 'observed_vcf_ref'},
                'Exact observation-only references')
        binding = dict(refs={k: refs[k] for k in ('sampled_registry_ref', 'inference_ref', 'observed_vcf_ref')},
                       tree_ref=refs['tree_ref'], panel_ref=refs['panel_ref'], coordinate_offset_bp=item['coordinate_offset_bp'])
        parents[parent_id] = dict(parent_ordinal=ordinal, binding=binding, observation_refs=refs,
                                  source_stats={key: stat_ref(value) for key, value in refs.items()})
    frozen = dict(schema='joint_model_native6_prediction_freeze_v1', recorded_utc=now(), model_id=MODEL_ID,
                  mode='structure', model=protocol['model'], checkpoint_ref=ref(path_for(protocol['checkpoint'])),
                  assessment_protocol_ref=ref(path_for(PROTOCOL)),
                  training_protocol_ref=ref(path_for(protocol['original_training_protocol'])),
                  readiness_metadata_ref=ref(path_for(READINESS)), runtime_source_refs=sources, dependency_refs=dependencies,
                  smoke_test_ref=smoke_ref, parents_order=list(TEST_PARENTS), parents=parents,
                  native_path_names=protocol['classes'], targets_per_parent=10000,
                  context=dict(seed=20261029, parent_ordinals=[3, 4, 5, 6, 7], radius_bp=50000.,
                               coordinate_count=5, weights=[0., .25, .25, .25, .25]),
                  gpu_uuids={str(k): v for k, v in GPU_UUIDS.items()}, workers=4, batch_sites=16,
                  precision='float32; TF32 disabled', persistent_prepared_cache=False,
                  probability_convention='raw native-six softmax computed in float64 from stored float32 logits, stored float32; no cohort postprocessing',
                  large_input_hash_policy='Inherited immutable source SHA256; freeze and run-time stat checks. Small registry/panel SHA256 verified on load.',
                  TEST_observation_payloads_opened_before_freeze=False, query_ancestry_truth_opened=False)
    output_ref = write_json(args.output, frozen)
    validate_freeze(args.output)
    print(json.dumps(output_ref), flush=True)


if __name__ == '__main__':
    main()
