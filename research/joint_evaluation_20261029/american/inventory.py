"""Bind existing American inputs and fix training exposure before held-out access."""
import argparse
import json
from pathlib import Path
from .common import BASE, MODEL_ID, MODEL_SOURCE, RUNTIME, CLASSES, path, read, ref, verify, write, source_refs, require, now

REGISTRY = dict(path='configs/benchmarks/fresh_american_parent_registry.json',
                sha256='3a2e096b4bc0763512075236d350d0baaf971936217cff0ed4e92dcca5c71b74')
TRAIN_BINDINGS = 'research/operations_resume_20261026/american/treeloom.replacement_task.json'
HELD_BINDINGS = 'research/benchmark_rebuild_20261024/american/expanded_delivery/task_dag.json'


def prepare():
    verify(MODEL_SOURCE)
    registry = read(REGISTRY)
    require(registry['class_order'] == list(CLASSES) and registry['sequence_length_bp'] == 30000000,
            'Exact registered American endpoint')
    old_train = json.loads(path(TRAIN_BINDINGS).read_text())['call']['spec']['arguments']['train_inputs']
    dag = json.loads(path(HELD_BINDINGS).read_text())
    held = {task['scientific_spec']['parent_id']: task['scientific_spec'] for task in dag['tasks']
            if task.get('scientific_spec', {}).get('stage') == 'prepare'}
    held_rows = {row['parent_id']: row for row in dag['parents']}
    parents, train_truth, score_paths = {}, {}, {}
    for ordinal, row in enumerate(registry['parents']):
        parent, split = row['parent_id'], row['split']
        if split == 'train':
            original = old_train[parent]
            binding = {key: original[key] for key in ('observation_ref', 'focal_registry_ref', 'native_preparation_ref')}
            train_truth[parent] = original['truth_ref']
        else:
            original = held_rows[parent]
            preparation_ref = (held[parent]['focal_preparation_ref'] if parent in held else
                               ref(path(original['focal_registry_ref']).parent / 'compact_focal_preparation.json'))
            binding = dict(observation_ref=original['observation_ref'], focal_registry_ref=original['focal_registry_ref'],
                           native_preparation_ref=preparation_ref)
        observation, preparation = read(binding['observation_ref']), read(binding['native_preparation_ref'])
        require(observation['parent_id'] == preparation['parent_id'] == parent and
                observation['split'] == preparation['split'] == {'validation': 'val'}.get(split, split) and
                observation['class_order'] == preparation['class_order'] == list(CLASSES) and
                observation['query_ancestry_truth_in_inputs'] is False and preparation['query_ancestry_truth_opened'] is False and
                preparation['focals_ref'] == binding['focal_registry_ref'] and preparation['observation_ref'] == binding['observation_ref'],
                'Exact original public observation/focal lineage')
        refs = {key: observation[key] for key in ('tree_ref', 'roster_ref', 'genotypes_ref', 'positions_ref', 'alleles_ref')}
        for reference in refs.values():
            require(not any(part in ('private', 'truth') for part in Path(reference['path']).parts), 'Observation-only input path')
            require(path(reference).is_file(), 'Existing original observed payload required')
        parents[parent] = dict(split=split, parent_ordinal=100 + ordinal, binding=binding, observed_refs=refs,
                               source_stats={key: dict(bytes=path(value).stat().st_size,
                                                       mtime_ns=path(value).stat().st_mtime_ns) for key, value in refs.items()})
        if split != 'train':
            inference = read(preparation['input_ref'])
            generation_ref = inference['input_ref']
            # Resolve only the public lineage now; the private completion is opened only after all15 predictions seal.
            generation = read(generation_ref)
            require(generation['parent_id'] == parent, 'Same-parent public generation lineage')
            score_paths[parent] = dict(generation_ready_ref=generation_ref,
                                      private_completion_path=str((path(generation_ref).parent / 'private/generation_completed.json').relative_to(BASE.parents[2])))
    require([sum(item['split'] == split for item in parents.values()) for split in ('train', 'validation', 'test')] == [20, 5, 10],
            'All original20 TRAIN /5 validation /10 TEST parents')
    write(BASE / 'train_truth_bindings.json', dict(schema='american_joint_TRAIN_only_truth_bindings_v1', parents=train_truth))
    write(BASE / 'scoring_bindings.json', dict(schema='american_joint_postseal_scoring_bindings_v1', parents=score_paths))
    protocol = dict(schema='american_joint_fixed_protocol_v1', recorded_utc=now(), model_id=MODEL_ID,
         model=dict(width=64, layers=3, classes=3), mode='structure', model_source_ref=MODEL_SOURCE,
         parent_registry_ref=REGISTRY, parents=parents, class_order=list(CLASSES), canonical_targets_per_parent=10000,
         epochs=13, training_target_presentations=2600000,
         budget_derivation='ceil(European 50 epochs * 50000 TRAIN targets / American 200000 TRAIN targets)=13 full epochs; 4% more target presentations, all original TRAIN rows each epoch',
         initialization_seed=101, batch_seed=20261032, batch_windows=16, workers=4,
         optimizer=dict(name='AdamW', learning_rate=.001, betas=[.9, .999], epsilon=1e-8, weight_decay=.0001, gradient_clip=1.),
         objective='Uniform shuffled coordinate windows; M/(S*N)*sum CE with actual last-batch window count S, all canonical targets equally weighted in expectation',
         checkpoint_policy='Fixed final epoch13; no held-out checkpoint/model selection',
         context=dict(seed=20261029, parent_ordinals='100..134 in original registry order', site_seed='rank of unique original focal coordinate within parent',
                      radius_bp=50000., coordinates=5, weights=[0., .25, .25, .25, .25]),
         input_adaptation=dict(reference_channels='EUR=0,EAS=1,AFR=2; fourth reference channel unused',
              cohort_channels='ADMIX=0,EUR=1,EAS=2,AFR=3; remaining observed cohort channels unused',
              focal_genotype='Use exact observed SNP if coordinate matches; otherwise all -1 missing with allele count1. Never move canonical midpoint.',
              full_native_trees=True, all_SNPs_in_complete_native_intervals=True, missing_query_ancestry=True),
         worker_cache_bytes=2 * 2**30, no_persistent_observation_cache=True,
         private_TRAIN_bindings_ref=ref(BASE / 'train_truth_bindings.json'),
         postseal_scoring_bindings_ref=ref(BASE / 'scoring_bindings.json'),
         original_binding_source_refs=[ref(TRAIN_BINDINGS), ref(HELD_BINDINGS)],
         heldout_payloads_opened_during_preparation=False, historical_TreeLoom_or_competitors_changed=False)
    print(write(BASE / 'protocol_v1.json', protocol), flush=True)


def seal(smoke):
    receipt = read(ref(smoke))
    require(receipt['passed'] is True and receipt['heldout_payloads_opened'] is False,
            'TRAIN-only CPU smoke must precede operational freeze')
    refs = source_refs()
    require(receipt['source_refs'] == refs, 'Seal only tested implementation bytes')
    print(write(BASE / 'freeze_v1.json', dict(schema='american_joint_runtime_freeze_v1', recorded_utc=now(),
          protocol_ref=ref(BASE / 'protocol_v1.json'), source_refs=refs, smoke_ref=ref(smoke),
          heldout_predictions_require_final_checkpoint_seal=True, model_id=MODEL_ID)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'seal'])
    parser.add_argument('--smoke', default=str(BASE / 'smoke_result.json'))
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else seal(args.smoke)
