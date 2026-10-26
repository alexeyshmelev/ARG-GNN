"""Seal the completed independent review only after all immutable evidence passes."""
from pathlib import Path
import hashlib
import json
from research.preprocessing_optimization_20261026.integration.hooks import source_refs

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/preprocessing_optimization_20261026'
def reference(path):
    path=Path(path)
    if not path.is_absolute():
        path=ROOT/path
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read(relative):
    path=BASE/relative
    assert path.stat().st_size <= 16*2**20, relative
    return json.loads(path.read_text())
def validate_reference(ref):
    assert reference(ref['path'])==ref, 'Changed reference '+ref['path']

evidence_paths=[
    'observed/qualification.json','observed/full_train_qualification.json',
    'observed/process_import_audit_final.json','observed/harness_serialization_review.json','review/parser_adversarial.json',
    'tree/parity.json','tree/process_parity.json','tree/private_safety.json',
    'review/tree_adversarial.json','integration/full_parent_qualification.json',
    'integration/gpu_parity_final.json','integration/resume_contract.json',
    'review/integration_hardening.json','review/fresh_review.json']
evidence={path:read(path) for path in evidence_paths}
assert all(value['status']=='pass' for value in evidence.values())
current_sources=source_refs()
for key in ['observed/process_import_audit_final.json','integration/gpu_parity_final.json']:
    assert evidence[key]['source_refs']==current_sources, key
for key in ['integration/resume_contract.json','review/integration_hardening.json','review/fresh_review.json']:
    assert evidence[key]['candidate_source_refs']==current_sources, key
fresh=evidence['review/fresh_review.json']
assert not fresh['findings']
harness_review=evidence['observed/harness_serialization_review.json']
for ref in harness_review['source_refs']:
    validate_reference(ref)
harness_ref=harness_review['source_refs'][0]
assert harness_ref['path']=='research/preprocessing_optimization_20261026/tree/full_process_parent.py'
assert harness_ref['sha256']=='fe66ce010b322137a51d4322a6ca54029384b1c79f8b5b97f2012c1bfab40a3e'
implementation_refs=[]
for ref in fresh['implementation_source_refs']:
    if ref['path']==harness_ref['path']:
        assert ref['sha256']=='eed3624acfc46e5073e23bc71930fa5443a20311ab35f8068b08ecb83fb405f5'
        implementation_refs.append(harness_ref)
    else:
        validate_reference(ref)
        implementation_refs.append(ref)
for ref in fresh['original_source_refs']:
    validate_reference(ref)
for ref in evidence['review/integration_hardening.json']['implementation_source_refs']:
    validate_reference(ref)
full=evidence['integration/full_parent_qualification.json']
for ref in full['source_refs']:
    validate_reference(ref)
for ref in full['record_refs'].values():
    validate_reference(ref)
records={key:json.loads((ROOT/ref['path']).read_text()) for key,ref in full['record_refs'].items()}
assert set(records)=={'reference','serial','process4'}
original=records['reference']
for name,record in records.items():
    assert record['full_parent'] is True and record['truth_opened'] is False and record['model_executed'] is False, name
    for key in ['parent','observed_inputs','bundle','metadata']:
        assert record[key]==original[key], name+': '+key
for ref in records['process4']['source_refs']:
    validate_reference(ref)
parser=evidence['observed/full_train_qualification.json']
assert parser['byte_identical_all_retained_calls'] and parser['callbacks_identical']
assert parser['truth_opened'] is False and parser['original_oracle_source_verified']
gpu=evidence['integration/gpu_parity_final.json']
assert gpu['inputs_byte_identical'] and gpu['same_unchanged_model_executor'] and gpu['baseline_repeat_included']
assert gpu['exact_training_output_claimed'] is False and gpu['production_checkpoint_modified'] is False
assert gpu['heldout_truth_opened'] is False and gpu['diagnostic_training_steps_not_benchmark_scores']
comparisons={row['phase']:row['comparison'] for row in gpu['records'] if 'comparison' in row}
assert set(comparisons)=={'reference_repeat','candidate'}
for comparison in comparisons.values():
    assert comparison['passed'] is True and comparison['atol']==1e-6 and comparison['rtol']==1e-5
contract=evidence['integration/resume_contract.json']
assert contract['model_executed'] is False and contract['production_files_changed'] is False
review=dict(
    schema='exact_preprocessing_independent_release_review_v1', status='pass',
    scope='Exact own preprocessing, ordered four-process encoder, authenticated checkpoint continuation and migration safety. The prepared European one-parent lookahead prediction wrapper is excluded from this deployment approval.',
    review_independence='Reviewer authored the observed parser and undeployed European wrapper, independently reviewed the tree process encoder and integration/migration changes. The parser, tree kernels and final integration were also independently examined by other agents. Fresh integration review is included as separate evidence.',
    candidate_source_refs=current_sources,
    implementation_source_refs=implementation_refs,
    review_updates='Fresh review remains evidence for its recorded source state. Only its full_process_parent.py harness and process_source_manifest.json were superseded by the separately reviewed serialization-only harness repair. Final harness and manifest pins are recorded in observed/harness_serialization_review.json.',
    original_source_refs=fresh['original_source_refs'],
    evidence_refs=[reference(BASE/path) for path in evidence_paths],
    review_script_ref=reference(Path(__file__).resolve()),
    findings=[], unresolved_release_blockers=[],
    checks=[
        'Complete real American TRAIN000 reference, serial and process4 preprocessing bundles, tensor bytes, shapes, dtypes and metadata are identical. Only pointer/version stamps are excluded by the documented semantic serializer.',
        'Final parser retains byte-identical complete European TRAIN000 calls and allele counts with exact callbacks and unchanged input source bindings.',
        'Final GPU proof uses final candidate sources, byte-identical preprocessing inputs and unchanged model execution, with unchanged-baseline repeat as the nondeterminism control.',
        'Ordered process delivery preserves tree and batch order. Chunk boundaries neither remove trees nor replace scientific arithmetic. Oversized complete trees use the qualified serial route.',
        'Final cold-import and four-process hook installation preserve Python, NumPy and Torch RNG, parent thread counts and uninitialized CUDA state.',
        'Bindings authenticate original owners and exact callables before assignment, roll back installation failure and verify completion bindings.',
        'Checkpoint continuation retains original trainer validation, model tensors, all 49 Adam states, RNG and remaining fixed600 update schedule.',
        'History restoration validates size/count/hash/path bounds before exclusive resource-checked writes and preserves original history paths.',
        'Migration verifies effective repair chain, PID identity/start/cwd/source pins and pidfd before signaling only the owned TreeLoom leader. Scheduler termination and absent owned process group precede archive/repair.',
        'The final implementation and reviewed original scientific source pins still match their reviewed hashes.'
    ],
    full_parent_timings=full['timings'], parser_speedup=parser['speedup'],
    gpu_numerical_comparisons=comparisons,
    limitations=[
        'No claim of permanent full GPU utilization, universal optimality or elimination of all future bottlenecks. Timings are measured under shared-host contention.',
        'Preprocessing equality is exact on qualified full parents and extensive edge cases. GPU arithmetic is within atol1e-6 and rtol1e-5, not a bitwise whole-training-trajectory guarantee.',
        'Memory protection consists of admission reserves and sampled RSS/PSS checks, not OS-enforced per-allocation limits.',
        'The final GPU check is a real 128-tree diagnostic training slice, not a heldout accuracy result or complete resumed training run.',
        'European lookahead wrapper has contract tests but no full real prediction qualification and is excluded from deployment.',
        'This reviewer performed no production migration, process signaling, queue mutation, Git operation or native competitor modification.'
    ], native_competitor_changed=False, simulation_or_scoring_changed=False,
    deployment_executed_by_reviewer=False)
output=BASE/'review/release_review.json'
with output.open('x') as handle:
    json.dump(review,handle,indent=2,sort_keys=True,allow_nan=False)
    handle.write('\n')
print(json.dumps(reference(output)))
