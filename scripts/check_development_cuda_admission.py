#!/usr/bin/env python3
"""Synthetic runtime admission only. Never load a cohort or fit parameters."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
import traceback
import uuid

import torch
from torch import Tensor
import torch.nn.functional as F

from arggnn import argformer_control, gst_baseline, graphssm_baseline  # preload pure factories
from arggnn.age_interface import precompute_age_interfaces
from arggnn.allrecord_boundary_temporal import AllRecordBoundaryGRU, AllRecordBoundaryTransformer
from arggnn.anchor_kernel import AnchorKernelModel, prepare_anchor_kernel_bank
from arggnn.anchor_temporal_baselines import (
    AnchorGRUModel, AnchorTemporalTransformerModel, ObservedReferenceCoordinateSmoother,
)
from arggnn.canonical_gru import CanonicalTreeGNNGRU, prepare_canonical_parent, prepare_canonical_gnn_input
from arggnn.data import GraphSequence
from arggnn.development_execution import configure_deterministic_development, deterministic_development_manifest
from arggnn.kron_interface import precompute_leaf_interfaces, with_query_reference_cache
from arggnn.reference_encoder import ReferenceFeatureAdapter
from arggnn.resources import ALLOWED_GPUS, PROCESS_TITLE, ROOT, project_path, require_storage, set_research_title
from arggnn.study_models import make_study_model

ADAPTER = dict(reference_columns=(0, 1, 2, 3), time_column=6, branch_column=7,
               time_scale=10000., kernel_scales=(.05, .1, .2))
NEURAL_FAMILIES = (
    'candidate', 'gru', 'transformer', 'allrecord_gru', 'allrecord_transformer',
    'canonical_gru', 'argmix_full_tree', 'gst_ca_span', 'graphssm_zoh',
)
SCOPE = ('Handmade observed trees and artificial external replay targets only. '
         'No optimizer, cohort access, fitting, ancestry performance, or scientific validation.')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_sources() -> dict[str, str]:
    paths = {Path(__file__).resolve(), ROOT/'scripts/launch_gpu.py', ROOT/'scripts/project_env.sh', ROOT/'AGENTS.md',
             ROOT/'src/arggnn/check_development_cuda_admission.py'}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name == 'arggnn' or name.startswith('arggnn.'):
            if filename and Path(filename).suffix == '.py':
                paths.add(Path(filename).resolve())
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def verify_sources(pins: dict[str, str]) -> None:
    changed = [name for name, expected in pins.items() if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
    if changed:
        raise RuntimeError('Source drift: ' + ', '.join(changed))


# Pin imported code immediately, then check before/after each case. No training module is imported.
IMPORTED_SOURCE_PINS = capture_sources()


@dataclass(frozen=True)
class Case:
    family: str
    hidden_dim: int = 32
    dropout: float = 0.
    geometry: str = 'plain'
    kernel_depth: int = 2
    bandwidth_bp: float | None = None
    pseudocount: float | None = None

    @property
    def identifier(self) -> str:
        if self.family == 'candidate':
            return f'candidate_{self.geometry}_depth{self.kernel_depth}_width{self.hidden_dim}'
        if self.family == 'reference_smoother':
            return f'reference_smoother_bandwidth{int(self.bandwidth_bp)}_pseudocount{self.pseudocount:g}'.replace('.', 'p')
        return f'{self.family}_width{self.hidden_dim}_dropout{self.dropout:g}'.replace('.', 'p')

    def as_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)} | {'case_id': self.identifier}


def admission_cases() -> tuple[Case, ...]:
    result = [Case('candidate', geometry=geometry, kernel_depth=depth)
              for geometry in ('plain', 'age') for depth in (1, 2)]
    result.extend(Case(family, hidden_dim=width, dropout=dropout)
                  for family in NEURAL_FAMILIES[1:] for width in (32, 64) for dropout in (0., .1))
    result.extend(Case('reference_smoother', bandwidth_bp=bandwidth, pseudocount=pseudocount)
                  for bandwidth in (10000., 30000., 100000., 300000.) for pseudocount in (.001, .01))
    assert len(result) == 44 and len({case.identifier for case in result}) == 44
    return tuple(result)


def handmade_observed_sequence(queries: int = 64, references: int = 24) -> GraphSequence:
    """Three balanced changing trees; persistent IDs are correspondence keys only."""
    if type(queries) is not int or type(references) is not int or queries < 2 or references < 4 or references % 4:
        raise ValueError('Require at least two queries and a positive multiple-of-four reference panel')
    leaves = queries + references
    xs, edges, identities, clades, graph_ids = [], [], [], [], []
    offset = 0
    for graph, rotation in enumerate((0, 1, 3)):
        ages = [0.] * queries + [250. + 7*(i % 12) for i in range(references)]
        roles = [[0.] * 4 for _ in range(queries)] + [[float(k == i % 4) for k in range(4)] for i in range(references)]
        descendant = [row[:] for row in roles]
        level_nodes = [(i + rotation) % leaves for i in range(leaves)]
        pairs, level = [], 0
        while len(level_nodes) > 1:
            following = []
            for index in range(0, len(level_nodes), 2):
                children = level_nodes[index:index+2]
                if len(children) == 1:
                    following.append(children[0])
                    continue
                parent = len(ages)
                ages.append(max(ages[child] for child in children) + 400. + 85*level + 30*graph)
                descendant.append([sum(descendant[child][k] for child in children) for k in range(4)])
                pairs.extend((child, parent) for child in children)
                following.append(parent)
            level_nodes, level = following, level + 1
        times = torch.tensor(ages, dtype=torch.float64)
        x = torch.zeros(len(ages), 8, dtype=torch.float64)
        x[:leaves, :4] = torch.tensor(roles, dtype=torch.float64)
        x[:queries, 4] = 1.; x[:leaves, 5] = 1.
        x[:, 6] = torch.log1p(times) / math.log1p(10000.)
        branches = torch.zeros_like(times)
        for child, parent in pairs:
            branches[child] = times[parent] - times[child]
        x[:, 7] = torch.log1p(branches) / math.log1p(10000.)
        edge = torch.tensor(pairs, dtype=torch.long).T
        xs.append(x); edges.append(torch.cat((edge, edge.flip(0)), 1) + offset)
        identities.extend([1000 + 11*i for i in range(queries)] + [5000 + 13*i for i in range(references)] + [-1]*(leaves-1))
        clades.extend(descendant); graph_ids.extend([graph]*len(ages)); offset += len(ages)
    intervals = torch.tensor([[0., .31], [.31, .72], [.72, 1.]], dtype=torch.float64)
    positions = torch.tensor([0., .08, .22, .31, .57, .72, .9, .999], dtype=torch.float64)
    target_graph = ((positions[:, None] >= intervals[None, :, 0]) & (positions[:, None] < intervals[None, :, 1])).long().argmax(1)
    # Reverse query order to exercise explicit ID correspondence rather than packed leaf order.
    sequence = GraphSequence(torch.cat(xs), torch.cat(edges, 1), torch.tensor(graph_ids), torch.tensor(identities),
                             (torch.tensor(clades, dtype=torch.float64) > 0).double(), intervals,
                             torch.tensor([1000 + 11*i for i in reversed(range(queries))]), positions, target_graph,
                             torch.tensor([0., 1.], dtype=torch.float64))
    sequence.validate()
    return sequence


def tensor_digest(value: Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    header = json.dumps({'dtype': str(cpu.dtype), 'shape': list(cpu.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class Fixture:
    raw: GraphSequence
    boundary: object
    anchors: object
    canonical: object
    targets: Tensor
    num_anchors: int
    manifest: dict


def prepare_fixture(queries: int = 64, references: int = 24, num_anchors: int = 32) -> Fixture:
    observed = handmade_observed_sequence(queries, references)
    augmented = ReferenceFeatureAdapter(8, **ADAPTER).precompute(observed)
    reference_ids = torch.tensor([5000 + 13*i for i in range(references)])
    plain = precompute_leaf_interfaces(augmented, time_column=6, parent_branch_column=7, resistance_scale=1e-4)
    plain = with_query_reference_cache(plain, reference_ids)
    age = precompute_age_interfaces(augmented, plain_bank=plain, reference_ids=reference_ids,
                                   time_column=6, parent_branch_column=7, resistance_scale=1e-4)
    anchors = prepare_anchor_kernel_bank(plain, age.operator_cache, num_anchors=num_anchors, num_scales=3)
    canonical = prepare_canonical_parent(observed, ADAPTER, source_provenance={'kind': 'handmade_synthetic_observed_fixture'})
    raw = replace(augmented, x=augmented.x.float(), clades=augmented.clades.float())
    canonical_sequence = replace(canonical.sequence, x=canonical.sequence.x.float(), clades=canonical.sequence.clades.float())
    canonical_input = prepare_canonical_gnn_input(canonical_sequence)
    # Artificial targets are external and defined only after all input preprocessing.
    targets = (torch.arange(observed.positions.numel())[:, None] + torch.arange(queries)[None, :]) % 4
    manifest = {
        'kind': 'handmade_synthetic_observed_fixture', 'generator': 'balanced pair construction, no simulation',
        'queries': queries, 'references': references, 'reference_classes': 4,
        'references_per_class': references//4, 'observed_holds': 3, 'num_anchors': num_anchors,
        'nodes_per_hold': 2*(queries+references)-1, 'raw_features': 48, 'boundary_features': 47,
        'feature_columns': list(plain.feature_columns), 'boundary_response_shape': list(plain.response.shape),
        'age_cqq_shape': list(age.operator_cache.cqq.shape), 'age_scales': list(age.operator_cache.operator_scales),
        'age_spectral_reconstruction_relative_max': float(age.operator_cache.spectral_reconstruction_relative.max()),
        'age_spectral_orthogonality_max': float(age.operator_cache.spectral_orthogonality_max.max()),
        'canonical_holds': canonical.sequence.num_graphs,
        'query_order': observed.query_node_ids.tolist(), 'reference_ids': reference_ids.tolist(),
        'intervals': observed.intervals.tolist(), 'readout_positions': observed.positions.tolist(),
        'observed_tensor_sha256': {field.name: tensor_digest(getattr(observed, field.name))
                                  for field in fields(observed) if isinstance(getattr(observed, field.name), Tensor)},
        'augmented_x_sha256': tensor_digest(augmented.x),
        'external_artificial_targets_sha256': tensor_digest(targets),
        'target_scope': 'Cyclic artificial categories for backward/replay only, no ancestry truth or prediction score',
        'labels_in_model_inputs': False, 'numeric_ID_features': False,
    }
    return Fixture(raw, plain, anchors, canonical_input, targets, num_anchors, manifest)


def make_case_model(case: Case, fixture: Fixture):
    common = dict(hidden_dim=case.hidden_dim, dropout=case.dropout)
    if case.family == 'candidate':
        return AnchorKernelModel(47, 4, hidden_dim=case.hidden_dim, num_layers=2, latent_dim=8,
                                 kernel_depth=case.kernel_depth, num_scales=3, geometry=case.geometry,
                                 coupling='full', num_anchors=fixture.num_anchors,
                                 initial_coupling=(5., 20., 80.), initial_damping=1., rate_parameterization='log')
    if case.family in ('gru', 'transformer'):
        cls = AnchorGRUModel if case.family == 'gru' else AnchorTemporalTransformerModel
        return cls(47, 4, **common, num_layers=2, latent_dim=8, temporal_depth=2, num_scales=3,
                   num_anchors=fixture.num_anchors, initial_coupling=(5., 20., 80.))
    if case.family in ('allrecord_gru', 'allrecord_transformer'):
        cls = AllRecordBoundaryGRU if case.family == 'allrecord_gru' else AllRecordBoundaryTransformer
        return cls(47, 4, **common, source_depth=2, temporal_depth=2)
    if case.family == 'canonical_gru':
        return CanonicalTreeGNNGRU(48, 4, **common, num_layers=2, temporal_layers=2)
    if case.family in ('argmix_full_tree', 'gst_ca_span', 'graphssm_zoh'):
        return make_study_model(case.family, 48, 4, common | {'num_layers': 2, 'reference_adapter': ADAPTER})
    if case.family == 'reference_smoother':
        return ObservedReferenceCoordinateSmoother(4, fixture.num_anchors,
                                                   bandwidth_bp=case.bandwidth_bp, pseudocount=case.pseudocount)
    raise ValueError('Unknown smoke family')


def case_input(case: Case, fixture: Fixture, device: str):
    if case.family in ('candidate', 'gru', 'transformer', 'reference_smoother'):
        value = fixture.anchors.to(device, torch.float32)
        features = value.bank.leaf_x.detach().clone().requires_grad_(case.family != 'reference_smoother')
        return replace(value, bank=replace(value.bank, leaf_x=features)), features
    if case.family in ('allrecord_gru', 'allrecord_transformer'):
        value = fixture.boundary.to(device, torch.float32)
        features = value.leaf_x.detach().clone().requires_grad_()
        return replace(value, leaf_x=features), features
    if case.family == 'canonical_gru':
        value = fixture.canonical.to(device)
        features = value.sequence.x.detach().clone().requires_grad_()
        return replace(value, sequence=replace(value.sequence, x=features)), features
    value = fixture.raw.to(device)
    features = value.x.detach().clone().requires_grad_()
    return replace(value, x=features), features


def raw_nll(logits: Tensor, artificial_targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.double().reshape(-1, 4), artificial_targets.reshape(-1), reduction='mean')


def gradient_evidence(model, family: str) -> dict:
    """Every trainable tensor must be connected/finite; each module group active."""
    groups, parameters = {}, {}
    expected_zero = {'blocks.0.gate.0', 'blocks.1.gate.0'} if family == 'gst_ca_span' else set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError('Missing or nonfinite gradient: ' + name)
        maximum = float(parameter.grad.detach().abs().max())
        group = name.rsplit('.', 1)[0] if '.' in name else name
        # Keep candidate/source rates individually visible rather than combining with maps.
        if name.endswith(('raw_coupling', 'raw_damping', 'mixture_logits')):
            group = name
        parameters[name] = {'elements': parameter.numel(), 'absolute_max': maximum}
        item = groups.setdefault(group, {'parameters': [], 'absolute_max': 0., 'expected_initial_zero': group in expected_zero})
        item['parameters'].append(name); item['absolute_max'] = max(item['absolute_max'], maximum)
    inactive = [name for name, value in groups.items() if value['absolute_max'] == 0 and name not in expected_zero]
    if inactive:
        raise RuntimeError('Structurally inactive gradient groups: ' + ', '.join(inactive))
    for name in expected_zero:
        if name not in groups or groups[name]['absolute_max'] != 0:
            raise RuntimeError('GST expected zero-initialized upstream gate gradient changed: ' + name)
    return {'groups': groups, 'parameters': parameters, 'all_trainable_gradients_present_and_finite': True,
            'all_required_groups_active': True, 'expected_initial_zero_groups': sorted(expected_zero)}


def write_json_fresh(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def save_fresh(path: Path, value) -> None:
    with path.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_launcher_lease(environ=None) -> dict:
    """Check inherited project reservation without querying or initializing CUDA."""
    environ = os.environ if environ is None else environ
    physical = environ.get('ARGGNN_PHYSICAL_GPU')
    if physical not in {str(value) for value in ALLOWED_GPUS}:
        raise RuntimeError('Use the project GPU launcher on physical GPU 2, 3, 4 or 5')
    expected = ROOT/'.runtime/locks'/f'gpu_{physical}.lock'
    try:
        lease = project_path(environ['ARGGNN_GPU_LEASE'])
        fd = int(environ['ARGGNN_GPU_LEASE_FD'])
        actual = os.fstat(fd); on_disk = lease.stat()
    except (KeyError, ValueError, OSError) as error:
        raise RuntimeError('Missing valid inherited launcher lease') from error
    if lease != expected.resolve() or not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        raise RuntimeError('GPU lease descriptor/path mismatch')
    contents = json.loads(lease.read_text())
    visible = environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if contents.get('gpu') != int(physical) or not isinstance(contents.get('uuid'), str) or not contents['uuid'].startswith('GPU-'):
        raise RuntimeError('GPU lease metadata mismatch')
    if len(visible) != 1 or visible[0] != contents['uuid'] or contents.get('job') != environ.get('ARGGNN_JOB_NAME'):
        raise RuntimeError('Must expose exactly the single GPU UUID and job from the launcher lease')
    # A second independent open file description must be unable to reserve it.
    with lease.open('r+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            raise RuntimeError('Launcher lease is not held')
    return {'physical_gpu': int(physical), 'visible_gpu_uuid': contents['uuid'], 'lease_path': str(lease),
            'lease_fd': fd, 'launcher_metadata': contents}


def _normalized_cuda_uuid(value):
    if not isinstance(value, str):
        raise RuntimeError("Malformed CUDA device UUID")
    text = value[4:] if value.startswith("GPU-") else value
    try:
        canonical = str(uuid.UUID(text))
    except (ValueError, AttributeError) as error:
        raise RuntimeError("Malformed CUDA device UUID") from error
    if canonical != text.lower():
        raise RuntimeError("Malformed CUDA device UUID")
    return canonical


def initialize_execution(device: str, receipt: dict | None = None) -> dict:
    """The title and policy precede lease admission and every device inspection.

    A caller-owned receipt preserves completed admission steps if startup fails.
    """
    result = {} if receipt is None else receipt
    result.update(device_kind=device, initialization_stage='process_title', device={'scope': 'not inspected'})
    set_research_title()
    import setproctitle
    actual_title = setproctitle.getproctitle()
    result['process_title'] = actual_title
    if actual_title != PROCESS_TITLE:
        raise RuntimeError('Required process title was not applied')
    result['initialization_stage'] = 'execution_policy'
    result['deterministic_execution_policy'] = configure_deterministic_development()
    result['software'] = {'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                          'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                          'cudnn': torch.backends.cudnn.version(), 'torch_build': torch.__config__.show(),
                          'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads()}
    result['initialization_stage'] = 'launcher_lease'
    if device == 'cuda':
        result.update(validate_launcher_lease())
        result['initialization_stage'] = 'visible_device_count'
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Must expose exactly one reserved CUDA GPU')
        result['initialization_stage'] = 'device_properties'
        props = torch.cuda.get_device_properties(0)
        result['device'] = {'logical_index': 0, 'name': props.name, 'total_memory_bytes': props.total_memory,
                            'compute_capability': [props.major, props.minor], 'runtime_uuid': str(getattr(props, 'uuid', 'unavailable'))}
        actual_uuid = result['device']['runtime_uuid']
        if actual_uuid != 'unavailable':
            result['device']['canonical_runtime_uuid'] = _normalized_cuda_uuid(actual_uuid)
            result['canonical_visible_gpu_uuid'] = _normalized_cuda_uuid(result['visible_gpu_uuid'])
            if result['device']['canonical_runtime_uuid'] != result['canonical_visible_gpu_uuid']:
                raise RuntimeError('Runtime device UUID does not match the reserved physical GPU')
        result['initialization_stage'] = 'driver_version'
        driver = subprocess.run(['nvidia-smi', '-i', result['visible_gpu_uuid'], '--query-gpu=driver_version',
                                 '--format=csv,noheader'], check=True, capture_output=True, text=True)
        result['nvidia_driver_version'] = driver.stdout.strip()
    elif device == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU validation requires CUDA_VISIBLE_DEVICES explicitly empty')
        result['device'] = {'name': platform.processor(), 'scope': 'CPU validation, not CUDA admission'}
    else:
        raise ValueError('Unknown device')
    result['initialization_stage'] = 'complete'
    return result


def model_manifest(model, prepared) -> dict:
    if hasattr(model, 'configuration_manifest'):
        if isinstance(model, CanonicalTreeGNNGRU):
            return model.configuration_manifest()
        return model.configuration_manifest(prepared)
    if hasattr(model, 'configuration_provenance'):
        return model.configuration_provenance()
    return {'class': type(model).__name__, 'parameters': sum(parameter.numel() for parameter in model.parameters())}


def run_case(case: Case, fixture: Fixture, device: str, directory: Path, *, source_pins=None) -> dict:
    """One no-update backward and exact evaluation/state replay; tests use CPU."""
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    memory_stats_reset = False
    record = {'case': case.as_dict(), 'started_utc': now(), 'scope': SCOPE, 'seed': 101,
              'device_kind': device, 'optimizer_steps': 0, 'status': 'failure', 'stage': 'source_guard',
              'source_sha256': source_pins, 'runtime_receipt': '../runtime.json'}
    write_json_fresh(directory/'started.json', record)
    model = prepared = features = logits = repeated = restored = training_logits = loss = loaded = replay = None
    try:
        if source_pins is not None:
            verify_sources(source_pins)
        record['applied_policy_at_start'] = deterministic_development_manifest()
        if device == 'cuda':
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
            memory_stats_reset = True
        record['stage'] = 'construct'
        torch.manual_seed(101)
        model = make_case_model(case, fixture).to(device)
        prepared, features = case_input(case, fixture, device)
        targets = fixture.targets.to(device)
        record['parameters'] = sum(parameter.numel() for parameter in model.parameters())
        record['model_manifest'] = model_manifest(model, prepared)
        record['stage'] = 'forward_backward'
        if record['parameters']:
            model.train()
            training_logits = model(prepared)
            if tuple(training_logits.shape) != (*targets.shape, 4) or not bool(torch.isfinite(training_logits).all()):
                raise RuntimeError('Nonfinite or wrong-shape training logits')
            loss = raw_nll(training_logits, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite artificial backward NLL')
            loss.backward()
            record['backward'] = gradient_evidence(model, case.family)
            if features.grad is None or not bool(torch.isfinite(features.grad).all()) or not bool(features.grad.abs().max() > 0):
                raise RuntimeError('Observed feature backward is missing, nonfinite, or inactive')
            record['observed_feature_gradient_absolute_max'] = float(features.grad.abs().max())
            record['artificial_training_raw_nll'] = float(loss.detach())
        else:
            record['backward'] = {'applicable': False, 'reason': 'deterministic observed-reference control has no trainable parameters'}
        record['stage'] = 'eval_save_load_repeat'
        model.eval()
        with torch.no_grad():
            logits = model(prepared)
            repeated = model(prepared)
            if tuple(logits.shape) != (*targets.shape, 4) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Nonfinite or wrong-shape evaluation logits')
            first_nll = float(raw_nll(logits, targets))
            repeated_nll = float(raw_nll(repeated, targets))
        save_fresh(directory/'eval_logits.pt', logits.detach().cpu())
        save_fresh(directory/'state.pt', {'state_dict': model.state_dict()})
        replay = make_case_model(case, fixture).to(device)
        loaded = torch.load(directory/'state.pt', map_location=device, weights_only=True)
        replay.load_state_dict(loaded['state_dict'], strict=True); replay.eval()
        with torch.no_grad():
            restored = replay(prepared)
            restored_nll = float(raw_nll(restored, targets))
        equality = torch.equal(logits, repeated) and torch.equal(logits, restored)
        nll_equality = first_nll == repeated_nll == restored_nll and math.isfinite(first_nll)
        record['replay'] = {'logits_bitwise_equal': equality, 'raw_nll_exact_equal': nll_equality,
                            'artificial_raw_nll': first_nll, 'repeat_artificial_raw_nll': repeated_nll,
                            'reloaded_artificial_raw_nll': restored_nll,
                            'repeat_max_absolute_logit_delta': float((logits-repeated).abs().max()),
                            'reload_max_absolute_logit_delta': float((logits-restored).abs().max()),
                            'logits_sha256': tensor_digest(logits), 'checkpoint_sha256': sha(directory/'state.pt'),
                            'serialized_logits_sha256': sha(directory/'eval_logits.pt')}
        if not equality or not nll_equality:
            raise RuntimeError('Evaluation/checkpoint replay is not exact')
        record['stage'] = 'final_source_policy_guard'
        if source_pins is not None:
            verify_sources(source_pins)
        record['applied_policy'] = deterministic_development_manifest()
        if device == 'cuda':
            torch.cuda.synchronize(0)
        record['status'] = 'success'; record['stage'] = 'complete'
    except Exception as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        if device == 'cuda' and memory_stats_reset:
            try:
                record['peak_memory'] = {'allocated_bytes': torch.cuda.max_memory_allocated(0),
                                         'reserved_bytes': torch.cuda.max_memory_reserved(0)}
            except Exception as error:
                record['peak_memory_error'] = str(error)
        else:
            record['peak_memory'] = {'allocated_bytes': None, 'reserved_bytes': None, 'scope': 'CUDA allocator metrics unavailable on CPU' if device == 'cpu' else 'case source/policy guard rejected before CUDA measurement'}
        record['ended_utc'] = now(); record['elapsed_seconds'] = time.monotonic() - started
        write_json_fresh(directory/'result.json', record)
        # Release every reference to live graphs and tensors before the next case.
        model = prepared = features = logits = repeated = restored = training_logits = loss = loaded = replay = None
        gc.collect()
        if device == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New in-project output directory, never reused')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cases = admission_cases()
    started = {'kind': 'synthetic_development_runtime_admission', 'started_utc': now(), 'scope': SCOPE,
               'requested_device': args.device, 'output': str(output), 'pid': os.getpid(),
               'source_sha256': IMPORTED_SOURCE_PINS, 'cases': [case.as_dict() for case in cases]}
    write_json_fresh(output/'run_started.json', started)
    successes = []; failures = []; runtime = None
    runtime_admitted = False
    run_started_monotonic = time.monotonic()
    exit_code = 1
    try:
        verify_sources(IMPORTED_SOURCE_PINS)
        storage = require_storage(headroom_bytes=100_000_000)
        runtime = {}
        initialize_execution(args.device, receipt=runtime)
        runtime_admitted = True
        runtime['storage_bytes_before'] = storage
        write_json_fresh(output/'runtime.json', runtime)
        fixture = prepare_fixture()
        write_json_fresh(output/'fixture.json', fixture.manifest)
        for case in cases:
            result = run_case(case, fixture, args.device, output/case.identifier, source_pins=IMPORTED_SOURCE_PINS)
            (successes if result['status'] == 'success' else failures).append(case.identifier)
            print(json.dumps({'event': 'synthetic_admission_case', 'case_id': case.identifier, 'status': result['status'],
                              'stage': result['stage'], 'error': result.get('error', {}).get('message')}), flush=True)
        verify_sources(IMPORTED_SOURCE_PINS)
        deterministic_development_manifest()
        exit_code = int(bool(failures))
    except BaseException as error:
        write_json_fresh(output/'run_failure.json', {'type': type(error).__name__, 'message': str(error),
                                                    'traceback': traceback.format_exc(), 'runtime_admitted': runtime_admitted,
                                                    'partial_runtime': runtime})
    finally:
        write_json_fresh(output/'completion.json', {'kind': started['kind'], 'ended_utc': now(), 'scope': SCOPE,
                          'status': 'success' if exit_code == 0 else 'failure',
                          'cuda_admitted': args.device == 'cuda' and exit_code == 0,
                          'successful_cases': successes, 'failed_cases': failures,
                          'uncompleted_cases': [case.identifier for case in cases if case.identifier not in successes + failures],
                          'required_case_count': len(cases), 'elapsed_seconds': time.monotonic()-run_started_monotonic,
                          'source_sha256': IMPORTED_SOURCE_PINS})
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
