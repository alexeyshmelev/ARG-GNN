"""Paired FP32 model step with independently reconstructed observed inputs."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import gc
import json
import os
import subprocess
import time
import numpy as np
import torch
import setproctitle
from research.preprocessing_optimization_20261026.integration.full_parent import signature
from research.preprocessing_optimization_20261026.integration.hooks import source_refs
from research.scalability_20261024.profiling import run_profile as reference
from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
from research.preprocessing_optimization_20261026.tree import observations, episodes, cut
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as memory
from arggnn.american_c3_gnn_dataset_v1 import load_observation

ROOT = reference.ROOT
OUT = ROOT / 'research/preprocessing_optimization_20261026/integration/gpu_parity_v2.json'
assert not OUT.exists()
assert os.environ['ARGGNN_PHYSICAL_GPU'] == '5'
expected = 'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
assert os.environ['CUDA_VISIBLE_DEVICES'] == expected
inventory = subprocess.check_output(['nvidia-smi', '--id=5', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
assert inventory[0].strip() == expected and int(inventory[1]) >= 24000
setproctitle.setproctitle('AI Auto Research Pipeline Test')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
sources = source_refs()
started = time.monotonic()
def check(stage):
    assert reference.rss() < 24 * 2**30, stage
    if torch.cuda.is_initialized():
        assert torch.cuda.memory_reserved() < 16 * 2**30, stage
    assert time.monotonic() - started < 900, stage
saved = torch.load(ROOT / reference.CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
rt = fresh.FreshTreeLoom(fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref']))
args = SimpleNamespace(mode='slice', trace_transfers=False, holds_per_block=128, haplotypes_per_gru=64)
baseline, metadata, labels, frozen, scope = reference.prepare(args, rt, saved, check)
plan = json.loads((ROOT / 'configs/benchmarks/paper_campaign_resume_20261026_v1.json').read_text())
job = next(t for t in plan['tasks'] if t['id'] == 'paper_american_treeloom_fit')
item = job['call']['spec']['arguments']['train_inputs']['argformer_american_train_000']
state = load_observation(item['observation_ref'], item['focal_registry_ref'], dict(
    native_preparation_ref=item['native_preparation_ref'], class_order=list(fresh.CLASSES),
    expected_haplotypes=1600, reference_counts=[200,200,200],
    input_reference_class_mask=None, input_reference_sample_indices=None))
workspace = observations.native_workspace_v1(state['ts'])
node_time = np.asarray(state['ts'].nodes_time)
trees, intervals = [], []
for tree in state['ts'].trees():
    if tree.index == 128:
        break
    trees.append(observations.native_tree(tree, state['nodes'], node_time=node_time, workspace=workspace))
    intervals.append(tuple(tree.interval))
lo, hi = np.searchsorted(state['positions'], [intervals[0][0], intervals[-1][1]])
bank, bank_metadata = observations.encode_block(trees, intervals, state['genotype'][lo:hi],
    state['allele_count'][lo:hi], state['positions'][lo:hi], state['labels'], state['ages'],
    num_classes=3, age_split=350., query_columns=state['query_columns'], resource_check=check)
isolated = tuple(bank_metadata.get('isolated_zero_site_holds', ()))
episode, episode_metadata = episodes.build_episodes(bank, max_payload_bytes=2**30, isolated_hold_indices=isolated)
cut_plan, cut_metadata = cut.prepare_cut_plan_v1(bank, expected_haplotypes=1600, max_nodes=400000,
    max_holds=128, isolated_hold_indices=isolated)
for name, a, b in [('bank', baseline.bank, bank), ('episodes', baseline.episodes, episode), ('cut', baseline.cut, cut_plan)]:
    assert signature(a) == signature(b), name + ' must be byte-identical'
candidate = replace(baseline, bank=bank, episodes=episode, cut=cut_plan,
    bank_stamps=memory._stamps(rt.frozen._bank_tensors(bank)),
    episode_stamps=memory._stamps(rt.frozen._episode_tensors(episode)))
candidate.validate()
del state, trees, workspace
gc.collect()
results = []
original_values = None
for name, bundle in [('reference', baseline), ('reference_repeat', baseline), ('candidate', candidate)]:
    receipt, values = reference.run_one('original', name, 0, args, rt, saved, bundle,
        metadata, labels, frozen, None, check)
    if original_values is None:
        original_values = values
    else:
        receipt['comparison'] = reference.compare(original_values, values, atol=1e-6, rtol=1e-5)
    results.append(receipt)
assert source_refs() == sources, 'Candidate source changed during paired test'
record = dict(status='pass', scope=scope, inputs_byte_identical=True,
    same_unchanged_model_executor=True, source_refs=sources, records=results,
    exact_training_output_claimed=False, baseline_repeat_included=True,
    production_checkpoint_modified=False, heldout_truth_opened=False,
    diagnostic_training_steps_not_benchmark_scores=True, seconds=time.monotonic()-started)
OUT.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(status='pass', comparison=results[-1]['comparison'])), flush=True)
