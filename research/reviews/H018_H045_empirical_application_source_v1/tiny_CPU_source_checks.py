"""Tiny FP32 CPU source tests only. No actual bank/checkpoint payload reads."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
from arggnn.anonymous_clade_flag_control_v1 import build_clade_flag_arrays_v1,AnonymousCladeFlagsV1
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodeGNNV1
from research.reviews.H018_H045_empirical_application_source_v1.variable_panel_model_v1 import VariablePanelEpisodeInterfaceV1,episode_inputs_v1
torch.set_num_threads(1);torch.manual_seed(101)
checks=[]
def check(name,value):
 assert bool(value),name
 checks.append(dict(name=name,passed=True))
def forest(patterns=('A', 'A', 'A', 'B', 'A', 'A', 'star'), h=4):
    parent, samples, reps, unary = ([], [], [], [])
    for pattern in patterns:
        start = len(parent)
        row = list(range(start, start + h))
        parent.extend([-2] * h)
        if pattern == 'star':
            root = len(parent)
            parent.append(-1)
            for node in row:
                parent[node] = root
            reps.append(None)
            unary.append(None)
        else:
            pair = row[:2] if pattern == 'A' else [row[0], row[2]]
            representative = len(parent)
            ancestor = representative + 1
            root = representative + 2
            parent.extend([ancestor, root, -1])
            for node in row:
                parent[node] = representative if node in pair else root
            reps.append(representative)
            unary.append(ancestor)
        samples.append(row)
    return (np.array(parent, dtype=np.int64), np.array(samples, dtype=np.int64), reps, unary)

def synthetic_bank(patterns=('A', 'A', 'B'), h=952):
    parent, sample, _, _ = forest(patterns, h)
    n, (t, _) = (len(parent), sample.shape)
    children = [[] for _ in range(n)]
    for node, ancestor in enumerate(parent):
        if ancestor >= 0:
            children[ancestor].append(node)
    height = np.zeros(n, dtype=np.int64)
    for node in range(n):
        if children[node]:
            height[node] = max((height[x] for x in children[node])) + 1
    levels = [np.flatnonzero(height == value) for value in range(int(height.max()) + 1)]
    up_children, ranks = ([], [])
    for level in levels:
        ch, rank = ([], [])
        for i, node in enumerate(level):
            ch.extend(children[node])
            rank.extend([i] * len(children[node]))
        up_children.append(np.array(ch, dtype=np.int64))
        ranks.append(np.array(rank, dtype=np.int64))
    depth = np.zeros(n, dtype=np.int64)
    for node in reversed(range(n)):
        if parent[node] >= 0:
            depth[node] = depth[parent[node]] + 1
    down = [np.flatnonzero(depth == value) for value in range(1, int(depth.max()) + 1)]
    edge = np.zeros((n, 4), dtype=np.float32)
    edge[parent >= 0, 2] = 0.1
    support = np.zeros((n, 5), dtype=np.float32)
    support[:, 4] = 1
    s = 2 * t
    arrays = dict(parent=parent, node_time=height.astype(np.float32), sample_index=sample, upward_levels=levels, upward_children=up_children, upward_parent_rank=ranks, downward_levels=down, edge_geometry=edge, branch_support_true=support, branch_support_sham=support.copy(), genotype=np.tile(np.arange(h, dtype=np.int16) % 2, (s, 1)), log_donor_allele_frequency=np.full((s, 2, 8), np.log(0.5), dtype=np.float32), donor_call_mask=np.ones((s, 8), dtype=np.bool_), panel_allele_frequency=np.full((s, 2), 0.5, dtype=np.float32), site_entropy=np.full(s, 0.5, dtype=np.float32), site_hold=np.repeat(np.arange(t, dtype=np.int64), 2), spans=np.array([1, 2, 5][:t], dtype=np.float32), readout_hold=np.arange(t, dtype=np.int64), query_column=np.arange(686, dtype=np.int64), reference_class=np.full(h, -1, dtype=np.int64), sample_age=np.zeros(h, dtype=np.float32))
    return MutationRoutedWholeTreeInputsV1.from_arrays(arrays)

start=time.monotonic()
original=OrderedCladeEpisodeGNNV1().float().eval()
with torch.no_grad():
 original.clade_coupling.O.weight.copy_(torch.randn_like(original.clade_coupling.O.weight)*.02)
 original.episode_coupling.O.weight.copy_(torch.randn_like(original.episode_coupling.O.weight)*.02)
state={k:v.detach().clone() for k,v in original.state_dict().items()}
adapted=VariablePanelEpisodeInterfaceV1(expected_haplotypes=952,expected_queries=686).eval()
adapted.load_state_dict(state,strict=True)
bank=synthetic_bank()
validate_inputs_v1(bank,expected_haplotypes=952)
flags_array,flag_meta=build_clade_flag_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy())
flags=AnonymousCladeFlagsV1.from_verified_arrays(flags_array,flag_meta)
episodes,meta=episode_inputs_v1(bank,known_flags=flags)
with torch.inference_mode():
 expected=original(bank,episodes=episodes)
 actual=adapted(bank,episodes=episodes)
 full_expected=original.forward_holds(bank,episodes=episodes)
 full_actual=adapted.forward_holds(bank,episodes=episodes)
check("original952_Q686_logits_bitwise_equal",torch.equal(expected,actual))
check("original952_all_haplotype_logits_bitwise_equal",torch.equal(full_expected,full_actual))
check("original952_output_shape",actual.shape==(3,686,4))
check("same31812_parameter_count",sum(p.numel() for p in adapted.parameters())==31812)
check("same_complete_state_keys_and_shapes",set(state)==set(adapted.state_dict()) and all(state[k].shape==adapted.state_dict()[k].shape for k in state))
check("nonzero_episode_and_clade_projection_tested",bool(original.episode_coupling.O.weight.count_nonzero()) and bool(original.clade_coupling.O.weight.count_nonzero()))
check("original_H045_encode_method_inherited",VariablePanelEpisodeInterfaceV1.encode_trees is OrderedCladeEpisodeGNNV1.encode_trees)
check("original_H045_query_readout_methods_inherited",VariablePanelEpisodeInterfaceV1.forward is OrderedCladeEpisodeGNNV1.forward and VariablePanelEpisodeInterfaceV1.forward_train_holds is OrderedCladeEpisodeGNNV1.forward_train_holds)
empirical=replace(synthetic_bank(h=40),query_column=torch.arange(16,40),reference_class=torch.tensor([k for k in range(4) for _ in range(4)]+[-1]*24))
validate_inputs_v1(empirical,expected_haplotypes=40)
flags_array,flag_meta=build_clade_flag_arrays_v1(empirical.parent.numpy(),empirical.sample_index.numpy())
flags=AnonymousCladeFlagsV1.from_verified_arrays(flags_array,flag_meta)
episodes40,meta40=episode_inputs_v1(empirical,known_flags=flags)
model40=VariablePanelEpisodeInterfaceV1().float().eval()
result=model40.load_state_dict(state,strict=True)
check("full31812_state_load_into40_Q24",not result.missing_keys and not result.unexpected_keys and sum(p.numel() for p in model40.parameters())==31812)
with torch.inference_mode():
 logits=model40(empirical,episodes=episodes40)
 probabilities=logits.softmax(-1)
check("synthetic40_Q24_shape",logits.shape==(3,24,4))
check("synthetic40_FP32_CPU_finite",logits.dtype==torch.float32 and logits.device.type=="cpu" and torch.isfinite(logits).all())
check("synthetic40_probabilities_unit_sum",torch.allclose(probabilities.sum(-1),torch.ones(3,24),atol=1e-6,rtol=0))
check("exact_prepared_flags_preserved",torch.equal(episodes40.flags.node_flags,flags.node_flags))
try:model40(empirical,episodes=episodes40,control="shuffled");bad=False
except ValueError:bad=True
check("non_aligned_control_rejected",bad)
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
check("tiny_synthetic_peak_RSS_below2GB",rss<2000000000)
base=ROOT/"research/reviews/H018_H045_empirical_application_source_v1"
ref=lambda p:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
receipt=dict(schema="H018_H045_variable_panel_tiny_CPU_source_tests_v1",passed=len(checks),checks=checks,
 source_ref=ref(base/"variable_panel_model_v1.py"),test_source_ref=ref(Path(__file__)),
 exact952_max_abs_difference=float((expected-actual).abs().max()),synthetic40_probability_shape=list(probabilities.shape),
 nonzero_episode_and_flag_weights_tested=True,actual_bank_or_checkpoint_payload_read=False,optimization_or_GPU_execution=False,
 measured_peak_RSS_bytes=rss,elapsed_seconds=time.monotonic()-start)
p=base/"tiny_CPU_source_receipt.json";assert not p.exists();p.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="checks"},sort_keys=True))
