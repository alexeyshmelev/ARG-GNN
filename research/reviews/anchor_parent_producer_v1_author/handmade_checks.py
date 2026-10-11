"""Finite H024 producer checks; all tensors and native inventory are handmade."""
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from arggnn import anchor_parent_producer_v1 as producer
from arggnn import anchor_parent_context_v3 as core
from arggnn import anchor_kernel as legacy
from arggnn import biological_candidate as frozen

HERE = Path(__file__).resolve().parent
checks = []
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def check(name, value):
    checks.append({'check': name, 'passed': bool(value)})
    if not value:
        raise AssertionError(name)


def reject(name, fn):
    try:
        fn()
    except (ValueError, AssertionError):
        check(name, True)
    else:
        check(name, False)


def inputs(full=False):
    n, q, r = (280, 184, 96) if full else (4, 2, 2)
    leaf_ids = torch.arange(n, dtype=torch.long) * 3 + 11
    qi = torch.arange(q - 1, -1, -1); ri = torch.arange(n - 1, q - 1, -1)
    tiles = []; records = []; proofs = []
    for i in range(8):
        tree_id = max(0, i - 1)
        generator = torch.Generator().manual_seed(101 + tree_id)
        features = torch.randn(1, n, 47, generator=generator) * .05
        response = (torch.eye(n) * 1.1 - torch.ones(n, n) * (.1 / n))[None]
        cache = NS(normalized_response=response, query_indices=qi, reference_indices=ri, query_ids=leaf_ids[qi], reference_ids=leaf_ids[ri])
        left, right = i / 8, (i + 1) / 8
        bank = NS(leaf_x=features, query_cache=cache, sample_ids=leaf_ids, query_node_ids=leaf_ids[qi], feature_columns=tuple(c for c in range(48) if c != 7), positions=torch.tensor([left + .03125, left + .09375], dtype=torch.float64), intervals=torch.tensor([[left, right]], dtype=torch.float64), target_graph=torch.zeros(2, dtype=torch.long), window_bounds=torch.tensor([left, right], dtype=torch.float64))
        layout = legacy._layout(bank, 32)
        vectors = torch.eye(q)[None, None].expand(1, 3, q, q).clone()
        values = torch.ones(1, 3, q)
        cqr = torch.full((1, 3, q, r), -.2 / r)
        plain = legacy.AnchorOperators(torch.diag_embed(values), cqr, values, vectors)
        age_values = values * torch.tensor([1., 1.2, 1.5])[None, :, None]
        age = legacy.AnchorOperators(torch.diag_embed(age_values), cqr * torch.tensor([1., 1.2, 1.5])[None, :, None, None], age_values, vectors)
        tiles.append(legacy.AnchorKernelBank(bank, layout, {'plain': plain, 'age': age}, 3))
        records.append({'tile_id': i, 'original_tree_ids': [tree_id]})
        proofs.append({'tile_id': i, 'original_tree_ids_verified': True, 'serialized_graph_reproduced': True, 'plain_bank_inputs_reproduced': True})
    return producer.VerifiedParentInputsV1(tuple(tiles), tuple(records), {'parent_id': 'bio_paths_pilot_train_000', 'split': 'train'}, 'bio_paths_pilot_train_000:' + '1' * 64, tuple(proofs), fixture=True)


report = producer.source_report_v1()
check('default source report opens no actual inputs', report['status'] == 'source_only' and report['actual_inputs_opened'] is False)
reject('truth input path rejected before IO', lambda: producer._input_path({'path': 'data/truth/secret.npz', 'sha256': '1' * 64}))
reject('TEST input path rejected before IO', lambda: producer._input_path({'path': 'data/bio_paths_pilot_test_000/observed/x.pt', 'sha256': '1' * 64}))
reject('oracle input path rejected before IO', lambda: producer._input_path({'path': 'data/oracle/x.trees', 'sha256': '1' * 64}))
reject('path escape rejected before IO', lambda: producer._input_path({'path': '../x.pt', 'sha256': '1' * 64}))
reject('missing digest rejected', lambda: producer._input_path({'path': 'configs/x.json'}))
full = inputs(True)
for geometry in ('age', 'plain'):
    assembled = producer.assemble_parent_v1(full, geometry)
    check(geometry + ' handmade280/Q184/R96/47 dimensions', assembled.leaf.shape == (8, 280, 47) and assembled.response.shape == (8, 280, 280) and len(assembled.parent.query_ids) == 184 and len(assembled.parent.reference_ids) == 96)
    check(geometry + ' exact256 anchor union', torch.equal(assembled.parent.layout.anchors, torch.cat([t.layout.anchors for t in full.loaded_tiles])) and len(assembled.parent.layout.anchors) == 256)
    check(geometry + ' exact256 weight union', torch.equal(assembled.parent.layout.weights, torch.cat([t.layout.weights for t in full.loaded_tiles])))
    check(geometry + ' full repeated tree occurrences kept', assembled.parent.original_tree_ids_by_cache_graph == (0, 0, 1, 2, 3, 4, 5, 6) and len(assembled.parent.operators.eigenvalues) == 8)
    check(geometry + ' explicit leaf order preserved', torch.equal(assembled.leaf_ids, full.loaded_tiles[0].bank.sample_ids))
    check(geometry + ' fixed CPU inputs', not assembled.leaf.requires_grad and not assembled.response.requires_grad and assembled.leaf.device.type == 'cpu')
age = producer.assemble_parent_v1(full, 'age'); plain = producer.assemble_parent_v1(full, 'plain')
check('age/plain encoder input identical', torch.equal(age.leaf, plain.leaf) and torch.equal(age.response, plain.response))
check('age/plain cached geometry differs', not torch.equal(age.parent.operators.cqq, plain.parent.operators.cqq))
# Mutation checks operate on independent handmade values.
bad = inputs(); bad.loaded_tiles[1].bank.leaf_x[0, 0, 0] += 1
reject('repeated-tree feature mutation rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad.loaded_tiles[1].bank.query_cache.normalized_response[0, 0, 0] += 1
reject('repeated-tree response mutation rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad.loaded_tiles[0].bank.query_cache.query_ids = bad.loaded_tiles[0].bank.query_cache.query_ids.flip(0)
reject('explicit sampled-ID join mismatch rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad.loaded_tiles[0].bank.feature_columns = tuple(range(47))
reject('changed47 feature columns rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad.loaded_tiles[0].bank.leaf_x.requires_grad_()
reject('learned input bank rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad = dataclasses.replace(bad, loaded_tiles=bad.loaded_tiles[:-1])
reject('partial parent rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
bad = inputs(); bad.source_checks[0]['plain_bank_inputs_reproduced'] = False
reject('missing original producer proof rejected', lambda: producer.assemble_parent_v1(bad, 'plain'))
reject('reduced handmade bank cannot masquerade as production', lambda: producer.assemble_parent_v1(dataclasses.replace(inputs(), fixture=False), 'plain'))
# Explicit original tree inventory on handmade native trees, including a crossing cut.
metadata = {'coverage': [0, 2000000], 'nodes': list(range(280))}
breaks = [0, 400000, 650000, 900000, 1150000, 1400000, 1650000, 2000000]
native_trees = [NS(index=i, num_roots=1, leaves=lambda: iter(range(280)), interval=NS(left=breaks[i], right=breaks[i+1])) for i in range(7)]
ts = NS(num_trees=7, sequence_length=2000000, samples=lambda: range(280), trees=lambda: iter(native_trees))
record = {'original_trees_preserved': 7, 'tiles': [{'physical_interval_bp0': [250000*i, 250000*(i+1)], 'original_tree_ids': [t.index for t in native_trees if max(t.interval.left, 250000*i) < min(t.interval.right, 250000*(i+1))]} for i in range(8)]}
check('explicit native inventory preserves crossing IDs', producer.verify_native_inventory_v1(ts, record, metadata)[0] == (0, 0, 400000))
bad = copy.deepcopy(record); bad['tiles'][1]['original_tree_ids'] = [1]
reject('approximate or swapped native tree IDs rejected', lambda: producer.verify_native_inventory_v1(ts, bad, metadata))
bad_ts = NS(**vars(ts)); bad_ts.num_trees = 6
reject('native inventory count drift rejected', lambda: producer.verify_native_inventory_v1(bad_ts, record, metadata))
# Exact original model/optimizer and connected reset/carry numerical integration.
small = inputs(); assemblies = {g: producer.assemble_parent_v1(small, g) for g in ('age', 'plain')}
metrics = []
for geometry in ('age', 'plain'):
    model, optimizer = producer.original_model_optimizer_v1(geometry, 101)
    check(geometry + ' unchanged original architecture', (model.input_dim, model.hidden_dim, model.num_layers, model.kernel_depth, model.latent_dim, model.num_scales, model.num_anchors) == (47, 32, 2, 2, 8, 3, 32) and model.coupling == 'full' and model.rate_parameterization == 'log')
    rate_ids = {id(p) for p in model.rate_and_mixture_parameters()}
    check(geometry + ' original AdamW grouping', isinstance(optimizer, torch.optim.AdamW) and [g['lr'] for g in optimizer.param_groups] == [.001, .001] and [g['weight_decay'] for g in optimizer.param_groups] == [.0001, 0.] and {id(p) for p in optimizer.param_groups[1]['params']} == rate_ids and len({id(p) for group in optimizer.param_groups for p in group['params']}) == len(list(model.parameters())))
    reset, _ = producer.forward_parent_v1(assemblies[geometry], model, mode='reset', checkpoint_blocks=False)
    clean = torch.cat([model(t) for t in small.loaded_tiles])
    maxdiff = float((reset - clean).abs().max().detach())
    check(geometry + ' reset equals eight unchanged full tile references', torch.allclose(reset, clean, atol=2e-5, rtol=2e-4))
    coeff = torch.linspace(-.2, .3, reset.numel()).reshape_as(reset)
    got = torch.autograd.grad((reset * coeff).sum(), tuple(model.parameters()), retain_graph=True)
    expected = torch.autograd.grad((clean * coeff).sum(), tuple(model.parameters()))
    check(geometry + ' reset complete original parameter VJP parity', all(torch.allclose(x, y, atol=1e-5, rtol=.001) for x, y in zip(got, expected)))
    carry, context = producer.forward_parent_v1(assemblies[geometry], model, mode='carry', checkpoint_blocks=True)
    check(geometry + ' full-support carry stays connected', carry.shape == reset.shape and carry.requires_grad and torch.isfinite(carry).all() and len(context.layers) == 2 and context.anchors.shape[0] == 256)
    clean_model = copy.deepcopy(model); clean_optimizer = torch.optim.AdamW(frozen.optimizer_groups(clean_model, json.loads((producer.ROOT/'configs/biological_candidate_pilot_v1.json').read_text())), lr=.001)
    def callback():
        output, _ = producer.forward_parent_v1(assemblies[geometry], model, mode='carry', checkpoint_blocks=True)
        return (output * coeff).sum(), 10.
    step = core.matched_parent_step_v3(optimizer, tuple(model.parameters()), [callback])
    clean_optimizer.zero_grad(set_to_none=True)
    output, _ = producer.forward_parent_v1(assemblies[geometry], clean_model, mode='carry', checkpoint_blocks=False)
    loss = (output * coeff).sum() / 10.; loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(clean_model.parameters(), 1., error_if_nonfinite=True); clean_optimizer.step()
    check(geometry + ' single aggregate backward/clip/update', (step['backward_calls'], step['clip_calls'], step['optimizer_steps']) == (1, 1, 1))
    check(geometry + ' carry checkpoint full parameter gradients', all(p.grad is not None and torch.isfinite(p.grad).all() and torch.allclose(p.grad, ref.grad, atol=1e-5, rtol=.001) for p, ref in zip(model.parameters(), clean_model.parameters())))
    check(geometry + ' carry AdamW clean parameter update', all(torch.allclose(p, ref, atol=1e-7, rtol=1e-5) for p, ref in zip(model.parameters(), clean_model.parameters())))
    got_state = list(optimizer.state.values()); expected_state = list(clean_optimizer.state.values())
    check(geometry + ' carry complete AdamW state', len(got_state) == len(expected_state) == len(list(model.parameters())) and all(set(x) == set(y) and all(torch.allclose(x[k], y[k], atol=1e-7, rtol=1e-5) for k in x) for x, y in zip(got_state, expected_state)))
    metrics.append({'geometry': geometry, 'handmade_leaf_count': 4, 'anchors': 256, 'reset_output_max_absolute_difference': maxdiff, 'carry_loss': step['loss'], 'carry_norm_before_single_clip': step['gradient_norm_before_single_clip']})
# Callback contamination reaches the sealed V3 guard through an original model.
model, optimizer = producer.original_model_optimizer_v1('plain', 101)
def contaminated():
    output, _ = producer.forward_parent_v1(assemblies['plain'], model, mode='carry', checkpoint_blocks=False)
    next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
    return output.sum(), 1.
reject('producer integration preserves callback gradient rejection', lambda: core.matched_parent_step_v3(optimizer, tuple(model.parameters()), [contaminated]))
result = {'kind': 'anchor_parent_producer_v1_handmade_CPU_checks', 'status': 'passed', 'checks': checks, 'check_count': len(checks), 'metrics': metrics, 'actual_dataset_opened': False, 'truth_opened': False, 'GPU_used': False, 'model_fitting': False, 'disposable_handmade_optimizer_steps': 4, 'sources_modified_by_checks': False}
(HERE/'handmade_checks.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'metrics':metrics},sort_keys=True))
