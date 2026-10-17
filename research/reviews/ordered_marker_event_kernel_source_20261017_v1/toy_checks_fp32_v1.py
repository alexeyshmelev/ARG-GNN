"""Bounded hand-authored CPU/FP32 API checks; no scientific payload readers."""
from pathlib import Path
import hashlib
import json
import torch
from event_kernel_fp32_v1 import (ToyTree, F32, G, A, MISSING, pathlog,
    dense_log_fields, features_h058, signed_event_fields,
    positive_cover_fields, hybrid_fields, logsum)

torch.set_default_dtype(F32)
torch.set_num_threads(1)
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = Path(__file__).resolve().parent
PIN = '081acadf8f55adec0a5d16c6430c30fe5771ee4a307f5387d32138e1659a5e61'
REPORT_PIN = 'af7d46068175d00f7b733e5abe29855b356fe66bacb3c7d449b207184a3a3dc6'
source = ROOT/'research/reviews/biological_operator_next_fresh_20261016_v1/packed_csr_v2/same_marker_packed_csr_v2.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == PIN
assert hashlib.sha256((ROOT/'research/reviews/boundary_architecture_novelty_20261017_v1/REPORT.md').read_bytes()).hexdigest() == REPORT_PIN


def enumeration_oracle(tree, lg, gt):
    """Positive path enumeration, independent of dense tree recursion."""
    effective = lg.clamp(-16., -.0001)
    empty = lg.new_full((G, A), -torch.inf)
    rows = []
    for s in range(gt.shape[0]):
        queries = []
        for qc in tree.query_column:
            q, pieces = tree.sample_node[qc], []
            for ref in tree.references:
                r, a = tree.sample_node[ref], gt[s, tree.route_permutation[ref]].item()
                if a < 0 or tree.inside(r, q): continue
                mask = (torch.arange(G)[:, None] == tree.group(ref)) & (torch.arange(A)[None, :] == a)
                pieces.append(torch.where(mask, pathlog(tree, effective, q, r), empty))
            queries.append(logsum(pieces, empty))
        rows.append(torch.stack(queries))
    return torch.stack(rows)


def compare(a, b, atol, rtol):
    assert a.dtype == b.dtype == F32
    finite_equal = torch.equal(torch.isfinite(a), torch.isfinite(b))
    both = torch.isfinite(a) & torch.isfinite(b)
    maxabs = (a[both] - b[both]).abs().max().item() if bool(both.any()) else 0.
    return {'pass': bool(finite_equal and torch.allclose(a, b, atol=atol, rtol=rtol)),
            'same_finite_support': finite_equal, 'max_abs': maxabs, 'atol': atol, 'rtol': rtol}


def loss(features):
    weights = torch.linspace(.01, .03, features.numel(), dtype=F32).reshape_as(features)
    return (features * weights).sum()


def direction_check(tree, gt, logs, analytic):
    direction = torch.tensor([0.] + [(-1. if i % 2 else 1.) for i in range(1, tree.n)], dtype=F32)
    direction = direction / direction.norm().clamp_min(torch.finfo(F32).eps)
    eps = logs.new_tensor(.002)
    plus = loss(features_h058(tree, hybrid_fields(tree, logs + eps*direction, gt)[0], gt)[0])
    minus = loss(features_h058(tree, hybrid_fields(tree, logs - eps*direction, gt)[0], gt)[0])
    fd = (plus - minus)/(2*eps)
    ad = (analytic * direction).sum()
    error = (fd-ad).abs()
    meaningful = bool(ad.abs() > .00001)
    return {'analytic': ad.item(), 'finite_difference': fd.item(), 'abs_error': error.item(),
            'pass': bool(meaningful and error <= .001 + .02*ad.abs()),
            'meaningful_nonzero': meaningful, 'epsilon': eps.item()}


def symbols_for_route(tree, gt, original_symbols):
    result = []
    for s, labels in enumerate(original_symbols):
        labels = list(labels)
        for ref in tree.references:
            a = gt[s, tree.route_permutation[ref]].item()
            labels[tree.sample_node[ref]] = a if a >= 0 else MISSING
        result.append(tuple(labels))
    return tuple(result)


def run_case(name, tree, gt, symbols, logs, hard=False):
    logs = torch.tensor(logs, dtype=F32, requires_grad=True)
    dense = dense_log_fields(tree, logs, gt)
    enumerated = enumeration_oracle(tree, logs, gt)
    cover, pieces = positive_cover_fields(tree, logs, gt)
    hybrid, selection = hybrid_fields(tree, logs, gt)
    signed, absolute, events = signed_event_fields(tree, logs, gt, symbols)
    safe = torch.where(signed > 0, signed, torch.ones_like(signed))
    signed_log = torch.where(signed > 0, safe.log(), torch.full_like(signed, -torch.inf))
    fd, md = features_h058(tree, dense, gt)
    fc, mc = features_h058(tree, cover, gt)
    fh, mh = features_h058(tree, hybrid, gt)
    fs, _ = features_h058(tree, signed_log, gt)
    gd = torch.autograd.grad(loss(fd), logs, retain_graph=True)[0]
    gc = torch.autograd.grad(loss(fc), logs, retain_graph=True)[0]
    gh = torch.autograd.grad(loss(fh), logs)[0]
    result = {'name': name, 'nodes': tree.n, 'sites': gt.shape[0],
        'references': len(tree.references), 'queries': len(tree.query_column),
        'signed_event_counts': events, 'positive_cover_piece_counts': [len(x) for x in pieces],
        'enumeration_vs_dense_log': compare(enumerated, dense, .00004, .00002),
        'positive_cover_vs_dense_log': compare(cover, dense, .00004, .00002),
        'positive_cover_vs_dense_features': compare(fc, fd, .00004, .00002),
        'hybrid_vs_dense_features': compare(fh, fd, .00004, .00002),
        'positive_cover_vs_dense_gradients': compare(gc, gd, .00008, .0004),
        'hybrid_vs_dense_gradients': compare(gh, gd, .00008, .0004),
        'signed_vs_dense_features': compare(fs, fd, .00004, .00002),
        'signed_negative_entries': int((signed < 0).sum()),
        'signed_minimum': signed.min().item(),
        'signed_finite_support_matches': torch.equal(torch.isfinite(signed_log), torch.isfinite(dense)),
        'masks_match': torch.equal(md, mc) and torch.equal(md, mh),
        'finite_gradients': bool(torch.isfinite(gd).all() and torch.isfinite(gc).all() and torch.isfinite(gh).all()),
        'hybrid_selection': selection}
    if not hard:
        result['gate_direction_check'] = direction_check(tree, gt, logs.detach(), gh)
        assert result['gate_direction_check']['pass']
    checks = [v['pass'] for k,v in result.items() if isinstance(v,dict) and k not in ['signed_vs_dense_features','hybrid_selection'] and 'pass' in v]
    result['pass_stable_path'] = all(checks) and result['masks_match'] and result['finite_gradients']
    assert result['pass_stable_path'], result
    return result, fd.detach()


def main():
    results = []
    parent = tuple([-1] + [(v-1)//2 for v in range(1,31)] + [14])
    nodes = tuple(list(range(15,31)) + [3,1,31,0,2])
    classes = tuple([(i//2)%4 for i in range(16)] + [2,-1,-1,-1,-1])
    ages = tuple([10 if i<8 else 400 for i in range(16)] + [10,100,0,700,200])
    perm = tuple([i^1 for i in range(16)] + list(range(16,21)))
    tree = ToyTree(parent,nodes,classes,ages,(17,18,19,20),perm)
    rows = []
    labels = [0]*32
    for v in range(32):
        if tree.inside(v,1): labels[v]=1
        if tree.inside(v,3): labels[v]=0
        if tree.inside(v,7): labels[v]=2
        if tree.inside(v,8): labels[v]=3
    labels[16]=MISSING
    rows.append(tuple(labels))
    labels = [3]*32
    for v in range(1,32): labels[v] = 2 if tree.inside(v,1) else 1
    labels[3],labels[1]=0,MISSING
    rows.append(tuple(labels))
    labels = [0]*32
    for v,a in [(15,1),(17,2),(18,3),(29,MISSING),(30,MISSING),(31,MISSING)]: labels[v]=a
    rows.append(tuple(labels))
    labels = [1]*32; labels[1]=2
    rows.append(tuple(labels))
    gt = torch.tensor([[(-1 if labels[v]==MISSING else labels[v]) for v in nodes] for labels in rows],dtype=torch.int64)
    symbols = symbols_for_route(tree,gt,rows)
    normal_logs = [-.05-.007*v for v in range(32)]
    result, feature = run_case('multifurcation_internal_query_all_groups_recurrent_multiallelic_missing_route',tree,gt,symbols,normal_logs)
    # Root is an internally sampled query: zero outside support keeps valid=1.
    assert bool((feature[:,:,16:] >= 0).all())
    assert bool((feature[0,2,16:] == 1).all())
    assert bool((feature[0,2,:8] < 0).any())
    # A missing original query masks its complete feature vector.
    assert bool((feature[1,0] == 0).all()) and bool((feature[2,1] == 0).all())
    identity_tree = ToyTree(parent,nodes,classes,ages,(17,18,19,20),tuple(range(len(nodes))))
    lg = torch.tensor(normal_logs,dtype=F32)
    routed_dense = dense_log_fields(tree,lg,gt)
    explicitly_routed_dense = dense_log_fields(identity_tree,lg,gt[:,list(perm)])
    result['route_matches_explicit_reference_reassignment'] = compare(routed_dense,explicitly_routed_dense,.00004,.00002)
    assert result['route_matches_explicit_reference_reassignment']['pass']
    marker_permutation = [3, 1, 0, 2]
    permuted_gt = gt[marker_permutation]
    permuted_log, _ = positive_cover_fields(tree, lg, permuted_gt)
    permuted_feature, _ = features_h058(tree, permuted_log, permuted_gt)
    marker_covariance = compare(permuted_feature, feature[marker_permutation], .00004, .00002)
    mean_pool = compare(permuted_feature.mean(0), feature.mean(0), .00004, .00002)
    max_pool = compare(permuted_feature.amax(0), feature.amax(0), .00004, .00002)
    positions = torch.arange(1, gt.shape[0]+1, dtype=F32)[:, None, None]
    ordered_moment_change = ((positions*permuted_feature).sum(0)-(positions*feature).sum(0)).abs().max()
    result['ordered_marker_api'] = {'permutation_covariance':marker_covariance,
        'mean_pool_invariant':mean_pool,'max_pool_invariant':max_pool,
        'position_weighted_moment_changes':bool(ordered_moment_change > .0001),
        'position_weighted_moment_max_change':ordered_moment_change.item(),
        'scope':'API retains order; this is not a fitted encoder or architecture efficacy test'}
    assert marker_covariance['pass'] and mean_pool['pass'] and max_pool['pass']
    assert result['ordered_marker_api']['position_weighted_moment_changes']
    results.append(result)
    # Near-unit external mass hides a strongly attenuated sole called reference.
    t = ToyTree((-1,0,0,1,1,3,4,6),(2,5,7,3),(0,0,0,-1),(10,10,10,100),(3,),(0,1,2,3))
    labels = (0,1,MISSING,1,1,MISSING,1,1)
    gt = torch.tensor([[-1,-1,1,1]],dtype=torch.int64)
    result,_ = run_case('ancestor_subtraction_tiny_sole_called_outside_reference',t,gt,(labels,),[-.0001,-.0001,-.0001,-.0001,-16.,-.0001,-16.,-.0001],True)
    results.append(result)
    # Large missing-reference subtraction leaves a spurious zero-allele residual.
    t = ToyTree((-1,0,0,0),(2,3,1),(0,0,-1),(10,10,0),(2,),(0,1,2))
    labels = (0,1,MISSING,1)
    gt = torch.tensor([[-1,1,1]],dtype=torch.int64)
    result,_ = run_case('signed_allele_subtraction_against_dominant_missing_reference',t,gt,(labels,),[-.0001,-.0001,-.0001,-16.],True)
    results.append(result)
    # exp(-144) is below FP32 probability range, while log-domain evidence survives.
    t = ToyTree(tuple([-1]+list(range(9))),(0,9),(0,-1),(400,0),(1,),(0,1))
    gt = torch.tensor([[2,2]],dtype=torch.int64)
    result,_ = run_case('long_attenuation_probability_underflow_logdomain_survives',t,gt,(tuple([2]*10),),[-.0001]+[-16.]*9,True)
    results.append(result)
    summary = {'scope':'hand-authored synthetic CPU/FP32 API and algebra correctness only',
        'no_scientific_payloads':True,'no_optimizer_or_training':True,'no_cuda':True,
        'floating_tensor_dtype':'torch.float32','torch_version':torch.__version__,
        'actual_h058_source_sha256':PIN,'preserved_report_sha256':REPORT_PIN,
        'stable_cases_pass':all(x['pass_stable_path'] for x in results),
        'signed_feature_failures':[x['name'] for x in results if not x['signed_vs_dense_features']['pass']],
        'cases':results}
    (OUT/'toy_results.fp32.v1.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))
    for x in results:
        print(json.dumps({'name':x['name'],'stable_pass':x['pass_stable_path'],'signed_pass':x['signed_vs_dense_features']['pass'],
            'signed_feature_max_abs':x['signed_vs_dense_features']['max_abs'],
            'positive_feature_max_abs':x['positive_cover_vs_dense_features']['max_abs'],
            'gradient_max_abs':x['positive_cover_vs_dense_gradients']['max_abs'],
            'dense_fallback_sites':x['hybrid_selection']['dense_fallback_sites']}))

if __name__ == '__main__': main()
