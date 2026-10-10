"""Handmade partitions/logits and tskit tables only; no simulation or data reads."""
from dataclasses import replace

import msprime
import numpy as np
import pytest
import tskit

from arggnn.dense_tract_metrics import (
    DenseGrid, Tract, TractPartition, coverage_clipped_midpoint_grid,
    dense_tract_metrics, extract_census_tracts,
)


def partition(rows, *, queries=("q0",), domain=(0., 100000.), classes=3):
    return TractPartition(tuple(queries), domain, classes,
                          tuple(tuple(Tract(*t) for t in row) for row in rows))


def grid(*, coverage=(0., 100000.), domain=(0., 100000.), n=10):
    return coverage_clipped_midpoint_grid(domain_bp=domain, coverage_bp=coverage,
                                         grid_points=n, grid_id="handmade-global-grid")


def logits_for(classes, k=3):
    classes = np.asarray(classes, dtype=int)
    if classes.ndim == 1:
        classes = classes[:, None]
    z = np.zeros(classes.shape+(k,), dtype=np.float64)
    np.put_along_axis(z, classes[..., None], 5., axis=-1)
    return z


def score(z, truth, g=None, **kwargs):
    return dense_tract_metrics(z, query_ids=truth.query_ids, grid=g or grid(),
                              true_partitions=truth, artificial_tile_edges_bp=[],
                              **kwargs)


def handmade_census():
    """q0: 0->0->1, q1: 1->0; extra topology change at 20."""
    tables = tskit.TableCollection(100.)
    for _ in range(2):
        tables.populations.add_row()
    q0 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    q1 = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    roots = [tables.nodes.add_row(time=1000), tables.nodes.add_row(time=1100)]
    intervals = [
        (q0, [(0., 20., 0), (20., 40., 0), (40., 100., 1)]),
        (q1, [(0., 70., 1), (70., 100., 0)]),
    ]
    for q, pieces in intervals:
        for left, right, population in pieces:
            census = tables.nodes.add_row(flags=msprime.NODE_IS_CEN_EVENT,
                                           time=350, population=population)
            tables.edges.add_row(left, right, census, q)
            for rl, rr, root in ((0., 20., roots[0]), (20., 100., roots[1])):
                l, r = max(left, rl), min(right, rr)
                if l < r:
                    tables.edges.add_row(l, r, root, census)
    tables.sort()
    return tables.tree_sequence(), (q0, q1)


def extract(ts, nodes, **kwargs):
    defaults = dict(query_ids=tuple(f"original-{q}" for q in nodes),
                    source_population_to_class={0: 0, 1: 1}, census_time=350,
                    num_classes=3)
    defaults.update(kwargs)
    return extract_census_tracts(ts, nodes, **defaults)


def test_exact_census_merges_topology_boundaries_and_preserves_explicit_order_mapping():
    ts, (q0, q1) = handmade_census()
    result = extract(ts, [q1, q0], query_ids=("observed-query-B", "observed-query-A"))
    assert ts.num_trees == 4
    assert result.query_ids == ("observed-query-B", "observed-query-A")
    assert result.tracts[0] == (Tract(0, 70, 1), Tract(70, 100, 0))
    assert result.tracts[1] == (Tract(0, 40, 0), Tract(40, 100, 1))
    g = grid(domain=(0, 100), coverage=(0, 100), n=10)
    predicted = np.column_stack((np.array(g.positions_bp) < 70,
                                np.array(g.positions_bp) >= 40)).astype(int)
    report = score(logits_for(predicted), result, g)
    assert [q["query_id"] for q in report["per_query"]] == list(result.query_ids)
    assert report["per_query"][0]["visible_true_boundary_positions_bp"] == [70.]
    assert report["per_query"][1]["visible_true_boundary_positions_bp"] == [40.]
    assert report["boundaries"]["matched"] == 2


@pytest.mark.parametrize("mode", ["missing", "ambiguous", "wrong_age"])
def test_missing_ambiguous_or_invalid_census_ancestor_rejected(mode):
    ts, nodes = handmade_census()
    tables = ts.dump_tables()
    if mode == "missing":
        flags = tables.nodes.flags.copy()
        flags[tables.nodes.time == 350] = 0
        tables.nodes.flags = flags
    elif mode == "ambiguous":
        flags = tables.nodes.flags.copy()
        flags[tables.nodes.time > 350] = msprime.NODE_IS_CEN_EVENT
        tables.nodes.flags = flags
    else:
        times = tables.nodes.time.copy()
        times[times == 350] = 349
        tables.nodes.time = times
    with pytest.raises(ValueError, match={"missing": "missing", "ambiguous": "ambiguous",
                                        "wrong_age": "age"}[mode]):
        extract(tables.tree_sequence(), nodes)


def test_census_nodes_queries_map_and_time_contract():
    ts, nodes = handmade_census()
    for bad_nodes in ([nodes[0], nodes[0]], [2], [0.5], [True]):
        with pytest.raises(ValueError):
            extract(ts, bad_nodes)
    with pytest.raises(ValueError, match="aligned"):
        extract(ts, nodes, query_ids=("only-one",))


def test_census_invalid_map_and_census_time_rejected():
    ts, nodes = handmade_census()
    for mapping in ({}, {0: 0}, {0: -1, 1: 1}, {0: 0., 1: 1},
                    {0: 0, 1: 3}, {2: 0}, {True: 0}):
        with pytest.raises(ValueError):
            extract(ts, nodes, source_population_to_class=mapping)
    for census in (0., -1., np.nan, np.inf, True, "350"):
        with pytest.raises(ValueError):
            extract(ts, nodes, census_time=census)
    with pytest.raises(ValueError, match="age"):
        extract(ts, nodes, census_time=349)
    # A sample may be valid in the tree but too old for the requested census.
    tables = ts.dump_tables()
    times = tables.nodes.time.copy()
    times[nodes[0]] = 340.
    tables.nodes.time = times
    with pytest.raises(ValueError, match="younger"):
        extract(tables.tree_sequence(), nodes, census_time=330)


def test_default_global_grid_is_4096_and_identity_is_deterministic():
    g = coverage_clipped_midpoint_grid(domain_bp=(0, 1000000), coverage_bp=(100, 999900),
                                         grid_id="original-dense4096")
    assert g.grid_points == 4096 and len(g.positions_bp) == 4096
    assert g.original_indices == tuple(range(4096))
    assert g.positions_bp[0] == pytest.approx(1000000/8192)
    assert len(g.identity_record()["sha256"]) == 64
    assert g.identity_record() == g.identity_record()


@pytest.mark.parametrize("mutation", [
    {"original_indices": (0, 2, 3)},
    {"original_indices": (3, 2, 1, 0)},
    {"original_indices": (0, 1, 2, 2)},
    {"original_indices": (0., 1., 2., 3.)},
    {"original_indices": (True, False, True, False)},
    {"positions_bp": (12500., 37500., 62500.1, 87500.)},
    {"positions_bp": (12500., 37500., np.nan, 87500.)},
    {"positions_bp": (12500., 37500., 62500., 87500., 90000.)},
    {"grid_id": ""},
    {"coverage_bp": (0., 110000.)},
    {"coverage_bp": (50000., 50000.)},
    {"coverage_bp": ((0., 40000.), (60000., 100000.))},
    {"domain_bp": (0., np.inf)},
    {"domain_bp": (-100000., 100000.)},
    {"grid_points": True},
    {"grid_points": 0},
])
def test_invalid_missing_gapped_nonuniform_or_reordered_grid_identity_rejected(mutation):
    base = grid(n=4)
    with pytest.raises(ValueError):
        replace(base, **mutation)


def test_original_grid_coordinates_cannot_be_regenerated_within_coverage():
    g = grid(n=4, coverage=(20000, 80000))
    assert g.original_indices == (1, 2)
    assert g.positions_bp == (37500., 62500.)
    with pytest.raises(ValueError):
        replace(g, positions_bp=(35000., 65000.))
    with pytest.raises(ValueError):
        replace(g, original_indices=(0, 1))


def test_partition_rejects_invalid_class_slots_gaps_and_unmerged_equal_labels():
    invalid_rows = [
        [(0, 50000, 0), (51000, 100000, 1)],
        [(0, 50000, 0), (50000, 100000, 0)],
        [(0, 100000, -1)], [(0, 100000, 3)],
        [(0, 100000, 0.)], [(0, 100000, True)], [(1, 100000, 0)],
        [(0, 100001, 0)], [(0, np.nan, 0)],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError):
            partition([row])
    with pytest.raises(ValueError):
        partition([[(0, 100000, 0)]], queries=("q0", "q1"))
    with pytest.raises(ValueError):
        partition([[(0, 100000, 0)]], classes=1)


def test_truth_exact_switches_clipping_censoring_without_coverage_edge_switches():
    g = grid(coverage=(12000, 90000))
    truth = partition([[(0, 10000, 0), (10000, 20000, 1),
                        (20000, 80000, 0), (80000, 95000, 1), (95000, 100000, 0)]])
    z = logits_for([1, 0, 0, 0, 0, 0, 0, 1])
    result = score(z, truth, g)
    query = result["per_query"][0]
    assert query["visible_true_boundary_positions_bp"] == [20000., 80000.]
    assert query["predicted_boundary_positions_bp"] == [20000., 80000.]
    assert query["boundaries"]["matched"] == 2
    assert query["boundaries"]["true_boundaries"] == 2
    tracts = query["true_tracts"]
    assert [(t["left_bp"], t["right_bp"]) for t in tracts] == [
        (12000., 20000.), (20000., 80000.), (80000., 90000.)]
    assert tracts[0]["original_length_bp"] == 10000
    assert tracts[0]["coverage_truncated_left"] and tracts[0]["coverage_censored_left"]
    assert tracts[-1]["coverage_truncated_right"] and tracts[-1]["right_censored"]
    assert tracts[1]["full_observed"]
    assert query["predicted_tracts"][0]["left_censored"]
    assert query["predicted_tracts"][-1]["right_censored"]
    assert result["quadrature_represented_physical_bp"] == 78000.


def test_true_boundary_exactly_at_coverage_edge_is_censored_and_not_visible():
    g = grid(coverage=(20000, 80000))
    truth = partition([[(0, 20000, 0), (20000, 80000, 1), (80000, 100000, 0)]])
    query = score(logits_for([1]*6), truth, g)["per_query"][0]
    assert query["visible_true_boundary_positions_bp"] == []
    tract = query["true_tracts"][0]
    assert not tract["coverage_truncated_left"] and not tract["coverage_truncated_right"]
    assert tract["coverage_censored_left"] and tract["coverage_censored_right"]
    assert not tract["full_observed"]


def test_exact_length_errors_require_two_matched_switches_and_same_class():
    truth = partition([[(0, 31000, 0), (31000, 40000, 1), (40000, 100000, 0)]])
    query = score(logits_for([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), truth)["per_query"][0]
    assert query["visible_true_boundary_positions_bp"] == [31000., 40000.]
    assert query["predicted_boundary_positions_bp"] == [30000., 40000.]
    assert query["boundaries"]["matched"] == 2
    assert query["tract_length_errors"]["matched_length_support"] == 1
    assert query["tract_length_errors"]["signed_errors_bp"] == [1000.]
    assert query["per_class"][1]["reconstructed_class_span_errors"]["signed_class_span_error_bp"] == 1000.
    assert query["reconstructed_tract_overlap"]["accuracy"] == pytest.approx(.99)
    assert query["span_quadrature"]["accuracy"] == 1.


@pytest.mark.parametrize("right,supported,span", [(35000., False, 0.), (36000., True, 6000.),
                                                (40000., True, 10000.)])
def test_short_full_true_tracts_have_counts_bp_and_zero_midpoint_support(right, supported, span):
    truth = partition([[(0, 30000, 0), (30000, right, 1), (right, 100000, 0)]])
    predictions = np.zeros(10, dtype=int)
    if supported:
        predictions[3] = 1
    short = score(logits_for(predictions), truth)["per_query"][0]["short_true_tracts"]
    assert short["true_tract_count"] == 1
    assert short["true_tract_bp"] == right-30000
    assert short["midpoint_supported_tracts"] == int(supported)
    assert short["zero_midpoint_tracts"] == int(not supported)
    assert short["evaluated_quadrature_bp"] == span
    assert short["readout"]["accuracy"] == (1. if supported else None)
    if supported:
        assert short["readout"]["nll"] is not None
    else:
        assert short["readout"]["nll"] is None
    assert short["per_class"][0]["true_tract_count"] == 0  # domain-censored flanks excluded
    assert short["per_class"][1]["true_tract_count"] == 1
    assert short["per_class"][2]["readout"]["accuracy"] is None


def test_over_cutoff_and_coverage_or_domain_censored_short_tracts_excluded():
    truth = partition([[(0, 1000, 1), (1000, 30000, 0), (30000, 40001, 1),
                        (40001, 99000, 0), (99000, 100000, 1)]])
    result = score(logits_for([0]*10), truth)
    short = result["per_query"][0]["short_true_tracts"]
    assert short["true_tract_count"] == 0
    assert short["readout"]["accuracy"] is None
    cutoff_truth = partition([[(0, 30000, 0), (30000, 36000, 1), (36000, 100000, 0)]])
    clipped = score(logits_for([1]+[0]*6), cutoff_truth, grid(coverage=(32000, 100000)))
    assert clipped["per_query"][0]["short_true_tracts"]["true_tract_count"] == 0


def test_readout_and_boundaries_near_artificial_tile_edges_versus_interior():
    truth = partition([[(0, 10000, 0), (10000, 40000, 1), (40000, 100000, 0)]])
    result = dense_tract_metrics(logits_for([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        query_ids=truth.query_ids, grid=grid(), true_partitions=truth,
        artificial_tile_edges_bp=[50000])
    query = result["per_query"][0]
    near, interior = (query["readout_location_groups"][key] for key in ("near_tile_edges", "interior"))
    assert near["evaluated_midpoints"] == 2 and near["denominator_bp"] == 20000.
    assert interior["evaluated_midpoints"] == 8 and interior["denominator_bp"] == 80000.
    group = query["boundary_location_groups"]
    assert group["near_tile_edges"]["true_boundary_support"] == 1  # 40kb is inclusive
    assert group["interior"]["true_boundary_support"] == 1
    assert group["near_tile_edges"]["matched_true_boundaries"] == 1
    assert group["interior"]["matched_true_boundaries"] == 1
    assert group["cross_group_matches"] == 0
    assert near["per_class"][2]["accuracy"] is None


def test_boundary_matches_crossing_tile_group_do_not_get_rematched_or_hidden():
    truth = partition([[(0, 39900, 0), (39900, 100000, 1)]])
    result = dense_tract_metrics(logits_for([0]*4+[1]*6), query_ids=truth.query_ids,
        grid=grid(), true_partitions=truth, artificial_tile_edges_bp=[50000])
    group = result["per_query"][0]["boundary_location_groups"]
    assert group["cross_group_matches"] == 1
    assert group["interior"]["true_boundary_support"] == 1
    assert group["interior"]["matched_true_boundaries"] == 1
    assert group["near_tile_edges"]["predicted_boundary_support"] == 1
    assert group["near_tile_edges"]["matched_predicted_boundaries"] == 1


def test_unequal_clipped_original_cells_control_quadrature_denominator_and_accuracy():
    g = grid(n=4, coverage=(10000, 80000))
    truth = partition([[(0, 100000, 0)]])
    z = np.array([[[2., 0., 0.]], [[0., 2., 0.]], [[0., 2., 0.]]])
    result = score(z, truth, g)
    assert result["quadrature_cell_bounds_bp"] == [[10000., 25000.], [25000., 50000.],
                                                 [50000., 75000.]]
    assert result["physical_coverage_bp"] == 70000.
    assert result["quadrature_represented_physical_bp"] == 65000.
    assert result["quadrature_unrepresented_physical_bp"] == 5000.
    readout = result["span_quadrature"]
    correct_loss = np.log(1+2*np.exp(-2))
    assert readout["accuracy"] == pytest.approx(15000/65000)
    assert readout["nll"] == pytest.approx(correct_loss+50000/65000*2)
    assert readout["denominator_bp"] == 65000.
    assert readout["per_class"][0]["true_support_bp"] == 65000.
    assert readout["per_class"][2]["accuracy"] is None
    assert readout["per_class"][2]["f1"] is None


def test_unsampled_coverage_edge_strips_never_gain_interpolated_quadrature_weight():
    g = grid(n=4, coverage=(20000, 80000))
    truth = partition([[(0, 100000, 0)]])
    result = score(logits_for([0, 0]), truth, g)
    assert result["quadrature_represented_physical_bp"] == 50000.
    assert result["quadrature_unrepresented_physical_bp"] == 10000.
    assert result["per_query"][0]["reconstructed_tract_overlap"]["denominator_bp"] == 60000.
    assert result["per_query"][0]["predicted_tracts"][0]["length_bp"] == 60000.
    assert result["span_quadrature"]["denominator_bp"] == 50000.


def test_exact_true_change_inside_a_cell_does_not_claim_integrated_nll_or_accuracy():
    g = grid(n=2)
    truth = partition([[(0, 10000, 1), (10000, 100000, 0)]])
    query = score(logits_for([0, 0]), truth, g)["per_query"][0]
    assert query["span_quadrature"]["accuracy"] == 1.
    assert query["reconstructed_tract_overlap"]["accuracy"] == .9
    assert query["span_quadrature"]["per_class"][1]["true_support_bp"] == 0.
    assert query["reconstructed_tract_overlap"]["per_class"][1]["true_support_bp"] == 10000.


def test_empty_retained_grid_has_explicit_na_and_no_fabricated_prediction_tract():
    g = grid(n=4, coverage=(0, 1000))
    truth = partition([[(0, 500, 0), (500, 100000, 1)]])
    result = score(np.empty((0, 1, 3)), truth, g)
    assert result["retained_midpoints"] == 0
    assert result["quadrature_represented_physical_bp"] == 0.
    assert result["quadrature_unrepresented_physical_bp"] == 1000.
    assert result["span_quadrature"]["nll"] is None
    query = result["per_query"][0]
    assert query["span_quadrature"]["accuracy"] is None
    assert query["predicted_tracts"] == []
    assert query["boundaries"]["missed"] == 1
    assert query["reconstructed_tract_overlap"]["accuracy"] is None
    assert query["per_class"][0]["reconstructed_class_span_errors"]["true_visible_support_bp"] == 500.
    assert query["per_class"][0]["reconstructed_class_span_errors"]["signed_class_span_error_bp"] is None


def test_raw_extreme_logits_have_uncapped_stable_nll_without_probability_floor():
    truth = partition([[(0, 100000, 1)]])
    result = score(np.array([[[1000., -1000., 0.]]]), truth, grid(n=1))
    assert result["span_quadrature"]["nll"] == 2000.
    assert result["span_quadrature"]["accuracy"] == 0.
    assert result["per_query"][0]["true_tracts"][0]["domain_censored_left"]
    assert result["per_query"][0]["true_tracts"][0]["domain_censored_right"]
    assert result["span_quadrature"]["nll"] > -np.log(1e-12)
    shifted = score(np.array([[[1000000., 998000., 999000.]]]), truth, grid(n=1))
    assert shifted["span_quadrature"]["nll"] == 2000.
    with pytest.raises(ValueError, match="representable"):
        score(np.array([[[1e308, -1e308, 0.]]]), truth, grid(n=1))


def test_no_true_switches_retains_visible_false_boundaries_and_switch_cost():
    truth = partition([[(0, 100000, 0)]])
    query = score(logits_for([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]), truth)["per_query"][0]
    match = query["boundaries"]
    assert match["true_boundaries"] == 0 and match["predicted_boundaries"] == 2
    assert match["extra"] == 2 and match["absolute_switch_count_error"] == 2
    assert match["recall"] is None and match["f1"] is None
    assert match["precision"] == 0.
    incidences = [c["boundary_incidence"]["predicted_boundaries"] for c in query["per_class"]]
    assert incidences == [2, 2, 0]  # each switch has two class incidences
    assert query["per_class"][2]["boundary_incidence"]["precision"] is None


def test_one_to_one_matching_reports_short_unresolved_switch_as_missed():
    truth = partition([[(0, 30000, 0), (30000, 30500, 1), (30500, 100000, 0)]])
    result = score(logits_for([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), truth)
    match = result["per_query"][0]["boundaries"]
    assert match["true_boundaries"] == 2 and match["predicted_boundaries"] == 1
    assert match["matched"] == 1 and match["missed"] == 1
    assert match["absolute_switch_count_error"] == 1
    short = result["per_query"][0]["short_true_tracts"]
    assert short["true_tract_count"] == 1 and short["zero_midpoint_tracts"] == 1


def test_query_order_domains_shapes_nonfinite_logits_and_tile_identity_rejected():
    truth = partition([[(0, 100000, 0)], [(0, 100000, 1)]], queries=("a", "b"))
    z = logits_for([[0, 1]]*10)
    kwargs = dict(query_ids=truth.query_ids, grid=grid(), true_partitions=truth,
                  artificial_tile_edges_bp=[])
    with pytest.raises(ValueError, match="order"):
        dense_tract_metrics(z, **dict(kwargs, query_ids=("b", "a")))
    for bad in (z[:, :1], z[:, :, :2], z[:-1], np.full_like(z, np.nan),
                np.full_like(z, np.inf)):
        with pytest.raises(ValueError):
            dense_tract_metrics(bad, **kwargs)
    with pytest.raises(ValueError, match="domains"):
        dense_tract_metrics(z, **dict(kwargs, grid=grid(domain=(0, 200000),
                                                       coverage=(0, 200000))))
    for edges in ([50000, 50000], [70000, 50000], [0], [100000], [np.nan]):
        with pytest.raises(ValueError):
            dense_tract_metrics(z, **dict(kwargs, artificial_tile_edges_bp=edges))
    for parameter in ("boundary_tolerance_bp", "short_tract_cutoff_bp", "tile_edge_radius_bp"):
        for bad in (-1., True, "1000"):
            with pytest.raises(ValueError):
                dense_tract_metrics(z, **kwargs, **{parameter: bad})


def test_aggregate_is_query_bp_only_with_clear_absent_class_support():
    truth = partition([[(0, 100000, 0)], [(0, 100000, 1)]], queries=("a", "b"))
    result = score(logits_for([[0, 1]]*10), truth)
    assert result["span_quadrature"]["denominator_bp"] == 200000.
    assert result["quadrature_represented_physical_bp"] == 100000.
    assert result["boundaries"]["query_support"] == 2
    assert result["per_class"][2]["span_quadrature"]["true_support_bp"] == 0.
    assert result["per_class"][2]["span_quadrature"]["recall"] is None
    assert "not independent parents" in result["policies"]["inference"]


def test_float64_unrepresentable_internal_grid_coordinates_are_rejected():
    with pytest.raises(ValueError, match="midpoint"):
        grid(domain=(1e16, 1e16+2), coverage=(1e16, 1e16+2), n=4)
    with pytest.raises(ValueError, match="midpoint"):
        grid(domain=(1e16, 1e16+2), coverage=(1e16, 1e16+2), n=1)


def test_grid_hash_binds_coverage_and_original_index_identity():
    a = grid(n=4, coverage=(10000, 80000))
    b = grid(n=4, coverage=(20000, 80000))
    assert a.identity_record()["sha256"] != b.identity_record()["sha256"]
    assert a.original_indices == (0, 1, 2)
    assert b.original_indices == (1, 2)
