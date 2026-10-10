"""Handmade observed/truth trees, source-only checks; no author pipeline/model run."""
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import msprime
import numpy as np
import pytest
import tskit

from arggnn.ancestralpaths_native_bridge import (
    AUTHOR_FILE_SHA256, AUTHOR_POPULATION_ORDER, COHORT_FEATURE_ORDER,
    PATH_TO_SOURCE_CLASS, ObservedPanel, build_native_keras_model,
    collapse_native_probabilities, derive_native_path_mapping,
    export_native_predictions, extract_native_features, inspect_author_source,
    native_accuracy_confusion, native_model_spec, parse_observed_panel,
    paths_from_census_populations, predict_native_probabilities,
    truth_paths_at_positions, validate_parent_partition,
)

SOURCE = Path(".runtime/author_sources/ancestralpaths_77db3cfae100")


def ordered_panel():
    text = "sample population group sex\n"
    for i, name in enumerate(("present_day",) + COHORT_FEATURE_ORDER):
        text += f"d{i} {name} {name} NA\n"
    return parse_observed_panel(text)


def chain_tree():
    """First ancestor has no ancient leaves; next8 shells have disjoint cohorts."""
    tables = tskit.TableCollection(100)
    for _ in range(18):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    child = tables.nodes.add_row(time=1.9)
    for leaf in (0, 1):
        tables.edges.add_row(0, 100, child, leaf)
    for group in range(8):
        pair = tables.nodes.add_row(time=2.3)
        for leaf in (2 + 2 * group, 3 + 2 * group):
            tables.edges.add_row(0, 100, pair, leaf)
        ancestor = tables.nodes.add_row(time=10.9 * (group + 1))
        for lower in (child, pair):
            tables.edges.add_row(0, 100, ancestor, lower)
        child = ancestor
    tables.sort()
    return tables.tree_sequence().first()


def star_tree():
    tables = tskit.TableCollection(100)
    for _ in range(6):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(time=120.9)
    for leaf in range(6):
        tables.edges.add_row(0, 100, root, leaf)
    tables.sort()
    text = "sample population group sex\np present_day present_day NA\nb Bronze_Age Bronze_Age NA\na BAA BAA NA\n"
    return tables.tree_sequence().first(), parse_observed_panel(text)


def truth_fixture(*, missing=False, wrong_time=False):
    """Two marginal truth trees; same topology after census collapse."""
    tables = tskit.TableCollection(100)
    tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
    for name in AUTHOR_POPULATION_ORDER:
        tables.populations.add_row(metadata={"name": name})
    for _ in range(6):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
    root = tables.nodes.add_row(time=1000, population=10)
    for left, right, first_pop in ((0, 50, 8), (50, 100, 7)):
        for leaf in range(6):
            pop = first_pop if leaf == 0 else (7 if leaf in (4, 5) else 8)
            census = tables.nodes.add_row(time=351 if wrong_time else 350,
                                         flags=0 if missing else msprime.NODE_IS_CEN_EVENT,
                                         population=pop)
            tables.edges.add_row(left, right, census, leaf)
            tables.edges.add_row(left, right, root, census)
    tables.sort()
    text = "sample population group sex\np present_day present_day NA\nb Bronze_Age Bronze_Age NA\na BAA BAA NA\n"
    return tables.tree_sequence(), parse_observed_panel(text)


def test_source_pin_and_published_equivalence():
    r = inspect_author_source(
        SOURCE, archive=".runtime/author_sources/ancestralpaths_77db3cfae100.tar.gz",
        published_root=".runtime/author_sources/ancestralpaths_published_zenodo8319452",
        published_archive=".runtime/author_sources/ancestralpaths_published_zenodo8319452.zip")
    assert r["published_source_byte_equivalence_checked"]
    assert r["overlapping_parents"] == [5, 6]
    assert not r["upstream_code_executed"] and not r["scientific_reproduction_verified"]
    assert not r["runtime_compatibility_verified"] and not r["code_licence_file_present"]
    assert r["source_sha256"] == AUTHOR_FILE_SHA256


def test_mapping_traced_in_source_not_class_name_guess():
    r = derive_native_path_mapping(SOURCE)
    assert [(row["path_1based"], row["query_cohort_case"], row["census_population_id"],
             row["census_population"], row["source_class_0based"])
            for row in r["paths"]] == [
                (1, "nonBAA", 8, "Ana", 0), (2, "nonBAA", 7, "CHG", 1),
                (3, "nonBAA", 4, "WHG", 2), (4, "nonBAA", 3, "EHG", 3),
                (5, "BAA", 8, "Ana", 0), (6, "BAA", 7, "CHG", 1)]


def copy_source(destination):
    destination.mkdir()
    for name in AUTHOR_FILE_SHA256:
        target = destination / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE / name, target)


def test_source_drift_is_rejected_before_ast_semantics(tmp_path):
    root = tmp_path / "source"
    copy_source(root)
    (root / "training.py").write_text((root / "training.py").read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="source drift: training.py"):
        inspect_author_source(root)


def test_population_order_drift_is_rejected_by_derivation(tmp_path):
    root = tmp_path / "source"
    copy_source(root)
    p = root / "simulate.py"
    p.write_text(p.read_text().replace('name="EHG"', 'name="other"'))
    with pytest.raises(ValueError, match="population order"):
        derive_native_path_mapping(root)


def test_path_semantics_drift_disagrees_between_train_and_test(tmp_path):
    root = tmp_path / "source"
    copy_source(root)
    p = root / "training.py"
    p.write_text(p.read_text().replace("path=5", "path=6"))
    with pytest.raises(ValueError, match="ambiguous|disagree"):
        derive_native_path_mapping(root)


def test_name_join_and_noncontiguous_haplotype_ids():
    text = "sample population group sex\nb BAA BAA NA\na present_day present_day NA\n"
    panel = parse_observed_panel(text, sample_order=("a", "b"), haplotype_node_ids=(7, 2, 18, 9))
    assert panel.group_by_node == {7: 8, 2: 8, 18: 1, 9: 1}
    with pytest.raises(TypeError):
        panel.cohort_by_node[7] = "Ana"


@pytest.mark.parametrize("text,order", [
    ("sample population group sex\na Ana Ana NA\na CHG CHG NA", None),
    ("sample population group sex\na AnA AnA NA", None),
    ("sample population group sex\na Ana Ana NA", ("x",)),
    ("population sample group sex\na Ana Ana NA", None),
])
def test_invalid_or_ambiguous_observed_population_order_rejected(text, order):
    with pytest.raises(ValueError):
        parse_observed_panel(text, sample_order=order)


def test_both_haplotypes_share_observed_cohort():
    with pytest.raises(ValueError, match="share"):
        ObservedPanel(("a",), (0, 1), {0: "Ana", 1: "CHG"})


def test_shell_order_skips_empty_nodes_and_does_not_recount_leaves():
    features = extract_native_features(chain_tree(), [0, 1], ordered_panel())
    expected = np.zeros((5, 9))
    expected[np.arange(5), np.arange(5)] = 1
    expected[:, 8] = np.asarray([10, 21, 32, 43, 54]) / 1500
    np.testing.assert_allclose(features[0], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(features[0], features[1])


def test_padding_and_source_native_focal_ancient_self_inclusion():
    tree, panel = star_tree()
    features = extract_native_features(tree, [0, 2, 4], panel)
    expected = np.zeros((3, 5, 9))
    expected[:, 0, :2] = .5
    expected[:, 0, 8] = 120 / 1500
    expected[:, 1:, :] = -15
    np.testing.assert_array_equal(features, expected)
    # Excluding the ancient focal would produce1/3 vs2/3, not these native values.


def test_age_variants_are_explicit_and_dtype_preserved():
    tree, panel = star_tree()
    training = extract_native_features(tree, [0], panel)
    paint = extract_native_features(tree, [0], panel, age_mode="painting_float", dtype=np.float32)
    assert training.dtype == np.float64 and paint.dtype == np.float32
    assert training[0, 0, 8] == 120 / 1500
    assert paint[0, 0, 8] == np.float32(120.9 / 1500)
    with pytest.raises(ValueError, match="age convention"):
        extract_native_features(tree, [0], panel, age_mode="guessed")


def test_features_never_read_hidden_population_flags_or_metadata():
    tree = chain_tree()
    class NoTruthTree:
        tree_sequence = SimpleNamespace(samples=tree.tree_sequence.samples)
        parent = tree.parent
        leaves = tree.leaves
        time = tree.time
        def get_population(self, *args):
            raise AssertionError("truth population read")
        def population(self, *args):
            raise AssertionError("truth population read")
        def node(self, *args):
            raise AssertionError("truth flags or metadata read")
    baseline = extract_native_features(tree, [0], ordered_panel())
    np.testing.assert_array_equal(extract_native_features(NoTruthTree(), [0], ordered_panel()), baseline)


def test_truth_counterfactual_changes_targets_but_not_cleaned_observation_features():
    truth, panel = truth_fixture()
    paths = truth_paths_at_positions(truth, [0, 2, 4], panel, [25, 50, 75])
    np.testing.assert_array_equal(paths, [[1, 1, 6], [2, 1, 6], [2, 1, 6]])
    # Removing artificial unary census nodes merges the two truth intervals.
    observed = truth.simplify(keep_unary=False, record_provenance=False)
    assert observed.num_trees == 1
    first = extract_native_features(observed.first(), [0, 2, 4], panel)
    tables = truth.dump_tables()
    populations = tables.nodes.population.copy()
    populations[7] = 7  # First query's hidden census changes Ana->CHG.
    tables.nodes.population = populations
    changed = tables.tree_sequence()
    assert truth_paths_at_positions(changed, [0], panel, [25])[0, 0] == 2
    changed_observed = changed.simplify(keep_unary=False, record_provenance=False)
    np.testing.assert_array_equal(first, extract_native_features(changed_observed.first(), [0, 2, 4], panel))


@pytest.mark.parametrize("kwargs,error", [
    ({"missing": True}, "missing native census"),
    ({"wrong_time": True}, "wrong census"),
])
def test_missing_or_wrong_census_does_not_loop_or_produce_fallback_label(kwargs, error):
    truth, panel = truth_fixture(**kwargs)
    with pytest.raises(ValueError, match=error):
        truth_paths_at_positions(truth, [0], panel, [25])


def test_source_mapping_six_paths_and_inadmissible_baa_truth():
    paths = paths_from_census_populations(np.asarray([8, 7, 4, 3, 8, 7]),
                                        ["present_day"] * 4 + ["BAA"] * 2)
    np.testing.assert_array_equal(paths, np.arange(1, 7))
    with pytest.raises(ValueError, match="BAA source"):
        paths_from_census_populations(np.asarray([4]), ["BAA"])
    with pytest.raises(ValueError, match="outside source"):
        paths_from_census_populations(np.asarray([6]), ["BAA"])


def test_secondary_probability_sum_precedes_argmax():
    p = np.asarray([[.29, .4, .0, .0, .28, .03]], dtype=np.float64)
    assert p.argmax(1)[0] == 1
    collapsed = collapse_native_probabilities(p)
    np.testing.assert_allclose(collapsed, [[.57, .43, .0, .0]], rtol=0, atol=2e-16)
    assert collapsed.argmax(1)[0] == 0
    assert PATH_TO_SOURCE_CLASS[p.argmax(1)[0]] == 1  # Hard-label collapse is different.
    np.testing.assert_array_equal(p, [[.29, .4, .0, .0, .28, .03]])


def test_native_accuracy_and_confusion_orientation_and_cohort_metrics():
    p = np.eye(6)[[0, 0, 0, 1, 5]]
    truth = np.asarray([1, 2, 2, 1, 6])
    r = native_accuracy_confusion(p, truth, cohorts=["present_day", "present_day", "Neo", "Neo", "BAA"])
    expected = np.zeros((6, 6), dtype=np.int64)
    expected[0, 0] = expected[1, 0] = expected[5, 5] = 1
    expected[0, 1] = 2
    np.testing.assert_array_equal(r["confusion_predicted_rows_truth_columns"], expected)
    assert r["accuracy_fraction"] == .4 and r["accuracy_percent"] == 40
    assert r["by_cohort"]["BAA"]["accuracy_fraction"] == 1
    assert r["by_cohort"]["Neo"]["accuracy_fraction"] == 0
    assert r["by_cohort"]["present_day"]["n_examples"] == 2
    with pytest.raises(ValueError, match="truth labels"):
        native_accuracy_confusion(p, truth - 1)


@pytest.mark.parametrize("p", [
    np.zeros((2, 6)), np.ones((1, 4)) / 4,
    np.asarray([[np.nan, 0, 0, 0, 0, 1]]),
    np.asarray([[-.1, 1.1, 0, 0, 0, 0]]),
    np.asarray([[1, 0, 0, 0, 0, 0]], dtype=np.int64),
])
def test_invalid_probabilities_are_rejected_without_renormalization(p):
    with pytest.raises(ValueError):
        collapse_native_probabilities(p)


def test_runtime_adapter_uses_rank3_predict_and_preserves_all_six():
    x = extract_native_features(chain_tree(), [0, 1], ordered_panel())
    class Model:
        def predict(self, data, *, batch_size, verbose):
            assert data.shape == (2, 5, 9) and batch_size == 128 and verbose == 0
            return np.asarray([[.1, .2, .3, .1, .2, .1]] * 2, dtype=np.float32)
    p = predict_native_probabilities(Model(), x)
    assert p.shape == (2, 6) and p.dtype == np.float32
    with pytest.raises(ValueError, match="model input"):
        predict_native_probabilities(Model(), x[:, None, :, :])


def test_model_adapter_declares_native_architecture_without_tensorflow():
    layers = []
    def layer(name):
        def create(*args, **kwargs):
            result = (name, args, kwargs)
            layers.append(result)
            return result
        return create
    class Model:
        def __init__(self, rows):
            assert rows == layers
        def compile(self, **kwargs):
            self.compilation = kwargs
    api = SimpleNamespace(models=SimpleNamespace(Sequential=Model),
                          layers=SimpleNamespace(**{name: layer(name) for name in ("Flatten", "Dense", "Dropout")}))
    model = build_native_keras_model(api)
    assert layers == [
        ("Flatten", (), {"input_shape": (5, 9)}),
        ("Dense", (512,), {"activation": "relu"}),
        ("Dropout", (.2,), {}),
        ("Dense", (6,), {"activation": "softmax"})]
    assert model.compilation == {"optimizer": "adam", "loss": "categorical_crossentropy", "metrics": ["accuracy"]}
    assert native_model_spec()["epochs"] == 5


def test_export_roundtrip_preserves_every_probability_and_order_without_pickle(tmp_path):
    p = np.asarray([[[.1, .2, .3, .1, .2, .1], [.3, .1, .1, .2, .2, .1]]], dtype=np.float32)
    path = tmp_path / "native.npz"
    meta = export_native_predictions(path, p, positions_bp=[12.5], query_node_ids=[7, 2],
                                     parent_id="handmade-only", age_mode="training_int")
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["native_probabilities"], p)
        np.testing.assert_array_equal(data["source_probabilities"], collapse_native_probabilities(p))
        np.testing.assert_array_equal(data["query_node_ids"], [7, 2])
        assert json.loads(str(data["metadata_json"])) == meta
    assert meta["upstream_exact_reproduction_claimed"] is False
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_native_predictions(path, p, positions_bp=[12.5], query_node_ids=[7, 2],
                                  parent_id="handmade-only", age_mode="training_int")
    assert path.read_bytes() == before


def test_independent_parent_partition_rejects_public_source_overlap_and_duplicates():
    assert validate_parent_partition(["p0", "p1"], ["p2"], ["p3"])["test"] == ["p3"]
    with pytest.raises(ValueError, match="overlap"):
        validate_parent_partition(["p3", "p5", "p6"], ["p0"], ["p5", "p6", "p7", "p8", "p9"])
    with pytest.raises(ValueError, match="unique"):
        validate_parent_partition(["p0", "p0"], ["p1"], ["p2"])


def test_query_permutation_preserves_requested_order():
    tree, panel = star_tree()
    first = extract_native_features(tree, [0, 2, 4], panel)
    permuted = extract_native_features(tree, [4, 0, 2], panel)
    np.testing.assert_array_equal(permuted, first[[2, 0, 1]])


@pytest.mark.parametrize("queries", [[0, 0], [-1], [1000], [False]])
def test_invalid_query_identity_is_rejected(queries):
    tree, panel = star_tree()
    with pytest.raises(ValueError):
        extract_native_features(tree, queries, panel)


def test_feature_api_has_no_truth_target_argument():
    import inspect
    assert tuple(inspect.signature(extract_native_features).parameters) == (
        "observed_tree", "query_node_ids", "panel", "age_mode", "dtype")


def test_truth_lookup_rejects_changed_author_population_order():
    truth, panel = truth_fixture()
    tables = truth.dump_tables()
    tables.populations.clear()
    for name in reversed(AUTHOR_POPULATION_ORDER):
        tables.populations.add_row(metadata={"name": name})
    with pytest.raises(ValueError, match="author population order"):
        truth_paths_at_positions(tables.tree_sequence(), [0], panel, [25])
