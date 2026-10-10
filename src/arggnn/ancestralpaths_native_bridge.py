"""Source-native AncestralPaths definitions, with explicit mechanical adapters.

This module imports no TensorFlow and executes no upstream script. Features read
only an observed tree and a declared observed cohort/order manifest. Target
helpers are separate and must never be called by an inference runner.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

AUTHOR_REPOSITORY = "https://github.com/AliPearson/AncestralPaths"
AUTHOR_COMMIT = "77db3cfae100641c35218b87e1451e14f3074666"
AUTHOR_ARCHIVE_SHA256 = "9624135ddb0d0ea3e2275a4ef463cbd0b62ecd13625cf2b2f3de653d778ba3a7"
PUBLISHED_ARCHIVE_SHA256 = "4d58adf9633d96ff9ef3b7f600d6192b4207c2e1551c1175367596e83b774e85"
AUTHOR_FILE_SHA256 = {
    "Model_schematic.pdf": "308674d7d2ef3eee7ba1e5e2fbff18c58aa443b519bff9e52da5bf757ec2bdc0",
    "README.md": "bd7890faaca2e5a6ffb95d909dbb60e134e69300b0e3a455af97258e96b49695",
    "admixture.py": "ab8ca5ab4b6c841b34c379ac12ffd8c5f48541a5af764ecd6b676592ced41cc2",
    "chunk.py": "7b3c1e13e064163a0f2c706924ccc070e04c148f00e7e8e36ccca5ddf65a733a",
    "example/mesoneo_ages_pop.txt": "3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de",
    "exp_fit.R": "132996048173bc1540d9019863c9d63f6af54a01737c65d3cadbbdf06142ad71",
    "exp_fit_bronze.R": "494bd01ad2cdb28402d2025012750526a82e820c53acea3199b17d8a89b23308",
    "find_nearest.py": "0fea04686441ce01202d5f86fa6f7253144423fbea3de313d2a55bbce8c903c4",
    "genetic_map_simulated.txt": "55328df62ee6cb3ae01083cbcca6b86b938b9275429b2a0d003c09b516f15045",
    "paint.py": "8abb6f82bac2905c9260c5329aff08c2e73d583af3ac7409d3b7025657857623",
    "painting.R": "16175a086728c5213dc09a59f3f75b5d84dc50762c55a67d494d07aff94d704b",
    "pipeline.sh": "e5c625bb9929e824fe4914265d0666068856958af0a640a89776f2eaa41df28e",
    "relate.sh": "691acafaefe4cc7185d5403a928e1fb125a1889b91b301f36daa50436345bf44",
    "simulate.py": "4ae51e2af2a7cc16393d2611d6745e47f63ed37d12c15631456abc135e6cd969",
    "testing_bygroup.py": "3c129d9a24695a1ecc31c2c903af2a86a4ade2dddaffd62b9c8d1fa8ae37a1bf",
    "tracts_prob.py": "755cda542371bfe61d23fbc6d80ac89ebf3a8af7b27077b386bca0858a3bbfe8",
    "training.py": "13b4f30ab0df75fc461b764ae2cdbd1fdc071ff45200e6401e84faa565508e4f",
}
AUTHOR_POPULATION_ORDER = (
    "present_bronze", "Yam", "Neo", "EHG", "WHG", "NE", "BAA",
    "CHG", "Ana", "WA", "trunk",
)
COHORT_FEATURE_ORDER = ("Bronze_Age", "BAA", "Yam", "Neo", "WHG", "EHG", "Ana", "CHG")
FEATURE_NAMES = COHORT_FEATURE_ORDER + ("ancestor_age_div_1500",)
NATIVE_PATH_NAMES = ("Ana_nonBAA", "CHG_nonBAA", "WHG", "EHG", "Ana_viaBAA", "CHG_viaBAA")
SOURCE_CLASS_NAMES = ("Ana", "CHG", "WHG", "EHG")
PATH_TO_SOURCE_CLASS = (0, 1, 2, 3, 0, 1)  # Proven by derive_native_path_mapping.
SOURCE_TRAIN_PARENTS = (3, 5, 6)
SOURCE_TEST_PARENTS = (5, 6, 7, 8, 9)
SOURCE_ISSUES = (
    "pipeline_passes_unsupported_training_poplab_argument",
    "training_hardcodes_952_haplotype_cohort_boundaries",
    "training_and_test_parent_overlap_5_6",
    "test_and_paint_rank4_input_for_rank3_model",
    "legacy_keras_tensorflow_backend_and_predict_classes",
    "training_no_explicit_simulation_or_training_seeds",
    "label_dependent_training_example_selection",
    "test_log_axis_reduction_mismatches_cohort_description",
    "paint_sibling_padding_and_unfilled_interval_columns",
    "training_int_age_vs_paint_float_age",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pop_path_pairs(script: str) -> dict[int, tuple[str, int]]:
    """Read pop->path assignments from the author AST; never import its code."""
    root = ast.parse(script)
    parents = {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}
    answer = {}
    for node in ast.walk(root):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "pop"
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)):
            continue
        assignments = [a for a in node.body if isinstance(a, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "path" for t in a.targets)
                       and isinstance(a.value, ast.Constant)]
        if len(assignments) != 1:
            continue
        cohort, child = "nonBAA", node
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.If) and isinstance(parent.test, ast.Compare):
                comparison = parent.test
                if any(isinstance(c, ast.Constant) and c.value == "BAA"
                       for c in comparison.comparators):
                    cohort = "BAA" if child in parent.body else "nonBAA"
                    break
            child = parent
        path = assignments[0].value.value
        population = test.comparators[0].value
        if not isinstance(path, int) or not isinstance(population, int) or path in answer:
            raise ValueError("ambiguous source path assignment")
        answer[path] = (cohort, population)
    return answer


def derive_native_path_mapping(source_root: str | Path) -> dict[str, Any]:
    """Trace class meanings in simulate/training/testing without executing them."""
    directory = Path(source_root)
    parsed = ast.parse((directory / "simulate.py").read_text())
    populations = []
    for node in parsed.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "add_population":
                names = [k.value.value for k in call.keywords
                         if k.arg == "name" and isinstance(k.value, ast.Constant)]
                if len(names) != 1:
                    raise ValueError("population declaration is not literal")
                populations.append(names[0])
    if tuple(populations) != AUTHOR_POPULATION_ORDER:
        raise ValueError("source population order differs from the native pin")
    train = _source_pop_path_pairs((directory / "training.py").read_text())
    test = _source_pop_path_pairs((directory / "testing_bygroup.py").read_text())
    if train != test or set(train) != set(range(1, 7)):
        raise ValueError("training/testing path assignments disagree or are incomplete")
    rows = []
    for path in range(1, 7):
        cohort, pop = train[path]
        name = populations[pop]
        if name not in SOURCE_CLASS_NAMES:
            raise ValueError("path source outside the four native census populations")
        rows.append({"path_1based": path, "query_cohort_case": cohort,
                     "census_population_id": pop, "census_population": name,
                     "source_class_0based": SOURCE_CLASS_NAMES.index(name)})
    if tuple(row["source_class_0based"] for row in rows) != PATH_TO_SOURCE_CLASS:
        raise ValueError("derived collapse differs from the declared native ontology")
    return {"population_order": populations, "paths": rows,
            "source_class_order": list(SOURCE_CLASS_NAMES)}


def inspect_author_source(source_root: str | Path, *, archive: str | Path | None = None,
                          published_root: str | Path | None = None,
                          published_archive: str | Path | None = None) -> dict[str, Any]:
    """Check all original bytes, optionally both archives; never execute source."""
    directory = Path(source_root)
    hashes = {}
    for name, expected in AUTHOR_FILE_SHA256.items():
        p = directory / name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"missing or non-regular native source: {name}")
        actual = _sha256(p)
        if actual != expected:
            raise ValueError(f"native source drift: {name}")
        hashes[name] = actual
        if published_root is not None:
            published = Path(published_root) / name
            if published.is_symlink() or not published.is_file() or _sha256(published) != actual:
                raise ValueError(f"published archive/source pin mismatch: {name}")
    for p, expected in ((archive, AUTHOR_ARCHIVE_SHA256),
                        (published_archive, PUBLISHED_ARCHIVE_SHA256)):
        if p is not None and _sha256(Path(p)) != expected:
            raise ValueError("author archive hash mismatch")
    dependencies = {}
    for package in ("numpy", "tskit", "msprime", "demes", "tqdm",
                    "tensorflow", "keras", "pandas", "scipy", "rpy2"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "schema": "ancestralpaths-source-inspection-v1",
        "repository": AUTHOR_REPOSITORY, "commit": AUTHOR_COMMIT,
        "source_sha256": hashes,
        "published_zenodo_doi": "10.5281/zenodo.8319452",
        "published_source_byte_equivalence_checked": published_root is not None,
        "path_mapping": derive_native_path_mapping(directory),
        "source_training_parents": list(SOURCE_TRAIN_PARENTS),
        "source_testing_parents": list(SOURCE_TEST_PARENTS),
        "overlapping_parents": sorted(set(SOURCE_TRAIN_PARENTS) & set(SOURCE_TEST_PARENTS)),
        "native_model": native_model_spec(),
        "script_execution_issues": list(SOURCE_ISSUES),
        "installed_package_metadata": dependencies,
        "code_licence_file_present": any((directory / n).exists()
                                       for n in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")),
        "upstream_code_executed": False,
        "runtime_compatibility_verified": False,
        "scientific_reproduction_verified": False,
    }


@dataclass(frozen=True)
class ObservedPanel:
    """Only observed sample/cohort identity in actual inferred haplotype order."""
    sample_order: tuple[str, ...]
    haplotype_node_ids: tuple[int, ...]
    cohort_by_node: Mapping[int, str]

    def __post_init__(self):
        object.__setattr__(self, "sample_order", tuple(self.sample_order))
        object.__setattr__(self, "haplotype_node_ids", tuple(self.haplotype_node_ids))
        if not self.sample_order or len(set(self.sample_order)) != len(self.sample_order):
            raise ValueError("sample names must be nonempty and unique")
        if any(not isinstance(s, str) or not s for s in self.sample_order):
            raise ValueError("sample names must be nonempty strings")
        ids = self.haplotype_node_ids
        if len(ids) != 2 * len(self.sample_order) or len(set(ids)) != len(ids):
            raise ValueError("require two distinct ordered haplotype IDs per diploid")
        if any(isinstance(u, bool) or not isinstance(u, (int, np.integer)) or u < 0 for u in ids):
            raise ValueError("haplotype IDs must be nonnegative integers")
        cohorts = dict(self.cohort_by_node)
        if set(cohorts) != set(ids) or any(not isinstance(v, str) or not v for v in cohorts.values()):
            raise ValueError("every haplotype needs exactly one observed cohort")
        if any(cohorts[ids[i]] != cohorts[ids[i + 1]] for i in range(0, len(ids), 2)):
            raise ValueError("both haplotypes of a diploid must share its observed cohort")
        object.__setattr__(self, "cohort_by_node", MappingProxyType(cohorts))

    @property
    def group_by_node(self) -> dict[int, int]:
        return {u: COHORT_FEATURE_ORDER.index(c) if c in COHORT_FEATURE_ORDER else 8
                for u, c in self.cohort_by_node.items()}


def parse_observed_panel(poplabels_text: str, *, sample_order: Sequence[str] | None = None,
                         haplotype_node_ids: Sequence[int] | None = None,
                         other_observed_cohorts: Sequence[str] = ("present_day", "GBR")) -> ObservedPanel:
    """Parse Relate poplabels, joining sample names rather than assuming IDs.

    Author else->group8 is retained only for explicitly declared other cohorts.
    A misspelled ancient label therefore cannot silently become present-day.
    """
    records = {}
    allowed = set(COHORT_FEATURE_ORDER) | set(other_observed_cohorts)
    lines = [line.split() for line in poplabels_text.splitlines() if line.strip()]
    if not lines or lines[0] != ["sample", "population", "group", "sex"]:
        raise ValueError("expected Relate poplabels header: sample population group sex")
    for fields in lines[1:]:
        if len(fields) != 4 or fields[0] in records:
            raise ValueError("duplicate sample or malformed poplabels row")
        if fields[1] not in allowed:
            raise ValueError("undeclared observed cohort")
        records[fields[0]] = fields[1]
    order = tuple(records) if sample_order is None else tuple(sample_order)
    if len(order) != len(records) or set(order) != set(records):
        raise ValueError("poplabels and sample-order file must have identical samples")
    ids = tuple(range(2 * len(order))) if haplotype_node_ids is None else tuple(haplotype_node_ids)
    if len(ids) != 2 * len(order):
        raise ValueError("haplotype count does not match diploid sample order")
    cohorts = {ids[2 * i + h]: records[name] for i, name in enumerate(order) for h in range(2)}
    return ObservedPanel(order, ids, cohorts)


def extract_native_features(observed_tree: Any, query_node_ids: Sequence[int],
                            panel: ObservedPanel, *, age_mode: str = "training_int",
                            dtype: Any = np.float64) -> np.ndarray:
    """Return [Q,5,9] native upward shells, without reading truth or populations.

    Feature order: Bronze_Age,BAA,Yam,Neo,WHG,EHG,Ana,CHG,age/1500.
    At each ancestor take *new* ancient leaves, skip empty shells, and pad all
    remaining rows with -15 at the root. Native ancient focal self-inclusion is
    retained. Exact per-query extraction avoids paint.py's sibling-copy bug.
    """
    if age_mode not in ("training_int", "painting_float"):
        raise ValueError("declare training_int or painting_float native age convention")
    ids = tuple(query_node_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("queries must be nonempty and unique")
    if any(isinstance(u, bool) or not isinstance(u, (int, np.integer)) for u in ids):
        raise ValueError("query IDs must be integers")
    samples = set(int(u) for u in observed_tree.tree_sequence.samples())
    if samples != set(panel.haplotype_node_ids) or not set(ids).issubset(samples):
        raise ValueError("observed panel/query IDs must match the inferred sampled nodes")
    groups = panel.group_by_node
    features = np.zeros((len(ids), 5, 9), dtype=dtype)
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("native feature dtype must be floating")
    for i, query in enumerate(ids):
        previous, visited = set(), set()
        parent, depth = observed_tree.parent(int(query)), 0
        while parent != -1 and depth < 5:
            if parent in visited:
                raise ValueError("observed ancestor chain contains a cycle")
            visited.add(parent)
            total = {int(leaf) for leaf in observed_tree.leaves(parent)
                     if groups[int(leaf)] != 8 and int(leaf) not in previous}
            if total:
                age = observed_tree.time(parent)
                if not np.isfinite(age) or age < 0:
                    raise ValueError("observed node age must be finite and nonnegative")
                features[i, depth, 8] = (int(age) if age_mode == "training_int" else float(age)) / 1500
                add = 1.0 / len(total)
                for leaf in sorted(total):
                    features[i, depth, groups[leaf]] += add
                previous.update(total)
                depth += 1
            parent = observed_tree.parent(parent)
        features[i, depth:, :] = -15
    return features


def native_model_spec() -> dict[str, Any]:
    return {
        "input_shape": [5, 9], "flattened_features": 45, "output_classes": 6,
        "layers": [{"type": "Flatten", "input_shape": [5, 9]},
                   {"type": "Dense", "units": 512, "activation": "relu"},
                   {"type": "Dropout", "rate": 0.2},
                   {"type": "Dense", "units": 6, "activation": "softmax"}],
        "optimizer": "adam", "loss": "categorical_crossentropy",
        "metrics": ["accuracy"], "epochs": 5,
        "batch_size": 32, "shuffle": True,
        "batch_size_and_shuffle_origin": "keras_model_fit_defaults_not_source_literals",
        "upstream_training_seed": None,
    }


def build_native_keras_model(keras_api: Any) -> Any:
    """Explicit adapter for a reviewed Keras runtime, without importing it here."""
    model = keras_api.models.Sequential([
        keras_api.layers.Flatten(input_shape=(5, 9)),
        keras_api.layers.Dense(512, activation="relu"),
        keras_api.layers.Dropout(0.2),
        keras_api.layers.Dense(6, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _probabilities(probabilities: Any) -> np.ndarray:
    p = np.asarray(probabilities)
    if p.ndim < 2 or p.shape[-1] != 6 or p.size == 0:
        raise ValueError("native probabilities must have a nonempty final axis of six")
    if not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all():
        raise ValueError("native probabilities must be finite floating values")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=-1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError("native softmax probabilities must be normalized without repair")
    return p


def predict_native_probabilities(model: Any, features: Any, *, batch_size: int = 128) -> np.ndarray:
    """Rank3 input and predict+argmax replace obsolete API, preserving all six."""
    x = np.asarray(features)
    if x.ndim != 3 or x.shape[1:] != (5, 9) or x.shape[0] < 1 or not np.isfinite(x).all():
        raise ValueError("native model input must be nonempty [examples,5,9]")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    p = _probabilities(model.predict(x, batch_size=batch_size, verbose=0))
    if p.shape != (x.shape[0], 6):
        raise ValueError("model output must match example order and six native paths")
    return p.copy()


def collapse_native_probabilities(probabilities: Any) -> np.ndarray:
    """Secondary source endpoint: sum 1+5,2+6,3,4 before argmax."""
    p = _probabilities(probabilities)
    return np.stack((p[..., 0] + p[..., 4], p[..., 1] + p[..., 5],
                     p[..., 2], p[..., 3]), axis=-1)


def _path_labels(labels: Any, shape: tuple[int, ...]) -> np.ndarray:
    y = np.asarray(labels)
    if y.shape != shape or not np.issubdtype(y.dtype, np.integer) or ((y < 1) | (y > 6)).any():
        raise ValueError("native truth labels must be integer 1..6 with matching example order")
    return y


def native_accuracy_confusion(probabilities: Any, truth_paths_1based: Any,
                             *, cohorts: Any | None = None) -> dict[str, Any]:
    """Author unweighted accuracy and predicted-row/truth-column confusion.

    Confusion matrices always retain six axes, including unobserved classes.
    Additional span/tract metrics belong to a separately named common endpoint.
    """
    p = _probabilities(probabilities)
    y = _path_labels(truth_paths_1based, p.shape[:-1])
    classes = p.argmax(axis=-1)
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (classes.ravel(), (y - 1).ravel()), 1)
    result = {"n_examples": int(y.size),
              "accuracy_fraction": float(np.mean(classes == y - 1)),
              "accuracy_percent": float(100 * np.mean(classes == y - 1)),
              "confusion_predicted_rows_truth_columns": confusion,
              "confusion_path_axis_1based": list(range(1, 7))}
    if cohorts is not None:
        c = np.asarray(cohorts)
        if c.shape != y.shape or not all(isinstance(v, str) and v for v in c.ravel().tolist()):
            raise ValueError("cohort labels must match each scored example")
        result["by_cohort"] = {name: native_accuracy_confusion(p[c == name], y[c == name])
                               for name in sorted(set(c.ravel().tolist()))}
    return result


def paths_from_census_populations(census_populations: Any, observed_cohorts: Any) -> np.ndarray:
    """External truth-only native source-population -> six-path assignment."""
    pop, cohort = np.asarray(census_populations), np.asarray(observed_cohorts)
    if pop.shape != cohort.shape or not np.issubdtype(pop.dtype, np.integer):
        raise ValueError("census population and observed cohort arrays must align")
    if not np.isin(pop, (3, 4, 7, 8)).all():
        raise ValueError("census lineage outside source populations EHG,WHG,CHG,Ana")
    if not all(isinstance(c, str) and c for c in cohort.ravel().tolist()):
        raise ValueError("observed cohorts must be nonempty strings")
    baa = cohort == "BAA"
    if np.any(baa & ~np.isin(pop, (7, 8))):
        raise ValueError("BAA source outside native Ana/CHG paths")
    paths = np.empty(pop.shape, dtype=np.int64)
    for population, path in ((8, 1), (7, 2), (4, 3), (3, 4)):
        paths[pop == population] = path
    paths[baa & (pop == 8)], paths[baa & (pop == 7)] = 5, 6
    return paths


def truth_paths_at_positions(truth_ts: Any, query_node_ids: Sequence[int],
                            panel: ObservedPanel, positions_bp: Sequence[float],
                            *, census_time: float = 350.0) -> np.ndarray:
    """Truth-only robust census lookup, kept separate from inference features.

    Reject malformed census targets instead of the author's unbounded while loop.
    Sample order here is the truth order and must be joined externally to Relate.
    """
    import msprime
    positions = np.asarray(positions_bp, dtype=np.float64)
    queries = tuple(query_node_ids)
    if positions.ndim != 1 or positions.size == 0 or not np.isfinite(positions).all():
        raise ValueError("truth positions must be a nonempty finite vector")
    if (positions < 0).any() or (positions >= truth_ts.sequence_length).any():
        raise ValueError("truth positions must be in [0,sequence_length)")
    if census_time != 350.0:
        raise ValueError("native source census is exactly generation350")
    if not queries or len(set(queries)) != len(queries) or not set(queries).issubset(panel.haplotype_node_ids):
        raise ValueError("truth queries must match the observed cohort manifest")
    if set(int(u) for u in truth_ts.samples()) != set(panel.haplotype_node_ids):
        raise ValueError("truth sample IDs and panel order disagree")
    names = tuple(p.metadata.get("name") for p in truth_ts.populations())
    if names != AUTHOR_POPULATION_ORDER:
        raise ValueError("truth population metadata must match the author population order")
    populations = np.empty((len(positions), len(queries)), dtype=np.int64)
    for i, pos in enumerate(positions):
        tree = truth_ts.at(float(pos))
        for j, query in enumerate(queries):
            if tree.time(query) >= census_time:
                raise ValueError("native query must be younger than census")
            u = query
            while u != -1:
                node = truth_ts.node(u)
                if node.flags & msprime.NODE_IS_CEN_EVENT:
                    if node.time != census_time:
                        raise ValueError("wrong census event time")
                    populations[i, j] = node.population
                    break
                u = tree.parent(u)
            else:
                raise ValueError("missing native census ancestor")
    cohorts = np.broadcast_to([panel.cohort_by_node[q] for q in queries], populations.shape)
    return paths_from_census_populations(populations, cohorts)


def validate_parent_partition(train: Sequence[str], validation: Sequence[str],
                              test: Sequence[str]) -> dict[str, list[str]]:
    """Independent-parent evaluation fix; no truth-based replicate selection."""
    result = {}
    seen = set()
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        values = list(values)
        if not values or any(not isinstance(p, str) or not p for p in values) or len(set(values)) != len(values):
            raise ValueError("each split requires unique nonempty parent IDs")
        if seen.intersection(values):
            raise ValueError("train/validation/test parent overlap")
        seen.update(values)
        result[name] = values
    return result


def export_native_predictions(path: str | Path, probabilities: Any, *,
                              positions_bp: Sequence[float], query_node_ids: Sequence[int],
                              parent_id: str, age_mode: str) -> dict[str, Any]:
    """Preserve all six probabilities plus a named secondary four-source view."""
    p = _probabilities(probabilities)
    positions, queries = np.asarray(positions_bp), np.asarray(query_node_ids)
    if p.ndim != 3 or p.shape[:2] != (positions.size, queries.size):
        raise ValueError("export arrays must align as [position,query,6]")
    if positions.ndim != 1 or positions.size == 0 or not np.isfinite(positions).all() or (positions < 0).any() or np.any(np.diff(positions) <= 0):
        raise ValueError("export positions must be finite nonnegative and strictly increasing")
    if queries.ndim != 1 or not np.issubdtype(queries.dtype, np.integer) or (queries < 0).any() or len(set(queries.tolist())) != queries.size:
        raise ValueError("export queries must be unique nonnegative integer IDs")
    if not isinstance(parent_id, str) or not parent_id or age_mode not in ("training_int", "painting_float"):
        raise ValueError("export requires parent identity and declared native feature age mode")
    source = collapse_native_probabilities(p)
    metadata = {
        "schema": "ancestralpaths-native-predictions-v1", "parent_id": parent_id,
        "author_commit": AUTHOR_COMMIT, "feature_age_mode": age_mode,
        "native_path_axis_1based": list(range(1, 7)),
        "native_path_names": list(NATIVE_PATH_NAMES),
        "secondary_source_class_axis_0based": list(range(4)),
        "secondary_source_class_names": list(SOURCE_CLASS_NAMES),
        "secondary_path_to_source_class_0based": list(PATH_TO_SOURCE_CLASS),
        "upstream_exact_reproduction_claimed": False,
    }
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, native_probabilities=p, source_probabilities=source,
                            native_path_1based=p.argmax(axis=-1) + 1,
                            source_class_0based=source.argmax(axis=-1),
                            positions_bp=positions, query_node_ids=queries,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return metadata
