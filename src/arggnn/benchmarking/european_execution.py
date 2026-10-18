"""Concrete European Relate/feature stages; no work or science imports at import.

These helpers repair only length-specific preparation. They do not supply a
campaign launcher, a paper sampler, or a complete-parent TreeLoom executor.
Caller-owned admission, process supervision and finite limits remain required.
"""
from __future__ import annotations

import importlib
import json

from . import european_recipes as recipes

NATIVE_HELPER_REF = {
    "path": "scripts/infer_biological_paths_relate_v1.py",
    "sha256": "a63a79b36a422f7346f2a35206f41b35fb440053edcfe4e080755c649ed737e9",
}
BRIDGE_REF = {
    "path": "src/arggnn/ancestralpaths_native_bridge.py",
    "sha256": "fb69904e3004ca7de6f16736e542949fb1b3695edb104cdf601187a7f8d33f2c",
}


def _approved(recipe, guard, execution_check, headroom):
    if (not callable(guard) or not callable(execution_check)
            or type(headroom) is not int or headroom <= 0):
        raise ValueError("Caller approval, existing resource guard and finite headroom required")
    registered = recipes.load_recipe("ancestralpaths_european")
    fields = ("id", "schema", "parameters", "generator", "evidence", "source_refs")
    if json.dumps({k: recipe[k] for k in fields}, sort_keys=True, allow_nan=False) != json.dumps(
            {k: registered[k] for k in fields}, sort_keys=True, allow_nan=False):
        raise ValueError("Canonical registered AP recipe required")
    execution_check()
    guard(headroom)
    recipes.verify_sources(recipe)
    recipes.verify_sources({"source_refs": [NATIVE_HELPER_REF, BRIDGE_REF]})
    return recipe["parameters"]["length_bp"]


def _runtime():
    np = importlib.import_module("numpy")
    tskit = importlib.import_module("tskit")
    full = importlib.import_module(recipes.AP_HELPER)
    _, helper = full.helpers()
    return np, tskit, full, helper


def prepare_relate_inputs(recipe, assets, destination, *, guard, execution_check,
                          write_headroom_bytes):
    """Prepare genuine full-length Relate inputs with the pinned VCF filter.

    assets is the authenticated observed-only structure, with paths, rows,
    names, position_join and source_sha256. Filtering retains the original
    algorithm and its complete in-memory row list; it is not a bounded-RAM
    streaming VCF reader. No Relate process is launched.
    """
    length = _approved(recipe, guard, execution_check, write_headroom_bytes)
    _, _, full, helper = _runtime()
    destination = full.confined(destination)
    if destination.exists():
        raise ValueError("Fresh admitted Relate input directory required")
    rows = assets["rows"]
    expected = full.roster()
    fields = ("sample_index", "sample_id", "haplotype", "observed_cohort",
              "age_generations", "role", "observed_reference_class")
    if (len(rows) != 952 or len(assets["names"]) != 476
            or assets["names"] != [r["sample_id"] for r in expected[::2]]
            or any(any(row[k] != source[k] for k in fields)
                   for row, source in zip(rows, expected))):
        raise ValueError("Complete published AP roster and sample order required")
    if any(type(p) is not int or not 1 <= p <= length
           or entry["VCF_position_1based"] != p
           or entry["tree_position_0based"] != p - 1
           for p, entry in assets["position_join"].items()):
        raise ValueError("Observed VCF coordinates must join the declared full domain")
    with assets["paths"]["genotypes.vcf"].open() as stream:
        header, retained, counts = helper.filter_vcf(
            stream, assets["names"], assets["position_join"])
    guard(write_headroom_bytes)
    destination.mkdir()
    full.text_write(destination / "input.vcf",
                    "\n".join(header) + "\n" +
                    "\n".join("\t".join(r["fields"]) for r in retained) + "\n", guard)
    rate = recipe["parameters"]["recombination_rate"]
    full.text_write(destination / "input.map",
                    "pos COMBINED_rate Genetic_Map\n"
                    f"1 {rate * 1e8:g} 0\n{length + 1} {rate * 1e8:g} {length * rate * 100:g}\n", guard)
    for output, source in (("sample_ages.txt", "sample_ages.generations"),
                           ("input.poplabels", "population_labels.txt")):
        full.text_write(destination / output, assets["paths"][source].read_text(), guard)
    full.json_write(destination / "original_sample_manifest.json", rows, guard)
    full.json_write(destination / "preparation.json", dict(
        length_bp=length, observed_source_sha256=assets["source_sha256"],
        site_filter_counts=counts, truth_or_oracle_files_read=False,
        map_adapter="canonical Relate header; full declared numeric domain",
        polarization="original observed VCF REF; no invented ancestral FASTA",
        helper_ref=NATIVE_HELPER_REF), guard)
    return retained, counts


def clean_converted_tree(recipe, native_file, assets, retained_vcf, destination, *,
                         guard, execution_check, write_headroom_bytes,
                         include_auxiliary_grid=False):
    """Apply the pinned converter coordinate/sanitization algorithm at full length.

    All native nodes/trees/sites and IDs are retained within actual observed
    coverage. No tail extrapolation. The optional physical grid is explicit;
    sampled native-six training needs no dense grid.
    """
    length = _approved(recipe, guard, execution_check, write_headroom_bytes)
    if type(include_auxiliary_grid) is not bool:
        raise ValueError("Explicit auxiliary-grid choice required")
    np, tskit, full, helper = _runtime()
    native_file, destination = full.confined(native_file), full.confined(destination)
    if destination.exists():
        raise ValueError("Fresh admitted sanitized tree output required")
    inferred = tskit.load(native_file)
    mapping = helper.inferred_sample_mapping(inferred, assets)
    positions = inferred.tables.sites.position
    if not np.array_equal(positions, positions.astype(np.int64)):
        raise ValueError("Converter SNP positions must remain discrete")
    actual = positions.astype(np.int64).tolist()
    if (not actual or any(p not in retained_vcf for p in actual)
            or len(set(actual)) != len(actual)):
        raise ValueError("Converted SNPs cannot join the complete observed haps")
    left, right = float(min(actual) - 1), float(max(actual))
    if not 0 <= left < right <= length:
        raise ValueError("Converted coverage is outside the declared full domain")
    guard(max(write_headroom_bytes, 2 * int(inferred.nbytes) + 1048576))
    tables = inferred.dump_tables()
    edge = tables.edges
    shifted_left = np.maximum(edge.left - 1, left)
    shifted_right = np.minimum(edge.right - 1, right)
    keep = shifted_right > shifted_left
    parent, child = edge.parent[keep], edge.child[keep]
    tables.edges.clear()
    tables.edges.set_columns(left=shifted_left[keep] - left, right=shifted_right[keep] - left,
                             parent=parent, child=child)
    tables.sites.position = positions - 1 - left
    tables.sequence_length = right - left
    tables.nodes.population = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.migrations.clear()
    tables.populations.clear()
    tables.individuals.clear()
    tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sort()
    clean = tables.tree_sequence()
    if helper.inferred_sample_mapping(clean, assets) != mapping or clean.num_nodes != inferred.num_nodes:
        raise ValueError("Clipping changed native inferred node IDs/order")
    clean.dump(destination)
    grid = {}
    if include_auxiliary_grid:
        step = recipe["parameters"]["auxiliary_scoring_grid_bp"]
        centers = np.arange(step / 2, length, step, dtype=np.float64)
        retained = np.flatnonzero((centers >= left) & (centers < right))
        grid = dict(retained_fixed100bp_indexes=retained.tolist(),
                    fixed_grid_total=len(centers), fixed_grid_retained=len(retained))
    query_rows = [r for r in assets["rows"] if r["role"] == "query"]
    if len(mapping) != 952 or len(query_rows) != 686:
        raise ValueError("Complete native-six sample/query mapping required")
    return dict(
        condition="post_five_size_iterations_primary",
        native_tree_sha256=full.sha(native_file), clean_tree_sha256=full.sha(destination),
        native_tree=str(native_file), clean_tree=str(destination),
        coverage_original_bp0=[left, right], physical_coverage_fraction=(right - left) / length,
        local_to_original_offset_bp=left, sequence_length_local_bp=clean.sequence_length,
        original_parent_interval_bp0=[0, length], normalization_length_bp=length,
        physical_coordinate_transform="physical_original = local_tree_position + coverage_left",
        sample_index_to_inferred_tree_node_id=mapping, node_ids_preserved_during_clipping=True,
        converter_VCF_to_original_offset_bp=-1, sample_age_order_preserved=True,
        auxiliary_grid_included=include_auxiliary_grid, samples=952,
        sites=clean.num_sites, trees=clean.num_trees,
        query_keys=[[r["sample_id"], r["haplotype"]] for r in query_rows],
        site_join=[dict(native_site_id=i, VCF_position_1based=p,
                        original_position_0based=p - 1, local_position_0based=p - 1 - left,
                        original_site_id=assets["position_join"][p]["site_id"])
                   for i, p in enumerate(actual)], **grid)


def iter_native_sampled_features(recipe, observed_ts, panel, query_node_ids, sampled_rows, *,
                                 guard, execution_check, write_headroom_bytes):
    """Yield (example_index, Float32[5,9]) while visiting observed trees once.

    sampled_rows is the frozen coordinate registry sorted by local position:
    each record has example_index, position_local_bp and query_node_id. Repeats
    are preserved. This extractor chooses no rows, reads no labels and creates
    no per-tree/full-grid cache. Source repeated-add arithmetic remains Float64
    until each final input row is cast to Float32 for native Keras.
    """
    _approved(recipe, guard, execution_check, write_headroom_bytes)
    np, _, _, _ = _runtime()
    bridge = importlib.import_module("arggnn.ancestralpaths_native_bridge")
    queries = set(query_node_ids)
    if len(queries) != 686 or len(panel.haplotype_node_ids) != 952:
        raise ValueError("Exact complete AP panel and query identity required")
    iterator = iter(sampled_rows)
    pending = next(iterator, None)
    previous = -1.
    seen = set()
    for tree in observed_ts.trees():
        selected = []
        while pending is not None:
            position = pending["position_local_bp"]
            if (not isinstance(position, (int, float)) or not 0 <= position < observed_ts.sequence_length
                    or position < previous or pending["query_node_id"] not in queries
                    or type(pending["example_index"]) is not int or pending["example_index"] < 0
                    or pending["example_index"] in seen):
                raise ValueError("Frozen sorted sampled registry has invalid position/query/example identity")
            if position >= tree.interval.right:
                break
            if position < tree.interval.left:
                raise ValueError("Sampled position lacks native observed tree coverage")
            selected.append(pending)
            seen.add(pending["example_index"])
            previous = position
            pending = next(iterator, None)
        if selected:
            guard(write_headroom_bytes)
            nodes = list(dict.fromkeys(row["query_node_id"] for row in selected))
            features = bridge.extract_native_features(tree, nodes, panel,
                                                       age_mode="training_int", dtype=np.float64)
            if features.shape != (len(nodes), 5, 9):
                raise ValueError("Native source feature shape changed")
            index = {node: i for i, node in enumerate(nodes)}
            for row in selected:
                yield row["example_index"], features[index[row["query_node_id"]]].astype(np.float32)
    if pending is not None:
        raise ValueError("Sampled position lacks native observed tree coverage")
