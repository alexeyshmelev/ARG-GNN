"""Explicit capacity-only bridge retry with unchanged clip/sanitize operations.

The original pinned source and recipe remain unchanged. ROOT must preserve the
failed output receipt before retry. This driver uses a 16 GiB serialization
budget; the external task owns the separate 128 GiB RSS reservation.
"""
from __future__ import annotations

import argparse
import json

from arggnn.benchmarking.paper_argmix import paper_argmix_generator as g
from arggnn.benchmarking.paper_argmix.paper_argmix_generator import (
    observed_assets, confined, metadata, need, file_ref, Writes, _fresh_directory)
from arggnn.benchmarking.paper_argmix.paper_argmix_features import ScopeGuard

BRIDGE_OUTPUT_BYTES = 16 * 2**30
ORIGINAL_SOURCE_REF = {
    "path": "src/arggnn/benchmarking/paper_argmix/paper_argmix_generator.py",
    "sha256": "b8a897df08c6f62152185742268915d1c4c7ebbd80034bfa50ada2364c52033c",
}


def finalize_relate(config_path, parent_id, parent_output, native_work, terminal_receipt, output, *, guard):
    """Consume root-confirmed native success, clip only to observed SNP support."""
    config, parent, obs, observation, rows = observed_assets(config_path, parent_id, parent_output)
    import numpy as np
    import tskit
    native_work = confined(native_work)
    commands = metadata(native_work / "commands.json")
    terminal = metadata(terminal_receipt)
    need(commands["parent_id"] == parent_id and commands["config_ref"] == file_ref(config_path), "Native command provenance differs")
    need(terminal["commands_ref"] == file_ref(native_work / "commands.json") and
         terminal["actual_wait_consumed"] is True and terminal["returncodes"] == [0,0],
         "Root must supply actual successful waits for both unchanged native argv")
    source = confined(commands["native_trees"])
    need(source == native_work / "inferred.trees" and source.stat().st_size <= config["bounds"]["native_relate_bytes"],
         "Native output path/cap differs")
    inferred = tskit.load(source)
    sample_ids = list(map(int, inferred.samples()))
    need(len(sample_ids) == len(rows), "Native sample count differs")
    need(np.allclose(inferred.tables.nodes.time[sample_ids], [r["age_generations"] for r in rows], atol=1e-9, rtol=0),
         "Native sample age/order differs")
    # The root executed commands use direct .haps columns and explicit .sample IDs, so no converter reordering is inferred.
    positions = np.fromfile(obs / "positions.u4", dtype='<u4')
    native_sites = inferred.tables.sites.position
    need(len(native_sites) > 1 and np.array_equal(native_sites, native_sites.astype(np.int64)) and
         np.array_equal(native_sites.astype(np.int64), positions.astype(np.int64)+1),
         "Native SNP subset/drop changed common support. Review explicitly before proceeding")
    left, right = float(positions[0]), float(positions[-1]+1)
    need(0 <= left < right <= config["map"]["length_bp"], "Invalid inferred original support")
    tables = inferred.dump_tables()
    edges = tables.edges
    lo, hi = np.maximum(edges.left-1, left), np.minimum(edges.right-1, right)
    keep = hi > lo
    par, child = edges.parent[keep], edges.child[keep]
    tables.edges.clear()
    tables.edges.set_columns(left=lo[keep]-left, right=hi[keep]-left, parent=par, child=child)
    tables.sites.position = native_sites-1-left
    tables.sequence_length = right-left
    tables.nodes.population = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.individual = np.full(inferred.num_nodes, tskit.NULL, dtype=np.int32)
    tables.nodes.flags = tables.nodes.flags & tskit.NODE_IS_SAMPLE
    for table in (tables.nodes, tables.edges, tables.sites, tables.mutations):
        table.drop_metadata()
    tables.migrations.clear(); tables.populations.clear(); tables.individuals.clear(); tables.provenances.clear()
    tables.metadata_schema = tskit.MetadataSchema(None)
    tables.metadata = b""
    tables.sort()
    clean = tables.tree_sequence()
    need(list(map(int, clean.samples())) == sample_ids and clean.num_nodes == inferred.num_nodes,
         "Clipping changed inferred identities")
    need(all(tree.num_roots == 1 for tree in clean.trees()), "Uncovered/disconnected inferred interval")
    budget = Writes(BRIDGE_OUTPUT_BYTES, guard)
    output = _fresh_directory(output, guard)
    budget.tree(output / "observed.trees", clean)
    joined = [dict(r, inferred_tree_node_id=node, observed_node_id=node) for r,node in zip(rows,sample_ids)]
    budget.json(output / "sample_manifest.json", joined)
    receipt = dict(schema="ARGMix_prospective1611_inferred_observation_v1", parent_id=parent_id, split=parent["split"],
                   config_ref=file_ref(config_path), source_ref=file_ref(__file__),
                   observed_manifest_ref=file_ref(obs / "observation_manifest.json"),
                   native_commands_ref=file_ref(native_work / "commands.json"), native_terminal_ref=file_ref(terminal_receipt),
                   native_tree_ref=file_ref(source), tree_ref=file_ref(output / "observed.trees"),
                   roster_ref=file_ref(output / "sample_manifest.json"), trees=clean.num_trees, sites=clean.num_sites,
                   observed_coverage_bp0=[left,right], local_to_rebased_offset_bp=left,
                   chromosome_map_length_bp=config["map"]["length_bp"],
                   no_tree_extrapolation=True, original_observed_genotypes_authoritative=True,
                   converted_mutations_forbidden_as_genotype_features=True, truth_read=False)
    receipt["capacity_override"] = dict(original_source_ref=ORIGINAL_SOURCE_REF,
        original_recipe_bridge_bytes=config["bounds"]["inferred_bridge_bytes"],
        actual_serialization_envelope_bytes=BRIDGE_OUTPUT_BYTES,
        clipping_sanitization_and_identity_checks_unchanged=True)
    budget.json(output / "completed.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "parent-id", "parent-output", "native-work", "terminal-receipt", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    need(args.max_output_bytes == BRIDGE_OUTPUT_BYTES, "Explicit 16 GiB serialization envelope required")
    g.verify_ref(ORIGINAL_SOURCE_REF)
    import setproctitle
    from arggnn.benchmarking.paper_campaign.runtime import TITLE
    setproctitle.setproctitle(TITLE)
    guard = ScopeGuard(args.output, args.max_output_bytes, args.timeout_seconds)
    result = finalize_relate(args.config, args.parent_id, args.parent_output, args.native_work,
                             args.terminal_receipt, args.output, guard=guard)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
