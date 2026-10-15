"""Bounded observation-only C3 to native phased FLARE VCF/map converter."""
from decimal import Decimal
import json
from .american_c3_common_v1 import CLASSES, require, read, bound, cli
from .american_c3_flare_common_v1 import (
    PIN_COMMIT, NATIVE_OPTIONS, PROJECTION, observation_ref, pair_roster,
    native_mac_threshold, validate_focals, raw_bound, gzip_bound, GzipVCF,
)

SOURCE = "src/arggnn/american_c3_flare_convert_v1.py"
DEFAULTS = dict(
    schema="american_c3_flare_conversion_v1", execution_enabled=False,
    observation_ref=None, simulation_config_ref=None, focal_registry_ref=None,
    external_truth_ref=None, chromosome="1", max_sites=1000000,
    max_haplotypes=3200, output_cap_bytes=8000000000,
    multiallelic_policy="preserve_all_supported_2to4_ACGT_alleles",
    unsupported_site_policy="reject_no_silent_filter",
    native_options=NATIVE_OPTIONS, projection=PROJECTION,
)


def execute(cfg, w):
    import numpy as np
    from .american_c3_evaluate_v1 import registry
    require(cfg["schema"] == DEFAULTS["schema"] and cfg["external_truth_ref"] is None
            and cfg["native_options"] == NATIVE_OPTIONS and cfg["projection"] == PROJECTION
            and cfg["multiallelic_policy"] == DEFAULTS["multiallelic_policy"]
            and cfg["unsupported_site_policy"] == DEFAULTS["unsupported_site_policy"],
            "Frozen observation-only native FLARE conversion contract required")
    m = read(observation_ref(cfg["observation_ref"]))
    simulation = read(observation_ref(cfg["simulation_config_ref"]))
    roster_record = read(observation_ref(m["roster_ref"]))
    rows = roster_record["samples"]
    require(m["schema"] == "american_c3_observed_v1"
            and m["query_ancestry_truth_in_inputs"] is False
            and m["class_order"] == roster_record["class_order"] == list(CLASSES)
            and m["parent_id"] == roster_record["parent_id"]
            and m["genealogy_kind"] == "inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
            and m.get("sites_and_sample_identities_verified") is True,
            "Original inferred C3 observation/focal roster required; no oracle focal namespace")
    require(simulation["schema"] == "american_c3_simulation_reconstruction_v1"
            and simulation["parent_id"] == m["parent_id"] and simulation["split"] == m["split"]
            and simulation["sequence_length_bp"] == m["sequence_length_bp"]
            and simulation["recombination_rate"] == 1e-8,
            "Map must derive from the bound recorded simulation rate, not a real-map placeholder")
    length = int(m["sequence_length_bp"])
    require(length == m["sequence_length_bp"] and 0 < length < 2**31-1
            and cfg["chromosome"] and not any(c.isspace() for c in cfg["chromosome"]),
            "Positive discrete chromosome/domain required")
    factor = 1 if simulation["sample_unit"] == "haploid" else 2
    require(simulation["sample_unit"] in ("haploid", "diploid")
            and {k: factor*v for k, v in simulation["sample_counts"].items()} == m["counts_haploid"],
            "Recorded sample units/counts must match observed panel")
    refs, queries = pair_roster(rows)
    require(len(rows) <= cfg["max_haplotypes"], "Declared haplotype cap exceeded")
    threshold = native_mac_threshold(2*len(refs))
    focals = registry(observation_ref(cfg["focal_registry_ref"]))
    validate_focals(focals, m, rows)
    genotype = np.load(bound(observation_ref(m["genotypes_ref"])), mmap_mode="r", allow_pickle=False)
    positions = np.load(bound(observation_ref(m["positions_ref"])), allow_pickle=False)
    require(genotype.dtype == np.int8 and genotype.shape == (len(positions), len(rows))
            and 0 < len(positions) <= cfg["max_sites"]
            and positions.dtype == np.float64 and np.isfinite(positions).all()
            and np.array_equal(positions, positions.astype(np.int64))
            and (np.diff(positions) > 0).all()
            and ((positions >= 0) & (positions < length)).all(),
            "Complete discrete phased genotype/coordinate axes required")
    ref_cap, gt_cap = raw_bound(len(positions), len(refs)), raw_bound(len(positions), len(queries))
    planned = gzip_bound(ref_cap) + gzip_bound(gt_cap) + 16*len(positions) + 16777216
    require(planned <= cfg["output_cap_bytes"], "Conversion count bound exceeds admitted output cap")
    ref_writer = GzipVCF(w, "reference.vcf.gz", ref_cap)
    gt_writer = GzipVCF(w, "query.vcf.gz", gt_cap)
    def header(pairs):
        return ("##fileformat=VCFv4.3\n"
                "##contig=<ID=%s,length=%d>\n" % (cfg["chromosome"], length)
                + '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n'
                + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(p["vcf_sample_id"] for p in pairs) + "\n")
    ref_writer.write(header(refs))
    gt_writer.write(header(queries))
    expected_indexes = []
    counts = {str(k): 0 for k in (2, 3, 4)}
    ref_columns = np.asarray([i for p in refs for i in p["sample_indices"]])
    with bound(observation_ref(m["alleles_ref"])).open() as stream:
        for site, position in enumerate(positions):
            line = stream.readline()
            require(line and len(line.encode()) <= 128, "Bounded complete allele metadata required")
            alleles = json.loads(line)
            require(2 <= len(alleles) <= 4 and len(set(alleles)) == len(alleles)
                    and all(a in ("A", "C", "G", "T") for a in alleles),
                    "Unsupported/non-SNP/duplicate allele site rejected; no silent biallelic filter")
            g = genotype[site]
            require((g >= 0).all() and (g < len(alleles)).all(), "Phased complete nonmissing genotypes required")
            counts[str(len(alleles))] += 1
            prefix = "%s\t%d\tc3_site_%d\t%s\t%s\t.\t.\t.\tGT\t" % (
                cfg["chromosome"], int(position)+1, site, alleles[0], ",".join(alleles[1:]),
            )
            def data(pairs):
                return prefix + "\t".join(
                    "%d|%d" % (int(g[p["sample_indices"][0]]), int(g[p["sample_indices"][1]]))
                    for p in pairs) + "\n"
            ref_writer.write(data(refs))
            gt_writer.write(data(queries))
            frequencies = np.bincount(g[ref_columns], minlength=len(alleles))
            if np.sort(frequencies)[-2] >= threshold:
                expected_indexes.append(site)
            if site % 4096 == 0:
                w.adapter.budget_check_v9(w.storage, boundary=True)
        require(not stream.readline(), "Extra observed allele records rejected")
    ref_vcf, gt_vcf = ref_writer.close(), gt_writer.close()
    require(expected_indexes, "No native-default eligible markers; no lowered-MAC fallback")
    ix = np.asarray(expected_indexes, dtype=np.int64)
    marker_ref = w.npz("expected_native_marker_grid.npz", source_site_indexes=ix,
        positions_bp0=positions[ix].astype(np.int64), parent_id=np.asarray(m["parent_id"]),
        class_order=np.asarray(CLASSES))
    # PLINK: chromosome, ID, cM, one-based physical bp. End anchor is the domain boundary.
    cm = Decimal(str(simulation["recombination_rate"])) * Decimal(100) * Decimal(length)
    map_text = "%s map_start 0 1\n%s map_end %s %d\n" % (
        cfg["chromosome"], cfg["chromosome"], format(cm, "f"), length+1,
    )
    path = w.prepare("uniform_simulation.map", len(map_text.encode()))
    with path.open("x") as stream: stream.write(map_text)
    map_ref = w.finish(path, len(map_text.encode()))
    panel_text = "".join("%s\t%s\n" % (
        p["vcf_sample_id"], CLASSES[p["observed_reference_class"]]) for p in refs)
    path = w.prepare("reference_panels.txt", len(panel_text.encode()))
    with path.open("x") as stream: stream.write(panel_text)
    panel_ref = w.finish(path, len(panel_text.encode()))
    bridge_ref = w.json("phased_identity_bridge.json", dict(
        schema="american_c3_flare_phased_bridge_v1", parent_id=m["parent_id"],
        class_order=list(CLASSES), chromosome=cfg["chromosome"], sequence_length_bp=length,
        observation_ref=cfg["observation_ref"], focal_registry_ref=cfg["focal_registry_ref"],
        simulation_config_ref=cfg["simulation_config_ref"], reference_pairs=refs, query_pairs=queries,
        haplotype_phase_order="GT_left=haplotype0;GT_right=haplotype1;original_sample_index",
    ))
    return w.json("completed.json", dict(
        status="completed_C3_FLARE_observation_conversion", parent_id=m["parent_id"],
        focal_genealogy_kind=m["genealogy_kind"],
        class_order=list(CLASSES), observation_ref=cfg["observation_ref"],
        simulation_config_ref=cfg["simulation_config_ref"], focal_registry_ref=cfg["focal_registry_ref"],
        ref_vcf_ref=ref_vcf, gt_vcf_ref=gt_vcf, ref_panel_ref=panel_ref, map_ref=map_ref,
        phased_bridge_ref=bridge_ref, expected_marker_grid_ref=marker_ref,
        reference_haplotypes=2*len(refs), query_haplotypes=2*len(queries),
        input_sites=len(positions), native_expected_markers=len(ix), sites_by_allele_count=counts,
        unsupported_sites_dropped=0, native_MAC_threshold=threshold,
        native_options=NATIVE_OPTIONS, native_source_commit=PIN_COMMIT,
        projection=PROJECTION, common_focal_records=len(focals["record_ids"]),
        common_focal_registry_unchanged=True, recorded_recombination_rate=simulation["recombination_rate"],
        uniform_map_cM_at_domain_end=str(cm), query_ancestry_truth_opened=False,
        storage_count_bound_bytes=planned, author_Table1_exact_reproduction=False,
    ))


if __name__ == "__main__":
    cli(SOURCE, DEFAULTS, execute, dependencies=(
        "src/arggnn/american_c3_flare_common_v1.py", "src/arggnn/american_c3_evaluate_v1.py",
    ))
