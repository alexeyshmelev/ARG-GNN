"""Inert H018 next-stage qualification helpers. No network/native/model work.

Root supplies admitted map bytes and a callable current storage guard. This file
never reads VCFs, trees, targets, checkpoints, process state, or accounting state.
"""
from __future__ import annotations
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OWN = ROOT / "research/reviews/H018_genealogy_next_source_v1"
CONDITION = "H018_R16_20_30Mb_extension_v1"
BEGIN, END = 20_000_000, 30_000_000
IDS = ("Bar8", "Bon002", "KK1", "SATP", "Loschbour", "Bichon", "Sidelkino", "NEO88",
       "NEO935", "NEO898", "NEO23", "NEO739", "NEO943", "NEO866", "NEO945", "NEO738",
       "Gjerrild5", "NEO28", "NEO942", "NEO790")
CLASSES = ("ANA", "CHG", "WHG", "EHG")
PROXIES = ("Ana_proxy", "CHG_proxy", "WHG_proxy", "EHG_proxy")
MAP_COMMIT = "103ca071ea9e4e648192916142e5ed353ac94e27"
MAP_PATH = "maps/genetic_maps.b37/chr22.b37.gmap.gz"
MAP_BLOB = "eec8ceff2fdff0d70badcb09c18e5c6cc5b04854"
MAP_BYTES = 322186
MAP_EXPANDED_CAP = 8_000_000
BUNDLE_CAP = 4_000_000
NOTICE = "Copyright © 2018, University of Oxford"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_file(path):
    p = Path(path)
    require(not p.is_absolute() and ".." not in p.parts and str(p) not in ("", "."),
            "Canonical project-relative file required")
    full = ROOT / p
    require(full.resolve() == full and full.is_file() and not full.is_symlink(),
            "Existing regular project file required")
    return full


def read_ref(record, cap=4_000_000):
    require(set(record) == {"path", "sha256"}, "Exact path/SHA256 reference required")
    p = canonical_file(record["path"])
    require(p.stat().st_size <= cap, "Small source/metadata read cap exceeded")
    raw = p.read_bytes()
    require(sha(raw) == record["sha256"], "Source/metadata hash drift: " + record["path"])
    return raw


def source_pins():
    return json.loads((OWN / "pins.json").read_text())


def metadata_bundle_v1():
    """Only pinned compact metadata, including no observed genotype payload."""
    pins = source_pins()
    raw = {k: read_ref(pins[k]) for k in (
        "completed", "qualification", "original_qualification", "original_endpoint",
        "extension_endpoint", "age_contract", "subject_metadata", "subject_ids", "GLIMPSE_tree")}
    j = {k: json.loads(v) for k, v in raw.items()
         if k not in ("subject_metadata", "subject_ids")}
    q, done = j["qualification"], j["completed"]
    require(q["condition_id"] == done["condition_id"] == CONDITION
            and q["interval_GRCh37_bp0"] == [BEGIN, END]
            and q["candidate_sites"] == done["native_index_count"] == 94892
            and q["fixed_H018_QC_pass_sites"] == 4148
            and q["minimum_required"] == 1000 and q["region_observation_QC_passed"] is True,
            "Exact completed extension metadata required")
    extension = j["extension_endpoint"]
    require(extension["condition_id"] == CONDITION
            and extension["region"]["interval_bp0"] == [BEGIN, END]
            and extension["region"]["normalization_length_bp"] == 10_000_000
            and extension["reference_haplotypes"] == 16 and extension["query_haplotypes"] == 24
            and extension["observation_QC"] == j["original_endpoint"]["observation_QC"],
            "Pinned extension10Mb coordinates and original QC required")
    original = j["original_qualification"]
    require(original["fixed_H018_QC_pass_sites"] == 434
            and original["candidate_sites"] == 13312
            and original["region_observation_QC_passed"] is False
            and q["original_20_22Mb_gate_passed"] is False
            and q["fixed_filters"] == j["original_endpoint"]["observation_QC"],
            "Preserve original failed 2Mb gate and unchanged QC")
    require(done["empirical_genealogy_inference"] is False
            and done["model_prediction"] is False and done["ancestry_truth_available"] is False
            and q["GT_modified"] is False and q["HS_sampled"] is False and q["AP_used"] is False,
            "Acquisition is not inference or a model result")
    require(tuple(raw["subject_ids"].decode().splitlines()) == IDS
            and q["source_header"]["selected_subjects"] == list(IDS), "Exact original subject order")
    rows = list(csv.DictReader(io.StringIO(raw["subject_metadata"].decode()), delimiter="\t"))
    require(len(rows) == 20 and tuple(x["subject_id"] for x in rows) == IDS,
            "Exact 20-subject metadata roster")
    age = j["age_contract"]
    require(age["dates"].startswith("calibrated BP relative1950")
            and age["generation_time"]["working_value_years"] == 28
            and age["generation_time"]["fractional_generations_retained"] is True,
            "Pinned calibrated BP1950 / 28 convention required")
    source_dates = {x["subject_id"]: x for group in age["public_age_groups"].values()
                    for x in group["subjects"]}
    planned = []
    for i, r in enumerate(rows):
        role = "reference" if i < 8 else "query"
        require(int(r["subject_order0"]) == i and r["role"] == role
                and r["dating_source"] == "direct" and r["pure_ancestry_verified"] == "False",
                "Source role/date provenance cannot imply ancestry purity")
        cls = i // 2 if i < 8 else None
        require(r["proxy_class"] == (PROXIES[cls] if cls is not None else ""),
                "Only original proxy labels are admitted")
        src = source_dates[r["subject_id"]]
        mean, low, high = [float(r[k]) for k in ("age_BP_mean", "age_BP_lower95", "age_BP_upper95")]
        require([low, high] == src["age_95CI_cal_BP"] and mean == src["age_mean_cal_BP"]
                and src["source_sheet"] == r["metadata_source_sheet"]
                and src["source_row"] == int(r["metadata_source_row"])
                and 0 <= low <= mean <= high, "Exact original source row/date join required")
        for hap in (0, 1):
            planned.append(dict(sample_index=2*i+hap, subject_id=r["subject_id"], haplotype=hap,
                role=role, observed_reference_class=cls,
                observed_cohort=CLASSES[cls] if cls is not None else r["source_group_label"],
                age_mean_cal_BP=mean, age_95CI_cal_BP=[low, high],
                age_generations=mean / 28.0,
                date_zero="calBP1950", years_per_generation=28,
                date_source_sheet=src["source_sheet"], date_source_row=src["source_row"],
                native_tree_node_id=None, query_ancestry_truth=None))
    require(len(planned) == 40
            and [sum(r["observed_reference_class"] == c for r in planned) for c in range(4)] == [4]*4,
            "R16/Q24 haplotype roster required")
    donors = [sum(r["observed_reference_class"] == c and (r["age_generations"] < 350) == younger
                  for r in planned) for younger in (True, False) for c in range(4)]
    require(donors == [2]*8, "Mean/28 original younger/older donor channel roster differs")
    tree = j["GLIMPSE_tree"]
    require(tree["sha"] == MAP_COMMIT, "Pinned GLIMPSE source tree commit")
    item = next(x for x in tree["tree"] if x["path"] == MAP_PATH)
    require(item["type"] == "blob" and item["sha"] == MAP_BLOB and item["size"] == MAP_BYTES,
            "Pinned map candidate metadata differs")
    summary = dict(schema="H018_genealogy_next_metadata_v1", condition_id=CONDITION,
        region_GRCh37_bp0=[BEGIN, END], normalization_length_bp=10_000_000, observed_candidate_sites=94892, fixed_QC_sites=4148,
        original_2Mb_fixed_QC_sites=434, original_2Mb_passed=False,
        original_endpoint_replaced=False, subject_count=20, haplotypes=40,
        reference_haplotypes=16, query_haplotypes=24,
        source_calibrated_dates_qualified=True, inference_tip_date_roundtrip_qualified=False,
        date_convention="calBP1950 / 28, fractional generations, no 2026 offset",
        younger_then_older_C4_donor_counts=donors,
        empirical_genealogy_inference=False, model_runtime_qualified=False, real_accuracy_available=False,
        observed_VCF_ref_from_completion_only=done["observed_VCF_ref"],
        site_QC_ref_from_completion_only=done["site_QC_ref"],
        input_payloads_read=False, query_ancestry_truth_available=False,
        map_candidate=dict(commit=MAP_COMMIT, path=MAP_PATH, git_blob_sha1=MAP_BLOB,
            source_size_bytes_from_retained_tree=MAP_BYTES,
            url=f"https://raw.githubusercontent.com/odelaneau/GLIMPSE/{MAP_COMMIT}/{MAP_PATH}",
            byte_identity_verified=False, source_semantics_qualified=False))
    ages = "".join(format(r["age_generations"], ".17g") + "\n" for r in planned).encode()
    sample = ("ID_1 ID_2 missing\n0 0 0\n"
              + "".join(f"{sid} {sid} 0\n" for sid in IDS)).encode()
    return summary, planned, ages, sample


def inspect_candidate_map_bytes_v1(raw):
    """Exact candidate identity and bounded header inspection, no semantic claim."""
    require(isinstance(raw, bytes) and len(raw) == MAP_BYTES, "Exact candidate map byte count")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    require(blob == MAP_BLOB, "Exact Git source blob identity required")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(MAP_EXPANDED_CAP + 1)
    require(len(expanded) <= MAP_EXPANDED_CAP, "Expanded map cap")
    text = expanded.decode("ascii")
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) >= 3, "Map header and >=2 rows required")
    return text, dict(source_compressed_bytes=len(raw), source_sha256=sha(raw),
        git_blob_sha1=blob, expanded_bytes=len(expanded), expanded_sha256=sha(expanded),
        nonempty_lines=len(lines), first_three_nonempty_lines=lines[:3],
        byte_identity_verified=True, source_semantics_qualified=False)


def parse_map_text_v1(text, contract):
    """Fail closed until root has reviewed the actual source units/coordinates."""
    require(contract.get("status") == "root_reviewed_source_semantics"
            and contract.get("assembly") == "GRCh37"
            and contract.get("chromosome") == "22"
            and contract.get("physical_units") == "bp"
            and contract.get("cumulative_distance_units") == "cM"
            and contract.get("position_basis") in ("absolute_bp1", "absolute_bp0")
            and contract.get("source_format") in ("pos_chr_cM", "pos_rate_cM")
            and contract.get("semantic_evidence_refs"), "Inactive or unreviewed source map semantics")
    for ref in contract["semantic_evidence_refs"]:
        read_ref(ref)
    lines = [line.split() for line in text.splitlines() if line.strip()]
    require(lines[0] == contract.get("exact_header_tokens"), "Exact root-reviewed source map header")
    positions, distance, reported = [], [], []
    for row in lines[1:]:
        require(len(row) == 3, "Exactly three source map columns")
        require(row[0].lstrip("+").isdigit(), "Integer source physical position")
        pos = int(row[0]) + (1 if contract["position_basis"] == "absolute_bp0" else 0)
        require(pos >= 1 and (not positions or pos > positions[-1]), "Strictly increasing bp1 map positions")
        if contract["source_format"] == "pos_chr_cM":
            require(row[1] in ("22", "chr22"), "Chromosome22 source map rows")
        else:
            rate = float(row[1])
            require(math.isfinite(rate) and rate >= 0, "Finite nonnegative cM/Mb map rates")
            reported.append(rate)
        cm = float(row[2])
        require(math.isfinite(cm) and cm >= 0 and (not distance or cm >= distance[-1]),
                "Finite nondecreasing source genetic positions in cM")
        positions.append(pos); distance.append(cm)
        require(len(positions) <= 200000, "Source map row cap")
    require(len(positions) >= 2 and positions[0] <= BEGIN+1 and positions[-1] >= END+1,
            "Public map must bracket both regional boundaries; no extrapolation")
    if reported:
        tolerance = contract.get("rate_rounding_tolerance_cM_per_Mb")
        require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= 0.02,
                "Explicit reviewed source rate-rounding tolerance required")
        for i in range(len(positions)-1):
            slope = (distance[i+1]-distance[i])*1e6/(positions[i+1]-positions[i])
            require(abs(reported[i]-slope) <= tolerance, "Rate and cumulative map units disagree")
    return positions, distance


def crop_piecewise_map_v1(positions, distance):
    """Preserve the public piecewise-linear cM curve; shift origin only."""
    left, right = BEGIN+1, END+1
    def at(pos):
        i = bisect.bisect_right(positions, pos) - 1
        if i == len(positions)-1:
            require(pos == positions[-1], "No map extrapolation")
            return distance[-1]
        require(i >= 0, "No map extrapolation")
        return distance[i] + (distance[i+1]-distance[i]) * ((pos-positions[i])/(positions[i+1]-positions[i]))
    knots = [left] + [p for p in positions if left < p < right] + [right]
    zero = at(left)
    cumulative = [at(p)-zero for p in knots]
    require(cumulative[-1] > 0, "Positive genetic distance over the admitted region")
    rates = [(cumulative[i+1]-cumulative[i])*1e6/(knots[i+1]-knots[i])
             for i in range(len(knots)-1)]
    rates.append(rates[-1])  # Terminal sentinel rate is unused past final boundary.
    out = "Position(bp) Rate(cM/Mb) Map(cM)\n"
    out += "".join(f"{p-BEGIN} {r:.17g} {cm:.17g}\n"
                   for p, r, cm in zip(knots, rates, cumulative))
    return out.encode(), dict(source_boundary_bp1=[left, right], local_boundary_bp1=[1, END-BEGIN+1],
        source_cM_at_left=zero, regional_distance_cM=cumulative[-1],
        regional_distance_Morgans=cumulative[-1]/100.0,
        native_source_knots_inside_region=len(knots)-2, output_rows=len(knots),
        operation="piecewise-linear interpolation only at boundaries; preserve all interior source knots; subtract constant genetic origin",
        uniform_map_substitution=False, source_positions_or_distances_rescaled=False,
        source_map_brackets_full_requested_interval=True,
        source_build_and_units_depend_on_root_semantic_receipt=True)


def qualify_candidate_map_bytes_v1(raw, semantics):
    text, identity = inspect_candidate_map_bytes_v1(raw)
    positions, distance = parse_map_text_v1(text, semantics)
    output, crop = crop_piecewise_map_v1(positions, distance)
    return output, dict(identity, source_semantics_qualified=True, **crop,
        map_sha256=sha(output), sites_and_haps_map_join_qualified=False,
        inference_or_model_executed=False)


def write_root_metadata_stage_v1(output_stage, *, guard, map_ref=None, semantics=None):
    """One preflight for one bounded new stage, using root's unchanged real guard.

    No default guard and no accounting imports. Root must bind this consumer,
    pins/config/map/semantic refs and the actual marker/repair authority under
    the current completed_scope_credits_v3 activation before calling.
    """
    require(callable(guard), "Root current actual storage guard callback required")
    name = Path(output_stage)
    require(len(name.parts) == 1 and name.name.startswith("root_metadata_stage_"),
            "Fresh owned root_metadata_stage_* basename required")
    output = OWN / name
    require(output.resolve() == output and not output.exists(), "Fresh isolated owned stage")
    summary, planned, ages, sample = metadata_bundle_v1()
    payloads = {
        "metadata_qualification.json": (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode(),
        "planned_haplotype_roster.json": (json.dumps(planned, indent=2, allow_nan=False)+"\n").encode(),
        "planned.sample": sample, "planned.sample_ages.generations": ages,
        "NOTICE.txt": (NOTICE+"\n").encode()}
    if map_ref is not None:
        raw = read_ref(map_ref, cap=MAP_BYTES)
        if semantics is None:
            _, report = inspect_candidate_map_bytes_v1(raw)
        else:
            native_map, report = qualify_candidate_map_bytes_v1(raw, semantics)
            payloads["qualified.regional.map"] = native_map
        payloads["map_qualification.json"] = (json.dumps(report, indent=2, allow_nan=False)+"\n").encode()
    total = sum(map(len, payloads.values()))
    require(total + 8192 <= BUNDLE_CAP, "One-stage compact bundle cap")
    # No file has been created before this one actual preflight.
    guard(BUNDLE_CAP)
    output.mkdir()
    for filename, raw in payloads.items():
        with (output/filename).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(output.relative_to(ROOT)), artifacts=len(payloads), output_bytes=total,
                guard_preflight_calls=1, scientific_runtime_executed=False)
