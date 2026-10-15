"""Independent, pure-byte H018 serialized preparation verifier.

No file/network/process/guard/native/model I/O. The public entry point has no
contract override. Root supplies bounded actual bytes after its own admission.
Synthetic tests exercise the same private engine with a separate tiny contract.
"""
import csv
import gzip
import hashlib
import io
import json
import math
from collections import namedtuple
from types import MappingProxyType

_Contract = namedtuple("_Contract", "begin end candidates retained subjects references queries map_rows pins production")
_PIN_VALUES = {
    "source_VCF": ("data/H018_R16_20_30Mb_extension_batched_v4/H018_R16_Q24.chr22_20_30Mb_extension.noAP.vcf.gz",
                   "8b2adb8436896d91a7fbfae63d610faf3b7fa0d90e72b6331c709ebd86a69e29"),
    "source_QC": ("data/H018_R16_20_30Mb_extension_batched_v4/regional_site_QC.tsv",
                  "30686795e290ece774a7a69e46e93a9f543d56a3917c74a85dc2527b7091b3b3"),
    "planned_haplotype_roster.json": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/planned_haplotype_roster.json",
                                     "7cffa116d842c9ba88e78e23f4aa01c61264f3250c68a96955c0f906edf7cde3"),
    "planned.sample": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/planned.sample",
                       "873cf2aa5a56dee5a532745aba752248ace172e1c27851b288673a0e875b1aa9"),
    "planned.sample_ages.generations": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/planned.sample_ages.generations",
                                       "6f6926250ce4ba37ba4f8e813586974b3e92cb1850636b759696cc1df0b813d7"),
    "qualified.regional.map": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/qualified.regional.map",
                               "9828ccd33b14c5540410aff7a14b367861f4f3962d597ad00d1278d68fa2bc08"),
    "map_qualification.json": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/map_qualification.json",
                               "73ea5e4f7bd2266fa70d6062c301b19799bb7c587ad7264e79b53467c8552722"),
    "metadata_completion": ("research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1/completed.json",
                            "c854720d2272d8c39e01c2348f90f0d01ecd883f57fb8390a00e1ee2d5aad3b3"),
    "metadata_independent_result_validation": ("research/reviews/H018_metadata_root_execution_20261015_v1/root_result_validation_after_control_failure.json",
                                               "622968f16c8d6740a0d3ed519a8ed468c03f7054d4870ae28713a5ce2efbec29"),
}
H018 = _Contract(20_000_000, 30_000_000, 94892, 4148, 20, 16, 24, 12280,
                 MappingProxyType(dict(_PIN_VALUES)), True)
ORIGINAL_BASIS_REFS = MappingProxyType({k: MappingProxyType(dict(path=v[0], sha256=v[1]))
                                       for k, v in H018.pins.items()})
OUTPUT_DIRECTORY = "data/H018_R16_observed_bridge_v1"
JOB = "H018-observed-baseline-GT-preparation-v1"
CONDITION = "H018_R16_20_30Mb_extension_v1"
BRIDGE_REF = {"path": "research/reviews/H018_observed_bridge_source_v1/bridge_core.py",
              "sha256": "a860ebf2f046834871d06a1938f0a1de0431cc20f698c1c5096b87bdc4ef89d2"}
INPUT_REFS_REF = {"path": "research/reviews/H018_observed_bridge_source_v1/input_refs.inactive.json",
                  "sha256": "dbc7ad96c9d91574c6c3137f0c0767380c4238116c15b7bb7c2dc953e13b26b7"}
ASSET_NAMES = frozenset(("baseline.haps", "baseline.sample", "sample_ages.generations",
    "frozen_site_identities.json", "polarization_equivalence.json", "bridge_preparation.json",
    "qualified.regional.map", "NOTICE.txt", "provenance.json"))
OUTPUT_NAMES = ASSET_NAMES | {"completed.json"}
FALSE_GATES = ("native_execution_authorized", "native_inference_executed", "HS_draws_materialized",
               "GPU_used", "model_prediction", "ancestry_truth_available")
MAX_OUTPUT = 32_000_000
MAX_LINE = 1_000_000
MAX_EXPANDED = 1_000_000_000
BASES = frozenset("ACGT")


class ReadbackError(ValueError):
    """A failed comparison never yields a successful or partial certificate."""


def _need(ok, message):
    if not ok:
        raise ReadbackError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            _need(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    def constant(value):
        raise ReadbackError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _equal(actual, expected):
    """JSON equality with exact types, including bool versus integer."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _ref(contract, name):
    path, digest = contract.pins[name]
    return dict(path=path, sha256=digest)


def _finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else True)


def _gates(obj):
    _need(all(obj.get(k) is False for k in FALSE_GATES), "Preparation-only false gates")
    _need(obj.get("separate_native_policy_and_admission_required") is True
          and obj.get("fractional_calBP1950_div28_native_age_roundtrip_policy_resolved") is False,
          "Unresolved separate native/date policy retained")


def _caveat(obj, retained_bytes=False):
    _need(obj.get("metadata_original_OS_exit") == "unknown"
          and obj.get("metadata_original_wait0_available") is False, "Original metadata OS exit remains unknown")
    if retained_bytes:
        _need(obj.get("metadata_original_reservation_retained_bytes") == 5_000_000,
              "Original metadata 5MB charge retained")


def _basis(contract, basis):
    _need(set(basis) == set(contract.pins), "Exact immutable original basis set")
    for name, raw in basis.items():
        cap = 80_000_000 if name in ("source_VCF", "source_QC") else (
            2_000_000 if name == "qualified.regional.map" else 65_536)
        _need(type(raw) is bytes and len(raw) <= cap, "Bounded immutable basis bytes: " + name)
        _need(_sha(raw) == contract.pins[name][1], "Immutable original SHA differs: " + name)
    done = _json(basis["metadata_completion"])
    _need(done["status"] == "completed_source_qualified_map_and_sample_metadata"
          and done["no_internal_OS_exit_or_wait0_claim"] is True
          and done["map_rows"] == contract.map_rows, "Original metadata completion")
    for name in ("qualified.regional.map", "planned.sample", "planned.sample_ages.generations",
                 "planned_haplotype_roster.json", "map_qualification.json"):
        _need(done["assets"][name] == _ref(contract, name), "Original completed asset tie: " + name)
    caveat = _json(basis["metadata_independent_result_validation"])
    _need(caveat["status"] == "all_outputs_verified_original_writer_absent_actual_child_OS_exit_unknown"
          and caveat["actual_direct_wait0_claim"] is False and caveat["reservation_remains_fully_charged"] is True
          and caveat["all_producer_asset_hashes_verified"] is True
          and caveat["completed_ref"] == _ref(contract, "metadata_completion")
          and caveat["map_qualification_ref"] == _ref(contract, "map_qualification.json"),
          "Independent metadata result caveat and exact refs")
    mq = _json(basis["map_qualification.json"])
    _need(mq["source_semantics_qualified"] is True and mq["uniform_map_substitution"] is False
          and mq["source_positions_or_distances_rescaled"] is False
          and mq["output_rows"] == contract.map_rows
          and mq["local_boundary_bp1"] == [1, contract.end-contract.begin+1]
          and mq["map_sha256"] == contract.pins["qualified.regional.map"][1],
          "Qualified exact map provenance")
    return done


def _roster(contract, raw):
    roster = _json(raw)
    h = 2*contract.subjects
    _need(type(roster) is list and len(roster) == h, "Exact qualified haplotype count")
    names = []
    for j, row in enumerate(roster):
        _need(type(row) is dict and type(row["sample_index"]) is int and row["sample_index"] == j,
              "Exact ordered haplotype sample_index")
        _need(type(row["subject_id"]) is str and row["subject_id"] and not any(c.isspace() for c in row["subject_id"]),
              "Literal subject ID")
        _need(type(row["haplotype"]) is int and row["haplotype"] == j % 2, "Adjacent haplotype 0/1 order")
        _need(_finite(row["age_mean_cal_BP"], True) and _finite(row["age_generations"], True)
              and row["age_generations"] == row["age_mean_cal_BP"]/28.0
              and row["date_zero"] == "calBP1950" and row["years_per_generation"] == 28,
              "Exact source calBP1950/28 fractional dates")
        _need(row["query_ancestry_truth"] is None and row["native_tree_node_id"] is None, "No query truth or inferred node substitution")
        _need(row["role"] == ("reference" if j < contract.references else "query"), "Original R/Q order")
        if j % 2:
            previous = dict(roster[j-1]); current = dict(row)
            for field in ("sample_index", "haplotype"):
                previous.pop(field); current.pop(field)
            _need(_equal(previous, current), "Same original diploid metadata for both haplotypes")
        else:
            names.append(row["subject_id"])
        if contract.production:
            _need(row["observed_reference_class"] == (j//4 if j < 16 else None), "Original C4 reference class roster")
    _need(len(set(names)) == contract.subjects
          and contract.references+contract.queries == h, "No duplicate/padded subjects")
    if contract.production:
        _need([sum(x["observed_reference_class"] == c and (x["age_generations"] < 350) == younger
                   for x in roster) for younger in (True, False) for c in range(4)] == [2]*8,
              "Original younger/older C4 donor channels")
    return roster, names


def _qc(contract, raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    _need(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Unique QC columns")
    candidates, retained = [], []
    previous = 0
    for row in reader:
        _need(None not in row and None not in row.values(), "Exact QC row cardinality")
        pos = int(row["POS_bp1"])
        _need(row["chrom"] == "22" and contract.begin <= pos-1 < contract.end
              and pos > previous and int(row["absolute_bp0"]) == pos-1
              and int(row["region_bp0"]) == pos-1-contract.begin, "Original candidate coordinates/order")
        ref, alt = row["REF"], row["ALT"]
        _need(ref in BASES and alt in BASES and len(ref) == len(alt) == 1 and ref != alt, "Original biallelic ACGT candidate")
        identity = ["22", pos, ref, alt]
        candidates.append(identity); previous = pos
        _need(row["H018_fixed_QC_pass"] in ("True", "False"), "Frozen QC Boolean spelling")
        if row["H018_fixed_QC_pass"] == "True":
            ancestor = row["pan_troglodytes"]
            _need(ancestor in (ref, alt), "Retained source ancestor belongs to original alleles")
            retained.append((identity, ancestor))
    _need(len(candidates) == contract.candidates and len(retained) == contract.retained, "Exact frozen candidate/retained counts")
    return candidates, retained


def _source_vectors(contract, raw, names, candidates, retained):
    selected = {tuple(identity): ancestor for identity, ancestor in retained}
    vectors, observed, header, nmain, expanded = [], [], False, False, 0
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        while True:
            line = stream.readline(MAX_LINE+1)
            if not line:
                break
            expanded += len(line)
            _need(len(line) <= MAX_LINE and expanded <= MAX_EXPANDED, "Original VCF line/expansion bounds")
            text = line.decode("utf-8").rstrip("\r\n")
            if text.startswith("##"):
                _need(not text.startswith(("##FORMAT=<ID=AP,", "##INFO=<ID=AP,")), "AP-free original header")
                if text.startswith("##NMAIN="):
                    _need(not nmain and text == "##NMAIN=15", "Unique original NMAIN15")
                    nmain = True
                continue
            fields = text.split("\t")
            if text.startswith("#CHROM\t"):
                _need(not header and nmain and fields[:9] == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
                      and fields[9:] == names, "Original exact 20-subject VCF header/order")
                header = True
                continue
            _need(header and not text.startswith("#") and len(fields) == 9+contract.subjects, "Original VCF record cardinality")
            identity = [fields[0], int(fields[1]), fields[3], fields[4]]
            observed.append(identity)
            tags = fields[8].split(":")
            _need(len(tags) == len(set(tags)) and "AP" not in tags, "Unique AP-free original FORMAT")
            info = {}
            for item in fields[7].split(";"):
                if item in ("", "."):
                    continue
                key, sep, value = item.partition("=")
                _need(key not in info and key != "AP", "Unique AP-free source INFO")
                info[key] = value if sep else True
            key = tuple(identity)
            if key not in selected:
                continue
            ancestor = selected[key]
            _need(info.get("pan_troglodytes") == ancestor and ancestor in BASES and "GT" in tags,
                  "Independent original source/frozen ancestor and GT join")
            gt_index, codes = tags.index("GT"), []
            for sample in fields[9:]:
                values = sample.split(":")
                _need(gt_index < len(values) and values[gt_index] in ("0|0", "0|1", "1|0", "1|1"),
                      "Independent original phased callable GT")
                first, second = values[gt_index].split("|")
                codes += [int(first), int(second)]
            vectors.append((identity, ancestor, codes))
    _need(header and observed == candidates and len(vectors) == contract.retained,
          "All original candidates and every retained vector in exact order")
    _need([x[0] for x in vectors] == [x[0] for x in retained], "No retained source reordering/intersection")
    return vectors


def _map(contract, raw):
    lines = raw.decode("ascii").splitlines()
    _need(lines and lines[0] == "Position(bp) Rate(cM/Mb) Map(cM)" and len(lines)-1 == contract.map_rows,
          "Qualified map header and all source knots")
    previous_pos, previous_cm, positions = 0, -1.0, []
    for line in lines[1:]:
        fields = line.split()
        _need(len(fields) == 3 and fields[0].isdigit(), "Written map physical coordinate")
        pos, rate, cm = int(fields[0]), float(fields[1]), float(fields[2])
        _need(pos > previous_pos and _finite(rate) and rate >= 0 and _finite(cm) and cm >= previous_cm,
              "Finite ordered nondecreasing qualified map")
        positions.append(pos); previous_pos, previous_cm = pos, cm
    _need(positions[0] == 1 and positions[-1] == contract.end-contract.begin+1
          and float(lines[1].split()[2]) == 0.0 and previous_cm > 0, "Qualified local map boundaries/origin")
    return positions[0], positions[-1]


def _verify_serialized(contract, basis, outputs):
    """Private generic engine; only tests may supply a non-H018 contract."""
    metadata_done = _basis(contract, basis)
    _need(set(outputs) == OUTPUT_NAMES and all(type(v) is bytes for v in outputs.values()), "Exact actual ten-file output set")
    _need(sum(map(len, outputs.values()))+131072 <= MAX_OUTPUT, "Bounded whole serialized output batch")
    completed = _json(outputs["completed.json"])
    _need(completed["schema"] == "H018_observed_bridge_completed_v1"
          and completed["status"] == "completed_baseline_GT_preparation_not_native_admitted"
          and completed["job_id"] == JOB and completed["condition_id"] == CONDITION
          and set(completed["assets"]) == ASSET_NAMES, "Preparation-only completed artifact closure")
    admitted_ref = completed["admitted_ref"]
    _need(type(admitted_ref) is dict and set(admitted_ref) == {"path", "sha256"}
          and type(admitted_ref["path"]) is str and type(admitted_ref["sha256"]) is str
          and len(admitted_ref["sha256"]) == 64
          and all(c in "0123456789abcdef" for c in admitted_ref["sha256"]), "Actual compact preparation admission ref")
    if contract.production:
        _need(admitted_ref["path"] == "research/reviews/H018_observed_bridge_root_source_v1/admission_v1/admitted.json",
              "Original preparation admission scope")
    for name in ASSET_NAMES:
        raw, rec = outputs[name], completed["assets"][name]
        _need(_equal(rec, dict(path=OUTPUT_DIRECTORY+"/"+name, bytes=len(raw), sha256=_sha(raw))),
              "Actual written asset bytes/hash/path: "+name)
    _gates(completed); _caveat(completed, True)
    _need(completed["no_internal_OS_exit_or_wait0_claim"] is True
          and completed["reservation_remains_fully_charged"] is True
          and completed["historical_charges_unchanged"] is True, "No fabricated worker wait/accounting credit")
    roster, names = _roster(contract, basis["planned_haplotype_roster.json"])
    expected_sample = ("ID_1 ID_2 missing\n0 0 0\n"+"".join(s+" "+s+" 0\n" for s in names)).encode()
    expected_ages = "".join(format(x["age_generations"], ".17g")+"\n" for x in roster).encode()
    _need(basis["planned.sample"] == expected_sample and outputs["baseline.sample"] == expected_sample,
          "Actual written sample headers, 20 subject IDs and haplotype join")
    _need(basis["planned.sample_ages.generations"] == expected_ages
          and outputs["sample_ages.generations"] == expected_ages, "Actual written exact source fractional ages/order")
    # Parse actual files as well as comparing bytes to immutable qualified text.
    sample_rows = [line.split() for line in outputs["baseline.sample"].decode("utf-8").splitlines()]
    _need(sample_rows[:2] == [["ID_1", "ID_2", "missing"], ["0", "0", "0"]]
          and sample_rows[2:] == [[sid, sid, "0"] for sid in names], "Parsed actual sample identities")
    ages = [float(x) for x in outputs["sample_ages.generations"].decode("ascii").splitlines()]
    _need(len(ages) == 2*contract.subjects and all(_finite(x, True) for x in ages)
          and ages == [x["age_generations"] for x in roster], "Parsed actual 40 fractional tip dates")
    _need(outputs["qualified.regional.map"] == basis["qualified.regional.map"], "No written map substitution/rescaling")
    map_left, map_right = _map(contract, outputs["qualified.regional.map"])
    notice_pin = metadata_done["assets"]["NOTICE.txt"]
    _need(_sha(outputs["NOTICE.txt"]) == notice_pin["sha256"], "Original licensed NOTICE copy")
    candidates, retained = _qc(contract, basis["source_QC"])
    vectors = _source_vectors(contract, basis["source_VCF"], names, candidates, retained)
    identities = [row[0] for row in vectors]
    _need(_equal(_json(outputs["frozen_site_identities.json"]), identities), "Every actual frozen identity, no drop/intersection")
    haps_rows = outputs["baseline.haps"].decode("ascii").splitlines()
    audits = _json(outputs["polarization_equivalence.json"])
    _need(len(haps_rows) == contract.retained and type(audits) is list and len(audits) == contract.retained,
          "Exact written haps/audit retained row counts")
    monomorphic, flips, vector_hasher = [], 0, hashlib.sha256()
    for number, ((identity, ancestor, original), line, audit) in enumerate(zip(vectors, haps_rows, audits)):
        chrom, pos, ref, alt = identity
        columns = line.split()
        _need(len(columns) == 5+2*contract.subjects, "Actual haps field/haplotype count")
        _need(all(x in ("0", "1") for x in columns[5:]), "Actual binary haps codes")
        actual_codes = [int(x) for x in columns[5:]]
        local_pos, flip = pos-contract.begin, ancestor == alt
        derived = ref if flip else alt
        expected_codes = [1-x if flip else x for x in original]
        _need(columns[:5] == [chrom, "H018_22_"+str(pos)+"_"+ref+"_"+alt, str(local_pos), ancestor, derived]
              and actual_codes == expected_codes, "Independent actual identity/AA-DA/phase/code comparison at retained row "+str(number))
        _need(map_left <= int(columns[2]) <= map_right, "Actual haps/map local coordinate join")
        original_nt = "".join(ref if code == 0 else alt for code in original)
        actual_nt = "".join(columns[3] if code == 0 else columns[4] for code in actual_codes)
        _need(actual_nt == original_nt, "Every actual serialized biological nucleotide/phase equals original GT")
        mono = len(set(actual_codes)) == 1
        expected_audit = dict(chrom=chrom, source_POS_bp1=pos, regional_bp0=pos-1-contract.begin,
            haps_POS_bp1=local_pos, source_REF=ref, source_ALT=alt, source_ancestral=ancestor,
            allele_orientation_flipped=flip, source_GT_haplotype_codes=original,
            native_haps_codes=actual_codes, nucleotide_equivalence_sha256=_sha(original_nt.encode()),
            exact_nucleotide_and_phase_order_equivalence=True, monomorphic_among_actual40=mono)
        _need(_equal(audit, expected_audit), "Independent exact actual audit/source-vector comparison at retained row "+str(number))
        flips += int(flip)
        if mono:
            monomorphic.append(pos)
        vector_hasher.update((str(pos)+":"+original_nt+"\n").encode())
    report = _json(outputs["bridge_preparation.json"])
    _need(report["schema"] == "H018_baseline_observed_bridge_prepared_v1"
          and report["status"] == "prepared_GT_inputs_not_native_admitted", "Preparation report schema/status")
    for obj in (report, completed):
        _need(obj["candidate_sites"] == contract.candidates and obj["retained_sites"] == contract.retained
              and obj["haplotypes"] == 2*contract.subjects and obj["reference_haplotypes"] == contract.references
              and obj["query_haplotypes"] == contract.queries and obj["counts_complete"] is True
              and obj["unsupported_site_count"] == 0 and obj["monomorphic_site_count"] == len(monomorphic)
              and obj["source_GT_biological_nucleotides_and_phase_unchanged"] is True,
              "Counts/GT/phase claims independently reproduced")
        _gates(obj)
    identity_sha = _sha(json.dumps(identities, separators=(",", ":")).encode())
    blockers = ["separate_native_policy_and_admission", "fractional_native_age_roundtrip_policy"]
    if monomorphic:
        blockers.append("monomorphic_actual40_sites")
    _need(report["monomorphic_sites"] == monomorphic and report["polarization_flipped_sites"] == flips
          and report["retained_identity_sha256"] == identity_sha
          and report["source_VCF_sha256"] == contract.pins["source_VCF"][1]
          and report["source_QC_sha256"] == contract.pins["source_QC"][1]
          and report["AP_used"] is False and report["original_QC_failure_replaced"] is False
          and report["native_baseline_admissible_if_no_other_blocker"] is (not monomorphic)
          and report["native_gate_blockers"] == blockers and report["map_qualification_required"] is True,
          "Exact actual report counts/digest/original source and native blockers")
    _need(completed["sites_dropped"] == 0 and completed["monomorphic_details_ref"] == completed["assets"]["bridge_preparation.json"],
          "No silent monomorphic/site dropping")
    provenance = _json(outputs["provenance.json"])
    _need(provenance["schema"] == "H018_observed_bridge_preparation_provenance_v1"
          and provenance["condition_id"] == CONDITION and provenance["admitted_ref"] == completed["admitted_ref"]
          and provenance["bridge_ref"] == BRIDGE_REF and provenance["input_refs_ref"] == INPUT_REFS_REF,
          "Actual preparation provenance")
    for key, name in (("source_VCF_ref", "source_VCF"), ("source_QC_ref", "source_QC"),
                      ("map_ref", "qualified.regional.map"), ("roster_ref", "planned_haplotype_roster.json"),
                      ("metadata_completion_ref", "metadata_completion"),
                      ("metadata_independent_validation_ref", "metadata_independent_result_validation")):
        _need(provenance[key] == _ref(contract, name), "Actual immutable provenance tie: "+key)
    _gates(provenance); _caveat(provenance, True)
    _need(provenance["dates"] == "source calBP1950 / 28, fractional, no 2026 offset, rounding or clamp"
          and provenance["coordinate_basis"] == "GRCh37 chr22 absolute bp0 [20000000,30000000); Oxford/map local bp1"
          and provenance["map_curve"] == "qualified piecewise public source curve copied unchanged; no uniform substitution"
          and provenance["extension_normalization_bp"] == 10_000_000
          and provenance["future_GNN_span_denominator"] == "log(2000001), unchanged"
          and provenance["original_2Mb_failure"] == dict(retained=434, candidates=13312, below_1000=True, replaced=False),
          "Original coordinate/date/map/normalization caveats")
    receipt = dict(schema="H018_bridge_serialized_readback_v1",
        status="verified_preparation_serialization_only_not_native_admitted" if contract.production else "synthetic_verifier_check_only",
        immutable_H018_contract=contract.production, preparation_admitted_ref=completed["admitted_ref"],
        preparation_completed_sha256=_sha(outputs["completed.json"]),
        candidate_sites=contract.candidates, retained_sites=contract.retained,
        haplotypes=2*contract.subjects, counts_complete=True, unsupported_site_count=0,
        monomorphic_site_count=len(monomorphic),
        monomorphic_source_POS_bp1_sha256=_sha(json.dumps(monomorphic, separators=(",", ":")).encode()),
        monomorphic_details_asset=completed["assets"]["bridge_preparation.json"], native_gate_blockers=blockers,
        polarization_flipped_sites=flips, retained_identity_sha256=identity_sha,
        original_nucleotide_vectors_sha256=vector_hasher.hexdigest(),
        original_source_vectors_independently_parsed=True, actual_serialized_site_nucleotide_phase_order_verified=True,
        actual_serialized_sample_haplotype_date_join_verified=True, actual_qualified_map_bytes_and_local_site_join_verified=True,
        basis_refs={k: _ref(contract, k) for k in contract.pins},
        actual_output_sha256={k: _sha(v) for k, v in outputs.items()},
        metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
        metadata_original_reservation_retained_bytes=5_000_000,
        native_execution_authorized=False, native_age_serialization_policy_resolved=False,
        native_sample_node_site_coverage_chronology_roundtrip_qualified=False,
        command_direct_wait_or_descendant_drain_certified=False, capacity_or_admission_certified=False,
        HS_execution_authorized=False, GPU_authorized=False, model_prediction=False, ancestry_truth_available=False)

    _need(len(json.dumps(receipt, sort_keys=True, indent=2).encode())+1 <= 65_536, "Compact independent readback receipt")
    return receipt


def verify_h018_serialized_v1(basis_bytes, actual_output_bytes):
    """Verify actual written bytes against immutable H018 originals.

    Root must admit/read/hash the inputs and output set, bind this exact verifier
    source, call this entry point without monkeypatching, and durably retain its
    returned compact receipt. Exceptions mean unqualified; never drop/rescue.
    This function neither supplies authority nor certifies an OS wait/native run.
    """
    return _verify_serialized(H018, basis_bytes, actual_output_bytes)
