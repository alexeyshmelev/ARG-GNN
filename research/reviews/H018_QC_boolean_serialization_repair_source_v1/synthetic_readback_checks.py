"""Meaningful tiny synthetic positives and semantic corruption checks only."""
from pathlib import Path
from types import MappingProxyType
import copy
import gzip
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_h018_independent_readback_synthetic_only", HERE/"readback_verifier_v2.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def enc(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(name, raw):
    return dict(path="synthetic/"+name, sha256=sha(raw))


def gates():
    return dict(native_execution_authorized=False, native_inference_executed=False,
        HS_draws_materialized=False, GPU_used=False, model_prediction=False, ancestry_truth_available=False,
        separate_native_policy_and_admission_required=True,
        fractional_calBP1950_div28_native_age_roundtrip_policy_resolved=False)


def fixture(monomorphic=True):
    # Literal original and serialized vectors include heterozygous phase, an
    # ancestral-ALT flip, and a retained monomorphic site. No bridge is imported.
    records = [
        (101, "A", "C", "A", [0, 1, 1, 0], True, [0, 1, 1, 0], "A", "C"),
        (110, "G", "T", "G", [1, 1, 0, 0], False, None, None, None),
        (120, "G", "T", "T", [0, 1, 0, 1], True, [1, 0, 1, 0], "T", "G"),
        (150, "C", "A", "A", [1, 1, 1, 1] if monomorphic else [0, 1, 1, 0], True,
         [0, 0, 0, 0] if monomorphic else [1, 0, 0, 1], "A", "C"),
        (180, "T", "G", "T", [0, 0, 1, 1], True, [0, 0, 1, 1], "T", "G"),
        (199, "A", "G", "A", [0, 0, 0, 0], False, None, None, None),
    ]
    names = ["ALPHA", "BETA"]
    roster = []
    for subject, name in enumerate(names):
        for hap in (0, 1):
            mean = (1000.125, 1234.5)[subject]
            roster.append(dict(sample_index=2*subject+hap, subject_id=name, haplotype=hap,
                role="reference" if subject == 0 else "query", observed_reference_class=0 if subject == 0 else None,
                age_mean_cal_BP=mean, age_generations=mean/28, date_zero="calBP1950", years_per_generation=28,
                native_tree_node_id=None, query_ancestry_truth=None))
    sample = b"ID_1 ID_2 missing\n0 0 0\nALPHA ALPHA 0\nBETA BETA 0\n"
    ages = "".join(format(row["age_generations"], ".17g")+"\n" for row in roster).encode()
    source = "##fileformat=VCFv4.2\n##NMAIN=15\n"
    source += "\t".join(["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]+names)+"\n"
    qc = "chrom\tPOS_bp1\tabsolute_bp0\tregion_bp0\tREF\tALT\tpan_troglodytes\tH018_fixed_QC_pass\n"
    identities, audits, haps = [], [], []
    for pos, ref_base, alt, ancestor, original, passed, native, aa, da in records:
        gt = [str(original[j])+"|"+str(original[j+1]) for j in (0, 2)]
        source += "\t".join(["22", str(pos), ".", ref_base, alt, ".", "PASS",
                            "pan_troglodytes="+ancestor, "GT:HS", gt[0]+":42", gt[1]+":17"])+"\n"
        qc += "\t".join(["22", str(pos), str(pos-1), str(pos-101), ref_base, alt, ancestor, str(int(passed))])+"\n"
        if passed:
            identities.append(["22", pos, ref_base, alt])
            haps.append(" ".join(["22", "H018_22_"+str(pos)+"_"+ref_base+"_"+alt, str(pos-100), aa, da]+[str(x) for x in native])+"\n")
            nucleotide = "".join((ref_base, alt)[x] for x in original)
            audits.append(dict(chrom="22", source_POS_bp1=pos, regional_bp0=pos-101, haps_POS_bp1=pos-100,
                source_REF=ref_base, source_ALT=alt, source_ancestral=ancestor,
                allele_orientation_flipped=ancestor == alt, source_GT_haplotype_codes=original,
                native_haps_codes=native, nucleotide_equivalence_sha256=sha(nucleotide.encode()),
                exact_nucleotide_and_phase_order_equivalence=True, monomorphic_among_actual40=len(set(native)) == 1))
    basis = {"source_VCF": gzip.compress(source.encode(), mtime=0), "source_QC": qc.encode(),
             "planned_haplotype_roster.json": enc(roster), "planned.sample": sample,
             "planned.sample_ages.generations": ages,
             "qualified.regional.map": b"Position(bp) Rate(cM/Mb) Map(cM)\n1 2000 0\n51 2000 0.10000000000000001\n101 2000 0.20000000000000001\n"}
    mq = dict(source_semantics_qualified=True, uniform_map_substitution=False,
              source_positions_or_distances_rescaled=False, output_rows=3,
              local_boundary_bp1=[1, 101], map_sha256=sha(basis["qualified.regional.map"]))
    basis["map_qualification.json"] = enc(mq)
    notice = b"Synthetic licence notice, not a retained production asset.\n"
    assets = {name: ref(name, basis[name]) for name in (
        "qualified.regional.map", "planned.sample", "planned.sample_ages.generations",
        "planned_haplotype_roster.json", "map_qualification.json")}
    assets["NOTICE.txt"] = ref("NOTICE.txt", notice)
    basis["metadata_completion"] = enc(dict(status="completed_source_qualified_map_and_sample_metadata",
        no_internal_OS_exit_or_wait0_claim=True, map_rows=3, assets=assets))
    basis["metadata_independent_result_validation"] = enc(dict(
        status="all_outputs_verified_original_writer_absent_actual_child_OS_exit_unknown",
        actual_direct_wait0_claim=False, reservation_remains_fully_charged=True,
        all_producer_asset_hashes_verified=True, completed_ref=ref("metadata_completion", basis["metadata_completion"]),
        map_qualification_ref=ref("map_qualification.json", basis["map_qualification.json"])))
    basis["qc_count_erratum"] = enc(dict(synthetic_fixture_only=True, candidate_sites=6, actual_retained_sites=4))
    pins = MappingProxyType({name: (ref(name, raw)["path"], sha(raw)) for name, raw in basis.items()})
    contract = v._Contract(100, 200, 6, 4, 2, 2, 2, 3, pins, False)
    mono = [150] if monomorphic else []
    report = dict(schema="H018_baseline_observed_bridge_prepared_v1", status="prepared_GT_inputs_not_native_admitted",
        candidate_sites=6, retained_sites=4, haplotypes=4, reference_haplotypes=2, query_haplotypes=2,
        counts_complete=True, unsupported_site_count=0, monomorphic_site_count=len(mono), monomorphic_sites=mono,
        source_GT_biological_nucleotides_and_phase_unchanged=True, polarization_flipped_sites=2,
        qc_count_erratum_ref=dict(v.QC_COUNT_ERRATUM_REF), qc_writer_ref=dict(v.QC_WRITER_REF),
        historical_reported_QC_counts=dict(extension=4148, original_2Mb=434),
        minimum_retained_SNPs=1000, prior_bridge_scope_full_charge_retained_bytes=50_000_000,
        retained_identity_sha256=sha(json.dumps(identities, separators=(",", ":")).encode()),
        source_VCF_sha256=sha(basis["source_VCF"]), source_QC_sha256=sha(basis["source_QC"]),
        AP_used=False, original_QC_failure_replaced=False,
        native_baseline_admissible_if_no_other_blocker=not monomorphic, map_qualification_required=True,
        native_gate_blockers=["separate_native_policy_and_admission", "fractional_native_age_roundtrip_policy"]
                             + (["monomorphic_actual40_sites"] if monomorphic else []), **gates())
    admission = dict(path="synthetic/admitted.json", sha256="0"*64)
    provenance = dict(schema="H018_observed_bridge_preparation_provenance_v1", condition_id=v.CONDITION,
        admitted_ref=admission, bridge_ref=dict(v.BRIDGE_REF), input_refs_ref=dict(v.INPUT_REFS_REF),
        qc_count_erratum_ref=dict(v.QC_COUNT_ERRATUM_REF), qc_writer_ref=dict(v.QC_WRITER_REF),
        historical_reported_QC_counts=dict(extension=4148, original_2Mb=434),
        prior_bridge_scope_full_charge_retained_bytes=50_000_000,
        source_VCF_ref=ref("source_VCF", basis["source_VCF"]), source_QC_ref=ref("source_QC", basis["source_QC"]),
        map_ref=ref("qualified.regional.map", basis["qualified.regional.map"]),
        roster_ref=ref("planned_haplotype_roster.json", basis["planned_haplotype_roster.json"]),
        metadata_completion_ref=ref("metadata_completion", basis["metadata_completion"]),
        metadata_independent_validation_ref=ref("metadata_independent_result_validation", basis["metadata_independent_result_validation"]),
        metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
        metadata_original_reservation_retained_bytes=5000000,
        dates="source calBP1950 / 28, fractional, no 2026 offset, rounding or clamp",
        coordinate_basis="GRCh37 chr22 absolute bp0 [20000000,30000000); Oxford/map local bp1",
        map_curve="qualified piecewise public source curve copied unchanged; no uniform substitution",
        extension_normalization_bp=10000000, future_GNN_span_denominator="log(2000001), unchanged",
        original_2Mb_failure=dict(retained=217, reported_retained=434, candidates=13312, minimum_required=1000, below_1000=True, replaced=False), **gates())
    output = {"baseline.haps": "".join(haps).encode(), "baseline.sample": sample, "sample_ages.generations": ages,
              "frozen_site_identities.json": enc(identities), "polarization_equivalence.json": enc(audits),
              "bridge_preparation.json": enc(report), "qualified.regional.map": basis["qualified.regional.map"],
              "NOTICE.txt": notice, "provenance.json": enc(provenance)}
    completed = dict(schema="H018_observed_bridge_completed_v1", status="completed_baseline_GT_preparation_not_native_admitted",
        job_id=v.JOB, condition_id=v.CONDITION, admitted_ref=admission, candidate_sites=6, retained_sites=4,
        qc_count_erratum_ref=dict(v.QC_COUNT_ERRATUM_REF), qc_writer_ref=dict(v.QC_WRITER_REF),
        historical_reported_QC_counts=dict(extension=4148, original_2Mb=434),
        prior_bridge_scope_full_charge_retained_bytes=50_000_000,
        haplotypes=4, reference_haplotypes=2, query_haplotypes=2, counts_complete=True, unsupported_site_count=0,
        monomorphic_site_count=len(mono), sites_dropped=0, source_GT_biological_nucleotides_and_phase_unchanged=True,
        metadata_original_OS_exit="unknown", metadata_original_wait0_available=False,
        metadata_original_reservation_retained_bytes=5000000, no_internal_OS_exit_or_wait0_claim=True,
        reservation_remains_fully_charged=True, historical_charges_unchanged=True, **gates())
    output["completed.json"] = enc(completed)
    refresh(output)
    return contract, basis, output


def refresh(output):
    """Rehash forged outputs so semantic failures cannot be just hash failures."""
    completed = json.loads(output["completed.json"])
    completed["assets"] = {name: dict(path=v.OUTPUT_DIRECTORY+"/"+name, bytes=len(output[name]), sha256=sha(output[name]))
                           for name in v.ASSET_NAMES if name in output}
    completed["monomorphic_details_ref"] = completed["assets"].get("bridge_preparation.json")
    output["completed.json"] = enc(completed)


def edit_json(output, name, edit):
    obj = json.loads(output[name]); edit(obj); output[name] = enc(obj)


def edit_haps(output, row, column, value):
    rows = output["baseline.haps"].decode().splitlines()
    fields = rows[row].split(); fields[column] = value; rows[row] = " ".join(fields)
    output["baseline.haps"] = ("\n".join(rows)+"\n").encode()


def run_checks():
    results = []
    def passed(name):
        results.append(dict(name=name, passed=True))
    for mono in (True, False):
        contract, basis, output = fixture(mono)
        receipt = v._verify_serialized(contract, basis, output)
        assert receipt["monomorphic_site_count"] == int(mono) and receipt["polarization_flipped_sites"] == 2
        assert receipt["actual_serialized_site_nucleotide_phase_order_verified"] is True
        assert receipt["native_execution_authorized"] is False and receipt["command_direct_wait_or_descendant_drain_certified"] is False
        assert receipt["status"] == "synthetic_verifier_check_only"
        passed("synthetic_full_roundtrip_with_"+("retained_monomorphic" if mono else "no_monomorphic")+"_site")
    def rejects(name, edit, expected):
        contract, basis, output = fixture()
        edit(basis, output)
        refresh(output)
        try:
            v._verify_serialized(contract, basis, output)
        except (v.ReadbackError, ValueError, KeyError, TypeError) as exc:
            assert expected in str(exc), (name, str(exc), expected)
            passed(name)
        else:
            raise AssertionError(name+" unexpectedly passed")
    rejects("phase_swap_with_same_allele_count_rejected", lambda b,o: (
        edit_haps(o,0,5,"1"), edit_haps(o,0,6,"0")), "phase/code")
    rejects("ancestral_ALT_without_XOR_rejected", lambda b,o: edit_haps(o,1,5,"0"), "phase/code")
    rejects("REF_assumed_ancestral_rejected", lambda b,o: edit_haps(o,1,3,"G"), "identity/AA-DA")
    rejects("regional_coordinate_off_by_one_rejected", lambda b,o: edit_haps(o,0,2,"0"), "identity/AA-DA")
    rejects("source_identity_name_change_rejected", lambda b,o: edit_haps(o,0,1,"other"), "identity/AA-DA")
    def reorder(b, o):
        rows=o["baseline.haps"].decode().splitlines(); rows[0],rows[1]=rows[1],rows[0]
        o["baseline.haps"]=("\n".join(rows)+"\n").encode()
    rejects("haps_row_reordering_rejected", reorder, "identity/AA-DA")
    def forge(b,o):
        edit_haps(o,0,5,"1")
        def change(a):
            a[0]["source_GT_haplotype_codes"][0]=1
            a[0]["native_haps_codes"][0]=1
            a[0]["nucleotide_equivalence_sha256"]=sha(b"CCCA")
        edit_json(o,"polarization_equivalence.json",change)
    rejects("joint_haps_and_audit_source_vector_forgery_rejected", forge, "phase/code")
    rejects("audit_source_vector_only_forgery_rejected", lambda b,o: edit_json(o,"polarization_equivalence.json",
        lambda a: a[0]["source_GT_haplotype_codes"].__setitem__(0,1)), "audit/source-vector")
    rejects("audit_phase_boolean_is_not_proof", lambda b,o: edit_json(o,"polarization_equivalence.json",
        lambda a: a[0].__setitem__("exact_nucleotide_and_phase_order_equivalence",False)), "audit/source-vector")
    def drop(b,o):
        rows=o["baseline.haps"].decode().splitlines(); rows.pop(2)
        o["baseline.haps"]=("\n".join(rows)+"\n").encode()
    rejects("retained_monomorphic_row_drop_rejected",drop,"row counts")
    rejects("new_identity_intersection_rejected", lambda b,o: edit_json(o,"frozen_site_identities.json",
        lambda a: a.pop(2)), "frozen identity")
    rejects("subject_sample_order_swap_rejected",lambda b,o:o.__setitem__("baseline.sample",
        b"ID_1 ID_2 missing\n0 0 0\nBETA BETA 0\nALPHA ALPHA 0\n"),"sample headers")
    rejects("fractional_age_rounding_rejected",lambda b,o:o.__setitem__("sample_ages.generations",
        b"36\n36\n44\n44\n"),"fractional ages")
    rejects("piecewise_map_substitution_rejected",lambda b,o:o.__setitem__("qualified.regional.map",
        o["qualified.regional.map"].replace(b"2000",b"1000")),"map substitution")
    rejects("false_zero_monomorphic_claim_rejected",lambda b,o:edit_json(o,"bridge_preparation.json",
        lambda a:a.__setitem__("monomorphic_site_count",0)),"independently reproduced")
    rejects("invented_original_metadata_wait0_rejected",lambda b,o:edit_json(o,"provenance.json",
        lambda a:a.__setitem__("metadata_original_wait0_available",True)),"remains unknown")
    rejects("native_authorization_escalation_rejected",lambda b,o:edit_json(o,"completed.json",
        lambda a:a.__setitem__("native_execution_authorized",True)),"false gates")
    rejects("original_source_bytes_drift_rejected",lambda b,o:b.__setitem__("source_QC",
        b["source_QC"].replace(b"\t1\n",b"\t0\n",1)),"Immutable original SHA")
    rejects("unexpected_serialized_artifact_rejected",lambda b,o:o.__setitem__("rescued.haps",b"x"),"ten-file")
    rejects("JSON_duplicate_audit_key_rejected",lambda b,o:o.__setitem__("polarization_equivalence.json",
        o["polarization_equivalence.json"].replace(b'"chrom": "22"',b'"chrom": "22", "chrom": "22"',1)),"Duplicate JSON key")
    # Reach original-source parsing with a synthetic-only re-pinned malformed GT.
    contract,basis,output=fixture()
    original=gzip.decompress(basis["source_VCF"])
    basis["source_VCF"]=gzip.compress(original.replace(b"0|1:42",b"0/1:42",1),mtime=0)
    newpins=dict(contract.pins);newpins["source_VCF"]=(newpins["source_VCF"][0],sha(basis["source_VCF"]))
    altered=contract._replace(pins=MappingProxyType(newpins))
    try:
        v._verify_serialized(altered,basis,output)
    except v.ReadbackError as exc:
        assert "phased callable GT" in str(exc)
        passed("independent_original_unphased_GT_rejected")
    else:
        raise AssertionError("unphased GT passed")
    # The public API cannot accept a synthetic pin/count override.
    contract,basis,output=fixture()
    try:
        v.verify_h018_serialized_v2(basis,output)
    except v.ReadbackError as exc:
        assert "Immutable original SHA" in str(exc)
        passed("public_H018_entry_rejects_synthetic_basis_and_has_no_override")
    else:
        raise AssertionError("synthetic fixture passed public H018")
    return dict(schema="H018_bridge_serialization_synthetic_checks_v1",status="PASS_SYNTHETIC_ONLY",
        checks=results,count=len(results),real_payloads_read=False,bridge_or_guard_imported=False,
        pipeline_or_admission_or_native_or_GPU_executed=False,synthetic_verifier_executed=True)


if __name__ == "__main__":
    print(json.dumps(run_checks(),sort_keys=True,indent=2))
