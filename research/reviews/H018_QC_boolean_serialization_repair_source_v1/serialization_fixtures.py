"""Synthetic byte fixtures only. Never imports acquisition, root driver or guard."""
from pathlib import Path
import ast
import copy
import csv
import hashlib
import importlib.util
import io
import json

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = Path(__file__).resolve().parent
WRITER = ROOT/"research/reviews/H018_R16_decode_performance_source_v1/prospective_worker_v3.py"
OLD_CORE = ROOT/"research/reviews/H018_observed_bridge_source_v1/bridge_core.py"


def load_tests():
    spec = importlib.util.spec_from_file_location("_h018_repair_synthetic_tests", HERE/"synthetic_readback_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def mask_namespace(path):
    # Tiny counts live only in this isolated AST fixture namespace. The public
    # producer/verifier contract remains fixed and accepts no count override.
    funcs = functions(path)
    tree = ast.Module(body=[copy.deepcopy(funcs[n]) for n in ("require", "frozen_identity", "frozen_mask")], type_ignores=[])
    ns = dict(csv=csv, io=io, BEGIN=100, END=200, CANDIDATES=6, RETAINED=4,
              MAX_QC_BYTES=10_000, BASES=frozenset("ACGT"))
    exec(compile(tree, "<synthetic-only exact parser AST>", "exec"), ns)
    return ns


def run_checks():
    tests = load_tests()
    v = tests.v
    checks = {}
    writer_raw = WRITER.read_bytes()
    assert hashlib.sha256(writer_raw).hexdigest() == v.QC_WRITER_REF["sha256"]
    tree = ast.parse(writer_raw)
    write = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and n.lineno == 438)
    explicit = next(n for n in ast.walk(tree) if isinstance(n, ast.AugAssign) and n.lineno == 429)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and n.lineno == 430)
    # Execute only these exact, audited in-memory serialization/counter nodes.
    serialize = compile(ast.Module(body=[copy.deepcopy(write)], type_ignores=[]), "<exact pinned writer serialization fixture>", "exec")
    accumulation = compile(ast.Module(body=[copy.deepcopy(explicit), copy.deepcopy(loop)], type_ignores=[]), "<exact pinned counter fixture>", "exec")
    raw = io.BytesIO()
    values = dict(qc=raw, qcfields=["yes", "no", "missing", "number", "text"],
                  row=dict(yes=True, no=False, missing=None, number=17, text="A"))
    exec(serialize, values)
    assert raw.getvalue() == b"1\t0\t\t17\tA\n"
    checks["exact_pinned_writer_AST_emits_01_preserves_other_fields"] = True
    from collections import Counter
    counters = Counter()
    for passed in (True, False, True):
        exec(accumulation, dict(counters=counters, row={"H018_fixed_QC_pass": passed, "other_QC": passed}))
    assert counters["H018_fixed_QC_pass_sites"] == 4 and counters["other_QC_sites"] == 2
    checks["exact_pinned_writer_AST_proves_double_accumulation"] = True

    contract, basis, output = tests.fixture()
    old = mask_namespace(OLD_CORE)
    new = mask_namespace(HERE/"bridge_core_v2.py")
    try:
        old["frozen_mask"](basis["source_QC"])
    except ValueError as exc:
        assert "Exact original serialized QC Boolean" in str(exc)
    else:
        raise AssertionError("Original parser unexpectedly accepted actual writer spelling")
    checks["original_parser_rejects_writer_01_bytes"] = True
    candidates, mask = new["frozen_mask"](basis["source_QC"])
    independent_candidates, independent_retained = v._qc(contract, basis["source_QC"])
    assert [list(x) for x in candidates] == independent_candidates
    assert [(list(k), value) for k, value in mask.items()] == independent_retained
    assert len(candidates) == 6 and len(mask) == 4
    assert [x[1] for x in mask] == [101, 120, 150, 180]
    checks["both_exact_01_parsers_preserve_every_candidate_and_frozen_decision_order"] = True
    for flag in ("True", "False", "01", "00", "1.0", "0.0", " 1", "1 ", "+1", "-1", "true", ""):
        rows = basis["source_QC"].decode().splitlines()
        first = rows[1].split("\t"); first[-1] = flag
        rows[1] = "\t".join(first)
        altered = ("\n".join(rows)+"\n").encode()
        for name, parse in (("core", new["frozen_mask"]), ("independent", lambda raw: v._qc(contract, raw))):
            try:
                parse(altered)
            except ValueError as exc:
                assert "Boolean" in str(exc), (name, flag, str(exc))
            else:
                raise AssertionError((name, flag, "noncanonical flag accepted"))
        checks["both_parsers_reject_"+repr(flag)] = True
    for field in ("qc_count_erratum_ref", "qc_writer_ref"):
        for asset in ("bridge_preparation.json", "provenance.json", "completed.json"):
            _, _, forged = tests.fixture()
            tests.edit_json(forged, asset, lambda obj: obj[field].__setitem__("sha256", "0"*64))
            tests.refresh(forged)
            try:
                v._verify_serialized(contract, basis, forged)
            except v.ReadbackError:
                pass
            else:
                raise AssertionError((field, asset, "false binding passed"))
            checks["forged_"+field+"_in_"+asset+"_rejected"] = True
    for field, value in (("historical_reported_QC_counts", {"extension": 2074, "original_2Mb": 217}),
                         ("prior_bridge_scope_full_charge_retained_bytes", 0)):
        _, _, forged = tests.fixture()
        tests.edit_json(forged, "completed.json", lambda obj: obj.__setitem__(field, value))
        tests.refresh(forged)
        try:
            v._verify_serialized(contract, basis, forged)
        except v.ReadbackError:
            pass
        else:
            raise AssertionError((field, "history/charge rewrite passed"))
        checks["historical_field_"+field+"_rewrite_rejected"] = True
    _, _, forged = tests.fixture()
    tests.edit_json(forged, "provenance.json", lambda obj: obj["original_2Mb_failure"].__setitem__("retained", 434))
    tests.refresh(forged)
    try:
        v._verify_serialized(contract, basis, forged)
    except v.ReadbackError as exc:
        assert "coordinate/date/map/normalization" in str(exc)
    else:
        raise AssertionError("Old incorrect 2 Mb count passed as actual")
    checks["reported_434_rejected_as_actual_original_count"] = True
    assert v.H018.candidates == 94892 and v.H018.retained == 2074
    assert v.ORIGINAL_BASIS_REFS["qc_count_erratum"] == v.QC_COUNT_ERRATUM_REF
    api = functions(HERE/"readback_verifier_v2.py")["verify_h018_serialized_v2"]
    assert len(api.args.args) == 2 and not api.args.kwonlyargs and api.args.kwarg is None and api.args.vararg is None
    checks["public_contract_fixed_94892_2074_erratum_pinned_no_override"] = True
    independent = tests.run_checks()
    assert independent["count"] >= 24
    return dict(schema="H018_QC_serialization_repair_synthetic_checks_v1", status="PASS_SYNTHETIC_ONLY",
                checks=checks, count=len(checks), independent_roundtrip_checks=independent,
                original_payloads_read=False, source_metadata_read=True, acquisition_driver_guard_or_native_imported=False,
                admission_or_worker_or_process_or_native_execution=False,
                statement="Executed only tiny in-memory synthetic fixtures and exact extracted serialization/counter/parser AST nodes. No production count override was added.")


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True, indent=2))
