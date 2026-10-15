"""Read-only source/manifest checks. Imports no producer, worker, guard or payload."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
REL = "research/reviews/H018_QC_boolean_serialization_repair_source_v1"
HERE = ROOT/REL
OLD = ROOT/"research/reviews/H018_observed_bridge_root_source_v1"
OLD_CORE = ROOT/"research/reviews/H018_observed_bridge_source_v1/bridge_core.py"
OLD_VERIFIER = ROOT/"research/reviews/H018_bridge_serialization_readback_source_v1/serialization_readback.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def read(name):
    return json.loads((HERE/name).read_bytes())


def tree(path):
    return ast.parse(path.read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


def functions(value):
    return {n.name: n for n in value.body if isinstance(n, ast.FunctionDef)}


def assignments(value):
    return {t.id: n.value for n in value.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


EXTRA = {"qc_count_erratum_ref", "qc_writer_ref", "historical_reported_QC_counts",
         "prior_bridge_scope_full_charge_retained_bytes"}
REMOVED_MESSAGES = {"Exact root count erratum and original QC writer source bindings",
                    "Root verified actual frozen QC counts; incorrect reported history preserved",
                    "Pure core bound to exact root count erratum and upstream writer"}


class ReverseDriverRepair(ast.NodeTransformer):
    """Undo only the enumerated scientific-count/provenance additions."""
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and any(isinstance(a, ast.Constant) and a.value in REMOVED_MESSAGES for a in node.value.args):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "erratum" for t in node.targets):
            return None
        if any(isinstance(t, ast.Tuple) and [x.id for x in t.elts] == ["extension", "original"] for t in node.targets):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) is int and node.value == 2074:
            node.value = 4148
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            is_original = any(k.arg == "reported_retained" for k in node.keywords)
            is_config = any(k.arg == "schema" and isinstance(k.value, ast.Constant)
                            and k.value.value == "H018_observed_bridge_root_config_v1" for k in node.keywords)
            drop = EXTRA | ({"reported_retained", "minimum_required"} if is_original else set())
            drop |= {"candidate_sites", "retained_sites", "minimum_retained_SNPs"} if is_config else set()
            node.keywords = [k for k in node.keywords if k.arg not in drop]
            if is_original:
                next(k for k in node.keywords if k.arg == "retained").value = ast.Constant(value=434)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        node.values = [v for v in node.values if not (
            isinstance(v, ast.Compare) and isinstance(v.left, ast.Subscript)
            and isinstance(v.left.value, ast.Name) and v.left.value.id == "result"
            and isinstance(v.left.slice, ast.Constant) and v.left.slice.value in EXTRA)]
        return node


def check():
    checks = {}
    old_driver = tree(OLD/"root_bridge_driver_v3.py")
    new_driver = tree(HERE/"root_bridge_driver.py")
    od, nd = functions(old_driver), functions(new_driver)
    oa, na = assignments(old_driver), assignments(new_driver)
    cfg, oldcfg = read("config.json"), json.loads((OLD/"config_v3.json").read_bytes())
    manifest, oldm = read("source_manifest.json"), json.loads((OLD/"source_manifest_v3.json").read_bytes())
    request = read("root_request.inactive.json")
    pins = {
        OLD_CORE: "a860ebf2f046834871d06a1938f0a1de0431cc20f698c1c5096b87bdc4ef89d2",
        OLD_VERIFIER: "eb32b6ad22b81276d88437ffe3c0bcef36e0ccb773788fe8fc19a4b11e12b6ff",
        OLD/"root_bridge_driver_v3.py": "b36af1a0bacf85bd46acb73a0350a923c1de0974e1b7f9eb21efd72736c2991b",
        OLD/"config_v3.json": "960205388a47a9a9a9ee1e809baf5747d59dd1b74013bd367fb2f62e89e841bb",
        OLD/"source_manifest_v3.json": "513c5dd292740e519c269b1979dd5be3713fad9751ebcbba7f021733606ff6d4",
        OLD/"root_request.inactive_v3.json": "f02f00455ff76b4d9f3f82ec6ce222e9026084ff730630fb8b1d80bb17e32d74",
    }
    checks["original_producer_wrapper_verifier_and_basis_source_pins_unchanged"] = all(sha(p) == h for p, h in pins.items())
    checks["all_functions_outside_three_explicit_repair_sites_AST_identical"] = set(od) == set(nd) and all(
        dump(v) == dump(nd[k]) for k,v in od.items() if k not in {"verify_request", "worker", "verify_output"})
    normalized = {k: ReverseDriverRepair().visit(copy.deepcopy(nd[k])) for k in ("verify_request", "worker", "verify_output")}
    checks["three_changed_driver_functions_revert_exactly_to_original_AST"] = all(dump(normalized[k]) == dump(od[k]) for k in normalized)
    changed_globals = {"REL", "SOURCE", "CONFIG", "MANIFEST", "PACKET", "AUX", "OUTPUT", "JOB", "KEY", "BRIDGE"}
    checks["all_budgets_guard_runtime_pins_and_other_globals_AST_identical"] = set(na) == set(oa)|{"QC_COUNT_ERRATUM_REF", "QC_WRITER_REF"} and all(
        dump(v) == dump(na[k]) for k,v in oa.items() if k not in changed_globals)
    checks["fresh_names_and_separate_output_packet_aux_scope"] = all(ast.literal_eval(na[k]) == val for k,val in dict(
        REL=REL, OUTPUT="data/H018_R16_observed_bridge_v2", JOB="H018-observed-baseline-GT-preparation-v2",
        KEY="H018/observed_baseline_GT_preparation_v2").items()) and all(
        isinstance(na[k], ast.BinOp) and isinstance(na[k].left, ast.Name) and na[k].left.id == "REL"
        and ast.literal_eval(na[k].right) == val for k,val in dict(SOURCE="/root_bridge_driver.py",CONFIG="/config.json",
        MANIFEST="/source_manifest.json",PACKET="/admission_v1",AUX="/worker_aux_v2").items())
    checks["both_worker_and_launch_keep_qualified_ROOT_plus_src_PYTHONPATH"] = (HERE/"root_bridge_driver.py").read_text().count('PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT / "src"))') == 2
    config_changes = {"auxiliary_directory","packet_directory","output_directory","job_id","bridge_ref",
                      "qc_count_erratum_ref","qc_writer_ref","candidate_sites","retained_sites","minimum_retained_SNPs"}
    checks["config_changes_only_fresh_identity_count_erratum_and_writer_bindings"] = {k:v for k,v in cfg.items() if k not in config_changes} == {k:v for k,v in oldcfg.items() if k not in config_changes}
    checks["exact_fifty_MB_budget_and_original_resource_bounds"] = cfg["reserved_bytes"] == 50_000_000 and cfg["output_batch_cap_bytes"] == 32_000_000 and cfg["core_payload_cap_bytes"] == 25_000_000 and cfg["auxiliary_guard_growth_bytes"] == 3_000_000 and cfg["worker_seconds"] == 300 and cfg["supervisor_seconds"] == 600 and cfg["CPU_threads"] == 1
    checks["actual_2074_count_and_unchanged_1000_threshold_configured"] = cfg["candidate_sites"] == 94892 and cfg["retained_sites"] == 2074 and cfg["minimum_retained_SNPs"] == 1000
    checks["entire_qualified_metadata_mapping_preserved_without_opening_targets"] = manifest["metadata_refs"] == oldm["metadata_refs"]
    allowed_refs = {"driver","config","pure_bridge","independent_serialization_verifier","qc_count_erratum","upstream_QC_writer"}
    checks["old_source_evidence_preserved_except_exact_new_producers_and_added_erratum_writer"] = set(manifest["source_refs"]) == set(oldm["source_refs"])|{"qc_count_erratum","upstream_QC_writer"} and all(manifest["source_refs"][k] == v for k,v in oldm["source_refs"].items() if k not in allowed_refs)
    checks["all_full_source_refs_hash_match_without_payload_reads"] = all(ref(ROOT/r["path"]) == r for r in manifest["source_refs"].values())
    checks["original_full_manifest_fields_and_transitive_exclusions_preserved"] = all(manifest[k] == v for k,v in oldm.items() if k != "source_refs")
    suffixes = ast.literal_eval(na["DIRECT_BINDING_SUFFIXES"])
    excluded = {g:{k:r for k,r in manifest[g].items() if Path(r["path"]).suffix not in suffixes} for g in ("source_refs","metadata_refs")}
    direct = [r for g in ("source_refs","metadata_refs") for r in manifest[g].values() if Path(r["path"]).suffix in suffixes]
    checks["same_direct_suffixes_and_all_four_excluded_refs_remain_transitive"] = suffixes == (".json",".py",".sh") and excluded == manifest["transitive_only_refs"] and sum(len(x) for x in excluded.values()) == 4
    checks["request_exact_new_pins_and_remains_inactive"] = request["status"] == "inactive_source_plan" and request["root_authorizer"] is None and request["registration_ref"] is None and request["registered_before_execution"] is False and request["replay_authorized"] is False and request["driver_ref"] == ref(HERE/"root_bridge_driver.py") and request["config_ref"] == ref(HERE/"config.json") and request["source_manifest_ref"] == ref(HERE/"source_manifest.json") and all(request[k] == cfg[k] for k in ("auxiliary_directory","packet_directory","output_directory","job_id","reserved_bytes"))
    err = cfg["qc_count_erratum_ref"]; writer = cfg["qc_writer_ref"]
    checks["canonical_exact_root_erratum_and_writer_pinned"] = err == dict(path="research/reviews/H018_QC_count_erratum_root_20261015_v1/actual_QC_site_count_erratum_v2.json",sha256="47db1404927848d2dfa836f5760ded106fe50f868e699616ff339ccc493b1f78") and writer == dict(path="research/reviews/H018_R16_decode_performance_source_v1/prospective_worker_v3.py",sha256="8fb225c48b791e3876eb51116a4ee1996c775c5794a092d5904f97e958bfeda6") and manifest["source_refs"]["qc_count_erratum"] == err and manifest["source_refs"]["upstream_QC_writer"] == writer
    oldcore,newcore = functions(tree(OLD_CORE)),functions(tree(HERE/"bridge_core_v2.py"))
    checks["all_other_core_functions_including_polarization_and_HS_AST_identical"] = set(oldcore) == set(newcore) and all(dump(v) == dump(newcore[k]) for k,v in oldcore.items() if k not in {"frozen_mask","assemble_baseline"})
    normalized_mask = ast.get_source_segment((HERE/"bridge_core_v2.py").read_text(),newcore["frozen_mask"]).replace('("0", "1"), "Exact original serialized QC Boolean 0/1"','("True", "False"), "Exact original serialized QC Boolean"').replace('if flag == "1":','if flag == "True":').replace("Exact94892/2074 original frozen mask; root counter erratum bound","Exact94892/4148 frozen mask")
    checks["core_frozen_mask_changes_only_exact_01_and_correct_count_diagnostic"] = dump(ast.parse(normalized_mask).body[0]) == dump(oldcore["frozen_mask"])
    core_assemble = copy.deepcopy(newcore["assemble_baseline"])
    for node in ast.walk(core_assemble):
        if isinstance(node,ast.Constant) and node.value == "Exact all-candidate and2074-site identity/order join":
            node.value = "Exact all-candidate and4148-site identity/order join"
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == "dict":
            node.keywords = [k for k in node.keywords if k.arg not in EXTRA|{"minimum_retained_SNPs"}]
    checks["core_assemble_science_AST_identical_except_count_diagnostic_and_erratum_report_fields"] = dump(core_assemble) == dump(oldcore["assemble_baseline"])
    oldv,newv = functions(tree(OLD_VERIFIER)),functions(tree(HERE/"readback_verifier_v2.py"))
    checks["all_other_independent_parser_functions_AST_identical"] = set(newv) == (set(oldv)-{"verify_h018_serialized_v1"})|{"verify_h018_serialized_v2"} and all(dump(v) == dump(newv[k]) for k,v in oldv.items() if k not in {"_basis","_qc","_verify_serialized","verify_h018_serialized_v1"})
    qc_source = ast.get_source_segment((HERE/"readback_verifier_v2.py").read_text(),newv["_qc"]).replace('("0", "1"), "Frozen QC Boolean spelling 0/1"','("True", "False"), "Frozen QC Boolean spelling"').replace('row["H018_fixed_QC_pass"] == "1"','row["H018_fixed_QC_pass"] == "True"')
    checks["independent_QC_parser_changes_only_exact_01"] = dump(ast.parse(qc_source).body[0]) == dump(oldv["_qc"])
    api = copy.deepcopy(newv["verify_h018_serialized_v2"]); api.name = "verify_h018_serialized_v1"
    checks["public_verifier_API_unchanged_except_version_name_no_overrides"] = dump(api) == dump(oldv["verify_h018_serialized_v1"])
    for filename in ("bridge_core_v2.py","root_bridge_driver.py","readback_verifier_v2.py"):
        text=(HERE/filename).read_text()
        checks["root_erratum_and_writer_literal_bound_in_"+filename] = err["sha256"] in text and writer["sha256"] in text
    diffs = [
        ("bridge_core_v1_to_v2.diff",OLD_CORE,HERE/"bridge_core_v2.py"),
        ("readback_verifier_v1_to_v2.diff",OLD_VERIFIER,HERE/"readback_verifier_v2.py"),
        ("root_driver_v3_to_repair_v2.diff",OLD/"root_bridge_driver_v3.py",HERE/"root_bridge_driver.py")]
    checks["all_exact_diffs_recorded"] = all((HERE/name).read_text() == "".join(difflib.unified_diff(
        a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile=a.relative_to(ROOT).as_posix(),tofile=b.relative_to(ROOT).as_posix())) for name,a,b in diffs)
    for p in HERE.glob("*.py"):
        compile(p.read_text(),str(p),"exec")
    checks["all_Python_sources_compile_as_text_without_import"] = True
    failed = [k for k,v in checks.items() if not v]
    assert not failed, failed
    return dict(schema="H018_QC_boolean_repair_source_checks_v1",all_passed=True,checks=checks,count=len(checks),
                source_ref_count=len(manifest["source_refs"]),metadata_ref_count=len(manifest["metadata_refs"]),known_direct_ref_count=len(direct),
                erratum_ref=err,writer_ref=writer,source_only=True,original_payloads_or_failed_packets_or_live_state_read=False,
                producer_driver_guard_imported=False,admission_accounting_worker_or_native_execution=False,
                actual_guard_consumer_count_or_admission_acceptance_claimed=False,
                statement="All supervision, wait/drain, admission and guard functions remain AST-identical. Enumerated count/erratum/provenance changes reverse exactly to original producer science. Source checks do not certify runtime admission.")


if __name__ == "__main__":
    print(json.dumps(check(),sort_keys=True,indent=2))
