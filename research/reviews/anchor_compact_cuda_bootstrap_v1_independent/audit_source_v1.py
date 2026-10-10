"""Independent source-only seals and review receipts. No live/runtime state inspection."""
import ast
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
B=ROOT/"research/reviews/anchor_compact_cuda_bootstrap_v1_independent"
E=ROOT/"research/reviews/gpu_pool_research_v2_independent"
EXPECTED={
"src/arggnn/anchor_compact_cuda_bootstrap_v1.py":"18ac5db1617d5cb51d20ee66f8fb559a014d5b834bcbe4929bd3974703f7020b",
"configs/anchor_compact_cuda_bootstrap_v1.json":"986577a0ad0da16b8fb73823f3441819828f8055e7b18280dabe2e27cff4a215",
"scripts/gpu_pool_research_v2.py":"e85a39c39c1f3292bd3c7f26bd4b45eaeb94d87a3d6d224ca9194d959322c749",
"configs/gpu_pool_research_v2.json":"e10899412e2ba671c6e8fe39de39e6471643a1ed92dbb314c566851e21b26908"}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
for path,h in EXPECTED.items():assert digest(ROOT/path)==h,(path,"assigned bytes changed")
preserved=json.loads((ROOT/"research/reviews/gpu_pool_research_v2_author/preserved_source_baseline.json").read_text())
assert len(preserved)==32
pins={p:dict(expected=h,actual=digest(ROOT/p)) for p,h in preserved.items()}
assert all(v["expected"]==v["actual"] for v in pins.values())
for owned in (B,E):
    snapshot=json.loads((owned/"source_snapshot_manifest.json").read_text())
    assert all(digest(ROOT/p)==h==digest(owned/"sealed_source"/p) for p,h in snapshot.items())
    save(owned/"preserved_dependency_audit_final_v1.json",dict(all_32_frozen_sources_unchanged=True,count=32,checks=pins))
boot_cfg=json.loads((ROOT/"configs/anchor_compact_cuda_bootstrap_v1.json").read_text())
source_pins={p:dict(expected=h,actual=digest(ROOT/p)) for p,h in boot_cfg["source_sha256"].items()}
assert all(v["expected"]==v["actual"] for v in source_pins.values())
boot_tree=ast.parse((ROOT/"src/arggnn/anchor_compact_cuda_bootstrap_v1.py").read_text())
boot_imports=[ast.unparse(n) for n in boot_tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
assert not any("torch" in n or "qualification_runner" in n for n in boot_imports)
runner_tree=ast.parse((ROOT/"src/arggnn/anchor_compact_qualification_runner_v1.py").read_text())
observer=next(n for n in runner_tree.body if isinstance(n,ast.FunctionDef) and n.name=="observe_live")
observer_text=ast.unparse(observer)
assert observer_text.index("torch.cuda.is_initialized()")<observer_text.index("ctypes.CDLL")
run=next(n for n in boot_tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_bootstrap")
run_text=ast.unparse(run)
assert run_text.index("apply_and_validate_title")<run_text.index("load_cuda_modules")<run_text.index("configure_deterministic_development")<run_text.index("recheck_inputs(admission, initial=True)")<run_text.index("initialize_context")<run_text.index("observe_live")<run_text.index("runner.main")
assert all(flag in run_text for flag in ["bootstrap_execution","nvtop_snapshot","adapter_admission.json","runner_admission.json"])
ext_tree=ast.parse((ROOT/"scripts/gpu_pool_research_v2.py").read_text())
ext_text=ast.unparse(ext_tree)
assert 'modules=MODULES' in ext_text and 'Direct runner submission is forbidden' in ext_text
template_path=ROOT/"research/reviews/anchor_compact_cuda_bootstrap_v1_author/root_permit_template.json"
template=json.loads(template_path.read_text())
assert template["status"]=="inactive_template_root_must_issue_fresh_receipt" and template["root_delegate_live_evidence_issuance"] is False
actions=dict(GPU_context_or_controller_or_worker_started=False,Git_access_or_mutation=False,
Mac_project_write=False,active_permit_file_written=False,actual_bank_target_truth_inferred_or_holdout_arrays_opened=False,
global_repository_storage_scan=False,installation=False,ledger_or_live_runtime_state_access=False,
authored_module_config_or_author_bundle_mutation=False,only_handmade_CPU_mock_fixtures=True,
writes_only_owned_review_directories=True)
boot_checks=[
"exact assigned source/config hashes and all32 preserved dependencies",
"standard-library module import scope and inactive author permit",
"fresh root/config/review/extension/experiment admission and ledger/decision hash bindings before context",
"controller own-parent UID PID start-tick session PGID identity and inherited lease equality",
"complete eligible driver inventory and declared project compute PID identity; own existing context rejects",
"full registry lifecycle controller logical job exact args argv source peak storage and log reservation binding",
"launching null-PID capture window requires exact parent/session/lease/argv",
"complete owned1GB and separate1MiB log reservation inventory and projected peak strictly below450GB",
"all 30 frozen source-map and extra bootstrap extension ledger/decision source pin failure injections before CUDA",
"fixed Python executable/hash Torch metadata and driver precontext guard",
"exact title and frozen deterministic policy precede context initialization",
"metadata-only adapter admission exact original candidate/prepared/observed provenance joins",
"same-PID own UUID and two NVTOP snapshot PID/title/index equality before scientific runner",
"frozen actual runner validate_domains body accepts generated domain shapes and rejects malformed own GPU evidence",
"exact same-PID runner main argv and domain hashes preserved",
"cache/temp/config plus stdout/stderr inside owned1GB output and exclusive immutable evidence writes",
"negative initial/context/observation/domain failures cannot reach scientific runner",
]
extension_checks=[
"exact assigned launcher/config and bootstrap/config bytes plus all32 preserved dependencies",
"source-only exact six original plus runner/bootstrap allowlist and no base-global rebinding",
"fixed unchanged Controller injection with explicit modules",
"base controller/entrypoint/launcher/guard and independent runner PASS bytes verified",
"extension config/review shape failures and canonical source root symlink regular size guards",
"CLI direct runner and arbitrary modules reject before IPC",
"raw IPC runner registration acceptance disclosed; frozen runner own-context guard blocks before driver/NVTOP",
"literal unchanged v1 submission shape and peak/storage/log reservations",
"base v1 exact types paths argument shapes module rejection protocol validation",
"base source/output rechecks before Popen for added bootstrap",
"status wait shutdown delegate unchanged even after source drift",
]
concerns=[
dict(id="C1",scope="both",severity="operational_remaining",source_blocker=False,
finding="PASS covers source semantics and handmade CPU/mock fixtures only. Fresh root authorization and actual original TRAIN qualification/profile remain unperformed.",
evidence="No controller, worker, CUDA context, active permit, ledger/live state, actual arrays or scientific probes were used."),
dict(id="C2",scope="extension",severity="boundary_disclosure",source_blocker=False,
finding="The extension CLI rejects direct runner submission; its inherited controller allowlist still accepts the runner module through raw v1 IPC registration.",
evidence="Independent _validate_spec fixture accepts WORKER without launching. Extracted sealed observe_live rejects uninitialized own context before driver/NVTOP. This cannot create an unqualified context or run scientific probes."),
dict(id="C3",scope="bootstrap",severity="root_inventory_obligation",source_blocker=False,
finding="Before CUDA, complete project-device attribution depends on truthful root project_physical_gpus/project_compute_identities. Bootstrap compares driver compute rows only on declared project devices.",
evidence="Source check_inventory, root permit contract, and root owner clarification. Root must inspect full live ownership inventory, all current project GPUs, eligible devices and max-two ceiling before admission. Postcontext exact-title NVTOP set must match permit and frozen runner domains enforce eligible/max-two."),
dict(id="C4",scope="both",severity="root_storage_obligation",source_blocker=False,
finding="Owned1GB stage checks and the separately bounded1MiB pool log are reservations/stage observations, not continuous OS quotas.",
evidence="Exact split1001048576 base reserve and complete inventory arithmetic exercised. Root must reconcile persistent pool storage accounting without double count and maintain complete whole-repository writer/reservation accounting."),
]
for owned in (B,E):
    applicable=[c for c in concerns if c["scope"]=="both" or c["scope"]==("bootstrap" if owned==B else "extension")]
    with (owned/"concerns.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","scope","severity","source_blocker","finding","evidence"]);writer.writeheader();writer.writerows(applicable)
    save(owned/"source_audit_v1.json",dict(actual_execution_admitted=False,all_32_frozen_sources_unchanged=True,
        assigned_source_sha256=EXPECTED,bootstrap_source_map_count=len(source_pins),bootstrap_source_map_checks=source_pins,
        module_scope_imports=boot_imports,source_blocking_findings=[],concerns=applicable,actions=actions,
        reviewed_contracts=boot_checks if owned==B else extension_checks,
        frozen_runner_receipt=dict(path="research/reviews/anchor_compact_qualification_runner_v1_independent/review_receipt.json",
        sha256=digest(ROOT/"research/reviews/anchor_compact_qualification_runner_v1_independent/review_receipt.json"))))
save(E/"test_failure_attempt01_receipt.json",dict(kind="independent_test_harness_attempt",actual_execution_admitted=False,
    source_verdict=None,test_collection_passed=False,returncode=2,
    log="pytest_independent_attempt01.txt",preserved_source="test_independent_attempt01.py",
    cause="pytest parameter name request is reserved",repair="Renamed own test parameter to request_value; no authored implementation changed",
    subsequent_result="57 passed in 2.73s",subsequent_log="pytest_independent_attempt02.txt"))
readme_b="""# Independent compact CUDA bootstrap v1 review

PASS applies to the exact assigned source/config bytes and handmade CPU/mock contract review. Actual execution is not admitted.

The focused replay passed 62 tests. The first independent suite passed 150. The expanded final suite passed 201 and includes all frozen source-map failure gates, adapter metadata provenance, the frozen runner domain-validation body, and its own UUID/two-NVTOP-snapshot observer body. Imports inside those two pure contract bodies were replaced by handmade dependencies without importing scientific code, opening actual arrays, reading live state, querying the driver or creating a CUDA context.

Focused author tests were copied into this owned directory only to replay the advertised coverage. PROJECT points at the canonical repository; all test temporary/cache paths remain under owned review directories. Those replays do not supply the independent verdict. The independent tests use the authored in-memory scaffold for setup and independently choose attacks and extracted frozen body validation.

All 32 baseline dependencies and all 30 bootstrap source-map pins still match. Snapshots, every test attempt, outputs and hash manifests are preserved. No authored file, source/config, author bundle, ledger, live runtime state, active permit, controller, GPU scheduling, installation or Git action was changed/accessed.

Before actual admission, root must supply truthful complete live ownership/device inventory, eligible/max-two/natural-drain checks, fresh experiment/runtime/storage/lease/controller/registry authorization and owned1GB plus separate1MiB pool-log accounting. Own UUID/title/PID evidence and exact-title NVTOP project-device coverage must be observed by the real same-PID bootstrap before the frozen runner. Stage caps are not continuous quotas. See concerns.csv for all remaining obligations.
"""
readme_e="""# Independent GPU pool research v2 review

PASS applies to the exact assigned launcher/config and compact bootstrap/config bytes, source orchestration, and handmade CPU/mock contract review. Actual execution is not admitted.

The focused replay passed 36 tests; the independent final suite passed 57. The first independent attempt failed during collection because an own test used pytest's reserved request parameter name. Its source/log/failure receipt are preserved. Renaming that test parameter fixed collection without changing authored code.

The CLI rejects direct sealed-runner submission before IPC. The fixed inherited controller allowlist includes the runner, so a raw v1 IPC registration may accept that module. This boundary is disclosed and is not a claim that raw IPC registration is rejected. The frozen runner observer requires an already qualified own CUDA context and has no context-creation path; the independent extracted guard-body fixture rejects before driver/NVTOP. Consequently the direct path cannot establish an unqualified CUDA context or execute scientific probes.

Original scheduler source, old six-module allowlist globals, launcher and guards remain unchanged. All 32 preserved dependencies match. All sockets used by focused IPC replay were isolated fixture sockets under this owned directory; no actual controller/socket/lease/live state or worker was used. All test temp/cache paths remain project-local and owned.

Root owns truthful full live project-device inventory and max-two checks, free-device selection/natural drain, fresh permits, complete experiment/runtime/lease/storage admission and persistent accounting. The paired bootstrap receipt grants no actual execution. See concerns.csv.
"""
(B/"README.md").write_text(readme_b)
(E/"README.md").write_text(readme_e)
def manifest(owned):
    result={}
    for path in owned.rglob("*"):
        if path.is_file() and not path.is_symlink() and "tmp" not in path.relative_to(owned).parts and "t" not in path.relative_to(owned).parts and path.name not in {"artifact_manifest_v1.json","review_receipt.json","review_seal.json"} and not path.name.startswith("audit_attempt"):
            result[str(path.relative_to(owned))]=digest(path)
    save(owned/"artifact_manifest_v1.json",result)
    return result
now=dt.datetime.now(dt.timezone.utc).isoformat()
for owned,kind,checks,tests in [
(B,"anchor_compact_cuda_bootstrap_v1_independent_source_CPU_review",boot_checks,[
dict(attempt=1,role="focused_replay",source="test_focused_replay_v1.py",log="pytest_focused_attempt01.txt",result="62 passed in 0.79s",returncode=0),
dict(attempt=1,role="independent",source="test_independent_attempt01.py",log="pytest_independent_attempt01.txt",result="150 passed in 1.88s",returncode=0),
dict(attempt=2,role="independent",source="test_independent_v1.py",log="pytest_independent_attempt02.txt",result="201 passed in 2.46s",returncode=0)]),
(E,"gpu_pool_research_v2_independent_source_CPU_review",extension_checks,[
dict(attempt=1,role="focused_replay",source="test_focused_replay_v1.py",log="pytest_focused_attempt01.txt",result="36 passed in 1.22s",returncode=0),
dict(attempt=1,role="independent",source="test_independent_attempt01.py",log="pytest_independent_attempt01.txt",result="test collection failed: reserved request parameter name",returncode=2),
dict(attempt=2,role="independent",source="test_independent_v1.py",log="pytest_independent_attempt02.txt",result="57 passed in 2.73s",returncode=0)])]:
    artifacts=manifest(owned)
    own_bytes=sum(p.stat().st_size for p in owned.rglob("*") if p.is_file() and not p.is_symlink())
    receipt=dict(kind=kind,verdict="PASS",verdict_scope="Exact assigned source/config bytes, source orchestration and independent handmade CPU/mock guard contracts only",
    actual_execution_admitted=False,project_root=str(ROOT),created_utc=now,
    bootstrap_sha256=EXPECTED["src/arggnn/anchor_compact_cuda_bootstrap_v1.py"],
    bootstrap_config_sha256=EXPECTED["configs/anchor_compact_cuda_bootstrap_v1.json"],
    all_32_frozen_sources_unchanged=True,all_reviewed_source_and_snapshot_bytes_unchanged=True,
    blocking_source_findings=[],reviewed_contracts=checks,actions=actions,
    independent_tests=dict(attempts=tests,final_passed=201 if owned==B else 57,
    author_focused_replay_substituted_for_independent_verdict=False,
    CUDA_VISIBLE_DEVICES="",no_scientific_or_Torch_import=True,
    only_handmade_CPU_or_mocked_context_operations=True,all_attempts_preserved=True),
    runtime=dict(python_import_scope="standard library and pytest only for reviewed guards; no scientific/Torch imports",CUDA_VISIBLE_DEVICES="",CUDA_context_initialized=False),
    artifact_sha256=artifacts,artifact_manifest_sha256=digest(owned/"artifact_manifest_v1.json"),
    source_audit_sha256=digest(owned/"source_audit_v1.json"),source_snapshot_manifest_sha256=digest(owned/"source_snapshot_manifest.json"),
    preserved_dependency_audit_sha256=digest(owned/"preserved_dependency_audit_final_v1.json"),
    concerns=dict(path="concerns.csv",source_blockers=0,remaining_root_operational_obligations=True),
    owned_review_regular_file_bytes=own_bytes,
    outstanding=["Fresh truthful root full live ownership/device eligible/max-two and natural-drain authorization",
    "Fresh exact current experiment/runtime/storage/controller/lease/registry permit; persistent owned1GB+1MiB log accounting",
    "Actual original TRAIN qualification and all frozen parity checks before synchronized profiling",
    "No efficiency biological accuracy novelty training-trajectory checkpoint or early-stopping claim established"])
    if owned==E:
        receipt.update(launcher_sha256=EXPECTED["scripts/gpu_pool_research_v2.py"],config_sha256=EXPECTED["configs/gpu_pool_research_v2.json"],
            direct_runner_property=dict(CLI_rejected_before_IPC=True,raw_IPC_registration_rejected=False,
            frozen_runner_requires_qualified_own_context=True,unqualified_context_or_scientific_execution_reachable=False))
    else:
        receipt.update(inactive_author_template_sha256=digest(template_path),
            root_precontext_inventory_completeness_is_authoritative_obligation=True,
            postcontext_exact_title_project_NVTOP_set_rechecked=True)
    save(owned/"review_receipt.json",receipt)
    save(owned/"review_seal.json",dict(actual_execution_admitted=False,
        review_receipt_sha256=digest(owned/"review_receipt.json"),artifact_manifest_sha256=digest(owned/"artifact_manifest_v1.json"),
        assigned_source_sha256=EXPECTED,all_32_frozen_sources_unchanged=True))
    print(str(owned.relative_to(ROOT)/"review_receipt.json"),receipt["verdict"],digest(owned/"review_receipt.json"),"owned_bytes",own_bytes)
