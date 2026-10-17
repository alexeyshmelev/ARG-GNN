"""Independent H058 v8 whole source review, source/metadata and RAM doubles only."""
from pathlib import Path
from types import SimpleNamespace
import ast,contextlib,copy,hashlib,io,json,os,runpy,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v8"
REVIEW="research/reviews/H058_GPU0_v8_whole_source_independent_20261017_v1"
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
rows=[];witnesses=[]
def check(name,condition=True,**evidence):
    if not condition:raise AssertionError(name)
    rows.append(dict(name=name,status="PASS",**evidence))
def rf(path):
    return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def read(path):return json.loads((ROOT/path).read_bytes())
def emit(nodes,scope):
    mod=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]))
    exec(compile(mod,"<independent_exact_source_AST_RAM>","exec"),scope)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
expected={
 "source_handoff.v8.json":"40f042d8ba0a61aea913765c2e46babed4ad87d4afd6b134d1d74c19092d5859",
 "source_manifest.v1.json":"2e1d18d4cfde8cf79be0dc9faffbdfed7619b05f94cf2a069f9ad65a8e9d87e4",
 "operational_v1.py":"a286793d0711f10c5e267df572d6c108ed9a914cfc00fc528670a58b6b6a9bf5"}
for name,digest in expected.items():check("exact_ROOT_supplied_candidate_"+name,rf(D+"/"+name)["sha256"]==digest)
manifest=read(D+"/source_manifest.v1.json")
check("all_402_declared_source_and_bounded_metadata_refs_match",
 len(manifest["refs"])==402 and all(rf(r["path"])==r for r in manifest["refs"]))
check("exact_source_map",manifest["source_sha256"]=={r["path"]:r["sha256"]for r in manifest["refs"]})
# Reproduce inspected author checks once without the optional result-writing flag.
captured=io.StringIO()
with contextlib.redirect_stdout(captured):
    author=runpy.run_path(str(ROOT/D/"targeted_checks_v8.py"),run_name="H058_v8_author_checks_reproduced_by_independent_reviewer")
check("inspected_author_AST_RAM_checks_reproduced_once",
 author["result"]["status"]=="PASS"and author["result"]["assertions"]==116
 and author["result"]["negative_fixtures"]==14,
 author_assertions=author["result"]["assertions"],author_negatives=author["result"]["negative_fixtures"])
c,d,p=author["c"],author["d"],author["p"]
check("actual_roles_separate_and_shared_file_pinned",d is not p and d.SOURCE==p.SOURCE==c.DRIVER
 and d.__file__==p.__file__==str(ROOT/c.DRIVER))
check("role_sources_byte_exact_mirrors",author["op"]._DRIVER_SOURCE==(ROOT/D/"driver_role.source.txt").read_text()
 and author["op"]._PRODUCER_SOURCE==(ROOT/D/"producer_role.source.txt").read_text())
protocol=p.fixed_protocol_v1();cfg=read(D+"/scientific_config.INACTIVE.template.v1.json")
check("exact_registered_scientific_protocol",cfg["scientific_protocol"]==protocol)
check("whole32_TRAIN_8VAL_1920updates_12checkpoints_96pretruth_predictions",
 len(protocol["TRAIN_parent_ids"])==32 and len(protocol["VAL_parent_ids"])==8
 and protocol["region_length_bp"]==2000000 and protocol["updates_all_new_arms"]==1920
 and protocol["updates_per_arm"]==960 and protocol["passes"]==30
 and protocol["scheduled_checkpoint_files"]==12
 and protocol["VAL_prediction_products_required_before_truth"]==96)
check("original_warmstart_and_fresh_optimizer_contract",protocol["warmstart"]=="completed_H045_own_state_pass30"
 and protocol["independent_reset_AdamW"]is True and protocol["all_parameters_trainable"]is True
 and protocol["first_step_qualification"]=="first_scheduled_step_retained_in_960_no_extra_updates")
check("TRAIN_only_no_VAL_TEST_truth_authority",c.FIXED_PHASES==("train",)
 and protocol["VAL_influences_optimization"]is False and protocol["outer_truth_read"]is False)
check("original_resource_and_cleanup_caps",c.CAPS["reservation_bytes"]==256*2**20
 and sum(c.CAPS[k]for k in ("scientific_output_bytes","auxiliary_bytes","directory_and_accounting_allowance_bytes"))==256*2**20
 and c.CAPS["max_seconds"]==86400 and c.CAPS["cleanup_seconds"]==30
 and c.CAPS["max_cuda_reserved_bytes"]==28*2**30 and c.CAPS["max_cpu_rss_bytes"]==128*2**30
 and c.CAPS["C3_fixed_envelope_bytes"]==26*2**30 and c.CAPS["headroom_bytes"]==4*2**30)
check("fresh27h_and_historical_clocks_source_bound",c.fit_clock_acceptance_v5()["interval_seconds"]==97200
 and c.fit_clock_acceptance_v5()["historical_clock_or_admission_mutation_authorized"]is False)
check("genuine_first_two_arm_math_qualification_source_pinned",
 hashlib.sha256(author["op"]._LEAN_MATH_SOURCE.encode()).hexdigest()=="fa5bddb450d2e531859f8d0c62bc6a1e49f60e82359baf97adede3e5cdb86966")
# Witness A: execute actual admission/mutation bodies, replacing only prior authority,
# paths, metadata writes and guard runtime with RAM doubles. The actual unchanged
# RootMutationMemo constructor/authenticator executes; no guard operation is called.
driver_tree=ast.parse(author["op"]._DRIVER_SOURCE)
deps=read(D+"/dependencies.v1.json");memo_ref=deps["mutation_memo"]["source_ref"]
memo=c.load(memo_ref,"_independent_H058_v8_ROOT_mutation_memo_source_only")
check("actual_selected_Census_path_differs_from_memo_Relate_pin",c.GUARD["path"]!=memo.PINS["guard"]["path"],
 selected_guard_ref=c.GUARD,memo_supported_guard_ref=memo.PINS["guard"])
guard=SimpleNamespace(__file__=str(ROOT/c.GUARD["path"]),
 budget=SimpleNamespace(v4=SimpleNamespace(),SOURCE_REL=memo.PINS["budget"]["path"],LAUNCHER_REL=memo.PINS["budget"]["path"]))
dummy=lambda name:dict(path="RAM_NOT_AUTHORITY/"+name,sha256="0"*64)
request=dummy("request.json");original_ref=dummy("original.json")
q=dict(packet_directory="RAM_NOT_AUTHORITY/packet",output_directory="RAM_NOT_AUTHORITY/output",
 preparation_directory="RAM_NOT_AUTHORITY/preparation",instance_id="ram",job_id="H058-GPU0-v8-fulltrain-ram",phase="train",
 science_handoff_ref=dummy("science.json"),config_ref=dummy("config.json"))
minimal_cfg=dict(runtime_equivalence_ref=dummy("runtime_equivalence.json"),source_manifest_ref=dummy("science_manifest.json"))
fake_protocol=SimpleNamespace(H050_INPUT_REF=original_ref,H050_TRAIN_BINDINGS_REF=dummy("bindings.json"))
saves=[];state_or_reserve_calls=[]
scope=dict(vars(d))
scope.update(validate_request_v1=lambda r:(q,minimal_cfg,rf(c.DRIVER),fake_protocol,dict(refs=[])),
 guard_v1=lambda **kwargs:guard,local=lambda path:SimpleNamespace(exists=lambda:False),
 save=lambda directory,name,value:(saves.append(dict(name=name,value=value))or dummy(name)),
 read=lambda r:dict(observation_entry_refs={},observation_cache_source_refs={})if r==original_ref else c.read(r),
 source_refs_v1=lambda:[])
emit([fn(driver_tree,"mutation_memo_v1"),fn(driver_tree,"admit_v1")],scope)
try:scope["admit_v1"](request)
except ValueError as exc:
    error=str(exc)
else:raise AssertionError("Expected actual memo path mismatch")
check("actual_admit_body_hits_memo_path_failure_before_state_or_reserve",
 error=="Exact supported module pathname"and len(saves)==2 and not state_or_reserve_calls)
witnesses.append(dict(id="B1",status="REPRODUCED_SOURCE_BLOCKER",
 actual_admission_error=error,metadata_save_names=[v["name"]for v in saves],
 selected_guard_ref=c.GUARD,memo_source_ref=memo_ref,memo_supported_guard_ref=memo.PINS["guard"],
 actual_guard_operations=0,actual_reservations=0,
 route="Exact admit_v1 and mutation_memo_v1 AST, source-authenticated unchanged RootMutationMemo; RAM authority/path/write/guard doubles"))
# Witness B: exact original ancestry backend and actual wrapper with a RAM process table.
fixture=author["fixture"];process=author["process"]
tree,table,children=fixture()
table[200]=process(200,20,group=100);children[100]=[200];tree.refresh()
check("baseline_discovered_child_same_group_tracked",tree.known[200]["start"]==20 and tree.groups=={100})
table[200]=process(200,20,group=200);tree.refresh()
check("same_exact_PID_start_new_group_not_recorded",tree.known[200]["group"]==200 and tree.groups=={100}and tree.complete is True)
witnesses.append(dict(id="B2",status="REPRODUCED_SOURCE_BLOCKER",pid=200,start=20,
 original_group=100,current_observed_group=tree.known[200]["group"],retained_groups=sorted(tree.groups),
 reason="Existing exact identity updates bypass _PendingV8 new-PID-only group tracking"))
limits=dict(d.OWNED_LIMITS_V8);d.OWNED_LIMITS_V8["groups"]=1
tree2,table2,children2=fixture()
table2[200]=process(200,20,group=100);children2[100]=[200];tree2.refresh()
table2[200]=process(200,20,group=200);tree2.refresh()
check("existing_identity_group_change_bypasses_group_bound",tree2.complete is True and tree2.known[200]["group"]==200 and tree2.groups=={100})
d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
# The real final group scanner can claim all groups absent without ever probing
# the newly observed group. This does not claim a real process was present.
probes=[]
def absent_old_only(group,sig):
    probes.append((group,sig))
    if group==100:raise ProcessLookupError()
scan_scope=dict(vars(d),os=SimpleNamespace(killpg=absent_old_only),time=SimpleNamespace(monotonic=lambda:0.))
emit([fn(driver_tree,"observe_groups_v8")],scan_scope)
scan=scan_scope["observe_groups_v8"](tree,1.)
check("final_actual_group_scanner_omits_new_observed_group",scan["all_owned_groups_observed_absent"]is True and probes==[(100,0)])
witnesses[-1].update(groups_limit_bypassed=True,final_scan_probes=probes,
 final_scan_all_retained_groups_absent=scan["all_owned_groups_observed_absent"])
# Known corrected operational terminal, bounded JSON only; no process observation.
outer_path="research/reviews/research_execution_ROOT_20261017_v27/H058_v7_actual_outer_terminal_consumed.root_v1.json"
outer=read(outer_path)
check("known_actual_v7_outer241_is_consumed",outer["actual_tool_terminal_consumed"]is True and outer["actual_tool_exit_code"]==241
 and outer["actual_worker_returncode"]==-15 and outer["completed_ref"]is None and outer["first_optimizer_update_verified"]is False)
check("no_science_framework_imports",not any(n=="torch"or n.startswith(("torch.","torch_geometric","numpy"))for n in sys.modules))
result=dict(schema="H058_GPU0_v8_independent_whole_source_AST_RAM_review_v1",verdict="SOURCE_BLOCKED",
 independent=True,source_only=True,source_ref=rf(c.DRIVER),source_manifest_ref=rf(c.MANIFEST),
 source_handoff_ref=rf(D+"/source_handoff.v8.json"),review_scope="Whole v8 package, entry/admission joins, original science, math integration, clocks, leases and owned lifecycle",
 blockers=[
 dict(id="B1",priority=1,title="Current Census guard cannot pass unchanged Relate ROOT mutation memo",
 source_locations=[D+"/driver_role.source.txt:8",D+"/driver_role.source.txt:46",D+"/common_v1.py:84",
 memo_ref["path"]+":9",memo_ref["path"]+":26",memo_ref["path"]+":130"],
 impact="Every genuine admission reaching mutation_memo_v1 rejects Exact supported module pathname before read_state_v1 or reserve_v1",
 required_repair="Authenticate an exact narrow current-Census-compatible mutation-memo adapter/bridge and exercise complete admission-to-memo route with RAM doubles"),
 dict(id="B2",priority=1,title="Existing owned identities can move to an unretained process group",
 source_locations=[D+"/driver_role.source.txt:110",D+"/driver_role.source.txt:169",
 c.OWNED["path"]+":52"],
 impact="An observed same PID/start group change bypasses group retention/bound; final all-owned-group claim can omit the new observed group",
 required_repair="Check/retain every newly observed group on all exact-identity updates, retain historical groups, fail closed on group overflow and test changed-group identities")],
 assertions=len(rows),checks=rows,reproduced_author_check_summary={k:v for k,v in author["result"].items()if k!="checks"},
 witnesses=witnesses,known_v7_actual_terminal_ref=rf(outer_path),
 known_v7_actual_outer_exit_code=241,known_v7_actual_worker_returncode=-15,
 prior_v7_actual_optimizer_totals="UNKNOWN_no_new_update_verified",
 accepted_source_properties=[
 "Shared driver/producer file and separate actual namespaces genuinely reduce direct roles to 64 with actual current47 guard roles",
 "Original science/protocol and full context guard preserved; exact reviewed math helper integrated only around train_one",
 "Both genuine first scheduled arm updates remain fully guarded; no extra scientific updates",
 "Fresh 27h scope with original full24h science, cleanup/evidence caps and historical clocks preserved",
 "Bounded pending identities, cumulative discoveries and history summaries repair cumulative64-identity failure for serial helpers",
 "Actual direct/adopted waits, nonzero failure, EOF, pidfd-only signalling and finite group-probe deadline remain required"],
 review_limitations=[
 "Actual guard selection/state/reserve/admission/lease/process/GPU/resource operations were not invoked",
 "Original authority, source and operational metadata reads plus exact AST/RAM doubles only",
 "No scientific payload, truth or checkpoint read/hash and no Torch/PyG/NumPy/numerical execution",
 "Author checks reproduced once; independent extra fixtures focus on actual admission-to-memo and same-identity process-group changes",
 "Known actual v7 outer terminal is consumed241; historical ownership truncation and unknown final optimizer totals remain preserved"],
 no_candidate_or_live_source_mutated=True,execution_authorized=False)
out=ROOT/REVIEW
with (out/"checks.result.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(verdict=result["verdict"],assertions=len(rows),blockers=[x["id"]for x in result["blockers"]],
 result_ref=rf(REVIEW+"/checks.result.v1.json"),witnesses=witnesses),sort_keys=True))
