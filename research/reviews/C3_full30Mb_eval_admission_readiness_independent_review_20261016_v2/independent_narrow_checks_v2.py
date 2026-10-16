"""Independent narrow v2 per-parent review using sealed v1 fake integration harness."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");sys.path.insert(0,str(ROOT))
BASE="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v2"
CAND="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v2"
OLD="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v1"
OLDREVIEW="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1"
def file_ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
oldreview=json.loads((ROOT/OLDREVIEW/"review.v1.json").read_text())
harness_path=OLDREVIEW+"/independent_fake_checks_v1.py"
harness_ref=next(r for section in oldreview["checks"] for r in section["evidence_refs"] if r["path"]==harness_path)
assert file_ref(harness_path)==harness_ref
oldsrc=(ROOT/OLD/"prepare_evaluation_root_v1.py").read_text()
newsrc=(ROOT/CAND/"prepare_evaluation_root_v2.py").read_text()
expected=oldsrc.replace('HERE="'+OLD+'"','HERE="'+CAND+'"').replace('SOURCE=HERE+"/prepare_evaluation_root_v1.py"','SOURCE=HERE+"/prepare_evaluation_root_v2.py"')
oldblock='''    require(set(supplied)==set(c.VAL),"All5 actual closed full30Mb plans required before this evaluation admission")
    plans={p:c.plan_ready_v1(supplied[p],p) for p in c.VAL}'''
newblock='''    required_parents=({c.VAL[int(phase[-3:])]} if phase.startswith("predict_val_") else set(c.VAL))
    require(set(supplied)==required_parents,
        "Prediction requires only its own actual closed plan;matrix/score require all5 actual closed full30Mb plans")
    plans={p:c.plan_ready_v1(supplied[p],p) for p in c.VAL if p in required_parents}'''
assert expected.count(oldblock)==1
assert expected.replace(oldblock,newblock)==newsrc
oldfunc={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(oldsrc).body if isinstance(n,ast.FunctionDef)}
newfunc={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(newsrc).body if isinstance(n,ast.FunctionDef)}
assert set(oldfunc)==set(newfunc)
assert [k for k in oldfunc if oldfunc[k]!=newfunc[k]]==["evidence_request_fields_v1"]
# Only AST-extract inert setup and FakeFlow class from the sealed independent v1 fixture.
# The original top-level test runner is excluded.
h=(ROOT/harness_path).read_text()
h=h.replace('REVIEW="'+OLDREVIEW+'"','REVIEW="'+BASE+'"')
h=h.replace('BUILDER="'+OLD+'"','BUILDER="'+CAND+'"')
h=h.replace('+".prepare_evaluation_root_v1"','+".prepare_evaluation_root_v2"')
nodes=[]
for node in ast.parse(h).body:
    nodes.append(node)
    if isinstance(node,ast.ClassDef) and node.name=="FakeFlow":break
else:raise AssertionError("No sealed fake harness class")
exec(compile(ast.Module(body=nodes,type_ignores=[]),"<sealed_v1_independent_harness_adapted_namespace_only>","exec"),globals())
check("exact_entire_source_delta_only_version_identity_and_required_parent_map")
check("all_other_function_ASTs_including_active_builder_review_caps_roles_storage_terminals_identical")
def own_only(flow):
    own=c.VAL[int(flow.phase[-3:])]
    flow.evidence["closed_plan_evidence_by_parent"]={own:copy.deepcopy(flow.plan_evidence[own])}
    return own
def trace_plans(flow,own=None):
    calls=[];original=flow.ns["plan_ready_v1"]
    def traced(e,parent):
        if own is not None:assert parent==own,"Unrelated parent touched before own prediction"
        calls.append(parent);return original(e,parent)
    flow.ns["plan_ready_v1"]=traced;flow.fakec.plan_ready_v1=traced
    return calls
summaries=[]
for phase in b.PHASES:
    f=FakeFlow(phase);own=own_only(f) if phase.startswith("predict_val_") else None
    plan_calls=trace_plans(f,own);result=f.run()
    req=f.value("request.root_active_v1.json");spec=f.value("input.json");storage=f.value("storage.json")
    assert result["status"]=="ADMITTED_C3_CPU_EVALUATION_STAGE_NOT_EXECUTED" and f.applies==1
    assert result["consumer_binding_count"]==58
    if own is not None:
        assert set(plan_calls)=={own} and req["plan_evidence"]==f.plan_evidence[own]
        assert "plan_evidence_by_parent" not in req
        # Full fake flow reads only acceptance/own-plan metadata, never any other plan's ref.
        for parent,e in f.plan_evidence.items():
            if parent!=own:
                assert not any(r["path"] in f.reads for r in (e["plan_ref"],e["completion_ref"],e["actual_wait_ref"],e["close_receipt_ref"],e["outer_terminal_ref"]))
    else:assert set(plan_calls)==set(c.VAL)
    assert ("VAL_truth_capability" in req)==(phase=="score")
    assert storage["reserved_new_bytes"]==f.value("config.json")["output_cap_bytes"]==33554432
    assert storage["reservation"]["auxiliary_scopes"]==[str(ROOT/req["packet_directory"]/"worker_aux"),str(ROOT/req["packet_directory"]/"worker_tmp")]
    assert storage["reservation"]==f.value("reserve_receipt.json")["value"] and storage["initial_ledger_pins"]==f.value("reserve_receipt.json")["initial_ledger_pins"]
    assert req["expires_at_utc"]==spec["expires_at_utc"]==storage["expires_at_utc"]==f.expiry
    assert not any((ROOT/k).exists() for k in f.mem)
    assert not any(Path(p).suffix in (".pt",".npy",".npz") for p in f.reads)
    argv=result["exact_one_ROOT_stage_execution_argv"]
    f.fakec.set_active_spec_v1=lambda record:f.ns["validate_input_v1"](record)
    dn=dict(argparse=importlib.import_module("argparse"),json=json,c=f.fakec,SOURCE=c.DRIVER,__doc__="fake v2")
    dn["bind_reviewed_supervisor_v1"]=lambda:lambda p:0
    assert extracted(MAIN+"/root_driver_v1.py","main",dn)(argv[4:])==0
    summaries.append(dict(phase=phase,required_plan_parent_count=1 if own else 5,actual_schema_plan_calls=sorted(set(plan_calls)),roles=result["consumer_binding_count"],
        total_packet_metadata_charge=f.charge,all_admission_writes_in_memory=True))
check("all7_full_fake_schema_flows_own_parent_predictions_all5_matrix_score_separate_argv",summaries)
# Use already sealed historical VAL000/001 compact metadata evidence with genuine plan_ready bodies.
# Acceptance, guards, authority, reservation and output remain fake.
historical=[]
for idx in (0,1):
    phase="predict_val_%03d"%idx;f=FakeFlow(phase);own=own_only(f)
    record=json.loads((ROOT/OLD/("plan_evidence_VAL%03d.actual_v1.json"%idx)).read_text())
    f.evidence["closed_plan_evidence_by_parent"]={own:record}
    calls=trace_plans(f,own);f.run()
    assert set(calls)=={own}
    historical.append(dict(phase=phase,own_plan_ref=record["plan_ref"],other_parent_plans_not_required_or_read=True,
        actual_acceptance_guard_reservation_or_execution=False))
check("VAL000_VAL001_historical_ready_closed_plans_work_independently_with_fake_acceptance_admission",historical)
for idx in range(5):
    for change in ("missing","wrong_parent","extra_other","own_close_missing","own_drain_missing","own_outer_unconsumed"):
        f=FakeFlow("predict_val_%03d"%idx);own=own_only(f)
        if change=="missing":f.evidence["closed_plan_evidence_by_parent"]={}
        elif change=="wrong_parent":f.evidence["closed_plan_evidence_by_parent"]={c.VAL[(idx+1)%5]:f.plan_evidence[c.VAL[(idx+1)%5]]}
        elif change=="extra_other":f.evidence["closed_plan_evidence_by_parent"][c.VAL[(idx+1)%5]]=f.plan_evidence[c.VAL[(idx+1)%5]]
        elif change=="own_close_missing":f.evidence["closed_plan_evidence_by_parent"][own]["close_receipt_ref"]=None
        else:
            key="actual_wait_ref" if change=="own_drain_missing" else "outer_terminal_ref"
            value=f.read(f.evidence["closed_plan_evidence_by_parent"][own][key])
            value.update(dict(all_owned_processes_and_groups_drained=False) if change=="own_drain_missing" else dict(actual_terminal_consumed=False))
            f.evidence["closed_plan_evidence_by_parent"][own][key]=f.put(value)
        f.rejected_before_writes()
check("all5_predictions_missing_wrong_extra_own_unclosed_undrained_unconsumed_negatives")
for phase in ("matrix","score"):
    for parent in c.VAL:
        for change in ("missing_map_entry","missing_actual_close"):
            f=FakeFlow(phase)
            if change=="missing_map_entry":f.evidence["closed_plan_evidence_by_parent"].pop(parent)
            else:f.evidence["closed_plan_evidence_by_parent"][parent]["close_receipt_ref"]=None
            f.rejected_before_writes()
check("matrix_score_still_require_every_one_of5_actual_closed_plans_before_writes")
for phase in b.PHASES:
    f=FakeFlow(phase)
    if phase.startswith("predict_val_"):own_only(f)
    acceptance=copy.deepcopy(f.acceptance);acceptance["outer_terminal_ref"]=None
    f.evidence["ROOT_artifact_acceptance_ref"]=f.put(acceptance);f.rejected_before_writes()
check("all7_phases_still_require_genuine_ROOT_saved600_own_terminal_acceptance")
for change in ("missing_cell","duplicate_cell","wrong_focal","truth_opened"):
    f=FakeFlow("score");matrix=copy.deepcopy(f.matrix)
    if change=="missing_cell":matrix["cells"].pop()
    elif change=="duplicate_cell":matrix["cells"][-1]=copy.deepcopy(matrix["cells"][0])
    elif change=="wrong_focal":matrix["cells"][-1]["focal_registry_ref"]=dict(path="SYNTHETIC_wrong.json",sha256="0"*64)
    else:
        seal=f.read(matrix["cells"][-1]["prediction_seal_ref"]);seal["private_truth_opened"]=True;matrix["cells"][-1]["prediction_seal_ref"]=f.put(seal)
    f.evidence["matrix"]=f.stage("matrix",f.put(matrix));f.reads=[];f.rejected_before_writes()
    assert not any(r["path"] in f.reads for r in f.simulations.values())
check("all30_original_pretruth_matrix_metadata_before_simulation_capability_reads")
for phase in ("predict_val_000","predict_val_001","matrix"):
    f=FakeFlow(phase)
    if phase.startswith("predict_val_"):own_only(f)
    f.evidence["VAL_truth_capability"]={};f.rejected_before_writes()
check("truth_capability_remains_score_only")
for mode in ("wrapped","overflow","race","late"):
    f=FakeFlow("predict_val_000",mode);own_only(f);reject(f.run)
    assert f.applies==(1 if mode in ("race","late") else 0)
    check("own_plan_full_flow_"+mode+"_rejects_no_retry_or_execution")
# Import author module without __main__, call its inert run, and write only our own evidence.
sp=importlib.util.spec_from_file_location("_independent_author_v2_inert",ROOT/CAND/"inert_per_parent_checks_v2.py")
author=importlib.util.module_from_spec(sp);sp.loader.exec_module(author);author_result=author.run()
expected_author=json.loads((ROOT/CAND/"source_static_inert_checks.v2.json").read_text())
for value in (author_result,expected_author):value.pop("finished_utc",None)
assert author_result==expected_author and author_result["check_count"]==111
with (ROOT/BASE/"author_fixture_reproduction.v2.json").open("x") as stream:json.dump(author_result,stream,sort_keys=True,indent=2);stream.write("\n")
check("author111_inert_checks_reproduced_without_candidate_write")
assert not any(name in sys.modules for name in ("torch","numpy","torch_geometric"))
print(json.dumps(dict(schema="C3_eval_builder_v2_independent_checks_v1",status="PASS_SOURCE_AND_SOFTWARE_ONLY",check_count=len(checks),checks=checks,
    sealed_v1_harness_ref=harness_ref,source_ref=file_ref(CAND+"/prepare_evaluation_root_v2.py"),
    all_active_current_guards_authorities_reservations_packets_and_supervisor_endpoints_fake=True,
    historical_VAL000001_compact_metadata_only=True,scientific_payloads_opened=False,
    actual_current_acceptance_plan_readiness_guard_resources_process_probe_accounting_admission_execution_evaluation_or_Git=False,
    v1_candidate_review_and_evidence_modified=False),sort_keys=True,indent=2))
