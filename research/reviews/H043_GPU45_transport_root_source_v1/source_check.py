"""Stdlib-only H043 root proposal checks. No guard/state/array/CUDA execution."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
HERE=ROOT/"research/reviews/H043_GPU45_transport_root_source_v1"
from research.reviews.H043_GPU45_transport_root_source_v1 import root_common_v1 as common
from arggnn import train_anonymous_exact_clade_transport_v1 as worker
assert "torch" not in sys.modules
assert not any("completed_scope_credits" in key for key in sys.modules)
cfg=json.loads((HERE/"config.json").read_text());worker.validate_protocol(cfg)
counts=common.read(cfg["count_manifest_ref"])
manifest=common.read(cfg["source_manifest_ref"])
for path,digest in manifest["source_sha256"].items():
    assert common.ref(path)["sha256"]==digest,path
for f in HERE.glob("*.py"):
    ast.parse(f.read_text());compile(f.read_text(),str(f),"exec")
checks=["all_source_compiles_without_execution","frozen57_source_closure_and_scientific_config_match"]

def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError("Expected rejection")

plans={p:common.count_plan_v1(counts,p) for p in ("train","evaluate")}
assert plans["train"]["total_output_scope_upper_bound_bytes"]==42524210
assert plans["evaluate"]["total_output_scope_upper_bound_bytes"]==23105482
assert all(plan["total_output_scope_upper_bound_bytes"]<200000000 for plan in plans.values())
rejected(lambda:common.count_plan_v1(counts,"outer"))
bad=copy.deepcopy(counts);bad["parents"][worker.TRAIN[0]]["native_holds"]=1000000
rejected(lambda:common.count_plan_v1(bad,"train"))
assert sum(counts["parents"][p]["native_holds"]*686*4*4 for p in worker.TRAIN)==335108256
assert sum(r["compact_payload_upper_bound_bytes"] for r in plans["train"]["parents"].values())<cfg["transport_cache_cap_bytes"]
assert 12*sum(counts["parents"][p]["native_holds"]*686*4*4 for p in worker.VAL)<cfg["RAM_prediction_cap_bytes"]
checks.append("count_derived_output_raw_cache_target_RAM_caps_and_oversized_metadata_rejection")
for pid,row in counts["parents"].items():
    bank=SimpleNamespace(parent=SimpleNamespace(numel=lambda n=row["native_nodes"]:n),
                         sample_index=SimpleNamespace(shape=(row["native_holds"],952)))
    raw=worker.transport_upper_bounds_v1(bank)
    proof=common.preparation_check_v1(pid,bank,raw,cfg,counts,rss_reader=lambda:1000000)
    assert proof["projected_RSS_planning_bytes"]<common.RSS_CAP
    rejected(lambda:common.preparation_check_v1(pid,bank,dict(raw,native_nodes=1),cfg,counts,
                                               rss_reader=lambda:0))
    rejected(lambda:common.preparation_check_v1(pid,bank,raw,dict(cfg,GPU_transport_cap_bytes=1),counts,
                                               rss_reader=lambda:0))
    rejected(lambda:common.preparation_check_v1(pid,bank,raw,cfg,counts,
                                               rss_reader=lambda:common.RSS_CAP))
checks.append("all40_dimension_only_prebuild_checks_reject_raw_cap_count_and_RSS_drift")
for phase in ("train","evaluate"):
    preview=json.loads((HERE/(phase+"_input.inactive.json")).read_text())
    assert preview["status"]=="inactive" and preview["physical_gpu"] is None
    assert preview["refs"]["runtime"] is None and preview["refs"]["storage"] is None
    assert preview["arm"]=="joint_pair"
checks.append("inactive_inputs_have_no_GPU_schedule_runtime_storage_or_admission")
driver=(HERE/"root_driver.py").read_text();context=(HERE/"context_factory_v1.py").read_text()
assert "predecessor_refs" not in driver and "H040" not in driver
assert '"--gpu",str(gpu)' in driver and "read1(65536)" in driver and "rc=proc.wait()" in driver
assert "transaction_memo_adapter_ref" in context and "before_transport_build=before_build" in context
assert "preparation_check_v1(pid,bank,declared,cfg,counts)" in context
assert not any("completed_scope_credits" in key for key in sys.modules)
assert "torch" not in sys.modules
checks.append("one_GPU_explicit_root_choice_direct_wait_memo2_and_prebuild_hook_source_contract")
driver_tree=ast.parse(driver)
functions={n.name:n for n in driver_tree.body if isinstance(n,ast.FunctionDef)}
def calls(node,name):
    return [n for n in ast.walk(node) if isinstance(n,ast.Call)
            and ((isinstance(n.func,ast.Name) and n.func.id==name)
              or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
admit=functions["admit"];launch=functions["launch"]
admit_load=calls(admit,"load_guard_v1");launch_load=calls(launch,"load_guard_v1")
assert len(admit_load)==len(launch_load)==1
assert ast.literal_eval(admit_load[0].keywords[0].value) is False
assert ast.literal_eval(launch_load[0].keywords[0].value) is True
common_tree=ast.parse((HERE/"root_common_v1.py").read_text())
loader=next(n for n in common_tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_guard_v1")
assert loader.args.kwonlyargs[0].arg=="consumer" and ast.literal_eval(loader.args.kw_defaults[0]) is False
original_branch=next(n for n in loader.body if isinstance(n,ast.If))
assert isinstance(original_branch.test,ast.UnaryOp) and isinstance(original_branch.test.op,ast.Not)
assert not calls(original_branch,"install") and isinstance(original_branch.body[-1],ast.Return)
assert len(calls(admit,"apply_reserve_v1"))==1 and not calls(launch,"apply_reserve_v1")
context_tree=ast.parse(context)
context_loader=calls(context_tree,"load_guard_v1")
assert len(context_loader)==1 and ast.literal_eval(context_loader[0].keywords[0].value) is True
checks.append("default_reserve_loader_unmemoized_consumer_only_launch_and_worker_memo")
protected=next(n for n in launch.body if isinstance(n,ast.Try) and calls(n,"wait"))
assert len(calls(ast.Module(body=protected.finalbody,type_ignores=[]),"wait"))==1
assert len(calls(ast.Module(body=protected.finalbody,type_ignores=[]),"close"))==1
live_body=ast.Module(body=protected.body+protected.handlers,type_ignores=[])
for forbidden in ("budget_check_v9","inspect_v1","open","write","rglob","stat"):
    assert not calls(live_body,forbidden),forbidden
assert calls(live_body,"read1") and calls(live_body,"extend")
postwait_log=next(n for n in launch.body if isinstance(n,ast.Try) and calls(n,"budget_check_v9"))
assert postwait_log.lineno>protected.end_lineno and calls(postwait_log,"write")
receipt_save=[n for n in calls(launch,"save") if len(n.args)>1
              and isinstance(n.args[1],ast.Constant) and n.args[1].value=="actual_wait.json"]
assert len(receipt_save)==1 and receipt_save[0].lineno>postwait_log.end_lineno
assert 'returncode=rc' in driver and 'logging_error=logging_error' in driver
assert 'descendant_drain_or_no_live_writer_claim=False' in driver
checks.append("live_child_RAM_capture_only_finally_wait_terminal_log_and_exact_rc_receipt")
assigns={n.targets[0].id:n.value for n in driver_tree.body if isinstance(n,ast.Assign)
         and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
cli_module=ast.literal_eval(assigns["CLI_MODULE"])
assert cli_module.startswith("arggnn.") and all(p.isidentifier() for p in cli_module.split("."))
cli_path="src/"+cli_module.replace(".","/")+".py"
declared_path=eval(compile(ast.Expression(assigns["CLI"]),"<source-only-CLI-path>","eval"),{"CLI_MODULE":cli_module})
assert declared_path==cli_path=="src/arggnn/H043_GPU45_transport_worker_v1.py"
canonical_cli=ROOT/cli_path
ast.parse(canonical_cli.read_text());compile(canonical_cli.read_text(),str(canonical_cli),"exec")
assert manifest["source_sha256"][cli_path]==common.ref(cli_path)["sha256"]
assert any(isinstance(n.args[0],ast.Name) and n.args[0].id=="CLI" for n in calls(admit,"ref") if n.args)
command=next(n.value for n in ast.walk(launch) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=="cmd" for t in n.targets))
assert isinstance(command,ast.List) and any(isinstance(n,ast.Name) and n.id=="CLI_MODULE" for n in command.elts)
for phase in ("train","evaluate"):
    preview=json.loads((HERE/(phase+"_input.inactive.json")).read_text())
    assert preview["worker_cli_ref"]==common.ref(cli_path)
launcher=(ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py").read_text()
assert 'args.module.startswith("arggnn.")' in launcher
assert 'module_path = "src/" + args.module.replace(".", "/") + ".py"' in launcher
assert 'expected not in record["reservation"]["consumer_bindings"]' in launcher
checks.append("exact_frozen_launcher_arggnn_module_src_path_and_direct_CLI_binding_contract")
launch_text=ast.get_source_segment(driver,launch)
assert "initial_ledger_pins" not in launch_text and "launch_storage.json" not in launch_text
assert not calls(launch,"inspect_v1")
assert 'storage=read(spec["refs"]["storage"])' in launch_text
checks.append("launch_preserves_original_reservation_prefix_pins_and_storage_ref_without_inspect")



receipt=dict(schema="H043_root_proposal_source_qualification_v1",status="passed",checks=checks,
 source_count=len(manifest["source_sha256"]),count_manifest_ref=cfg["count_manifest_ref"],
 forecasts={phase:{k:v for k,v in plan.items() if k!="parents"} for phase,plan in plans.items()},
 guard_import_or_configure_or_state_read=False,real_arrays_checkpoints_labels_metrics_read=False,
 GPU_admission_or_journal_action=False,root_context_driver_or_CLI_execution=False)
(HERE/"source_qualification_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,indent=2,sort_keys=True))
