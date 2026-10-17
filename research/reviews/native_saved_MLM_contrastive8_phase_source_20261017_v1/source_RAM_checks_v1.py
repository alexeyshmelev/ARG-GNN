"""Bounded AST and in-RAM checks only. Never imports adapter/guard/scientific runtime."""
import ast,copy,hashlib,json,pathlib,sys,traceback,types
HERE=pathlib.Path(__file__).parent
BASE=pathlib.Path("research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/native64")
CHECKS=[]
def need(v,msg):
 if not v:raise ValueError(msg)
def check(name,value):need(value,name);CHECKS.append(name)
def bad(name,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):CHECKS.append(name);return
 raise ValueError("Did not reject: "+name)
def source(name,base=False):return ((BASE if base else HERE)/name).read_text()
def tree(name,base=False):return ast.parse(source(name,base))
def fn(name,fun,base=False,cls=None):
 nodes=tree(name,base).body
 if cls:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef)and n.name==cls).body
 return copy.deepcopy(next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==fun))
def dump(n):return ast.dump(n,include_attributes=False)
def isolated(node,ns):
 exec(compile(ast.Module(body=[node],type_ignores=[]),"<RAM-only-extracted-source>","exec"),ns);return ns[node.name]
def literal(name):
 return next(ast.literal_eval(n.value)for n in tree("contract_v1.py").body if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id==name for a in n.targets))
def ref(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
def main():
 for path in HERE.glob("*.py"):ast.parse(path.read_text(),filename=str(path))
 check("all_python_AST_valid",True)
 oldmod=str(BASE).replace("/",".");newmod=str(HERE).replace("/",".")
 old=source("execution_v1.py",True).replace(oldmod,newmod)
 oldtree=ast.parse(old)
 oldexec=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=="execute_v1")
 check("entire_execute_supervisor_AST_identical_after_module_relocation",dump(oldexec)==dump(fn("execution_v1.py","execute_v1")))
 for name in ["_fresh_status_v1","_registry_v1","controller_v1","capacity_v1","attach_v1"]:
  check("unchanged_GPU0_"+name,dump(fn("direct_gpu0_v1.py",name))==dump(fn("direct_gpu0_v1.py",name,True)))
 check("scientific_binding_byte_equivalent",source("science_binding_v1.py")==source("science_binding_v1.py",True))
 for name in ["validated_composer_timestamp_v1","checkpoint_metadata_bridge_v1","_bind_private_scientific_helper_v1","stable_reference_v1"]:
  check("unchanged_context_"+name,dump(fn("worker_context_v1.py",name))==dump(fn("worker_context_v1.py",name,True)))
 caps={}
 for isbase in [True,False]:
  n=next(n for n in tree("contract_v1.py",isbase).body if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id=="BASE_CAPS"for a in n.targets))
  ns={};exec(compile(ast.Module(body=[n],type_ignores=[]),"<literal-caps>","exec"),ns);caps[isbase]=ns["BASE_CAPS"]
 check("all_finite_lifecycle_and_checkpoint_caps_preserved",dict(caps[True],reservation_bytes=8*2**30)==caps[False])
 config=json.loads((HERE/"scientific_config.FROZEN.v1.json").read_text())
 original=json.loads((BASE/"scientific_config.FROZEN.v1.json").read_text())
 check("unchanged_full2135_contrastive_scientific_protocol",config["scientific_protocol"]["contrastive"]==original["scientific_protocol"]["contrastive"])
 check("only_phase_contrastive",literal("GPU_PHASES")==("contrastive",)and config["allowed_phases"]==["contrastive"])
 check("whole_cap8GiB_including_all_AUX",config["output_cap_bytes"]==8*2**30 and config["group_scope_caps"]=={"contrastive":8*2**30})
 check("no_runtime_fit_claim",config["actual_GPU_fit_qualified"]is False and config["actual_storage_fit_qualified"]is False)
 check("no_finalizer_worker_exposed",not any(isinstance(n,ast.FunctionDef)and n.name=="finalize_worker_v1"for n in tree("worker_context_v1.py").body))
 run=fn("worker_context_v1.py","run",cls="NativeContext")
 check("sole_science_dispatch_original_saved_MLM_contrastive","contrastive_keep1_saved_MLM_v1"in ast.unparse(run)and "embedding_v1"not in ast.unparse(run)and "RAM_retrieval"not in ast.unparse(run))
 data={"out":[],"aux":[]}
 original_sizes=types.SimpleNamespace(_sizes=lambda path,**kw:data[path])
 census=isolated(fn("contract_v1.py","charged_census_v1"),dict(require=need,STATE=[dict(output_directory="out",worker_auxiliary_scope="aux")],local=lambda p:p,original=original_sizes))
 check("root_nodes_are_charged",sum(census({"phase":"contrastive"}).values())==8192)
 data["out"]=[(("contrastive","blob"),8*2**30-8192,"regular")]
 check("exact_whole_cap_passes",sum(census({"phase":"contrastive"}).values())==8*2**30)
 bad("prospective_one_byte_overflow",lambda:census({"phase":"contrastive"},required_write_bytes=1))
 data["aux"]=[(("contrastive","TMP","x"),1,"regular")]
 bad("AUX_is_inside_whole_cap",lambda:census({"phase":"contrastive"}))
 data["out"]=[];data["aux"]=[]
 bad("negative_planned_growth",lambda:census({"phase":"contrastive"},required_write_bytes=-1))
 bad("bool_planned_growth",lambda:census({"phase":"contrastive"},required_write_bytes=True))
 bad("no_embedding_phase",lambda:census({"phase":"embedding_contrastive_train"}))
 data["out"]=[(("embedding_contrastive_train","x"),4096,"directory")]
 bad("no_unadmitted_sibling_scope",lambda:census({"phase":"contrastive"}))
 data["out"]=[(("contrastive","checkpoints","a.pt"),3*2**30+4096,"regular"),(("contrastive","checkpoints","b.pt"),3*2**30+4096,"regular")]
 check("two3GiB_checkpoint_peak_within8",sum(census({"phase":"contrastive"}).values())<8*2**30)
 data["out"].append((("contrastive","checkpoints","c.pt"),4096,"regular"))
 bad("third_regular_checkpoint_rejected",lambda:census({"phase":"contrastive"}))
 data["out"]=[(("contrastive","checkpoints","a.pt"),3*2**30+4097,"regular")]
 bad("checkpoint_larger3GiB_rejected",lambda:census({"phase":"contrastive"}))
 data["out"]=[(("completed.json",),4096,"regular")]
 check("honest_operational_root_completion_allowed",sum(census({"phase":"contrastive"}).values())==12288)
 data["out"]=[];data["aux"]=[(("completed.json",),4096,"regular")]
 bad("no_AUX_root_completion",lambda:census({"phase":"contrastive"}))
 # Actual baseline metadata is immutable input; never call guard/state methods.
 baseline=json.loads(pathlib.Path(literal("BASELINE_AUTHORITY_REF")["path"]).read_text())["bindings"]
 check("genuine_baseline47roles40unique",len(baseline)==47 and len({r["path"]for r in baseline.values()})==40)
 c=types.SimpleNamespace(ROOT=pathlib.Path("/RAM"),HERE=str(HERE),MANIFEST=str(HERE/"source_manifest.v1.json"),
  PHASES=("contrastive",),GPU_PHASES=("contrastive",),BASE_CAPS=caps[False],require=need,
  GUARD=literal("GUARD"),POLICY=ref("policy"),SELECTOR_REF=ref("selector"),SAVED_INTEGRATION_REF=literal("SAVED_INTEGRATION_REF"),
  SAVED_SCIENTIFIC_ENTRY_REF=literal("SAVED_SCIENTIFIC_ENTRY_REF"),SAVED_SCIENCE_REVIEW_REF=literal("SAVED_SCIENCE_REVIEW_REF"),
  SCIENCE_BINDING=str(HERE/"science_binding_v1.py"),CONTEXT=str(HERE/"worker_context_v1.py"),
  IO_SOURCE_REF=literal("IO_SOURCE_REF"),IO_POLICY_REF=literal("IO_POLICY_REF"),OWNED_ROSTER_POLICY_REF=literal("OWNED_ROSTER_POLICY_REF"),
  original=types.SimpleNamespace(RUNTIME_REF=ref("runtime")),ref=ref,local=lambda p:pathlib.Path("/RAM")/p)
 for name in ["COORDINATOR_REF","BATCH_INPUT_REF","OWNER4_SOURCE_REF","OWNER4_MANIFEST_REF","OWNER4_REVIEW_REF"]:setattr(c,name,literal(name))
 ns=dict(c=c,copy=copy,datetime=__import__("datetime").datetime,re=__import__("re"))
 isolated(fn("root_builder_v1.py","_record_directory_v1"),ns)
 masterfn=isolated(fn("root_builder_v1.py","master_metadata_v1"),ns)
 q=dict(instance="ram",source_review_ref=ref("review"),created_at_utc="2026-10-17T01:00:00+00:00",expires_at_utc="2026-10-17T10:00:00+00:00",
  storage_policy_selector_ref=c.SELECTOR_REF,ROOT_saved_MLM_artifact_acceptance_ref=ref("acceptance"))
 master=masterfn(q,config_ref=ref(str(HERE/"scientific_config.FROZEN.v1.json")),storage_ref=ref("storage"),request_ref=ref("request"))
 check("master_whole8_fresh_roots",master["future_storage_reserved_bytes"]==8*2**30 and "contrastive8"in master["output_directory"]and "contrastive8"in master["worker_auxiliary_scope"])
 check("ROOT_records_inside_owned_namespace",master["ROOT_record_directory"]==str(HERE/"ROOT_records_ram"))
 phasefn=isolated(fn("root_builder_v1.py","phase_metadata_v1"),ns)
 pq=dict(q,phase="contrastive",prerequisites={},max_cuda_reserved_bytes=60*2**30,direct_attachment_authority_ref=ref("attachment"),controller_binding_ref=ref("binding"))
 phase=phasefn(pq,master,ref("master"))
 check("phase_original_caps_and_historical_outer1",phase["caps"]==dict(caps[False],max_cuda_reserved_bytes=60*2**30)and phase["prerequisites"]["MLM"]["historical_MLM_outer_exit_code"]==1)
 for cap in [0,True,75*2**30]:bad("bad_GPU_cap_"+str(cap),lambda cap=cap:phasefn(dict(pq,max_cuda_reserved_bytes=cap),master,ref("master")))
 bad("no_embedding_builder",lambda:phasefn(dict(pq,phase="embedding_contrastive_train"),master,ref("master")))
 bad("no_full_pipeline_finalizer_builder",lambda:phasefn(dict(pq,phase="finalize"),master,ref("master")))
 bad("no_expiry_extension",lambda:phasefn(dict(pq,expires_at_utc="2026-10-18T10:00:00+00:00"),master,ref("master")))
 check("coordinator_encloses_complete_capture_and_apply",any(isinstance(n,ast.With)and "accounting_transaction"in ast.unparse(n.items[0].context_expr)and "_state_locked"in ast.unparse(n)and "apply_reserve_v1"in ast.unparse(n)for n in ast.walk(fn("root_builder_v1.py","prepare_master_v1"))))
 executed=[ref(str(HERE/n))for n in ["__init__.py","contract_v1.py","execution_v1.py","direct_gpu0_v1.py","worker_context_v1.py","science_binding_v1.py","root_builder_v1.py"]]
 direct=[c.GUARD,ref(c.MANIFEST),master["config_ref"],master["runtime_ref"],ref("request"),q["source_review_ref"],q["ROOT_saved_MLM_artifact_acceptance_ref"],
  c.SAVED_INTEGRATION_REF,c.SAVED_SCIENTIFIC_ENTRY_REF,c.SAVED_SCIENCE_REVIEW_REF]+executed+[c.COORDINATOR_REF,c.BATCH_INPUT_REF,c.OWNER4_SOURCE_REF,c.OWNER4_MANIFEST_REF,c.OWNER4_REVIEW_REF]
 union={r["path"]:r for r in list(baseline.values())+direct}
 check("actual47role_baseline_plus_direct_union_le64",len(union)<=64)
 check("no_checkpoint_data_or_raw_lock_binding",not any(r["path"].endswith((".pt",".npz",".lock",".pkl"))for r in direct))
 for n in ["root_builder_v1.py","execution_v1.py"]:
  body=ast.unparse(fn(n,"main"))
  check("inactive_default_"+n,body.index("SOURCE_ONLY_INACTIVE")<body.index("sys.platform"))
 check("lineage_exact_child_wait_and_producer_sources","actual_producer_manifest_join"in source("root_builder_v1.py")and "prerequisite_metadata_v1"in ast.unparse(fn("root_builder_v1.py","seal_producer_lineage_v1")))
 check("lineage_no_close_or_credit_claim",'close_or_credit_authorized=False'in source("root_builder_v1.py"))
 check("owner4_dual_source_gate",all(s in ast.unparse(fn("direct_gpu0_v1.py","configure_controller_binding_v1"))for s in ["OWNER4_SOURCE_REF","OWNER4_MANIFEST_REF","OWNER4_REVIEW_REF","ENDPOINT_SOURCE_REF"]))
 check("original8GiB_scope_not_relabelled",str(BASE)not in source("execution_v1.py")and "native_saved_MLM_contrastive8"in source("contract_v1.py"))
 # Genuine metadata joins exercised wholly in RAM.
 documents={}
 c.read=lambda r:documents[r["path"]]
 c.raw=lambda r:None
 c.active=lambda value:need(value.get("status")=="active","active RAM authority")
 c.GPU0_UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
 binding_ref=ref("binding");cap_ref=ref("capability")
 endpoint=next(ast.literal_eval(n.value)for n in tree("direct_gpu0_v1.py").body if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id=="ENDPOINT_SOURCE_REF"for a in n.targets))
 review=json.loads(pathlib.Path(c.OWNER4_REVIEW_REF["path"]).read_text())
 documents[c.OWNER4_REVIEW_REF["path"]]=review
 binding=dict(status="active",schema="native_VAL000_downstream_GPU0_controller_binding_v1",
  source_review_passed=True,endpoint_source_ref=endpoint,physical_gpu=0,GPU_UUID=c.GPU0_UUID,
  additional_controller_and_other_borrower_submissions_revoked=True,controller_preserved_until_downstream_drain=True,
  independent_source_review_ref=c.OWNER4_REVIEW_REF,owner_service_source_ref=c.OWNER4_SOURCE_REF,owner_service_manifest_ref=c.OWNER4_MANIFEST_REF,
  route="reviewed_same_protocol_successor",protected_job=None,protected_envelope_bytes=0,
  successor_capability_explicitly_ROOT_authorized=True,controller_pid=12345,controller_start_ticks=67890,controller_capability_ref=cap_ref)
 capability=dict(source_refs={endpoint["path"]:endpoint,c.OWNER4_SOURCE_REF["path"]:c.OWNER4_SOURCE_REF},
  source_manifest_ref=c.OWNER4_MANIFEST_REF,source_review_ref=c.OWNER4_REVIEW_REF,jobs={},submit_authorized=False)
 documents[binding_ref["path"]]=copy.deepcopy(binding);documents[cap_ref["path"]]=copy.deepcopy(capability)
 dns=dict(c=c,read=c.read,raw=c.raw,active=c.active,require=need,GPU0_UUID=c.GPU0_UUID,ENDPOINT_SOURCE_REF=endpoint,
  BINDING_REF=None,CONTROLLER_PID=None,CONTROLLER_START=None,C3_JOB=None,CURRENT_CAP=None,C3_ENVELOPE_MIB=0)
 directfn=isolated(fn("direct_gpu0_v1.py","configure_controller_binding_v1"),dns)
 check("actual_owner4_review_shape_and_dual_pin_join",directfn(dict(controller_binding_ref=binding_ref))==binding and dns["BINDING_REF"]==binding_ref)
 documents[cap_ref["path"]]["source_refs"].pop(c.OWNER4_SOURCE_REF["path"])
 bad("missing_owner4_service_pin_rejected",lambda:directfn(dict(controller_binding_ref=binding_ref)))
 documents[cap_ref["path"]]=copy.deepcopy(capability);documents[cap_ref["path"]]["source_refs"].pop(endpoint["path"])
 bad("missing_executed_v3_pin_rejected",lambda:directfn(dict(controller_binding_ref=binding_ref)))
 documents[cap_ref["path"]]=copy.deepcopy(capability);documents[binding_ref["path"]]["route"]="current_C3"
 bad("old_C3_route_rejected",lambda:directfn(dict(controller_binding_ref=binding_ref)))
 c.PARENT="VAL000"
 comp_ref=ref("completion");wait_ref=ref("wait");outer_ref=ref("outer");input_ref=ref("input")
 complete=dict(parent_id="VAL000",query_ancestry_truth_opened=False,child_input_ref=input_ref,stage="contrastive",checkpoint_ref=ref("handle"),
  actual_execution_source_manifest_ref=ref(c.MANIFEST))
 wait=dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,
  complete_owned_roster_known=True,completed_ref=comp_ref,operational_failure=None,group_observations=[dict(absent=True)],
  input_ref=input_ref,actual_direct_child_wait=True,child_completion_ref=comp_ref,process_group_ESRCH=True,adapter_manifest_ref=ref(c.MANIFEST))
 outer=dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,
  all_owned_processes_and_groups_drained=True,actual_wait_ref=wait_ref,actual_tool_session_id=444,
  schema="native_saved_MLM_downstream_actual_outer_terminal_v1",terminal_consumed=True,actual_tool_terminal_exit_code=0,
  tool_session_id=444,child_completion_ref=comp_ref)
 original_docs={comp_ref["path"]:complete,wait_ref["path"]:wait,outer_ref["path"]:outer}
 documents.update(copy.deepcopy(original_docs))
 bundle=dict(completion_ref=comp_ref,wait_ref=wait_ref,outer_terminal_ref=outer_ref)
 bundlefn=isolated(fn("worker_context_v1.py","_bundle_v1"),dict(c=c))
 check("genuine_completion_wait_outer0_source_join",bundlefn(bundle,"contrastive")==tuple(documents[r["path"]]for r in [comp_ref,wait_ref,outer_ref]))
 for name,r,key,value in [
  ("historical_manifest_relabel",wait_ref,"adapter_manifest_ref",ref(str(BASE/"source_manifest.v1.json"))),
  ("worker_nonzero",wait_ref,"returncode",1),("raw_OS_wait_nonzero",wait_ref,"actual_OS_wait_status",256),
  ("incomplete_owned_drain",wait_ref,"all_owned_processes_and_groups_drained",False),
  ("outer_nonzero",outer_ref,"actual_exit_code",1),("outer_session_mismatch",outer_ref,"tool_session_id",555),
  ("wrong_input_completion",comp_ref,"child_input_ref",ref("other_input"))]:
  documents.update(copy.deepcopy(original_docs));documents[r["path"]][key]=value
  bad("bundle_rejects_"+name,lambda:bundlefn(bundle,"contrastive"))
 result=dict(schema="native_saved_MLM_contrastive8_source_RAM_checks_v1",status="PASS_SOURCE_RAM_ONLY",checks=CHECKS,check_count=len(CHECKS),
  baseline_role_count=len(baseline),baseline_unique_count=40,direct_role_count=len(direct),union_unique_count=len(union),
  actual_resource_operations=False,scientific_imports_or_payload_reads=False,guard_or_process_GPU_socket_lock_probes=False,
  actual_fit_or_execution_proven=False)
 print(json.dumps(result,sort_keys=True,indent=2));return result
if __name__=="__main__":
 try:main()
 except BaseException:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
