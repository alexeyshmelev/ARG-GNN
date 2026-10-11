"""Fresh independent v2 source, authority and handmade retrieval review only."""
from __future__ import annotations
import ast, copy, hashlib, importlib.util, json, os, pickle, sys, types
from pathlib import Path
from unittest.mock import patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REVIEW=ROOT/"research/reviews/native_argformer_tsinfer_fit_v2_independent"
sys.dont_write_bytecode=True
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="4"
checks=[]; counterexamples=[]; details={}
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load("_independent_nativefit2",ROOT/"src/arggnn/native_argformer_tsinfer_fit_v2.py")
c=m.policy();old=load("_independent_sealed_nativefit1",ROOT/"src/arggnn/native_argformer_tsinfer_fit_v1.py");oc=old.policy()
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError,AssertionError) as e:
        checks.append(name);return {"type":type(e).__name__,"message":str(e)}
    raise AssertionError("unexpected acceptance: "+name)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:json.dump(value,f,sort_keys=True,indent=2)
    return m.reference(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
expected={"src/arggnn/native_argformer_tsinfer_fit_v2.py":"8cbbd74992a6e3f730937a17b33f32b5d112e8f89aa967e99a2538588925fb7b",
"configs/native_argformer_tsinfer_fit_v2.json":"29c506b7dcb44a2544c5a0dcf8e2211c3ee9b242a09cc9e6984ffab5255b3129",
"research/reviews/native_argformer_tsinfer_fit_v2_author/source_receipt.json":"a7182ff22ca3be2bbbb4e941dcf0966d47300df7c60a8b75851932b675c2c335"}
for p,h in expected.items():check("requested_pin_"+p,sha(ROOT/p)==h)
author=json.loads((ROOT/"research/reviews/native_argformer_tsinfer_fit_v2_author/source_receipt.json").read_text())
v1=json.loads((ROOT/author["previous_sealed_v1_source_receipt"]["path"]).read_text())
check("sealed_v1_receipt_hash",sha(ROOT/author["previous_sealed_v1_source_receipt"]["path"])==author["previous_sealed_v1_source_receipt"]["sha256"])
for record in author["files"]+v1["files"]:check("sealed_file_"+record["path"],sha(ROOT/record["path"])==record["sha256"])
src=m.SCRIPT.read_text().replace("native_argformer_tsinfer_fit_v2","native_argformer_tsinfer_fit_v1").replace("_admitted_native_tsinfer_fit_v2","_admitted_native_tsinfer_fit_v1").replace("_admitted_native_tsinfer_retrieval_v2","_admitted_native_tsinfer_retrieval_v1").replace("project_budget_v4_root_reserve_admission","project_budget_v3_root_reserve_admission").replace(m.CONFIG_SHA256,old.CONFIG_SHA256)
check("complete_executor_AST_binding_only",ast.dump(ast.parse(src),include_attributes=False)==ast.dump(ast.parse(old.SCRIPT.read_text()),include_attributes=False))
nc=copy.deepcopy(c)
for k in ("kind","result_root","short_tmp_root"):nc[k]=oc[k]
nc["storage"]["reservation_key_prefix"]=oc["storage"]["reservation_key_prefix"]
for r in nc["storage"]["companion_contract"].values():nc["frozen_files_sha256"].pop(r["path"])
nc["storage"]["companion_contract"]=oc["storage"]["companion_contract"]
for r in oc["storage"]["companion_contract"].values():nc["frozen_files_sha256"][r["path"]]=r["sha256"]
check("complete_configuration_binding_only",m.exact(nc,oc))
proof={}
for source,names in [(m.FIT,c["source_AST"]["fit_functions"]),(m.RETRIEVAL,c["source_AST"]["retrieval_classes"]+c["source_AST"]["retrieval_functions"])]:
    tree=ast.parse((ROOT/source).read_text())
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    proof[source]={name:hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest() for name in names}
    check("independent_scientific_AST_"+source,proof[source]==v1["native_ASTs"][source])
check("exact_30_native_scientific_definitions",sum(map(len,proof.values()))==30)
check("inherited_116_checks_hash",sha(ROOT/author["checks"]["reused_scientific_fixture_report"]["path"])==author["checks"]["reused_scientific_fixture_report"]["sha256"])
check("inherited_retrieval_fixture_size_provenance",all(text in (ROOT/"scripts/check_native_argformer_tsinfer_fit_v1.py").read_text() for text in ["26542080","4608","5760"]))
m.verify_source(c)
check("original_author_source_inventory_and_all_frozen_files",True)
check("no_truth_or_fallback_or_paper_reproduction",c["claims"]["query_truth_authorized"] is False and c["claims"]["gradient_cache_or_fallback_authorized"] is False and c["claims"]["AmericanAdmixture_paper_reproduction"] is False)
details["native_ASTs"]=proof
# Only the eight expressly allowed prepared metadata files are read.
details["prepared_scalar_metadata"]={}
for parent in c["parent_ids"]:
    dref=c["prepared_dispatchers"][parent];qref=c["qualified_corpus_metadata"][parent]
    for ref in (dref,qref):check("prepared_metadata_hash_"+ref["path"],sha(ROOT/ref["path"])==ref["sha256"])
    d=json.loads((ROOT/dref["path"]).read_text());q=json.loads((ROOT/qref["path"]).read_text())
    check("prepared_parent_scalar_qualified_"+parent,d["parent_id"]==parent and d["query_truth_used"] is False and q["native_original_node_IDs_used"] is True and q["query_truth_used"] is False)
    b=q["budget"];n=b["actual_counts"]
    check("scalar_exact_source_update_and_fullpair_budgets_"+parent,b["MLM_optimizer_updates"]==5*(n["MLM_train"]//512) and b["contrastive_optimizer_updates"]==n["pure_reference_train"]//4096 and b["exhaustive_query_corpus_pairs"]==n["query_full_paths_scored"]*(n["pure_reference_train"]+n["pure_reference_val"]) and b["exhaustive_query_corpus_pairs"]<=40000000000)
    details["prepared_scalar_metadata"][parent]={"counts":n,"exhaustive_pairs":b["exhaustive_query_corpus_pairs"]}
fixture=REVIEW/"fixtures_final";fixture.mkdir()
parent=c["parent_ids"][0]
inventory=write(fixture/"empty_inventory.json",{})
def authorities(stage,run="fresh"):
    a=m.admission_bindings(c,parent,stage,run)
    r={"kind":"native_argformer_tsinfer_fit_v2_independent_review","review_role":"independent_reviewer","verdict":"PASS","implementation":a["implementation"],"config":a["config"],"fixture_only_not_authority":True}
    rt={"kind":"native_argformer_tsinfer_fit_v2_root_runtime","authorizer":"root_agent","active":True,"CPU_only":stage=="retrieve",**c["runtime"],"python_executable_sha256":sha(ROOT/c["runtime"]["interpreter"]),"environment_inventory_receipt":inventory,"additive_helpers":[],"fixture_only_not_authority":True}
    reg={"kind":"native_argformer_tsinfer_fit_v2_root_experiment_registration","authorizer":"root_agent","active":True,"condition":c["condition"],"scope_parents":[parent],"stage":stage,"implementation":a["implementation"],"config":a["config"],"query_truth_authorized":False,"hypothesis_id":"H013","repeat_reason":"independent handmade only","fixture_only_not_authority":True}
    return a,r,rt,reg
for stage in m.STAGES:
    a,r,rt,reg=authorities(stage)
    check("complete_handmade_authority_field_contract_"+stage,m.validate_admission(c,parent,stage,"fresh",a,r,rt,reg)==a)
    for field,value in [("kind","native_argformer_tsinfer_fit_v1_root_admission"),("active",False),("query_truth_authorized",True),("scientific_fallback",True),("contrastive_execution","gradient_cache"),("output",str(fixture/"borrowed")),("short_tmp",str(fixture/"wrong_tmp"))]:
        bad=copy.deepcopy(a);bad[field]=value
        reject("reject_"+stage+"_admission_"+field,lambda:m.validate_admission(c,parent,stage,"fresh",bad,r,rt,reg))
    for obj,field,val in [(r,"kind","native_argformer_tsinfer_fit_v1_independent_review"),(rt,"kind","native_argformer_tsinfer_fit_v1_root_runtime"),(reg,"active",False)]:
        altered=copy.deepcopy(obj);altered[field]=val
        rr=altered if obj is r else r;tt=altered if obj is rt else rt;gg=altered if obj is reg else reg
        reject("reject_"+stage+"_"+field+"_"+obj["kind"],lambda:m.validate_admission(c,parent,stage,"fresh",a,rr,tt,gg))
    scope,output,tmp,log=m.scopes(c,parent,stage,"fresh")
    check("source_scope_all_cached_outputs_"+stage,scope in output.parents and scope in log.parents and tmp.parent==ROOT/c["short_tmp_root"])
    check("short_AF_UNIX_path_"+stage,len(os.fsencode(tmp/"pymp-12345678/listener-12345678"))<=107)
# Empty inventory is accepted, and actual preflight does not inspect it or scientific distributions.
a,r,rt,reg=authorities("retrieve")
accepted=m.validate_admission(c,parent,"retrieve","fresh",a,r,rt,reg)
with patch.dict(os.environ,{"PYTHONPATH":"","CUDA_VISIBLE_DEVICES":""}):
    preflight=m.runtime_preflight(c,rt,True)
check("counterexample_empty_inventory_passes_runtime_gate",accepted==a and preflight["CPU_only"])
counterexamples.append({"id":"RUNTIME_INVENTORY_NOT_ENFORCED","source_lines":[235,236,237,387,399],"accepted_fixture":inventory,
"observation":"An empty immutable JSON qualifies as environment_inventory_receipt and native runtime_preflight passes. No expected inventory schema, complete native environment bytes, retrieval numpy/torch versions, or tskit version is compared. Native GPU path later checks distribution versions only and not inventory bytes."})
for value in ["../escape","/tmp/escape","data/query_truth.json","data/oracle.json"]:
    reject("confined_reject_"+value,lambda value=value:m.confined(value))
for parent in c["parent_ids"]:
    n=m.scopes(c,parent,"fit","fresh");o=old.scopes(oc,parent,"fit","fresh")
    check("fresh_versioned_parent_scopes_"+parent,n[0]!=o[0] and n[2]!=o[2])
# Complete bootstrap uses pinned native interpreter, real preparation and owned caches, exec intercepted only.
bc=copy.deepcopy(c);bc["result_root"]=str(fixture/"bootstrap");bc["short_tmp_root"]=".runtime/tmp/afts2"
arg=types.SimpleNamespace(parent=parent,stage="retrieve",run_id="bootstrap",admission=str(fixture/"bootstrap_admission.json"))
write(Path(arg.admission),{"fixture_only_not_authority":True})
storage=types.SimpleNamespace(check=lambda:{"fixture_only":True})
owner={"owns_output":False};calls=[]
def stop_exec(path,argv,environ):
    calls.append((str(path),argv,environ));raise RuntimeError("independent deliberate stop before actual child execution")
with patch.dict(os.environ,dict(os.environ),clear=True),patch.object(m.os,"execve",side_effect=stop_exec):
    reject("complete_bootstrap_exec_intercept",lambda:m.bootstrap(bc,arg,{},rt,storage,owner))
scope,out,tmp,log=m.scopes(bc,arg.parent,arg.stage,arg.run_id)
check("bootstrap_preserves_fresh_owned_cache_and_short_tmp",owner["owns_output"] and out.is_dir() and calls[0][2]["TMPDIR"]==str(tmp) and calls[0][2]["CUDA_VISIBLE_DEVICES"]=="")
check("bootstrap_only_exact_native_interpreter",calls[0][0]==str(ROOT/c["runtime"]["interpreter"]))
check("bootstrap_no_unadmitted_PYTHONPATH",calls[0][2]["PYTHONPATH"]=="")
reject("bootstrap_output_reuse_is_terminal",lambda:m.bootstrap(bc,arg,{},rt,storage,{"owns_output":False}))
tmp.rmdir()
# Complete execute(native=True) rejection route: a runtime preflight error escapes outside its owned-stage receipt handler.
ec=copy.deepcopy(c);ec["result_root"]=str(fixture/"child_failure");ec["short_tmp_root"]=str(fixture/"child_tmp")
arg=types.SimpleNamespace(parent=c["parent_ids"][0],stage="retrieve",run_id="child",admission=str(fixture/"child_admission.json"),expected_admission_sha256="",expected_source_sha256=sha(m.SCRIPT),expected_config_sha256=sha(m.CONFIG),native=True)
aa,rr,tt,gg=authorities("retrieve","child");aa.update(m.admission_bindings(ec,arg.parent,arg.stage,arg.run_id))
aa.update(independent_review={"path":"fixture_review","sha256":"1"*64},runtime={"path":"fixture_runtime","sha256":"2"*64},experiment_registration={"path":"fixture_reg","sha256":"3"*64},prepared_dispatcher={"path":"fixture_dispatcher","sha256":"4"*64},storage={"path":"fixture_storage","sha256":"5"*64})
ar=write(Path(arg.admission),aa);arg.expected_admission_sha256=ar["sha256"]
_,out,_,_=m.scopes(ec,arg.parent,arg.stage,arg.run_id);out.mkdir(parents=True);(out/"exec_admission.json").write_text("{}")
mapping={str(arg.admission):aa,"fixture_review":rr,"fixture_runtime":tt,"fixture_reg":gg,"fixture_storage":{}}
def fixture_read(record,*args):return mapping[record["path"]]
with patch.object(m,"read_json",side_effect=fixture_read),patch.object(m,"prepared_metadata",return_value={}),patch.object(m,"Storage",return_value=storage),patch.object(m,"runtime_preflight",side_effect=ValueError("independent child runtime mismatch")):
    details["owned_native_preflight_failure"]=reject("complete_execute_native_child_preflight_rejected",lambda:m.execute(ec,arg))
check("counterexample_owned_child_failure_no_terminal_receipt",not (out/"receipt.json").exists() and not (out/"exit_status").exists())
counterexamples.append({"id":"OWNED_CHILD_PREFLIGHT_FAILURE_NO_RECEIPT","source_lines":[645,646,647,648,649,650,656,663],"fixture_output":str(out.relative_to(ROOT)),
"observation":"Complete execute(native=True) rejects a child runtime-preflight mismatch after bootstrap owns output, but the exception lies outside try/finally and leaves exec_admission.json without receipt.json or exit_status."})
# The real fit wrapper must reject bad memory authority before native execution. No placeholder model or fake fit success.
fitargs=types.SimpleNamespace(parent=c["parent_ids"][0],stage="fit",run_id="rejected_fit")
data={"dispatcher":{"parent_id":fitargs.parent,"prepared_manifest":{"path":str(fixture/"never_read/prepared_manifest.json"),"sha256":"6"*64}},"prepared":{"namespace":"handmade"}}
guard=types.SimpleNamespace(apply_and_validate_title=lambda:None,validate_lease=lambda:{"canonical_uuid":"handmade"})
# This negative-only guard substitution never counts as model/scientific success.
with patch.object(m,"load_frozen_module",return_value=guard),patch.object(m,"read_json",return_value={"status":"failed"}),patch.dict(os.environ,{"ARGGNN_LOGICAL_JOB_NAME":"handmade"}):
    details["fit_rejected_before_model"]=reject("real_fit_wrapper_rejects_failed_memory_before_native_model",lambda:m.execute_fit(c,fitargs,{"logical_job_name":"handmade","full_batch_memory_receipt":{"path":"fixture","sha256":"7"*64}},data,storage,{"qualified_GPU_UUID":"handmade"},{}))
# Fresh small exhaustive scientific retrieval test: every280 leaf, all4608 pure rows,240 query paths and1,105,920 pairs.
import numpy as np
import torch
if importlib.util.find_spec("threadpoolctl") is None:
    helper=ROOT/".runtime/env/lib/python3.11/site-packages/threadpoolctl.py"
    dm=load("threadpoolctl",helper);sys.modules["threadpoolctl"]=dm
    details["test_only_helper"]={"path":str(helper.relative_to(ROOT)),"sha256":sha(helper),"version":dm.__version__}
directory=fixture/"retrieval_prepared";directory.mkdir()
nodes=[7+3*i for i in range(280)]
roster=[{"sample_index":i,"inferred_tree_node_id":n,"source_observed_node_id":i+2000,"role":"pure_source_reference" if i<96 else "mixed_reference_only" if i<160 else "query","observed_reference_class":i//24 if i<96 else None} for i,n in enumerate(nodes)]
paths=[];occurrences=[];at={}
for i,node in enumerate(nodes):
    count=48 if i<96 else 2
    for j in range(count):
        pid=len(paths);paths.append((node,10000+j));occurrences.append((pid,node,200*j/count,200*(j+1)/count));at[(i,j)]=pid
paths=tuple(paths);occurrences=tuple(occurrences);namespace=parent+":"+"8"*64
corpus={"paths":paths,"occurrences":occurrences,"namespace":namespace}
with (directory/"corpus.pkl").open("xb") as f:pickle.dump(corpus,f,protocol=4)
raw=directory/"embedding/all/raw_all_sequences_and_vocab.pkl";raw.parent.mkdir(parents=True)
vocab={"special":{"[PAD]":0,"[CLS]":1,"[SEP]":2,"[MASK]":3},"node_id_tokens":{"node_"+str(n):i+4 for i,n in enumerate(sorted({n for p in paths for n in p}))}}
with raw.open("xb") as f:pickle.dump({"raw_sequences":[list(p) for p in paths],"vocab":vocab},f,protocol=4)
positions=np.asarray([50,150])
joins={"grid_indexes":np.asarray([0,1]),"original_positions_bp":positions,"local_positions_bp":positions,"query_nodes":np.asarray(nodes[160:]),"reference_nodes":np.asarray(nodes[:96]),"reference_classes":np.asarray([i//24 for i in range(96)]),
"query_path_ids":np.asarray([[at[(i,j)] for i in range(160,280)] for j in (0,1)]),"reference_path_ids":np.asarray([[at[(i,j)] for i in range(96)] for j in (12,36)])}
np.savez_compressed(directory/"coordinate_path_joins.npz",**joins);write(directory/"observed_roster.json",roster)
prepared={"parent_id":parent,"namespace":namespace,"unique_full_paths":len(paths),"coordinate_contract":{"retained_fixed100bp_indexes":[0,1],"coverage_original_bp0":[0,200],"local_to_original_offset_bp":0},"files_sha256":{str(p.relative_to(directory)):sha(p) for p in directory.rglob("*") if p.is_file()}}
pref=write(directory/"prepared_manifest.json",prepared)
data={"dispatcher":{"parent_id":parent,"prepared_manifest":pref},"prepared":prepared,"budget":{"MLM_optimizer_updates":5,"contrastive_optimizer_updates":1,"pure_reference_corpus_paths_all_train_plus_val":4608,"actual_counts":{"query_full_paths_scored":240},"exhaustive_query_corpus_pairs":1105920}}
rc=copy.deepcopy(c);rc["result_root"]=str(fixture/"retrieval_results");rc["short_tmp_root"]=str(fixture/"retrieval_tmp")
args=types.SimpleNamespace(parent=parent,stage="retrieve",run_id="handmade")
_,output,_,_=m.scopes(rc,parent,"retrieve","handmade");output.mkdir(parents=True)
fitdir=Path(rc["result_root"])/parent/"fit"/"handmade";fitdir.mkdir(parents=True)
vectors=torch.zeros((len(paths),768),dtype=torch.float32);vectors[:,0]=1;vectors[0]=0
torch.save({"embeddings":vectors,"native_pooling":"trained_author_mean_head"},fitdir/"native_path_embeddings.pt")
np.savez_compressed(fitdir/"downstream_C4_local_readout.npz",fixture_only=np.asarray([1]))
fit={"fixture_only_not_native_fit_success":True,"status":"passed","action":"fit","exit_code":0,"root_actual_execution":True,"true_query_ancestry_read":False,"implementation_sha256":sha(m.SCRIPT),"config_sha256":sha(m.CONFIG),"parent_id":parent,"namespace":namespace,"prepared_manifest_sha256":pref["sha256"],"published_native_task_reproduction":False,"classifier_unchanged_unused":True,"encoder_initialization":{"all_encoder_keys_loaded_equal":["fixture_only"],"missing_encoder":[],"unequal_encoder":[]},"readout":{"native_embedding_sha256":hashlib.sha256(vectors.numpy().tobytes()).hexdigest()},"artifacts_sha256":{p.name:sha(p) for p in fitdir.iterdir()}}
for phase,mode,ep,key in [("pretraining","mlm",5,"MLM_optimizer_updates"),("contrastive","contrastive",1,"contrastive_optimizer_updates")]:
    fit[phase]={"mode":mode,"execution":"full_batch","completed_epochs":ep,"actual_optimizer_updates":data["budget"][key],"backbone_unchanged":True,"true_query_ancestry_read":False,"initial_model_state_sha256":"1"*64,"final_model_state_sha256":"2"*64}
fitref=write(fitdir/"receipt.json",fit)
result=m.execute_retrieval(rc,args,{"completed_fit_receipt":fitref},data,storage,{})
seal=result["prediction_seal"];diag=seal["diagnostics"]
check("real_retrieval_wrapper_all_pure_full_paths_and_queries",diag["pure_reference_full_paths"]==4608 and diag["unique_query_paths"]==240 and diag["exhaustive_pairs"]==1105920)
check("duplicate_embeddings_retained_zero_rows_and_crossclass_ties",diag["native_embedding_zero_or_epsilon_rows"]==1 and diag["top1_cross_class_exact_max_ties"]==240)
with np.load(m.bound(seal["neighbors"]),allow_pickle=False) as z:
    check("exact_original_fullpath_tie_order",np.array_equal(z["neighbor_full_path_ids"][0],np.arange(1,7)))
    check("full_pure_reference_ids_no_query_selfmatch",np.array_equal(z["corpus_full_path_ids"],np.arange(4608)))
with np.load(m.bound({"path":seal["path"],"sha256":seal["sha256"]}),allow_pickle=False) as z:
    check("real_k1_k5_sealed_all_retained_query_intervals",z["C4_probabilities_raw_k1"].shape==(2,120,4) and z["C4_probabilities_raw_k5"].shape==(2,120,4))
    check("raw_vote_zero_preserved",np.any(z["C4_probabilities_raw_k1"]==0))
    check("query_native_nonconsecutive_IDs_retained",np.array_equal(z["query_nodes"],nodes[160:]))
check("no_GPU_or_truth_in_handmade_retrieval",not torch.cuda.is_initialized() and result["external_truth_access_authorized"] is False and seal["truth_opened"] is False)
ns=m.compile_retrieval(c)
bad=copy.deepcopy(joins);bad["query_path_ids"][0,0]=-1
reject("real_support_rejects_missing_query_path",lambda:ns["validate_support"](paths,occurrences,namespace,namespace,bad,200))
reject("real_support_rejects_cross_ARG_namespace",lambda:ns["validate_support"](paths,occurrences,"other",namespace,joins,200))
bad=copy.deepcopy(joins);bad["reference_path_ids"][0,0]=1
reject("real_support_rejects_wrong_reference_interval_path",lambda:ns["validate_support"](paths,occurrences,namespace,namespace,bad,200))
bad=copy.deepcopy(roster);bad[160]["observed_reference_class"]=0
reject("real_roster_rejects_query_ancestry_label",lambda:m.retrieval_parent(data,joins,bad,np))
scores=np.asarray([[1,1,0,1,0,1,1]],dtype=np.float32);ids=np.asarray([70,2,9,11,1,5,3])
v,i=ns["lexicographic_topk"](scores,ids,5)
check("independent_nonordered_tie_counterexample",i.tolist()==[[2,3,5,11,70]])
reject("real_topk_rejects_nonfinite_scores",lambda:ns["lexicographic_topk"](np.asarray([[np.nan]],dtype=np.float32),np.asarray([1]),1))
details["fresh_retrieval_fixture"]={"full_paths":len(paths),"observed_leaves":280,"pure_reference_full_paths":4608,"query_full_paths":240,"exhaustive_pairs":1105920,"seal":seal}
for p,h in expected.items():check("pins_preserved_after_tests_"+p,sha(ROOT/p)==h)
report={"kind":"native_argformer_tsinfer_fit_v2_independent_handmade_checks","status":"completed_with_concrete_guard_counterexamples","check_count":len(checks),"checks":checks,"counterexamples":counterexamples,"details":details,"implementation":m.reference(m.SCRIPT),"config":m.reference(m.CONFIG),"test_source":m.reference(__file__),"source_only":True,"actual_model_fit_executed":False,"released_model_instantiated":False,"placeholder_model_counted_as_native_success":False,"actual_scientific_arrays_opened":False,"query_truth_opened":False,"GPU_inventory_or_context_used":False,"actual_live_ledger_read":False,"runtime_installation_or_authority_activation":False,"fresh_full_retrieval_kernel_on_handmade_only":True}
write(REVIEW/"handmade_checks.json",report)
print(json.dumps({"status":report["status"],"check_count":len(checks),"counterexamples":counterexamples},indent=2))
