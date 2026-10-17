"""Independent H059 whole-source review checks using static metadata and RAM doubles only."""
import ast,copy,hashlib,json,os,pathlib,stat
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H059_outer000_Relate64_execution_source_20261017_v1"
OLD="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
OUT=ROOT/"research/reviews/H059_outer000_Relate64_independent_review_20261017_v1"
def need(ok,message):
 if not ok:raise ValueError(message)
def raw(name):
 p=ROOT/name;s=p.lstat();need(p.resolve()==p and ROOT in p.parents and stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=32*2**20,"Only exact bounded source/metadata")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:opened=os.fstat(f.fileno());data=f.read(32*2**20+1);after=os.fstat(f.fileno())
 identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(identity(s)==identity(opened)==identity(after)==identity(p.lstat())and len(data)==s.st_size,"Stable exact metadata/source")
 return data
def ref(name):return dict(path=name,sha256=hashlib.sha256(raw(name)).hexdigest())
def bound(record):
 data=raw(record["path"]);need(hashlib.sha256(data).hexdigest()==record["sha256"],"Exact bound SHA "+record["path"]);return data
def get(tree,name):
 found=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name];need(len(found)==1,"Single source function "+name);return found[0]
def dump(node):return ast.dump(node,include_attributes=False)
source_ref=ref(HERE+"/prepare_outer000_v1.py");manifest_ref=ref(HERE+"/source_manifest.v1.json");handoff_ref=ref(HERE+"/source_handoff.v1.json")
need(source_ref["sha256"]=="876f026ac8f6ef4dd441cd02d961a81320032368524aca250b993d1349cf5ef5"and manifest_ref["sha256"]=="fed146aa3f35a2532afec7092f73352eb1d606f314c7b276812d6ecdf1b60ba9"and handoff_ref["sha256"]=="7d54ce00d96c3381a97eb0623f5de736322901ebf0203183d20aa4488cc544dc","Requested immutable inputs")
manifest=json.loads(bound(manifest_ref));predecessor=json.loads(bound(manifest["predecessor_manifest_ref"]));need(len(manifest["refs"])==330 and len(predecessor["refs"])==241,"Complete source ref counts")
mapping={r["path"]:r["sha256"]for r in manifest["refs"]};need(len(mapping)==330 and all(mapping.get(r["path"])==r["sha256"]for r in predecessor["refs"]),"Entire unchanged241-ref predecessor closure retained")
bytes_read=python_files=0
for record in manifest["refs"]:
 data=bound(record);bytes_read+=len(data)
 need(pathlib.Path(record["path"]).suffix in(".py",".sh",".json",".md",".patch",".diff"),"No model/array/tree/binary payload reference opened")
 if pathlib.Path(record["path"]).suffix==".py":ast.parse(data,filename=record["path"]);python_files+=1
need(len(manifest["expected_original_metadata_and_assets"])==6,"Original6 expected asset declarations retained without reads")
new=ast.parse(bound(source_ref));old=ast.parse(raw(OLD+"/prepare_outer000_v1.py"))
source_map=json.loads(raw(HERE+"/source_map.v1.json"));unchanged=source_map["unchanged_definitions"]
for name in unchanged:need(dump(get(new,name))==dump(get(old,name)),"Exact unchanged original function "+name)
for name in ("worker","execute"):
 node=copy.deepcopy(get(new,name));removed=[]
 node.body=[n for n in node.body if not(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=="verify_generation_roles_v1"and not removed.append(n))]
 need(len(removed)==1 and dump(node)==dump(get(old,name)),"Entire original "+name+" restored after sole role assertion removal")
need(all(isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.If))or isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)for n in new.body),"Inert driver top-level shape")
bindings=json.loads(raw(HERE+"/source_bindings.v1.json"));cfg=json.loads(raw("configs/expanded_full_panel_v1.json"));registry=json.loads(raw("research/reviews/European_outer8_preparation_source_20261016_v3/outer8_registry.v1.json"));protocol=json.loads(raw("research/reviews/European_outer8_preparation_source_20261016_v3/protocol.v1.json"));methods=json.loads(raw("research/reviews/European_outer8_preparation_source_20261016_v3/frozen_method_ids.v1.json"))
outer=[r for r in cfg["parent_registry"]if r["split"]=="outer"]
need(outer==registry["outer_rows"]and len(outer)==8 and all(r["downstream_role"]=="UNTOUCHED_OUTER"and r["evaluation_authorized"]is False for r in outer),"All original8 seed-domain rows retained")
need(cfg["roster"]["haplotypes"]==952 and cfg["roster"]["diploids"]==476 and cfg["roster"]["pure_reference_haplotypes"]==266 and cfg["roster"]["query_haplotypes"]==686 and cfg["simulation"]["length_bp"]==2000000,"Full original roster/length")
need(cfg["simulation"]==protocol["scientific_identity"]["simulation"]and cfg["roster"]==protocol["scientific_identity"]["roster"]and cfg["demography"]==protocol["scientific_identity"]["demography"]and cfg["relate"]==protocol["scientific_identity"]["relate"],"Original biological/Relate settings match frozen protocol")
need(cfg["relate"]["population_size_iterations"]==5 and cfg["relate"]["final_branch_length_reestimate"]is True and cfg["relate"]["timeout_seconds_per_step"]==7200,"Original full Relate inference")
expected_methods=["H050_stable_leaf_memory","H050_h045_continued_training","AncestralPaths_native_Dense5","ARGMix_current_native","ARGformer_native_within_ARG"]
freeze=json.loads(bound(bindings["frozen_method_freeze_ref"]));need(methods["primary_method_ids"]==freeze["method_ids"]==expected_methods and freeze["H058_automatically_included"]is False and protocol["all8_predictions_for_all5_methods_before_any_outer_truth"]is True,"Original five frozen slots and full pretruth barrier")
caps=json.loads(raw(HERE+"/caps.INACTIVE.v1.json"));need(caps["reservation_bytes"]==4294967296 and caps["output_bytes"]==4000000000 and caps["auxiliary_bytes"]==134217728 and caps["root_packet_bytes"]==8388608 and caps["wall_seconds"]==43200 and caps["cleanup_seconds"]==30,"Original finite whole caps")
need(caps["output_bytes"]+caps["auxiliary_bytes"]+caps["root_packet_bytes"]==4142606336<caps["reservation_bytes"],"Packet/AUX/science caps fit unchanged whole cap without fit claim")
admit=ast.unparse(get(new,"admit"))
need(admit.index("not packet.exists()")<admit.index("g.budget.v4.scope_spec")<admit.index("ar = put(prep")<admit.index("g.apply_reserve_v1")<admit.index("packet.mkdir()"),"Whole absent scopes and exact schema/role preflight precede reserve and packet writes")
need("auxiliary_scopes=[str(packet)]"in admit and "not packet.exists()"in admit[admit.index("g.apply_reserve_v1"):],"Entire packet charged and absent through successful reserve")
need("registration_ledger_ref"not in ast.unparse(get(new,"generation_direct_refs_v1"))and "registration_ledger_ref"in ast.unparse(get(new,"registration")),"Forbidden Markdown indirect behind bound JSON, still raw SHA verified")
# Only source-derived pure role function and original consumer/schema predicates
# run below. All filesystem/canonical/bound readers here are handwritten doubles.
constants={}
for n in new.body:
 if isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name)and t.id in("CONFIG","PRODUCER","ORIGINAL","PIDFD","OWNED","NAMESPACE","V4"):
    need(isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=="dict","Literal constant ref")
    constants[t.id]={k.arg:ast.literal_eval(k.value)for k in n.value.keywords}
def future(name):return dict(path="RAM_FUTURE/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
environment=dict(HERE=HERE,SOURCE=source_ref["path"],MANIFEST=manifest_ref["path"],REGISTRY="research/reviews/European_outer8_preparation_source_20261016_v3/outer8_registry.v1.json",METHODS="research/reviews/European_outer8_preparation_source_20261016_v3/frozen_method_ids.v1.json",PROTOCOL="research/reviews/European_outer8_preparation_source_20261016_v3/protocol.v1.json",SELECTOR=bindings["guard_source_ref"],ORIGINAL_SELECTOR=bindings["original_selector_ref"],ref=ref,**constants)
exec(compile(ast.Module(body=[copy.deepcopy(get(new,"generation_direct_refs_v1"))],type_ignores=[]),"<independent_source_role_builder_RAM>","exec"),environment)
q={k:future(k)for k in("runtime_ref","source_review_ref","runtime_qualification_ref")}
q.update(method_freeze_ref=bindings["frozen_method_freeze_ref"],policy_selector_ref=bindings["actual_selector_ref"],registration_ref=bindings["registration_ref"],registration_decisions_ref=bindings["registration_decisions_ref"],registration_ledger_ref=bindings["registration_ledger_ref"])
roles=environment["generation_direct_refs_v1"](q,future("request"));guard_names=json.loads(bound(bindings["guard_role_snapshot_ref"]))["bindings"]
guardtree=ast.parse(bound(bindings["guard_source_ref"]));negative=[]
def bound_RAM(record):
 need(isinstance(record,dict)and set(record)=={"path","sha256"}and isinstance(record["path"],str)and isinstance(record["sha256"],str)and len(record["sha256"])==64,"Handwritten RAM ref shape")
 return record
consumer_env=dict(copy=copy,require=need,_raw=bound_RAM)
exec(compile(ast.Module(body=[copy.deepcopy(get(guardtree,"consumer_bindings_v1"))],type_ignores=[]),"<original_consumer_bindings_RAM_only>","exec"),consumer_env)
combine=consumer_env["consumer_bindings_v1"];union=combine(list(guard_names.values()),roles)
need(len(guard_names)==46 and len(combine(list(guard_names.values())))==39 and len(union)==60 and all(r in union for r in guard_names.values())and all(pathlib.Path(r["path"]).suffix in(".py",".sh",".json")for r in union),"Complete exact60/64 metadata role union")
try:combine(list(guard_names.values())+[future("extra"+str(i))for i in range(26)],roles)
except ValueError:negative.append("All actual guard roles plus distinct overflow rejects")
else:raise ValueError("Original consumer role overflow accepted")
v4=ast.parse(raw("src/arggnn/project_budget_v4.py"))
scope_env=dict(Path=pathlib.Path,integer=lambda v,n,minimum=0:need(type(v)is int and v>=minimum,"Exact integer"),canonical=lambda root,p:root/p if not pathlib.Path(p).is_absolute()else pathlib.Path(p),paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,accounting_paths=lambda root:[],bound=lambda root,r:bound_RAM(r))
exec(compile(ast.Module(body=[copy.deepcopy(get(v4,"scope_spec"))],type_ignores=[]),"<original_scope_spec_RAM_only>","exec"),scope_env)
own=dict(key=bindings["proposed_reservation_key"],output=str(ROOT/bindings["output_directory"]),auxiliary_scopes=[str(ROOT/bindings["packet_directory"])],reserved_bytes=4294967296,consumer_bindings=union)
need(scope_env["scope_spec"](ROOT,own)==[pathlib.Path(own["output"]),pathlib.Path(own["auxiliary_scopes"][0])],"Original scope schema accepts complete2root reservation")
for label,change in (("Direct Markdown",lambda r:r["consumer_bindings"].append(bindings["registration_ledger_ref"])),("65 roles",lambda r:r["consumer_bindings"].extend(future("over"+str(i))for i in range(5))),("Scope overlap",lambda r:r.update(auxiliary_scopes=[r["output"]]))):
 wrong=copy.deepcopy(own);change(wrong)
 try:scope_env["scope_spec"](ROOT,wrong)
 except(ValueError,KeyError,TypeError):negative.append(label)
 else:raise ValueError("Invalid original scope schema accepted "+label)
# Exact selection accepts one immutable actual capsule; mutations stay rejected
# before configuring a guard. Reader and raw functions are RAM metadata doubles.
selected=json.loads(bound(bindings["actual_selector_ref"]));selection_env=dict(need=need,ACTUAL_SELECTOR_REF=bindings["actual_selector_ref"],SELECTOR=bindings["guard_source_ref"],APPROVED={selected["accounting_mode"]:dict(source_ref=selected["source_ref"],manifest_ref=selected["source_manifest_ref"],policy_ref=selected["policy_ref"])},read=lambda r:copy.deepcopy(selected),raw=lambda r:r)
exec(compile(ast.Module(body=[copy.deepcopy(get(new,"selection"))],type_ignores=[]),"<selection_metadata_RAM_only>","exec"),selection_env)
need(selection_env["selection"](bindings["actual_selector_ref"])==selected,"Exact e27d47a current Relate capsule")
for label,alter in (("Stale selector",None),("Inactive selector",("active",False)),("Old mode",("accounting_mode","native7_H050_stopped_retention_v1")),("Wrong policy",("policy_ref",future("wrong_policy")))):
 original=copy.deepcopy(selected)
 if alter:selected[alter[0]]=alter[1]
 try:selection_env["selection"](future("old_selector")if alter is None else bindings["actual_selector_ref"])
 except(ValueError,KeyError,TypeError):negative.append(label)
 else:raise ValueError("Bad selection accepted "+label)
 selected.clear();selected.update(original)
consumer_source=ast.unparse(get(new,"consumer_direct_refs_v1"));need("raise ValueError" in consumer_source,"No downstream scoring/pretruth bypass interface")
result=dict(schema="H059_outer000_Relate64_independent_source_checks_v1",status="SOURCE_PASS_STATIC_RAM_ONLY",source_ref=source_ref,source_manifest_ref=manifest_ref,source_handoff_ref=handoff_ref,source_refs_verified=330,preserved_predecessor_refs_verified=241,static_source_metadata_bytes=bytes_read,Python_ASTs_parsed=python_files,original_unchanged_functions=unchanged,worker_and_execute_whole_AST_restored_except_one_current_role_assertion_each=True,full_historical_named_guard_roles=46,full_historical_distinct_guard_refs=39,full_generation_union_refs=60,direct_role_ceiling=64,full_generation_union=union,original_scope_schema_accepts_whole_packet_output=True,negative_cases=negative,all_original_science_and_frozen_protocol_preserved=True,all8_original_seed_rows_retained=True,numeric_map_age_original_assets_left_unopened=6,model_or_numerical_payloads_read=False,project_guard_science_modules_imported=False,actual_resource_process_mutex_probe_admission_journal_Git_operations=False,actual_current_guard_state_capacity_key_absence_scope_fit_observed=False)
p=OUT/"independent_checks.v1.json"
with p.open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(dict(status=result["status"],result_ref=ref(str(p.relative_to(ROOT))),source_refs=330,predecessor_refs=241,union=len(union),negative_cases=len(negative))))
