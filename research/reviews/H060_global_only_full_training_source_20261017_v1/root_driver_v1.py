"""Inactive common contract/driver; exact child roles and reviewed reserve transaction memo."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,os,stat,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H060_global_only_full_training_source_20261017_v1"
SOURCE=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
def need(ok,message):
 if not ok:raise ValueError(message)
def local(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and ".."not in p.parts and p.resolve()==p,"Canonical contained source/metadata")
 return p
def metadata_ref(path):
 p=local(path);st=p.lstat()
 need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=32*2**20,"Compact regular source/metadata")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def raw(record):
 need(isinstance(record,dict)and set(record)=={"path","sha256"}and metadata_ref(record["path"])==record,"Exact immutable source/metadata SHA")
 return local(record["path"]).read_bytes()
def read(record):return json.loads(raw(record))
def bindings():return read(metadata_ref(HERE+"/source_bindings.v1.json"))
def load(record,name):
 raw(record);sp=importlib.util.spec_from_file_location(name,local(record["path"]))
 module=importlib.util.module_from_spec(sp);sys.modules[name]=module;sp.loader.exec_module(module);return module
def append_selected_source(roles,source,bound_source):
 """Pure, order-preserving addition. Actual unchanged consumer API enforces <=64."""
 need(isinstance(roles,list)and isinstance(source,dict)and set(source)=={"path","sha256"}
  and source==bound_source,"Exact metadata-selected and immutable lifetime-bound source only")
 result=list(roles)
 if source not in result:result.append(copy.deepcopy(source))
 return result
def global_scientific_review(c,record):
 """Whole-route independent scientific review is mandatory, separate from component PASS."""
 b=c.binding();v=c.read(record)
 c.require(v.get("schema")=="H060_global_only_complete_scientific_review_v1"
  and v.get("status")=="SOURCE_PASS"and v.get("independent_review")is True and v.get("blockers")==[]
  and all(v.get(k)==b[bkey]for k,bkey in (
   ("candidate_manifest_ref","source_manifest_ref"),("candidate_protocol_ref","protocol_ref"),
   ("candidate_handoff_ref","source_handoff_ref"),("candidate_scientific_pins_ref","scientific_source_pins_ref"),
   ("component_review_ref","global_component_review_ref")))
  and v.get("candidate_outer_manifest_ref")==metadata_ref(MANIFEST)
  and all(r in v.get("candidate_source_refs",[])for r in b["global_preparation_source_refs"]+[b["runner_source_ref"]]),
  "New complete independent global-only scientific review of actual full route/source closure")
 review=c.read(b["global_component_review_ref"])
 c.require(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_ref")==b["global_component_source_ref"]
  and review.get("source_manifest_ref")==b["global_component_manifest_ref"],
  "Exact ROOT independent typed-global component pass is evidence, never whole-route verdict")


def contract(consumer):
 """Actual new contract and driver share THIS implementation file, saving one real role."""
 spec=bindings()["consumers"].get(consumer);need(spec is not None,"Exact H060 or H061 consumer")
 c=load(spec["contract_ref"],"_child_source_repair_"+consumer+"_contract")
 # Isolated module globals only. Frozen files/modules/guards are never patched.
 # Original functions keep their original globals in this fresh private module.
 c.HERE=c.REL=HERE;c.SOURCE=c.DRIVER=SOURCE;c.ENTRY=ENTRY;c.MANIFEST=MANIFEST;c.HANDOFF=HANDOFF
 c.binding=lambda:read(spec["scientific_binding_ref"])
 c.scientific_review=lambda record:global_scientific_review(c,record)
 selector=load(bindings()["selector_bridge_ref"],"_H060_exact_Census_selector_bridge")
 selector.install_v1(c,bindings())
 original_roles=c.direct_roles
 def repaired_roles(request_ref,q,cfgref,closure):
  roles=original_roles(request_ref,q,cfgref,closure)
  chosen=c.selected(q["storage_policy_selector_ref"])
  c.require(c.SELECTOR_REF==q["storage_policy_selector_ref"]and c._CHOSEN==chosen
   and c.GUARD_REF==chosen["source_ref"],"Same exact lifetime-selected source before role union")
  return append_selected_source(roles,chosen["source_ref"],c.GUARD_REF)
 c.direct_roles=repaired_roles
 original_validate=c.validate_request
 def validate_request(record):
  q=c.read(record)
  capsule=c.read(q["storage_policy_selector_ref"]);child=bindings()["selected_child_source_ref"]
  c.require(q["storage_policy_selector_ref"]==bindings()["actual_selector_metadata_ref"]
   and capsule.get("source_ref")==child and capsule.get("selector_source_ref")==child,
   "Exact current ROOT Census selector capsule and child only; no fallback")
  prefixes=bindings()["fresh_namespace_prefixes"]
  c.require(all(isinstance(q.get(key),str)and q[key].startswith(prefix)
   for key,prefix in prefixes.items()),"Fresh global-only namespaces only, never an old scope")
  c.require(all(all(q.get(key)!=old[key]for key in("preparation_directory","packet_directory","output_directory"))
   for old in bindings()["excluded_scope_sets"]),"Every original failed/current scope remains separate")
  c.require(q.get("fresh_full_unfinished_cell_successor")is True
   and q.get("old_scope_replay_or_live_replacement")is False
   and q.get("new_global_observation_protocol_required")is True and q.get("transfer_resume_forbidden")is True,
   "Separate full unfinished-cell successor, no live replacement or old-scope replay")
  c.require(q.get("guard_role_repair_consumer")==consumer
   and q.get("source_manifest_ref")==c.ref(MANIFEST),"Genuine ROOT fresh successor request for this consumer")
  c.require(all(q.get(key)!=spec["excluded_"+key]for key in("preparation_directory","packet_directory","output_directory")),
   "Never retry/replay old H060 failure or old H061 preparation namespaces")
  result=original_validate(record)
  directive=c.read(q["root_directive_ref"])
  c.require(directive.get("fresh_full_unfinished_cell_successor")is True
   and directive.get("old_scope_replay_or_live_replacement")is False
   and directive.get("new_global_observation_protocol_required")is True and directive.get("transfer_resume_forbidden")is True,
   "ROOT directive explicitly binds the fresh full unfinished-cell successor")
  return result
 c.validate_request=validate_request
 return c
def driver(consumer,c):
 d=load(bindings()["consumers"][consumer]["driver_ref"],"_child_source_repair_"+consumer+"_driver")
 # Reuse the exact reviewed admission/execution functions, including pre-authority
 # union+scope_spec and all finite lifecycle/scientific completion predicates.
 for name,value in dict(c=c,ROOT=ROOT,HERE=HERE,SOURCE=SOURCE,PIDFD=c.PIDFD,OWNED=c.OWNED,
  need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=c.put,current=c.current,guard=c.guard).items():
  setattr(d,name,value)
 return d
def _make_root_mutation_memo(g,q,request_ref):
 """Accepted memo2 with exact reviewed current-Census operands and raw control paths."""
 b=bindings();manifest=read(q["source_manifest_ref"]);pins=manifest["source_sha256"]
 for key in ("ROOT_memo_source_ref","ROOT_memo_manifest_ref","ROOT_memo_handoff_ref","ROOT_memo_review_ref","ROOT_memo_bridge_ref"):
  record=b[key];need(pins.get(record["path"])==record["sha256"],"Exact transitive memo2/bridge/review pin");raw(record)
 review=read(b["ROOT_memo_review_ref"])
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_ref")==b["ROOT_memo_source_ref"],"Genuine exact accepted memo2 review")
 base=load(b["ROOT_memo_source_ref"],"_H060_exact_ROOT_memo2")
 bridge=load(b["ROOT_memo_bridge_ref"],"_H060_exact_Census_memo_bridge")
 paths=set()
 def collect(value):
  if type(value)is dict:
   if set(value)=={"path","sha256"}and type(value["path"])is str and Path(value["path"]).suffix==".json":paths.add(value["path"])
   else:
    for child in value.values():collect(child)
  elif type(value)in(list,tuple):
   for child in value:collect(child)
 collect(request_ref);collect(q);collect(manifest)
 for path in pins:
  if Path(path).suffix==".json":paths.add(path)
 for key in ("root_directive_ref","training_context_ref","registration_ref","runtime_ref","runtime_qualification_ref","source_review_ref","scientific_review_ref"):
  collect(read(q[key]))
 if q.get("ROOT_metadata_preparation_instruction_ref")is not None:collect(read(q["ROOT_metadata_preparation_instruction_ref"]))
 selector=read(q["storage_policy_selector_ref"]);policy=read(selector["policy_ref"])
 collect(selector);collect(policy);collect(read(policy["root_activation_ref"]))
 prep=q["preparation_directory"];stem=Path(q["packet_directory"]).name
 for suffix in (".config.json",".source_and_input_closure.json",".reserve_authority.json"):paths.add(prep+"/"+stem+suffix)
 need(len(paths)<=2048,"Bounded uncached current metadata/control exclusions")
 methods,counts=bridge.derive_memo_methods_v1(base,raw(b["ROOT_memo_source_ref"]),b["selected_child_source_ref"],tuple(sorted(paths)),b["ROOT_memo_bridge_ref"])
 need(counts==[1,1],"Exact accepted memo2 Census authentication and current exclusions only")
 Memo=type("H060CensusRootMutationMemoV2",(base.RootMutationMemo,),methods)
 cache=[dict(path=path,sha256=sha)for path,sha in sorted(pins.items())if Path(path).suffix in(".py",".sh")and Path(path).parts[0]in("src","scripts","research")]
 return Memo(g,integration_manifest_ref=q["source_manifest_ref"],adapter_ref=b["ROOT_memo_source_ref"],cache_refs=cache)

def _memo_admission_node(node,consumer):
 """Derive only the three reviewed admission substitutions from exact pinned source."""
 need(consumer in("H060","H061")and isinstance(node,ast.FunctionDef)and node.name=="admit"
  and not node.decorator_list,"Exact original admission function")
 result=copy.deepcopy(node);prefix=""if consumer=="H060"else"c."
 same=lambda a,b:ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 parse=lambda text:ast.parse(text).body[0]
 guard_node=parse("g = "+prefix+"guard(False)")
 state_node=parse("with g.budget.v4.mutex("+prefix+"ROOT):\n state = g._state_locked(implementation['sha256'], launcher['sha256'], fresh=True)")
 reserve_node=parse("receipt = g.apply_reserve_v1(ar, expected_source_sha256=implementation['sha256'], expected_launcher_sha256=launcher['sha256'])")
 found=[[i for i,n in enumerate(result.body)if same(n,wanted)]for wanted in(guard_node,state_node,reserve_node)]
 need(all(len(indices)==1 for indices in found),"One exact original guard/state/reserve site")
 gi,si,ri=[indices[0]for indices in found]
 need(gi<si<ri and sum(isinstance(n,ast.With)for n in ast.walk(result))==1,
  "Exact original operation order and only original initial mutex")
 result.body[si]=parse("state = memo.read_state_v1(expected_source_sha256=implementation['sha256'], expected_launcher_sha256=launcher['sha256'])")
 result.body[ri].value.func=ast.Attribute(value=ast.Name(id="memo",ctx=ast.Load()),attr="reserve_v1",ctx=ast.Load())
 result.body.insert(gi+1,parse("memo = _make_root_mutation_memo(g,q,request_ref)"))
 return ast.fix_missing_locations(result)
def _install_memo_admit(d,consumer):
 record=bindings()["consumers"][consumer]["driver_ref"];tree=ast.parse(raw(record))
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="admit"]
 need(len(nodes)==1,"One pinned original admission")
 derived=_memo_admission_node(nodes[0],consumer);d.__dict__["_make_root_mutation_memo"]=_make_root_mutation_memo
 exec(compile(ast.Module(body=[derived],type_ignores=[]),SOURCE+"#exact-original-admit-with-memo","exec"),d.__dict__)
def execute_or_admit(consumer,record,admission):
 c=contract(consumer);d=driver(consumer,c)
 if admission:_install_memo_admit(d,consumer)
 result=d.admit(record)if admission else d.execute(record)
 if admission:
  # This exact emitted successor module requires its consumer discriminator.
  result["execution_argv"]+=["--consumer",consumer]
  result["guard_role_repair_consumer"]=consumer
 return result
def consumer_for_admitted(record):
 s=read(record);matches=[k for k,v in bindings()["consumers"].items()if s.get("parent_id")==v["parent_id"]]
 need(len(matches)==1 and s.get("root_driver_ref")==metadata_ref(SOURCE)
  and s.get("main_source_ref")==metadata_ref(ENTRY),"Exact admitted common successor identity")
 return matches[0]
def worker(record,gate_fd):
 consumer=consumer_for_admitted(record);c=contract(consumer)
 w=load(bindings()["consumers"][consumer]["worker_ref"],"_child_source_repair_"+consumer+"_worker")
 w.c=c;w.ROOT=ROOT
 # Original worker validates the full new admitted input and real gate before
 # every scientific import; exact budgets, runner, settings and waits unchanged.
 return w.worker(record,gate_fd)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--consumer",choices=("H060",))
 group=p.add_mutually_exclusive_group();group.add_argument("--admit-ref");group.add_argument("--execute-ref");a=p.parse_args(argv)
 if not(a.admit_ref or a.execute_ref):
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False,guard_imported=False)));return 0
 need(a.consumer is not None and Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,
  "Exact ROOT consumer/-B/-s/repository interpreter")
 result=execute_or_admit(a.consumer,json.loads(a.admit_ref or a.execute_ref),a.admit_ref is not None)
 print(json.dumps(result,sort_keys=True),flush=True);return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
