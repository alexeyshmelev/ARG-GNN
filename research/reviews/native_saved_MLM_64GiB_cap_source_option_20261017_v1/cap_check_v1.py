"""Pure AST and RAM boundary checks for an aggregate storage failstop option only."""
from pathlib import Path
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");D=Path(__file__).parent
OLD=ROOT/"research/reviews/native_VAL000_downstream_execution_source_20261016_v2"
checks=[];rejects=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(call,label):
 try:call()
 except ValueError:rejects.append(label);return
 raise AssertionError("Accepted invalid RAM cap fixture "+label)
old=(OLD/"contract_v1.py").read_text();new=(D/"contract_CAP64.PROPOSED.v1.py").read_text()
a=ast.parse(old);b=ast.parse(new);changed=[]
target=ast.dump(ast.parse("64*2**30",mode="eval").body,include_attributes=False)
class Normalize(ast.NodeTransformer):
 def visit_BinOp(self,node):
  if ast.dump(node,include_attributes=False)==target:
   changed.append(node.lineno);return ast.copy_location(ast.parse("65*2**30",mode="eval").body,node)
  return self.generic_visit(node)
 def visit_Constant(self,node):
  if isinstance(node.value,str):node.value=node.value.replace("64GiB","65GiB")
  return node
normalized=Normalize().visit(copy.deepcopy(b))
need(len(changed)==6,"Exactly six numeric65GiB-to64GiB whole-scope operands")
need(ast.dump(normalized,include_attributes=False)==ast.dump(a,include_attributes=False),"Entire original AST unchanged apart from six cap operands and descriptive cap text")
oldcfg=json.loads((OLD/"scientific_config.FROZEN.v1.json").read_text())
newcfg=json.loads((D/"scientific_config.CAP64.PROPOSED.v1.json").read_text())
expected=copy.deepcopy(oldcfg);expected["output_cap_bytes"]=64*2**30
need(newcfg==expected,"Only frozen config output_cap_bytes changes")
need(newcfg["scientific_protocol"]==oldcfg["scientific_protocol"],"Every author scientific setting identical")
need(newcfg["allowed_phases"]==oldcfg["allowed_phases"]and len(newcfg["allowed_phases"])==7,"All six scientific stages plus finalizer preserved")
need(newcfg["group_scope_caps"]==oldcfg["group_scope_caps"]and sum(newcfg["group_scope_caps"].values())==65*2**30,
 "Unchanged8+56+1GiB group upper bounds coexist with stricter64GiB aggregate")
def assignment(t,name):return next(n.value for n in t.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==name for x in n.targets))
oldcaps=eval(compile(ast.Expression(assignment(a,"BASE_CAPS")),"<old_BASE_CAPS>","eval"),{"dict":dict})
newcaps=eval(compile(ast.Expression(assignment(b,"BASE_CAPS")),"<new_BASE_CAPS>","eval"),{"dict":dict})
want=dict(oldcaps,reservation_bytes=64*2**30)
need(newcaps==want,"All nonstorage resource caps identical")
need(newcaps["max_cpu_rss_bytes"]==384*2**30 and newcaps["max_checkpoint_bytes"]==3*2**30 and newcaps["max_checkpoint_files"]==2,
 "RSS and checkpoint guards preserved")
def fn(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
def require(ok,message):
 if not ok:raise ValueError(message)
rows={"out":[],"aux":[]};master=dict(output_directory="out",worker_auxiliary_scope="aux",config_ref={"path":"RAMconfig","sha256":"a"*64},runtime_ref={"path":"RAMruntime","sha256":"b"*64})
storage={"RAM":"admitted"};actual={"job_id":"RAM/stage"}
ref=lambda p:dict(path=p,sha256=hashlib.sha256(p.encode()).hexdigest())
def sizes(path,**kw):return copy.deepcopy(rows[path])
def environment(tree):
 env=dict(STATE=(master,storage,{},actual),require=require,local=lambda p:p,
  original=types.SimpleNamespace(_sizes=sizes),active_amendment_v1=lambda:None,DRIVER="RAMdriver",ref=ref)
 exec(compile(ast.Module(body=[fn(tree,"charged_census_v1"),fn(tree,"MasterStorageGuard")],type_ignores=[]),"<actual_cap_functions_with_RAM_sizes_and_guard>", "exec"),env)
 return env
env=environment(b);oldenv=environment(a)
def run(e=env,phase="embedding_contrastive_train",n=0):return e["charged_census_v1"](dict(phase=phase),required_write_bytes=n)
row=lambda group,n:((group,"unopened.bin"),n,"regular")
rows["out"]=[];rows["aux"]=[]
need(run()=={"out":0,"aux":0},"Empty RAM scope accepts without resource calls")
rows["out"]=[row("contrastive",8*2**30),row("embedding_contrastive_train",55*2**30),row("RAM_retrieve_and_seal",2**30)]
need(sum(run().values())==64*2**30,"Exact64GiB aggregate accepted with unchanged group limits")
reject(lambda:run(n=1),"one_more_byte_exceeds64GiB_before_write")
rows["out"]=[row("contrastive",8*2**30),row("embedding_contrastive_train",56*2**30),row("RAM_retrieve_and_seal",2**30)]
need(sum(run(oldenv).values())==65*2**30,"Original65GiB baseline accepts full group ceiling sum")
reject(run,"aggregate65GiB_rejected_without_lowering_any_group_cap")
for group,cap,phase in[("contrastive",8*2**30,"contrastive"),("embedding_contrastive_train",56*2**30,"embedding_contrastive_train"),("RAM_retrieve_and_seal",2**30,"RAM_retrieve_and_seal")]:
 rows["out"]=[row(group,cap+1)];rows["aux"]=[]
 reject(lambda phase=phase:run(phase=phase),"original_group_cap_"+group+"_still_enforced")
rows["out"]=[row("contrastive",8*2**30),row("embedding_contrastive_train",52*2**30)]
rows["aux"]=[row("embedding_contrastive_train",4*2**30),row("RAM_retrieve_and_seal",2**30)]
need(sum(run(oldenv).values())==65*2**30,"RAM original guard counts output and auxiliary roots")
reject(run,"64GiB_aggregate_includes_all_auxiliary_bytes")
rows["aux"]=[]
rows["out"]=[(("contrastive","checkpoints",str(i)+".pt"),3*2**30+4096,"regular")for i in range(2)]
need(sum(run(phase="contrastive").values())==6*2**30+8192,"Two original3GiB transient checkpoint files preserved")
rows["out"]=[(("contrastive","checkpoints",str(i)+".pt"),2**30+4096,"regular")for i in range(3)]
reject(lambda:run(phase="contrastive"),"third_regular_checkpoint_still_rejected")
rows["out"]=[(("contrastive","checkpoints","too_big.pt"),3*2**30+4097,"regular")]
reject(lambda:run(phase="contrastive"),"single_checkpoint_over3GiB_still_rejected")
rows["out"]=[row("unknown_stage",1)]
reject(run,"unknown_scientific_stage_still_rejected")
calls=[]
class Guard:
 def validate_storage_v9(self,s,config,stored,**kw):calls.append(config);return "RAM_only"
 def budget_check_v9(self,*a,**kw):return "RAM_only"
obj=env["MasterStorageGuard"](Guard())
need(obj.validate_storage_v9(actual,{},storage,source_ref=ref("RAMdriver"),config_ref=master["config_ref"],runtime_ref=master["runtime_ref"])=="RAM_only"
 and calls==[dict(output_cap_bytes=64*2**30)],"Master guard forwards exact64GiB whole cap to unchanged real-guard interface")
for name,field in[("ROOT_master.INACTIVE.template.v1.json","future_storage_reserved_bytes"),("ROOT_reserve_request.INACTIVE.template.v1.json","reserved_bytes"),("ROOT_phase_authority.INACTIVE.template.v1.json","caps")]:
 candidate=json.loads((D/name.replace(".v1.json",".CAP64.PROPOSED.v1.json")).read_text())
 original=json.loads((OLD/name).read_text())
 candidate.pop("CAP64_OPTION_ONLY_NOT_OPERATIONAL_INTEGRATION")
 if field=="caps":original["caps"]["reservation_bytes"]=64*2**30
 else:original[field]=64*2**30
 need(candidate==original,"Inactive metadata changes only exact operational cap "+name)
need(951276853248-948133576555==3143276693,"Historical arithmetic example is consistent")
need(3143276693-268435456-524288==2874316949,"Optional v6 charge example remains explicitly hypothetical")
need(65*2**30-64*2**30==1073741824,"Cap option saves exactly1GiB independently of later projection")
print(json.dumps(dict(schema="native_saved_MLM64GiB_cap_option_AST_RAM_check_v1",status="PASS_SOURCE_OPTION_ONLY",
 positive_assertions=len(checks),checks=checks,negative_cases=len(rejects),negative_fixtures=rejects,
 six_changed_numeric_operand_lines=changed,whole_current_route_implementation_review=False,
 scientific_options_reduced=False,all_original_stage_group_and_checkpoint_caps_preserved=True,
 real_guard_import_or_state_or_scope_measurement_calls=0,process_GPU_payload_checkpoint_or_truth_reads=0,
 activation_admission_credit_launch_Git=0,measured_fit=False),indent=2,sort_keys=True))
