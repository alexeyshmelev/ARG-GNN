"""Fresh-scope fake integration reproduction and independent cap/role/race cases."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND="research/reviews/native_VAL000_full_state_resume_source_20261016_v11"
BASE="research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v11"
text=(ROOT/CAND/"fresh_scope_integration_inert_v11.py").read_text()
tree=ast.parse(text)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
run.body.insert(len(run.body)-1,ast.parse('globals()["_independent_VFS_locals"]=locals()').body[0]);ast.fix_missing_locations(tree)
ns={"__name__":"_independent_native11_VFS_reproduction"}
exec(compile(tree,"<sealed_author_VFS_code_only_local_capture>","exec"),ns)
result=ns["run"]();expected=json.loads((ROOT/CAND/"fresh_scope_integration_inert.final2.v11.json").read_text())
for value in (result,expected):value.pop("finished_utc",None)
assert result==expected and result["check_count"]==58
def save(name,value):
 p=ROOT/BASE/name
 with p.open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
repro=save("author_fresh58_reproduction.v11.json",result)
L=ns["_independent_VFS_locals"];c=L["c"];pp=L["pp"];g=L["g"];args=L["args"];checks=[]
def ok(name):checks.append(dict(name=name,status="PASS_SOFTWARE_ONLY"))
def rejects(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError("Expected rejection")
# Required roles and fresh inspection roles are retained with real binding helper code.
guardtext=(ROOT/c.FRESH_GUARD_REF["path"]).read_text()
node=next(n for n in ast.parse(guardtext).body if isinstance(n,ast.FunctionDef) and n.name=="consumer_bindings_v1")
bn=dict(copy=copy,require=ns["require"],_raw=lambda record:None)
exec(compile(ast.Module(body=[node],type_ignores=[]),"<real_native14_binding_body_fake_raw>","exec"),bn)
required=c.fresh_required_roles_v11(c.FRESH_STATE[0]["root_stage_request_ref"],c.FRESH_STATE[1],c.FRESH_STATE[0]["consumer_manifest_ref"])
view=dict(bindings=copy.deepcopy(L["current"]))
guard=types.SimpleNamespace(consumer_bindings_v1=bn["consumer_bindings_v1"])
base=pp.role_union_v11(c,guard,view,c.FRESH_STATE[0]["root_stage_request_ref"],c.FRESH_STATE[1],c.FRESH_STATE[0]["consumer_manifest_ref"])
assert len(base)==51
for i in range(13):
 r=dict(path="VIRTUAL/INDEPENDENT_extra_"+str(i)+".json",sha256="0"*64);view["bindings"]["extra"+str(i)]=r
assert len(pp.role_union_v11(c,guard,view,c.FRESH_STATE[0]["root_stage_request_ref"],c.FRESH_STATE[1],c.FRESH_STATE[0]["consumer_manifest_ref"]))==64
view["bindings"]["extra65"]=dict(path="VIRTUAL/INDEPENDENT_65.json",sha256="1"*64)
rejects(lambda:pp.role_union_v11(c,guard,view,c.FRESH_STATE[0]["root_stage_request_ref"],c.FRESH_STATE[1],c.FRESH_STATE[0]["consumer_manifest_ref"]))
ok("actual_native14_union51_all_returned_extra_through64_and65reject")
rosters=L["rosters"];rosters.clear();rosters.update(copy.deepcopy(L["baseline_rosters"]))
downspec=L["downspec"];rosters[downspec["output_directory"]]=[]
prospective=2*(2*2**30+4096)+256*1024
aux=17*2**28-prospective
rosters[downspec["worker_auxiliary_scope"]]=[(("cache",),aux,"regular")]
c.charged_census_v1(downspec)
rejects(lambda:c.charged_census_v1(downspec,required_write_bytes=1))
assert c.CAPS["reservation_bytes"]==5*2**30 and c.CAPS["auxiliary_bytes"]==768*2**20
ok("lower_selected_cap_exact_prospective_two_slots_plus_actual_AUX_boundary_and_next_byte_reject")
# Failed reservation is terminal and never creates a fake receipt/input.
original_apply=g.apply_reserve_v1;calls=[]
def race(*a,**kw):calls.append(1);raise ValueError("SYNTHETIC changed current prefix")
g.apply_reserve_v1=race;args.instance="INDEPENDENT_RACE";args.reservation_bytes=5*2**30
rejects(lambda:pp.prepare_v11(args))
assert len(calls)==1 and not any(k.endswith("INDEPENDENT_RACE/reserve_receipt.json") for k in L["files"])
g.apply_reserve_v1=original_apply
ok("actual_prepare_prefix_race_single_attempt_no_receipt_or_command")
original_inspect=g.inspect_v1
g.inspect_v1=lambda **kw:dict(projected_bytes=980000000000-5*2**30-2*1048576,bindings=copy.deepcopy(L["current"]))
count=len(L["dirs"]);args.instance="INDEPENDENT_NO_FORECAST_CREDIT"
rejects(lambda:pp.prepare_v11(args))
assert len(L["dirs"])==count
g.inspect_v1=original_inspect
ok("actual_current_projection_limit_rejects_before_packet_no_future_credit")
assert not any(n in sys.modules for n in ("torch","numpy","composer","torch_geometric"))
print(json.dumps(dict(schema="native11_fresh_scope_independent_VFS_checks_v1",status="PASS_SOFTWARE_ONLY_WITH_SEPARATE_CONFIRMED_MEMO_BLOCKER",
 author_result_ref=repro,author_checks=58,independent_check_groups=len(checks),checks=checks,
 real_guard_factories_not_exercised_by_VFS=True,actual_guards_payloads_resources_processes_accounting_admissions_or_jobs=False),sort_keys=True,indent=2))
