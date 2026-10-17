"""Reproduce sealed actual metadata routes and add independent invalid-record challenges.
Uses real inert candidate/original modules; only future records are RAM doubles.
"""
from pathlib import Path
import hashlib
REL="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
p=Path(REL)/"actual_metadata_route_checks_v1.py"
assert hashlib.sha256(p.read_bytes()).hexdigest()=="cc627341e71ae5e17b56f19567a4faed236b85b73fb2dda028e0eb604d3ef852"
s=p.read_text()
s=s.replace("c,d=w.initialize();real_cread=c.read;real_cref=c.ref","c,d=w.initialize();c.cpu_slot=forbidden;d.execute=forbidden;d.admit=forbidden;d.close=forbidden;real_cread=c.read;real_cref=c.ref")
helper="""
independent_challenges=[]
def must_reject(call,label,message):
    before_guards=(w._guard,w._memo)
    try:call()
    except ValueError as error:
        check(message in str(error),"Independent rejection at expected boundary: "+label)
        independent_challenges.append(label)
    else:raise AssertionError("Accepted invalid independent metadata challenge: "+label)
    check((w._guard,w._memo)==before_guards,"Independent invalid metadata leaves guards absent: "+label)
def predecessor_challenge(label,mutate):
    saved_ram=copy.deepcopy(ram)
    try:
        target=ram[cr["path"]]
        inp=target["metadata"]["input_ref"]
        req=ram[inp["root_stage_request_ref"]["path"]]
        mutate(target,inp,req)
        ram[target["refs"]["input_ref"]["path"]]=copy.deepcopy(inp)
        must_reject(lambda:w.predecessor_capsule(cr,manifest["condition"]),label+parent+"/"+stage,
            "Exact complete immutable original predecessor manifest/source closure")
    finally:ram.clear();ram.update(saved_ram)
def future_request_challenge(label,mutate,message):
    backup=copy.deepcopy(ram[qr["path"]])
    try:
        mutate(ram[qr["path"]])
        must_reject(lambda:w.validate_request(qr),label+parent+"/"+stage,message)
    finally:ram[qr["path"]]=backup
def missing_prior_map(target,inp,req):
    source=dict(inp["source_sha256"]);source.pop(next(iter(source)))
    inp["source_sha256"]=source;req["source_sha256"]=dict(source)
"""
s=s.replace("checks=[]","checks=[]\n"+helper,1)
after_request="""
    predecessor_challenge("Self-consistent incomplete predecessor source map ",missing_prior_map)
    predecessor_challenge("Wrong predecessor manifest despite complete map ",lambda target,inp,req:req.update(source_manifest_ref=rr("ram/wrong_manifest")))
    future_request_challenge("Missing current source binding ",lambda x:x["source_sha256"].pop(next(iter(x["source_sha256"]))),"Complete source closure")
    future_request_challenge("GPU authorization true ",lambda x:x.update(GPU_authorized=True),"Exact finite ROOT original TRAIN")
    future_request_challenge("Non-qualified stage runtime ",lambda x:x.update(runtime_ref=manifest["condition"]["stage_runtime_refs"]["native_paths"if stage=="infer"else"infer"]),"Exact qualified stage runtime")
    future_request_challenge("Future actual capacity ",lambda x:x["actual_capacity"].update(created_at_utc=(now+timedelta(seconds=60)).isoformat()),"Actual current ROOT authority")
    future_request_challenge("Authority expiry exceeded ",lambda x:x.update(expires_at_utc=(end+timedelta(seconds=1)).isoformat()),"Finite parent authority")
"""
# Original c.active diagnostic is read from the actual original contract below; match semantic check text.
from pathlib import Path as _Path
contract=(_Path("research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1/root_contract_v1.py")).read_text()
import ast as _ast
active=_ast.unparse(next(n for n in _ast.parse(contract).body if isinstance(n,_ast.FunctionDef)and n.name=="active"))
# This is diagnostic adaptation for an exact existing boundary, not a substitute validator.
if "'Fresh"not in active and '"Fresh'not in active:
    after_request=after_request.replace('),"Fresh")', '),"Finite")')
s=s.replace('check(got==q and gotcaps==caps,"Actual whole request validator "+parent+"/"+stage)',
    'check(got==q and gotcaps==caps,"Actual whole request validator "+parent+"/"+stage)\n'+after_request)
after_input="""
    saved_ram=copy.deepcopy(ram)
    try:
        broken=copy.deepcopy(storage)
        broken["reservation"]["consumer_bindings"].remove(roles[-1])
        ram[storeref["path"]]=broken;ram[ir["path"]]["storage"]=broken
        must_reject(lambda:d.input_spec(ir),"Missing admitted role "+parent+"/"+stage,"Every actual direct source and predecessor metadata role")
    finally:ram.clear();ram.update(saved_ram)
"""
s=s.replace('check(ss==s and qq==q and cc==cfg,"Actual originaldriver→patchedcontract input route "+parent+"/"+stage)',
 'check(ss==s and qq==q and cc==cfg,"Actual originaldriver→patchedcontract input route "+parent+"/"+stage)\n'+after_input)
after_close="""
    for label,record,field,value,message in (
        ("Supervisor boolfalse OS status",sw,"actual_OS_wait_status",False,"Separate genuine direct supervisorOSwait0"),
        ("Outer UI terminal claimed",outerref,"UI_tool_terminal_claimed",True,"Genuine batch ROOT direct supervisorOSwait0"),
        ("Worker boolfalse OS status",wr,"actual_OS_wait_status",False,"Genuine workerOSwait0"),
        ("Worker incomplete drain",wr,"all_owned_processes_and_groups_drained",False,"Genuine workerOSwait0")):
        backup=copy.deepcopy(ram[record["path"]]);ram[record["path"]][field]=value
        try:must_reject(lambda:k.metadata_join(p),label+parent+"/"+stage,message)
        finally:ram[record["path"]]=backup
"""
s=s.replace('check(joined[-2:]==([1001,1002],[1001,1002]),"Actual ordinaryclose complete metadata/PID join "+parent+"/"+stage)',
 'check(joined[-2:]==([1001,1002],[1001,1002]),"Actual ordinaryclose complete metadata/PID join "+parent+"/"+stage)\n'+after_close)
s=s.replace('result=dict(schema="TreeLoom_actual_source_metadata_routes_v1"',
 'result=dict(independent_challenges=independent_challenges,independent_challenge_count=len(independent_challenges),reproduced_original_author_route_checks=39,schema="TreeLoom_independent_actual_source_metadata_routes_v1"')
exec(compile(s,str(p)+"#independent-challenges","exec"),{"__name__":"__independent_actual_metadata__"})
