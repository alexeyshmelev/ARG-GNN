"""Independent narrow GST V2 source/config and saved-metadata preservation proof."""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=Path(__file__).resolve().parent
AUTHOR=ROOT/"research/reviews/gst_uspto_payload_copy_v2_author"
SOURCE=ROOT/"src/arggnn/gst_uspto_payload_copy_v2.py"
CONFIG=ROOT/"configs/gst_uspto_payload_copy_v2.json"
DOC=ROOT/"docs/gst_uspto_payload_copy_v2.md"
OLD_SOURCE=ROOT/"src/arggnn/gst_uspto_payload_copy_v1.py"
OLD_CONFIG=ROOT/"configs/gst_uspto_payload_copy_v1.json"
REG=ROOT/"research/reviews/gst_uspto_payload_copy_v2_registration_root.json"
CHECKS=[];HASHED={};NOTES=[]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):
 raw=path.read_bytes();item={"path":str(path.relative_to(ROOT)),"sha256":sha(raw)}
 HASHED[item["path"]]={**item,"bytes":len(raw)}
 return item
def ck(label,value):
 CHECKS.append({"check":label,"passed":bool(value)})
 if not value:raise AssertionError(label)
def load(path):return json.loads(path.read_bytes())
def verify(reference):
 ck("hash "+reference["path"],ref(ROOT/reference["path"])==reference)
pins={
 SOURCE:"ae0f621c7ca4c546dce93fb433d045b5ca2444b60b9afb9f328a877659c394ea",
 CONFIG:"301253ac507efad0a70d0024972c24b9f81d704e8713caf9f08c2fb467628855",
 DOC:"903944b0401fe69a8fac0afddf4e3043133469487f60f90783b7b1df744434f7",
 AUTHOR/"author_receipt.json":"480fdea1ca2458e1c1ca7722b009c9a400e8279c91dfafc9b197241fde04ce79",
 REG:"562a13ce65d530734fe991b900c4c9bb93b8c6efda3c0d97637bd8b87cdb4d4f",
 OLD_SOURCE:"01a995a0c4cbe413d91d2af8820dff79ec9efddd7208ca326f955d2ae31ddf02",
 OLD_CONFIG:"9c89b25bf07bb4ed049a0884e318215c1c4cea93cd4dc76df9f5ec1a448afd97"}
for path,pin in pins.items():ck("exact assigned pin "+str(path.relative_to(ROOT)),ref(path)["sha256"]==pin)
newtext=SOURCE.read_text();oldtext=OLD_SOURCE.read_text();newtree=ast.parse(newtext);oldtree=ast.parse(oldtext)
newfuncs={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
oldfuncs={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
newclass=next(n for n in newtree.body if isinstance(n,ast.ClassDef) and n.name=="CopyScopeV2")
oldclass=next(n for n in oldtree.body if isinstance(n,ast.ClassDef) and n.name=="CopyScopeV1")
newmethods={n.name:n for n in newclass.body if isinstance(n,ast.FunctionDef)}
oldmethods={n.name:n for n in oldclass.body if isinstance(n,ast.FunctionDef)}
same_functions=("_require","_same","_active","_inventory_matches","_output_fd","_member_chunks")
same_methods=("stable","directories","finish","close")
ast_ref=lambda node:sha(ast.dump(node,include_attributes=False).encode())
preserved=[]
for name in same_functions:
 ck("unchanged AST function "+name,ast.dump(newfuncs[name],include_attributes=False)==ast.dump(oldfuncs[name],include_attributes=False))
 preserved.append({"function":name,"AST_sha256":ast_ref(newfuncs[name])})
for name in same_methods:
 ck("unchanged AST method "+name,ast.dump(newmethods[name],include_attributes=False)==ast.dump(oldmethods[name],include_attributes=False))
 preserved.append({"method":name,"AST_sha256":ast_ref(newmethods[name])})
normalize=lambda node:ast.parse(ast.get_source_segment(newtext,node).replace("_v2","_v1").replace("ScopeV2","ScopeV1")).body[0]
admit=normalize(newfuncs["admit_v2"])
ck("exact normalized root admission AST preserved",ast.dump(admit,include_attributes=False)==ast.dump(oldfuncs["admit_v1"],include_attributes=False))
# Remove only the newly added sticky-denial gate to compare the original write semantics.
def sticky_expr(node):
 return (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
  and node.value.func.id=="_require" and node.value.args and isinstance(node.value.args[0],ast.UnaryOp)
  and isinstance(node.value.args[0].operand,ast.Attribute) and node.value.args[0].operand.attr=="_guard_rejected")
for name in ("file","write"):
 node=deepcopy(newmethods[name])
 ck("one leading sticky gate "+name,sticky_expr(node.body[0]))
 node.body.pop(0)
 ck("normalized original "+name+" semantics unchanged",ast.dump(node,include_attributes=False)==ast.dump(oldmethods[name],include_attributes=False))
node=deepcopy(newmethods["check"]);ck("leading sticky gate check",sticky_expr(node.body[0]));node.body.pop(0)
ck("check only wraps original guard call",isinstance(node.body[-1],ast.Try))
node.body[-1:]=node.body[-1].body
ck("normalized original check semantics unchanged",ast.dump(node,include_attributes=False)==ast.dump(oldmethods["check"],include_attributes=False))
node=deepcopy(newmethods["event"])
idx=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="check")
del node.body[idx]
ck("event only adds guarded check before original write",ast.dump(node,include_attributes=False)==ast.dump(oldmethods["event"],include_attributes=False))
node=deepcopy(newmethods["__init__"])
node.body=[n for n in node.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=="_guard_rejected" for t in n.targets))]
ck("constructor only adds sticky state",ast.dump(node,include_attributes=False)==ast.dump(oldmethods["__init__"],include_attributes=False))
node=normalize(newfuncs["copy_payload_v2"])
node.body=[n for n in node.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=="result" and isinstance(t.slice,ast.Constant) and t.slice.value=="authority_rejected" for t in n.targets))]
ck("copy primitive only version identities and authority diagnostic changed",ast.dump(node,include_attributes=False)==ast.dump(oldfuncs["copy_payload_v1"],include_attributes=False))
newimports=[ast.dump(n,include_attributes=False) for n in newtree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
oldimports=[ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
ck("no new parser scientific model or GPU imports",newimports==oldimports)
cfg=load(CONFIG);oldcfg=load(OLD_CONFIG)
for key,value in oldcfg.items():
 if key!="kind":ck("original config field unchanged "+key,cfg[key]==value)
ck("exact new config fields predecessor and failure policy",set(cfg)-set(oldcfg)=={"predecessor","failure_write_policy"})
ck("only versioned config kind changed",cfg["kind"]=="gst_uspto_payload_copy_v2_policy" and oldcfg["kind"]=="gst_uspto_payload_copy_v1_policy")
for key in ("actual_archive_payload_open_authorized","actual_rows_parse_authorized","native_model_run_authorized","GPU_authorized","execution_enabled","archive_permissions_restored"):
 ck("source-only authorization flag "+key,cfg[key] is False)
ck("exact unchanged raw ZIP identity",cfg["raw"]==oldcfg["raw"])
# Do not open cfg.raw. Only saved inventory metadata is hashed and parsed.
for key in ("budget_source","budget_config","inventory_source","inventory_config","inventory","native_protocol"):
 verify(cfg[key])
for reference in cfg["predecessor"].values():verify(reference)
frozen=load(ROOT/cfg["inventory"]["path"])
ck("saved inventory complete metadata only",frozen["status"]=="metadata_only_complete")
ck("saved raw identity bound without archive read",frozen["archive_sha256"]==cfg["raw"]["sha256"] and frozen["archive_bytes"]==cfg["raw"]["bytes"])
ck("frozen five names and order retained",[x["name"] for x in frozen["members"]]==cfg["frozen_member_order"])
ck("frozen unchanged payload total",frozen["member_count"]==5 and frozen["declared_total_uncompressed_bytes"]==12710870==cfg["payload_bytes"])
ck("receipt progress block and reservation caps unchanged",(cfg["receipt_max_bytes"],cfg["progress_max_bytes"],cfg["stream_block_bytes"],cfg["reservation_bytes"])==(1000000,256000,65536,16000000))
prior_receipt=load(ROOT/cfg["predecessor"]["independent_REJECT"]["path"])
ck("V1 REJECT seal preserved",prior_receipt["verdict"]=="REJECT")
receipt=load(AUTHOR/"author_receipt.json")
for key in ("implementation","config","documentation","authority_checks","handmade_checks","source_delta_proof","root_source_only_registration"):verify(receipt[key])
ck("current author receipt exact source config docs",receipt["implementation"]==ref(SOURCE) and receipt["config"]==ref(CONFIG) and receipt["documentation"]==ref(DOC))
ck("author source registration current bound",receipt["root_source_only_registration"]==ref(REG) and receipt["root_source_only_registration_pending"] is False)
reg=load(REG)
ck("root registration H022 source handmade scope",reg["hypothesis_id"]=="H022" and reg["registration"]["id"]=="gst_uspto_payload_copy_v2_source" and reg["registration"]["scope"]=="source_handmade_ZIP_only_no_actual_payload_parse_TEST_scoring_model_GPU")
ck("root source registration no actual payload/model authority",reg["actual_archive_payload_parse_model_GPU_authorized"] is False)
ck("root registration frozen inventory preserved",reg["inventory"]==cfg["inventory"])
ck("author94 checks exact counts",receipt["checks"]=={"denial_and_caller_diagnostic_regressions":32,"preserved_V1_byte_semantics":62,"total":94})
# The author manifest is an earlier snapshot; admission does not consume it.
manifest=load(AUTHOR/"artifact_manifest.json")
prior_hash=manifest["files"]["author_receipt.json"]["sha256"]
ck("manifest receipt hash equals disclosed pre-registration receipt",prior_hash==receipt["prior_author_receipt_sha256_before_registration_binding"])
ck("manifest old receipt hash differs current bound receipt",prior_hash!=ref(AUTHOR/"author_receipt.json")["sha256"])
NOTES.append({"id":"N1","severity":"informational","file":str((AUTHOR/"artifact_manifest.json").relative_to(ROOT)),
 "body":"The manifest records the disclosed pre-registration author-receipt hash. The current assigned author receipt and root registration are independently hash-verified here. The manifest is not consumed by admit_v2; this preserved historical entry is not an execution binding."})
# Byte-identical author harnesses were replayed in owned folders.
for orig,copy in (("handmade_checks.py","replay_handmade/handmade_checks.py"),("authority_checks.py","replay_authority/authority_checks.py")):
 ck("replayed author source bytes unchanged "+orig,(AUTHOR/orig).read_bytes()==(OUT/copy).read_bytes())
 ref(AUTHOR/orig)
result={"kind":"gst_payload_copy_v2_independent_source_metadata_preservation","verdict":"PASS",
 "implementation":ref(SOURCE),"config":ref(CONFIG),"documentation":ref(DOC),
 "author_receipt":ref(AUTHOR/"author_receipt.json"),"root_source_only_registration":ref(REG),
 "check_count":len(CHECKS),"checks":CHECKS,"reference_hashes_verified":len(HASHED),"verified_references":list(HASHED.values()),
 "unchanged_AST_records":preserved,"normalized_root_admission_and_byte_copy_core_preserved":True,
 "native_inventory_CRC_SHA_names_order_caps_raw_identity_and_parser_model_routes_unchanged":True,
 "notes":NOTES,"actual_archive_raw_payload_runtime_model_GPU_budget_proc_launch_or_Git_actions":False}
with (OUT/"source_metadata_checks.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
print(json.dumps({"verdict":"PASS","checks":len(CHECKS),"reference_hashes_verified":len(HASHED),"preserved_AST_records":len(preserved),"informational_notes":len(NOTES)},indent=2))
