"""Focused mocked v11 namespace checks. No cleanup, guard or native execution."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types,unittest
from unittest.mock import Mock,patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_native_MLM_amendment_v11"
sys.path[:0]=[str(HERE),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
 str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
import amendment_v11 as a
import root_namespace_v11 as n
import root_native_contract_v1 as c
class Checks(unittest.TestCase):
 def test_known_roots_only_and_fail_closed(self):
  calls=[]
  def gone(pid,sig):calls.append((pid,sig));raise ProcessLookupError
  result=n.known_roots_groups_absent(kill=gone,killpg=gone)
  self.assertEqual(calls,[(p,0) for p in (*n.KNOWN_ROOTS,*n.KNOWN_GROUPS)])
  self.assertFalse(result["complete_old_writer_absence_claimed"])
  self.assertFalse(result["historical_ids_complete"])
  self.assertEqual(result["actual_optimizer_updates"],"UNKNOWN")
  with self.assertRaises(ValueError):n.known_roots_groups_absent(kill=lambda *x:None,killpg=gone)
  def denied(*x):raise PermissionError
  with self.assertRaises(PermissionError):n.known_roots_groups_absent(kill=denied,killpg=gone)
 def test_kernel_match_is_exact(self):
  text="header\n000: 2 0 10000 1 1 10 /tmp/unrelated\n001: 2 0 10000 1 1 11 /tmp/target\n"
  self.assertEqual(n.listeners(["/tmp/target"],read_text=text),["/tmp/target"])
  self.assertEqual(n.listeners(["/tmp/targe"],read_text=text),[])
 def test_exact_original_eight_metadata_pins(self):
  rows=n.original_socket_rows()
  self.assertEqual(len(rows),8)
  self.assertTrue(all(n.a.PARENT_TMP in Path(row["path"]).parents for row in rows))
  self.assertEqual([a.ref(x["path"]) for x in n.PRESERVED_REFS],n.PRESERVED_REFS)
 def test_writer_keeps_original_parent_storage(self):
  storage={"reserved_new_bytes":a.CAP}
  cls=type("Writer",(),{})
  guard=types.SimpleNamespace(PilotOutputV9=cls,budget_check_v9=Mock())
  path=Mock();path.parent=ROOT/a.PARENT_OUTPUT;path.exists.return_value=False
  reader=types.SimpleNamespace(root=ROOT)
  with patch.object(a,"read",return_value=storage),patch.object(a,"local",return_value=path),patch.object(a,"check_lifetime"):
   writer=a.child_writer(guard,a.OUTPUT,storage,reader)
   self.assertIs(writer.storage,storage)
   guard.budget_check_v9.assert_called_once_with(storage,boundary=True,required_write_bytes=8192)
   path.mkdir.assert_called_once_with()
   with self.assertRaises(ValueError):a.child_writer(guard,a.PARENT_OUTPUT,storage,reader)
   path.exists.return_value=True
   with self.assertRaises(ValueError):a.child_writer(guard,a.OUTPUT,storage,reader)
 def test_original_science_and_lifecycle_retained(self):
  original=a.read(a.ORIGINAL_INPUT_REF);req=a.read(a.ORIGINAL_REQUEST_REF);caps=a.read(a.ORIGINAL_CAPS_REF)
  self.assertEqual(c.make_config(req,caps),a.read(a.ORIGINAL_CONFIG_REF))
  self.assertEqual(original["expires_at_utc"],a.EXPIRES)
  self.assertEqual(original["refs"]["storage"],a.ORIGINAL_STORAGE_REF)
  old=ROOT/"research/reviews/American_C3_native_MLM_amendment_v10"
  def node(folder,file,name):
   source=(folder/file).read_text()
   item=next(x for x in ast.parse(source).body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==name)
   return ast.dump(item,include_attributes=False)
  for file,name in [("native_child_v1.py","_io_config"),("owned_lifecycle_v2.py","OwnedTree"),
      ("native_child_v1.py","_final_checkpoint"),("native_child_v1.py","optimizer_counter_evidence"),("terminal_capture_v10.py","WaitRelay")]:
   actual=file.replace("_v10","_v11")
   self.assertEqual(node(old,file,name),node(HERE,actual,name))
  self.assertLessEqual(len(__import__("os").fsencode(a.TMP/"mlm"/"pymp-12345678/listener-12345678")),107)
 def test_parent_envelope_requires_actual_full_counter_receipt(self):
  storage={"original":True};child={"path":a.OUTPUT+"/completed.json","sha256":"dummy"}
  value=dict(status="completed_C3_native_full_MLM_v1",actual_optimizer_updates=224775,completed_epochs=5,observed_counters_verified=True)
  guard=types.SimpleNamespace(budget_check_v9=Mock())
  def read(ref):return storage if ref==a.ORIGINAL_STORAGE_REF else value
  with patch.object(a,"read",side_effect=read),patch.object(a,"save",return_value={"created":"exclusive"}) as save:
   result=a.publish_parent_completion(guard,storage,child)
   self.assertEqual(result,{"created":"exclusive"})
   self.assertEqual(save.call_args[0][0],ROOT/a.PARENT_OUTPUT/"completed.json")
   self.assertEqual(save.call_args[0][1]["namespace_child_completion_ref"],child)
   self.assertIs(guard.budget_check_v9.call_args[0][0],storage)
   value["actual_optimizer_updates"]=3
   with self.assertRaises(ValueError):a.publish_parent_completion(guard,storage,child)
 def test_exact_cleanup_flow_in_memory_and_rechecks(self):
  rows=[dict(path=str(a.PARENT_TMP/("pymp-x%d"%i)/("listener-y%d"%i)),lstat=[67,i+1,49645,1,0,1,1]) for i in range(8)]
  source_ref={"path":n.SOURCE,"sha256":"mock"}
  directive_ref={"path":"directive","sha256":"mock"}
  authority_ref={"path":"authority","sha256":"mock"}
  closure_ref={"path":"closure","sha256":"mock"}
  auth=dict(schema="native_MLM_v11_exact_namespace_cleanup_authority",root_authorizer="root_agent",active=True,
    root_directive_ref=directive_ref,helper_ref=source_ref,original_admin_storage_ref=n.ADMIN_STORAGE_REF,
    preserved_metadata_refs=n.PRESERVED_REFS,no_new_reservation=True,all_prior_capabilities_revoked=True,source_closure_ref=closure_ref)
  closure=dict(schema="native_MLM_fullfit_execution_source_closure_v11",original_storage_ref=a.ORIGINAL_STORAGE_REF,source_sha256={n.SOURCE:"mock"})
  class Node:
   def __init__(self,row):self.row=row
   def lstat(self):return tuple(self.row["lstat"])
   def __str__(self):return self.row["path"]
   def unlink(self):events.append(("unlink",self.row["path"]))
  nodes={row["path"]:Node(row) for row in rows};events=[];saves=[]
  archive=types.SimpleNamespace(iterdir=lambda:[types.SimpleNamespace(name=Path(x["path"]).name) for x in n.PRESERVED_REFS])
  attempt=Mock();attempt.exists.return_value=False
  m=types.SimpleNamespace(budget_check_v9=Mock())
  def read(ref):
   if ref==authority_ref:return auth
   if ref==closure_ref:return closure
   return {"storage":"original"}
  def absent():events.append(("known",None));return {"known":"absent"}
  def save(m,storage,name,value):saves.append((name,value));return {"path":name,"sha256":"mock"}
  with patch.object(a,"read",side_effect=read),patch.object(a,"ref",return_value=source_ref),patch.object(a,"local",side_effect=lambda p:nodes[p]),\
      patch.object(n,"validate_directive"),patch.object(n,"original_socket_rows",return_value=rows),\
      patch.object(n,"socket_ancestors"),patch.object(n,"listeners",return_value=[]),patch.object(n,"known_roots_groups_absent",side_effect=absent),\
      patch.object(n,"stamp",side_effect=lambda x:list(x)),patch.object(n,"admin_guard",return_value=(m,{"admin":"original"})),\
      patch.object(n,"admin_save",side_effect=save),patch.object(n,"ARCHIVE",archive),patch.object(n,"ATTEMPT",attempt):
   n.cleanup(authority_ref)
  self.assertEqual([e[1] for e in events if e[0]=="unlink"],[row["path"] for row in rows])
  self.assertEqual(len([e for e in events if e[0]=="known"]),2+2*8)
  self.assertEqual([x[0] for x in saves],["before_cleanup.json"]+["before_unlink_%02d.json"%i for i in range(8)]+["cleanup_receipt.json"])
  self.assertFalse(saves[-1][1]["new_reservation_created"])
  self.assertFalse(saves[-1][1]["global_unrelated_metadata_census_used"])
if __name__=="__main__":
 result=unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 print(json.dumps(dict(schema="native_MLM_v11_focused_source_checks",tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
  actual_cleanup=False,actual_native_execution=False,scientific_payload_opened=False)),flush=True)
 raise SystemExit(0 if result.wasSuccessful() else 1)
