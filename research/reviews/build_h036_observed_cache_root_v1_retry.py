"""Build one observation-only H036 cache under a root-admitted 500 MB scope."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util, json, sys, traceback, time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v9 as b
from arggnn.census_tsinfer_fit_v1 import ConsumerReaderV1
from arggnn.mutation_routed_observation_efficiency_v1 import prepare_parents_v1
from arggnn.mutation_routed_observation_cache_v1 import write_observation_cache_v1
from arggnn.project_storage_full_reservation_v1 import budget_check_v9
P=ROOT/"research/reviews/mutation_routed_observation_cache_v1_root_retry"
OUT="data/mutation_routed_observation_cache_v1"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP=500000000
def ref(p):return b.v4.reference(ROOT,p)
def read(p):return json.loads((ROOT/p).read_text())
def save(name,obj):
 p=P/name
 with p.open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write(chr(10))
 return ref(p)
def run():
 start=time.monotonic()
 P.mkdir(exist_ok=False)
 old=read("research/reviews/mutation_routed_whole_tree_gnn_continue_v1_root/aligned_GPU2/input.json")
 sources={k:ref(p) for k,p in dict(
  model="src/arggnn/mutation_routed_whole_tree_gnn_v1.py",
  preprocessing="src/arggnn/mutation_routed_tree_observations_v1.py",
  loader="src/arggnn/mutation_routed_parent_loader_v1.py",
  efficiency="src/arggnn/mutation_routed_observation_efficiency_v1.py",
  cache="src/arggnn/mutation_routed_observation_cache_v1.py").items()}
 assert sources["cache"]["sha256"]=="f4adf6832b6d80bdf3233083e0aacc87ae27818314ca185021a9f4c8b0e97150"
 save("registration.json",dict(hypothesis_id="H036",condition="observation_cache_for_PyG_optimization_continuation",
  linked_prior=old["refs"]["registration"],targets_read=False,GPU_used=False,
  scientifically_identical_observed_inputs=True,source_refs=sources,cap_bytes=CAP,
  all_new_model_experiments_require="PyTorch_and_PyTorch_Geometric"))
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE,BASE)
  before=b.state(ROOT,c,bindings,verify_closed=False)[4]
 authority=read("research/reviews/interleaved_tree_tv_gnn_v1_root/budget_admission.json")
 authority["expected_before"]=b.pins(before)
 authority["reservation"]=dict(key="H036/observed_PyG_cache_v1",output=str(ROOT/OUT),
  auxiliary_scopes=[],reserved_bytes=CAP,consumer_bindings=[*sources.values(),ref("src/arggnn/project_storage_full_reservation_v1.py"),ref("src/arggnn/project_budget_v9_consumer.py")])
 ar=save("budget_admission.json",authority)
 spec=importlib.util.spec_from_file_location("root_cache_admit",ROOT/"research/reviews/full_reservation_admission_v1_root/admission.py")
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 result=mod.apply(ROOT,ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
 save("budget_receipt.json",result)
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE,BASE)
  current=b.state(ROOT,c,bindings,verify_closed=False)[4]
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
  accounting_mode="full_reserved_no_closed_scope_credit",job_id="mutation-routed-observation-cache-v1",
  output_directory=OUT,reserved_new_bytes=CAP,reservation=result["value"],
  expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=23)).isoformat(),
  initial_ledger_pins=b.pins(current),
  budget_components=dict(implementation=bindings["implementation"],launcher=bindings["launcher"],
   consumer_adapter=ref("src/arggnn/project_storage_full_reservation_v1.py")))
 save("storage.json",storage)
 budget_check_v9(storage,first=True)
 directory=ROOT/OUT;directory.mkdir(exist_ok=False)
 scratch=directory/"observation_tmp"
 reader=ConsumerReaderV1()
 bundle=prepare_parents_v1(old["observed_parents"],reader,scratch_directory=scratch)
 def guard(n):
  if n>CAP:raise ValueError("Exact compressed cache exceeds its 500 MB admission")
  budget_check_v9(storage,boundary=True,required_write_bytes=n)
  print(json.dumps(dict(stage="cache_prepared",compressed_bytes=n,elapsed_seconds=time.monotonic()-start)),flush=True)
  return True
 manifest=write_observation_cache_v1(bundle,directory,observed_parent_refs=old["observed_parents"],
  source_refs=sources,guard_exact_bytes=guard)
 done=dict(status="completed_observation_only_cache",hypothesis_id="H036",manifest_ref=manifest,
  source_refs=sources,observed_parents=old["observed_parents"],targets_read=False,GPU_used=False,
  elapsed_seconds=time.monotonic()-start,root_storage=ref(P/"storage.json"))
 save("completed.json",done)
 print(json.dumps(done),flush=True)
if __name__=="__main__":
 try:run()
 except Exception:
  traceback.print_exc(file=sys.stdout);raise SystemExit(1)

