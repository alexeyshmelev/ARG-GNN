"""Independent tiny CPU FP32 scanner and fake-I/O seal compatibility checks."""
from pathlib import Path
import os,sys,ast,copy,json,hashlib,importlib,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
CAND="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1"
REVIEW="research/reviews/C3_CPU_six_candidate_interleave_independent_review_20261016_v1"
m=importlib.import_module(CAND.replace("/",".")+".synthetic_fp32_v1")
scan=importlib.import_module(CAND.replace("/",".")+".six_candidate_scan_v1")
bridge=importlib.import_module(CAND.replace("/",".")+".prediction_bridge_v1")
m.HERE=ROOT/REVIEW
if not (m.HERE/"synthetic_result.v1.json").exists():m.main()
else:
 m.torch.set_num_threads(1);m.torch.set_num_interop_threads(1)
author=json.loads((ROOT/REVIEW/"synthetic_result.v1.json").read_text())
expected=json.loads((ROOT/CAND/"synthetic_result.v1.json").read_text())
for value in (author,expected):value.pop("elapsed_seconds",None)
assert author==expected
checks=[]
def ok(name,detail=None):checks.append(dict(name=name,result="PASS_TINY_SYNTHETIC_ONLY",**({"detail":detail} if detail is not None else {})))
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError("Expected independent rejection")
ok("reproduced_two_author_actual_FP32_cases_original_receipts_and_successful_CPU_RNG")
extra=[m.run_case([(0,3),(3,5)],8),m.run_case([(0,1),(1,2),(2,3),(3,4),(4,5)],4)]
assert max(x["max_tensor_abs_difference"] for x in extra)==0.
ok("independent_two_and_five_block_partitions_exact_serial_FP32_match",extra)
torch=m.torch;np=m.np
def models():
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(1837)
  return [m.AmericanC3WholeTreeGNNV1(hidden_dim=4,expected_haplotypes=6,site_chunk_size=2).eval() for _ in range(6)]
for mode in ("plan_mismatch","class_order","custom_generator","FP64","missing_focal_support"):
 ms=models();count=dict(forward=0,reverse=0);a,b=m.loaders([(0,2),(2,3),(3,5)],count)
 records=copy.deepcopy(m.RECORDS)
 if mode=="plan_mismatch":b.dataset.plan_ref=dict(path="synthetic_other_plan.json",sha256="0"*64)
 elif mode=="class_order":records["class_order"]=np.asarray(["AFR","EUR","EAS"])
 elif mode=="custom_generator":b.generator=torch.Generator()
 elif mode=="FP64":ms[0].double()
 else:
  records={k:(np.concatenate((v,v[:1])) if getattr(v,"ndim",0)==1 and k not in ("class_order",) else v) for k,v in records.items()}
 reject(lambda:scan.interleaved_six_focal_predictions_v1(ms,a,b,records=records,runtime_guard=lambda e:None))
ok("independent_plan_class_generator_dtype_and_missing_support_negatives")
# Real zero-worker drain function calls are traced on a mid-reverse deadline error.
calls=[];original_drain=scan.shutdown_dataloader
def traced(loader):calls.append(loader.dataset.direction);return original_drain(loader)
scan.shutdown_dataloader=traced
ms=models();counts=dict(forward=0,reverse=0);a,b=m.loaders([(0,2),(2,3),(3,5)],counts)
def guard(event):
 if b.dataset.counts["reverse"]>=2 and event=="before_original_candidate_block":raise ValueError("SYNTHETIC reverse deadline")
reject(lambda:scan.interleaved_six_focal_predictions_v1(ms,a,b,records=m.RECORDS,runtime_guard=guard))
scan.shutdown_dataloader=original_drain
assert calls[-2:]==["forward","reverse"]
ok("mid_reverse_deadline_drains_both_zero_worker_loaders")

# Compare actual original seal function bodies to the new bridge with all
# checkpoint/registry/loader/writer endpoints explicitly synthetic in RAM.
from arggnn import american_c3_gnn_predict_isolated_v2 as oldseal
from arggnn import american_c3_gnn_shared30mb_isolated_v2 as oldshared
VAL=oldshared.VAL;parent=VAL[0]
def r(label):return dict(path="SYNTHETIC_ONLY/"+label,sha256=hashlib.sha256(label.encode()).hexdigest())
training_ref=r("training.json");plan_ref=r("plan.json")
training=dict(checkpoint_candidates=[dict(epoch=e,global_step=e*20,checkpoint_ref=r("epoch"+str(e)+".pt")) for e in (5,10,15,20,25,30)],
 trained_parent_ids=["synthetic_train_"+str(i) for i in range(20)],train_sequence_lengths_bp=[2000000]*20,protocol_sha256="a"*64)
plan=dict(parent_id=parent,haplotypes=1600,sequence_length_bp=30000000,trees=5,class_order=["EUR","EAS","AFR"],
 focal_registry_ref=r("registry.npz"),observation_ref=r("observed.json"),config={})
records=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=(np.arange(10000)%1000+600).astype(np.int64),
 sample_indices=(np.arange(10000)%1000+600).astype(np.int64),positions_bp0=np.arange(10000,dtype=np.float32)*2999,
 parent_id=np.asarray(parent),class_order=np.asarray(["EUR","EAS","AFR"]))
fullmodels=bridge.new_candidate_models_v1()
class Dataset:
 def __init__(self,record,direction,worker_tmp_directory):self.plan_ref=record;self.plan=plan;self.direction=direction;self._state=None;self._pid=None
class Loader:
 def __init__(self,dataset):self.dataset=dataset;self._iterator=None
def loader(dataset,**kw):assert kw==dict(num_workers=0,pin_memory=False);return Loader(dataset)
def checked(record,ref):
 assert ref==training_ref
 candidate=next(row for row in training["checkpoint_candidates"] if row["checkpoint_ref"]==record)
 return training,dict(candidate)
def prediction(k):
 value={key:records[key] for key in scan.IDENTITY}
 value["probabilities"]=np.tile(np.asarray([.2+.01*k,.3,.5-.01*k],dtype=np.float32),(10000,1))
 return value
def receipt(k):
 return dict(schema="american_c3_exact_streaming_focal_prediction_v1",parent_id=parent,class_order=list(fullmodels[k].class_order),
 block_index_ref=plan_ref,focal_registry_ref=plan["focal_registry_ref"],observation_ref=plan["observation_ref"],
 focal_records=10000,trees=5,forward_blocks=3,reverse_blocks=3,tree_encoding_passes=2,
 recurrent_state_reset_only_at_true_chromosome_endpoints=True,geometry_uses_global_neighbor_spans=True,
 finite_halo_approximation=False,CPU_focal_state_bytes=10000*32*4,full_parent_GPU_graph_or_history=False,
 private_truth_opened=False,checkpoint_contract=fullmodels[k].checkpoint_contract(),author_Table1_exact_reproduction=False,
 worker_drain=dict(forward=dict(drained=True),reverse=dict(drained=True)))
class Writer:
 tmp="SYNTHETIC_ONLY/TMP"
 def __init__(self):self.docs={};self.saved=[]
 def save(self,name,value,kind="json"):
  assert name not in self.docs
  self.docs[name]=copy.deepcopy(value);self.saved.append((name,kind))
  return r(name)
def fake_scan(models,a,b,**kw):
 assert len(models)==6 and kw["records"] is records
 return [(prediction(k),receipt(k)) for k in range(6)]
bn=dict(bridge.__dict__,new_candidate_models_v1=lambda:fullmodels,
 completed_training_v1=lambda ref:training,eligible_checkpoint_v1=checked,
 load_c3_checkpoint=lambda record,model:checked(record,training_ref)[1],registry=lambda record:records,
 validate_shared_plan_v1=lambda value,**kw:None,validate_shared_metadata_v1=lambda value,**kw:None,
 AmericanC3WholeTreeDatasetV1=Dataset,make_dataloader=loader,interleaved_six_focal_predictions_v1=fake_scan)
newpredict=types.FunctionType(bridge.predict_v1.__code__,bn)
w=Writer();contract=types.SimpleNamespace(VAL=VAL,read=lambda ref:plan)
newpredict(dict(phase="predict_val_000",plan_evidence=dict(plan_ref=plan_ref)),dict(training_completion_ref=training_ref),training,w,lambda event:None,contract=contract)
on=dict(oldseal.__dict__,eligible_checkpoint_v1=checked,validate_plan_v1=lambda *a,**kw:None,registry=lambda ref:records,
 streamed_focal_predictions=lambda model,a,b,**kw:(prediction(next(i for i,row in enumerate(training["checkpoint_candidates"]) if row["checkpoint_ref"]==kw["checkpoint_ref"])),receipt(0)),
 shutdown_dataloader=lambda loader:dict(drained=True))
sealfn=types.FunctionType(oldseal.seal_prediction_v1.__code__,on)
sealfn.__kwdefaults__=oldseal.seal_prediction_v1.__kwdefaults__
sn=dict(oldshared.__dict__,eligible_checkpoint_v1=checked,validate_shared_plan_v1=lambda *a,**kw:None,
 validate_shared_metadata_v1=lambda *a,**kw:None,seal_prediction_v1=sealfn)
sharedfn=types.FunctionType(oldshared.seal_shared_val_prediction_v1.__code__,sn)
oldw=Writer()
for candidate in training["checkpoint_candidates"]:
 ds=[Dataset(plan_ref,d,"SYNTHETIC") for d in ("forward","reverse")]
 sharedfn(fullmodels[0],*[Loader(d) for d in ds],parent_id=parent,checkpoint_ref=candidate["checkpoint_ref"],training_completion_ref=training_ref,writer=oldw,device="cpu",runtime_guard=lambda event:None)
for name,value in oldw.docs.items():
 new=w.docs[name]
 if name.endswith(".npz"):
  assert set(new)==set(value) and all(np.array_equal(new[k],value[k]) for k in value)
 else:
  assert all(new[k]==v for k,v in value.items()),name
  assert new["learned_contexts_shared"] is False and new["checkpoint_independent_features_shared"] is True
assert [name for name,kind in w.saved]==[name for name,kind in oldw.saved]+[parent+"_all6_prediction_seals.json"]
aggregate=w.docs[parent+"_all6_prediction_seals.json"]
assert len(aggregate["prediction_seal_refs"])==6 and aggregate["private_truth_opened"] is False and aggregate["TEST_deferred"] is True
ok("actual_original_seal_bodies_all_output_fields_names_identity_probabilities_and_disclosures_preserved_fake_IO")
bad=copy.deepcopy(training);bad["checkpoint_candidates"][0],bad["checkpoint_candidates"][1]=bad["checkpoint_candidates"][1],bad["checkpoint_candidates"][0]
reject(lambda:newpredict(dict(phase="predict_val_000",plan_evidence=dict(plan_ref=plan_ref)),dict(training_completion_ref=training_ref),bad,Writer(),lambda event:None,contract=contract))
reject(lambda:newpredict(dict(phase="predict_val_005",plan_evidence=dict(plan_ref=plan_ref)),dict(training_completion_ref=training_ref),training,Writer(),lambda event:None,contract=contract))
ok("bridge_rejects_wrong_candidate_order_and_non_original_VAL_phase")
assert not torch.cuda.is_initialized() and torch.get_num_threads()==1
out=dict(schema="C3_six_interleave_independent_tiny_FP32_checks_v1",status="PASS_TINY_SYNTHETIC_ONLY",checks=checks,check_count=len(checks),
 author_fixture_reproduced=True,extra_partition_cases=extra,actual_checkpoint_dataset_truth_or_prediction_payloads_opened=False,
 actual_guard_admission_live_source_mutation_or_job=False,torch_threads=1,CUDA_initialized=False,
 full30Mb_speed_memory_fit_or_32thread_bitwise_equality_claimed=False)
print(json.dumps(out,sort_keys=True,indent=2))
