"""H039 v3 dense-row CRF fix and sealed witness reuse; root lease required."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import census_tsinfer_fit_v1 as fit
from . import train_mutation_routed_whole_tree_gnn_v1 as original
from . import train_mutation_routed_full_panel_pyg_v1 as h038
from .project_storage_terminal_retired_scopes_v1 import PilotOutputV9,budget_check_v9,validate_storage_v9
from .mutation_routed_full_panel_observations_v1 import validate_registry_v1
from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1,FullPanelPrimaryValidationV1,PRIMARY
from .directed_clade_exchange_witness_v1 import PARENTS
from .directed_clade_exchange_crf_v1 import TRAIN_ARMS
from .directed_clade_exchange_continuation_v3 import COUNTS,load_selected_warmstart_v1,build_family_v1,run_shared_continuations_v1
ROOT=Path(__file__).resolve().parents[2]
SOURCE="src/arggnn/train_directed_clade_exchange_full_panel_v3.py"
CONFIG="configs/directed_clade_exchange_full_panel_v3.json"
NEW_SOURCES=tuple("src/arggnn/"+p for p in ("directed_clade_exchange_witness_v1.py",
 "directed_clade_exchange_crf_v1.py","directed_clade_exchange_continuation_v1.py",
 "train_directed_clade_exchange_full_panel_v1.py","directed_clade_exchange_continuation_v2.py",
 "train_directed_clade_exchange_full_panel_v2.py","directed_clade_exchange_crf_v2.py",
 "directed_clade_exchange_continuation_v3.py","train_directed_clade_exchange_full_panel_v3.py"))
ALLOWED_SOURCE_PATHS=h038.ALLOWED_SOURCE_PATHS|frozenset(NEW_SOURCES)


def validate_protocol_v1(cfg):
 if (cfg.get("hypothesis_id")!="H039" or cfg.get("execution_guard_ready") is not True
  or cfg.get("budget_guard")!="journal_terminal_retired_scopes_credit_v1"
  or cfg.get("arms")!=list(TRAIN_ARMS) or cfg.get("seed")!=101 or cfg.get("dtype")!="float32"
  or cfg.get("continuation_passes")!=10 or cfg.get("evaluation_passes")!=[5,10]
  or cfg.get("updates_per_pass")!=32 or cfg.get("additional_updates_per_arm")!=320
  or cfg.get("model")!=dict(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True)
  or cfg.get("parameter_counts")!=COUNTS or cfg.get("learning_rate")!=.001
  or cfg.get("weight_decay")!=.0001 or cfg.get("gradient_clip_norm")!=1.
  or cfg.get("early_stopping") is not False
  or cfg.get("optimizer_state_policy")!="reset_AdamW_moments_and_step_for_all_three_arms"
  or cfg.get("selection")!="earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
  or cfg.get("output_cap_bytes")!=650000000
  or type(cfg.get("workspace_reserve_bytes")) is not int or cfg["workspace_reserve_bytes"]<=0):
  raise ValueError("Root-activated exact H039 shared320-update/three-arm continuation required")


def inputs_v1(record):
 from .native_argformer_exec import validate_lease,apply_and_validate_title
 lease=validate_lease();apply_and_validate_title()
 h038.validated_metadata_ref_v1(record);reader=fit.ConsumerReaderV1();spec=reader.decode_json(record)
 if (spec.get("schema")!="directed_clade_exchange_full_panel_input_v3"
  or spec.get("hypothesis_id")!="H039" or spec.get("root_authorizer")!="root_agent"
  or spec.get("fit_authorized") is not True or spec.get("pyg_authorized") is not True
  or spec.get("status")!="active" or spec.get("arm")!="joint_three_arm"
  or spec.get("arms")!=list(TRAIN_ARMS) or spec.get("seed")!=101 or spec.get("device")!="cuda"
  or not fit.canonical_job_v1(spec.get("job_id"))):
  raise ValueError("Active root H039 one-process three-arm capability required")
 fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
 refs=spec["refs"]
 if set(refs)!={"config","registration","generation_config","runtime","storage","observation_cache",
  "target_bindings","warmstart_completion","witness_sidecar_manifest"}:raise ValueError("Exact H039 metadata roles required")
 expected={"config":CONFIG,"registration":h038.METADATA_REF_PATHS["registration"],
  "generation_config":h038.METADATA_REF_PATHS["generation_config"]}
 for name,ref in refs.items():h038.validated_metadata_ref_v1(ref,expected_path=expected.get(name))
 cfg=reader.decode_json(refs["config"]);validate_protocol_v1(cfg)
 if cfg.get("reused_witness_manifest_ref")!=refs["witness_sidecar_manifest"]:
  raise ValueError("Root config must pin the complete sealed witness reuse manifest")
 sources=spec["source_sha256"]
 if set(sources)!=ALLOWED_SOURCE_PATHS:raise ValueError("Original reviewed closure plus six frozen H039v1/v2 plus three H039v3 sources required")
 for path,digest in sources.items():
  if (not isinstance(path,str) or not path.endswith(".py") or Path(path).as_posix()!=path
   or ".." in Path(path).parts or not path.startswith(("src/arggnn/","scripts/"))
   or not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)):
   raise ValueError("Canonical Python source/SHA closure required")
  reader.read({"path":path,"sha256":digest})
 if cfg["source_refs"]!=[{"path":p,"sha256":sources[p]} for p in NEW_SOURCES]:
  raise ValueError("All nine H039 source refs must match activation")
 if spec["observation_cache_source_refs"]!=cfg["observation_cache_source_refs"]:
  raise ValueError("Original observed cache source lineage required")
 if any(ref!={"path":p,"sha256":sources[p]} for p,ref in spec["observation_cache_source_refs"].items()):
  raise ValueError("Original cache source pins differ")
 import torch_geometric
 if (str(torch.__version__)!=cfg["runtime"]["torch"] or torch.version.cuda!=cfg["runtime"]["torch_cuda"]
  or torch_geometric.__version__!=cfg["runtime"]["torch_geometric"]):
  raise ValueError("Pinned original Torch/CUDA/PyG runtime required")
 generation=reader.decode_json(refs["generation_config"]);registry=generation["parent_registry"]
 roles=validate_registry_v1(registry)
 if tuple(roles["train"]+roles["val"])!=PARENTS or set(spec["observation_entry_refs"])!=set(PARENTS):
  raise ValueError("Exact original32TRAIN/eightVAL, no outer observations required")
 for p,ref in spec["observation_entry_refs"].items():
  h038.validated_metadata_ref_v1(ref,expected_path=generation["output_root"]+"/"+p+"/full_panel_observed_entry.json")
 registration=reader.decode_json(refs["registration"])
 if (registration["config_ref"]!=refs["generation_config"] or registration["parent_registry"]!=registry):
  raise ValueError("Unchanged original full-panel generation registration required")
 bindings=reader.decode_json(refs["target_bindings"])["parents"]
 if set(bindings)!=set(PARENTS):raise ValueError("No outer target bindings allowed")
 for p,binding in bindings.items():
  if (binding["truth_npz"]["path"]!=generation["output_root"]+"/"+p+"/truth/full_truth_intervals.npz"
   or binding["split"]!=("train" if p in PARENTS[:32] else "val")):
   raise ValueError("Original same-parent target path/split required")
  h038.validated_metadata_ref_v1(binding["generation_receipt"])
  provenance=reader.decode_json(binding["generation_receipt"])
  if provenance["parent_id"]!=p or provenance["files"]["truth/full_truth_intervals.npz"]["sha256"]!=binding["truth_npz"]["sha256"]:
   raise ValueError("Original same-parent target provenance differs")
 for name in ("max_cpu_rss_bytes","max_cuda_reserved_bytes"):
  if type(spec.get(name)) is not int or spec[name]<=0:raise ValueError("Positive original resource admission required")
 reader.read(refs["runtime"]);reader.read(cfg["base_fit_config"])
 storage=reader.decode_json(refs["storage"])
 launcher=cfg["helper_refs"]["actual_full_storage_launcher"]
 if launcher not in storage["reservation"]["consumer_bindings"]:raise ValueError("Existing real launcher must be reserved")
 validate_storage_v9(spec,cfg,storage,source_ref={"path":SOURCE,"sha256":sources[SOURCE]},
  config_ref=refs["config"],runtime_ref=refs["runtime"])
 if cfg.get("warmstart_completion_ref")!=refs["warmstart_completion"]:
  raise ValueError("Root config must bind actual final aligned H038 completion")
 original.science_v1()
 return spec,cfg,registry,roles,bindings,reader,storage


class CompressedSealOutputV1(PilotOutputV9):
 """Existing single-scope compressed writer; pending immutable arrays aid scoring."""
 def __init__(self,*args):super().__init__(*args);self.current_arrays={}
 def seal(self,name,arrays):
  expected={"C4_probabilities","positions_bp","bin_edges_bp","retained_fixed100bp_indexes","query_sample_indexes"}
  if set(arrays)!=expected:raise ValueError("Original complete probability-only seal schema required")
  immutable={k:np.array(v,copy=True,order="C") for k,v in arrays.items()}
  for a in immutable.values():a.setflags(write=False)
  ref=self.save(name,immutable,"npz");self.current_arrays[ref["path"]]=(dict(ref),immutable)
  return ref
 def discard(self,ref):super().discard(ref);self.current_arrays.pop(ref["path"],None)


class CurrentSealReaderV1:
 def __init__(self,base,output):self.base,self.output=base,output
 def read(self,ref):return self.base.read(ref)
 def current_probability_seal_v1(self,ref,parent):
  known=self.output.current_arrays.get(ref["path"])
  if known is None:return None
  if known[0]!=ref:raise ValueError("Immutable current compressed seal identity differs")
  arrays=known[1]
  checks={"positions_bp":parent["positions"],"bin_edges_bp":parent["edges"],
   "retained_fixed100bp_indexes":parent["retained_indexes"],"query_sample_indexes":parent["query_slots"]}
  if any(not np.array_equal(arrays[k],v) for k,v in checks.items()):
   raise ValueError("Original scored physical support/query identity differs")
  probability=arrays["C4_probabilities"]
  if (probability.dtype!=np.float32 or probability.shape!=(len(parent["positions"]),686,4)
   or not np.isfinite(probability).all() or np.any(probability<0)
   or not np.allclose(probability.sum(-1),1,rtol=1e-5,atol=1e-6)):
   raise ValueError("Complete finite FP32 raw probabilities required")
  return probability


def reuse_full_sidecars_v2(cache_manifest_ref,sidecar_manifest_ref,reader,*,registry,entry_refs,source_refs,output,execution_guard):
 """Read original banks once and reuse sealed sidecar arrays; no LCA generation."""
 from dataclasses import fields,replace
 from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
 from .mutation_routed_parent_loader_v1 import ObservationOnlyParentReaderV1
 from .directed_clade_exchange_witness_v1 import read_parent_sidecar_v1,SOURCE as WITNESS_SOURCE
 from .directed_clade_exchange_crf_v1 import CladeExchangeInputsV1
 boundary=ObservationOnlyParentReaderV1(reader);manifest=boundary.decode_json(sidecar_manifest_ref)
 builder={"path":WITNESS_SOURCE,"sha256":hashlib.sha256((ROOT/WITNESS_SOURCE).read_bytes()).hexdigest()}
 if (manifest.get("schema")!="directed_clade_exchange_full_observation_sidecars_v1"
  or manifest.get("builder_ref")!=builder or manifest.get("observation_cache_ref")!=cache_manifest_ref
  or manifest.get("entry_refs")!=entry_refs or manifest.get("source_refs")!=source_refs
  or manifest.get("parent_ids")!=list(PARENTS) or set(manifest.get("parents",{}))!=set(PARENTS)
  or manifest.get("all40_observed_banks_complete") is not True or manifest.get("targets_opened") is not False
  or manifest.get("outer_included") is not False):
  raise ValueError("Exact complete original sealed observation-only witness manifest required")
 index=read_full_cache_index_v1(cache_manifest_ref,reader,registry=registry,entry_refs=entry_refs,source_refs=source_refs)
 banks,parents={},{};started=time.monotonic()
 for pid in PARENTS:
  execution_guard();item=read_full_cache_parent_v1(pid,index,reader)
  arrays,sidecar=read_parent_sidecar_v1(manifest["parents"][pid],reader,parent_id=pid,
   observation_entry_ref=entry_refs[pid],cache_parent_ref=index["parents"][pid]["array_ref"])
  bank=CladeExchangeInputsV1.attach(item["model_inputs"],arrays,record=sidecar)
  values={}
  for f in fields(bank):
   value=getattr(bank,f.name)
   values[f.name]=tuple(v.pin_memory() for v in value) if isinstance(value,tuple) else value.pin_memory()
  banks[pid]=replace(bank,**values);parents[pid]=item["scoring"]
  print(json.dumps(dict(stage="H039_reuse_verified_witness_parent",parent_id=pid,
   native_holds=int(bank.sample_index.shape[0]),witness_arrays_recomputed=False)),flush=True)
  del item,arrays,sidecar,bank,values
  execution_guard()
 seal=output.save("all40_reused_observation_witness_sidecars.json",dict(
  original_sidecar_manifest_ref=sidecar_manifest_ref,observation_cache_ref=cache_manifest_ref,
  parent_ids=list(PARENTS),all40_observed_banks_complete=True,targets_opened=False,outer_included=False,
  witness_arrays_recomputed=False,source_sidecar_refs=manifest["parents"]))
 return banks,parents,seal,dict(raw_witness_array_bytes=manifest["raw_witness_array_bytes"],
  witness_arrays_recomputed=False,new_witness_output_bytes=0,bank_and_sidecar_read_seconds=time.monotonic()-started,
  parent_native_holds={p:int(b.sample_index.shape[0]) for p,b in banks.items()},
  parent_readout_rows={p:int(b.readout_hold.numel()) for p,b in banks.items()})


def storage_byte_math_v1(banks,parents):
 witness=sum(38416*(b.sample_index.shape[0]-1)+96 for b in banks.values())
 one_generation=sum(3*(len(p["positions"])*686*4*4+
  p["positions"].nbytes+p["edges"].nbytes+p["retained_indexes"].nbytes+p["query_slots"].nbytes) for p in parents.values() if p["split"]=="val")
 return dict(raw_witness_arrays=witness,raw24_VAL_arrays=one_generation,
  raw_selected_plus_pending48_VAL_arrays=2*one_generation,
  uncompressed_joint_arrays_including_reused_scope=witness+2*one_generation,
  new_witness_output_bytes=0,
  output_cap_bytes=650000000,compression="existing_PilotOutputV9_lossless_NPZ_DEFLATE_original_FP32_grid",
  compression_ratio_assumed_for_admission=False,actual_compressed_write_size_guard=True,
  at_most_selected_plus_pending_seals=True)


def run_v1(record):
 spec,cfg,registry,roles,bindings,reader,storage=inputs_v1(record)
 from .biological_candidate import configure_fit
 output=CompressedSealOutputV1(spec["output_directory"],storage,reader)
 budget_check_v9(storage,boundary=True,required_write_bytes=32000000)
 base=reader.decode_json(cfg["base_fit_config"]);device,runtime,lease=configure_fit("cuda",base,output.path)
 output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob("*") if p.is_file())
 if output.used>32000000:raise ValueError("Qualification exceeds original32MB admission")
 guard=lambda:original.lightweight_resources_v1(spec,lease)
 started=time.monotonic();helpers=fit.metric_helpers_v1();selected_refs={a:[] for a in TRAIN_ARMS}
 receipt=dict(schema="directed_clade_exchange_full_panel_evidence_v3",hypothesis_id="H039",
  status="running",arms=list(TRAIN_ARMS),seed=101,input_ref=record,config_ref=spec["refs"]["config"],
  source_sha256=spec["source_sha256"],runtime=runtime,outer_truth_read=False,outer_predictions_executed=False,
  scientific_control="B+A versus L=B+A+J; generic persistence and same-W unary evidence retained")
 output.save("started.json",receipt)
 try:
  payload,warmstart=load_selected_warmstart_v1(spec["refs"]["warmstart_completion"],reader)
  if cfg.get("warmstart_checkpoint_ref")!=warmstart["checkpoint_ref"]:
   raise ValueError("Root config must pin actual selected aligned checkpoint")
  banks,parents,observation_seal,preparation=reuse_full_sidecars_v2(spec["refs"]["observation_cache"],
   spec["refs"]["witness_sidecar_manifest"],reader,registry=registry,entry_refs=spec["observation_entry_refs"],
   source_refs=spec["observation_cache_source_refs"],output=output,execution_guard=guard)
  output.save("storage_byte_math.json",storage_byte_math_v1(banks,parents))
  targets={}
  for p in roles["train"]:
   guard();target=FullPanelExternalTargetV1(parents[p],bindings[p],"TRAIN_loss",reader)
   mass=h038.native_hold_train_mass_v1(parents[p],banks[p],target,helpers)
   targets[p]=torch.from_numpy(mass).pin_memory();del target,mass
  family,optimizers=build_family_v1(payload,cfg["model"],device=device,execution_guard=guard);del payload
  free,_=torch.cuda.mem_get_info(device);reserved=torch.cuda.memory_reserved(device)
  cap=min(int(free),spec["max_cuda_reserved_bytes"]-int(reserved))-cfg["workspace_reserve_bytes"]
  if cap<=0:raise ValueError("No admitted complete-parent shared residency headroom")
  residency=h038.FullParentResidencyV1(banks,targets,device,cap)
  all_bytes=sum(residency.needed(p) for p in PARENTS)
  if all_bytes<=cap:
   for p in PARENTS:residency.prefetch(p)
  seal_reader=CurrentSealReaderV1(reader,output)
  validations={a:FullPanelPrimaryValidationV1(parents,bindings,seal_reader,helpers,cfg) for a in TRAIN_ARMS}
  shared_caches={}
  for v in validations.values():v.caches=shared_caches
  sealed_passes=set()
  output.save("initialization_and_admission.json",dict(warmstart=warmstart,observation_seal=observation_seal,
   preparation=preparation,shared_read_only_residency=True,all_bank_and_TRAIN_mass_bytes=all_bytes,
   residency_cap_bytes=cap,all40_banks_resident=all_bytes<=cap,optimizer_state_reset_for_all=True))
  def seal(parent,arm,at,probability):
   guard();raw=probability.detach().cpu().numpy();p=parents[parent]
   return output.seal("%s.pass_%02d.%s.npz"%(arm,at,parent),dict(C4_probabilities=raw,
    positions_bp=p["positions"],bin_edges_bp=p["edges"],retained_fixed100bp_indexes=p["retained_indexes"],
    query_sample_indexes=p["query_slots"]))
  def score(seals):
   name=Path(seals[0][1]["path"]).name;arm=name.split(".")[0];at=int(name.split(".")[1].split("_")[1])
   if at not in sealed_passes:raise ValueError("All24 predictions must be sealed before any VAL score")
   guard();return validations[arm].score(seals)
  def checkpoint(arm,at,model,optimizer,steps):
   return output.save("%s.pass_%02d.pt"%(arm,at),dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
    optimizer_state=optimizer.state_dict(),arm=arm,seed=101,pass_index=at,additional_optimizer_steps=steps,
    original_optimizer_steps=warmstart["original_optimizer_steps"],warmstart=warmstart,input_ref=record,
    teacher_used=False,optimizer_state_reset=True),"torch")
  def register(name,value):
   guard();registered=output.save(name+".json",value)
   if name.endswith("_all24_VAL_sealed"):sealed_passes.add(value["pass_index"])
   if name.endswith("_selection"):
    row=value["row"];arm=row["arm"];current=[ref for _,ref in row["prediction_seals"]]+[row["checkpoint"]]
    previous=selected_refs[arm]
    discard=current if value["discard_current"] else previous
    if not value["discard_current"]:selected_refs[arm]=current
    for ref in discard:output.discard(ref)
    for _,ref in row["prediction_seals"]:output.current_arrays.pop(ref["path"],None)
    print(json.dumps(dict(stage="H039_VAL_selection",arm=arm,pass_index=row["pass_index"],
     additional_updates=row["optimizer_steps"],selected_pass=value["selected"]["pass_index"],
     primary=row["equal_parent_metrics"][PRIMARY],registered=registered,owned_output_bytes=output.used)),flush=True)
   elif name.startswith("TRAIN_pass_"):
    print(json.dumps(dict(stage="H039_TRAIN_pass",**value)),flush=True)
   return registered
  result=run_shared_continuations_v1(cfg,family,optimizers,residency,execution_guard=guard,
   seal=seal,score=score,record=register,checkpoint=checkpoint)
  full={}
  for arm in TRAIN_ARMS:
   guard();cells,means=validations[arm].full_final(result["selected"][arm]["prediction_seals"])
   full[arm]=output.save(arm+".selected_full_diagnostics.json",dict(
    selected_pass=result["selected"][arm]["pass_index"],selection_revised=False,cells=cells,equal_parent_metrics=means))
  receipt.update(status="completed_H039_seed101_shared_three_arm_continuation_v3",result=result,
   warmstart=warmstart,observation_seal=observation_seal,preparation=preparation,selected_full_diagnostics=full,
   retained_selected_records=selected_refs,retained_records=[v[0] for v in output.owned.values()],
   elapsed_seconds=time.monotonic()-started,owned_output_bytes=output.used,
   storage_compression_lossless=True,novelty_claim_established=False)
  guard();output.save("completed.json",receipt);return receipt
 except Exception as error:
  receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
   retained_selected_records=selected_refs,elapsed_seconds=time.monotonic()-started,automatic_replay=False)
  output.save("failed.json",receipt);raise


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
 p.add_argument("--input");p.add_argument("--expected-input-sha256");args=p.parse_args(argv)
 if not args.execute:
  print(json.dumps(dict(status="source_only_H039_not_executed",source=SOURCE,config=CONFIG,
   arms=list(TRAIN_ARMS),additional_updates_per_arm=320,output_scopes=1,output_cap_bytes=650000000)));return 0
 if not args.input or not args.expected_input_sha256:raise ValueError("Root hash-pinned activation required")
 run_v1({"path":args.input,"sha256":args.expected_input_sha256});return 0
if __name__=="__main__":raise SystemExit(main())
