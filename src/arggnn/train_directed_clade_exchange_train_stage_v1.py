"""H039 TRAIN-only fixed schedule and six actual checkpoints; root activation required."""
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from . import census_tsinfer_fit_v1 as fit
from . import train_mutation_routed_whole_tree_gnn_v1 as original
from . import train_mutation_routed_full_panel_pyg_v1 as h038
from .project_storage_terminal_retired_scopes_v1 import PilotOutputV9,budget_check_v9,validate_storage_v9
from .mutation_routed_full_panel_observations_v1 import validate_registry_v1
from .mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
from .directed_clade_exchange_witness_v1 import PARENTS
from .directed_clade_exchange_crf_v1 import TRAIN_ARMS
from .directed_clade_exchange_continuation_v3 import COUNTS,TRAIN,load_selected_warmstart_v1,build_family_v1,train_shared_parent_v1
ROOT=Path(__file__).resolve().parents[2]
SOURCE="src/arggnn/train_directed_clade_exchange_train_stage_v1.py"
CONFIG="configs/directed_clade_exchange_train_stage_v1.json"
NEW_SOURCES=tuple("src/arggnn/"+p for p in ("directed_clade_exchange_witness_v1.py",
 "directed_clade_exchange_crf_v1.py","directed_clade_exchange_continuation_v1.py",
 "train_directed_clade_exchange_full_panel_v1.py","directed_clade_exchange_continuation_v2.py",
 "train_directed_clade_exchange_full_panel_v2.py","directed_clade_exchange_crf_v2.py",
 "directed_clade_exchange_continuation_v3.py","train_directed_clade_exchange_full_panel_v3.py","train_directed_clade_exchange_train_stage_v1.py"))
ALLOWED_SOURCE_PATHS=h038.ALLOWED_SOURCE_PATHS|frozenset(NEW_SOURCES)


def validate_protocol_v1(cfg):
 if (cfg.get("hypothesis_id")!="H039" or cfg.get("execution_guard_ready") is not True
  or cfg.get("budget_guard")!="journal_terminal_retired_scopes_credit_v1"
  or cfg.get("arms")!=list(TRAIN_ARMS) or cfg.get("seed")!=101 or cfg.get("dtype")!="float32"
  or cfg.get("phase")!="train_only" or cfg.get("evaluation_deferred") is not True
  or cfg.get("qualification_allowance_bytes")!=2000000 or cfg.get("checkpoint_passes")!=[5,10]
  or cfg.get("VAL_influences_optimization") is not False
  or cfg.get("continuation_passes")!=10 or cfg.get("evaluation_passes")!=[5,10]
  or cfg.get("updates_per_pass")!=32 or cfg.get("additional_updates_per_arm")!=320
  or cfg.get("model")!=dict(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True)
  or cfg.get("parameter_counts")!=COUNTS or cfg.get("learning_rate")!=.001
  or cfg.get("weight_decay")!=.0001 or cfg.get("gradient_clip_norm")!=1.
  or cfg.get("early_stopping") is not False
  or cfg.get("optimizer_state_policy")!="reset_AdamW_moments_and_step_for_all_three_arms"
  or cfg.get("selection")!="earliest_maximum_equal_parent_Present182_exact_span_accuracy_then_lower_NLL_then_earlier_pass"
  or cfg.get("output_cap_bytes")!=15000000
  or type(cfg.get("workspace_reserve_bytes")) is not int or cfg["workspace_reserve_bytes"]<=0):
  raise ValueError("Root-activated H039 TRAIN-only15MB stage with six frozen checkpoints required")

def inputs_v1(record):
 from .native_argformer_exec import validate_lease,apply_and_validate_title
 lease=validate_lease();apply_and_validate_title()
 h038.validated_metadata_ref_v1(record);reader=fit.ConsumerReaderV1();spec=reader.decode_json(record)
 if (spec.get("schema")!="directed_clade_exchange_train_stage_input_v1"
  or spec.get("hypothesis_id")!="H039" or spec.get("root_authorizer")!="root_agent"
  or spec.get("phase")!="train_only" or spec.get("split_execution_authorized") is not True
  or spec.get("fit_authorized") is not True or spec.get("pyg_authorized") is not True
  or spec.get("status")!="active" or spec.get("arm")!="joint_three_arm"
  or spec.get("arms")!=list(TRAIN_ARMS) or spec.get("seed")!=101 or spec.get("device")!="cuda"
  or not fit.canonical_job_v1(spec.get("job_id"))):
  raise ValueError("Active root H039 one-process three-arm capability required")
 fit._check_active_v1(spec);fit.require_logical_job_v1(lease,spec)
 refs=spec["refs"]
 if set(refs)!={"config","registration","generation_config","runtime","storage","observation_cache",
  "target_bindings","warmstart_completion","witness_sidecar_manifest","phase_registration"}:raise ValueError("Exact H039 metadata roles required")
 expected={"config":CONFIG,"registration":h038.METADATA_REF_PATHS["registration"],
  "generation_config":h038.METADATA_REF_PATHS["generation_config"]}
 for name,ref in refs.items():h038.validated_metadata_ref_v1(ref,expected_path=expected.get(name))
 cfg=reader.decode_json(refs["config"]);validate_protocol_v1(cfg)
 validate_split_phase_v1(reader.decode_json(refs["phase_registration"]),cfg,refs)
 if cfg.get("reused_witness_manifest_ref")!=refs["witness_sidecar_manifest"]:
  raise ValueError("Root config must pin the complete sealed witness reuse manifest")
 sources=spec["source_sha256"]
 if set(sources)!=ALLOWED_SOURCE_PATHS:raise ValueError("Original reviewed closure plus nine frozen H039sources plus actual TRAIN-stage source required")
 for path,digest in sources.items():
  if (not isinstance(path,str) or not path.endswith(".py") or Path(path).as_posix()!=path
   or ".." in Path(path).parts or not path.startswith(("src/arggnn/","scripts/"))
   or not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest)):
   raise ValueError("Canonical Python source/SHA closure required")
  reader.read({"path":path,"sha256":digest})
 if cfg["source_refs"]!=[{"path":p,"sha256":sources[p]} for p in NEW_SOURCES]:
  raise ValueError("All ten H039 source refs must match activation")
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
 if set(bindings)!=set(TRAIN):raise ValueError("Only32TRAIN target bindings; no VAL or outer bindings")
 for p,binding in bindings.items():
  if (binding["truth_npz"]["path"]!=generation["output_root"]+"/"+p+"/truth/full_truth_intervals.npz"
   or binding["split"]!="train"):
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

def reuse_train_sidecars_v1(cache_manifest_ref,sidecar_manifest_ref,reader,*,registry,entry_refs,source_refs,output,execution_guard):
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
 for pid in TRAIN:
  execution_guard();item=read_full_cache_parent_v1(pid,index,reader)
  arrays,sidecar=read_parent_sidecar_v1(manifest["parents"][pid],reader,parent_id=pid,
   observation_entry_ref=entry_refs[pid],cache_parent_ref=index["parents"][pid]["array_ref"])
  bank=CladeExchangeInputsV1.attach(item["model_inputs"],arrays,record=sidecar)
  values={}
  for f in fields(bank):
   value=getattr(bank,f.name)
   values[f.name]=tuple(v.pin_memory() for v in value) if isinstance(value,tuple) else value.pin_memory()
  banks[pid]=replace(bank,**values);parents[pid]=item["scoring"]
  print(json.dumps(dict(stage="H039_TRAIN_stage_reuse_verified_witness_parent",parent_id=pid,
   native_holds=int(bank.sample_index.shape[0]),witness_arrays_recomputed=False)),flush=True)
  del item,arrays,sidecar,bank,values
  execution_guard()
 seal=output.save("all32_TRAIN_reused_observation_witness_sidecars.json",dict(
  original_sidecar_manifest_ref=sidecar_manifest_ref,observation_cache_ref=cache_manifest_ref,
  parent_ids=list(TRAIN),all32_TRAIN_observed_banks_complete=True,targets_opened=False,outer_included=False,
  witness_arrays_recomputed=False,source_sidecar_refs={p:manifest["parents"][p] for p in TRAIN}))
 return banks,parents,seal,dict(reused_full_manifest_raw_witness_array_bytes=manifest["raw_witness_array_bytes"],
  witness_arrays_recomputed=False,new_witness_output_bytes=0,bank_and_sidecar_read_seconds=time.monotonic()-started,
  parent_native_holds={p:int(b.sample_index.shape[0]) for p,b in banks.items()},
  parent_readout_rows={p:int(b.readout_hold.numel()) for p,b in banks.items()})

def validate_split_phase_v1(phase,cfg,refs):
 """Root prospectively registers fixed optimization and deferred selection."""
 if (phase.get("schema")!="directed_clade_exchange_split_phase_registration_v1"
  or phase.get("root_authorizer")!="root_agent" or phase.get("status")!="registered"
  or phase.get("hypothesis_id")!="H039" or phase.get("arms")!=list(TRAIN_ARMS)
  or phase.get("seed")!=101 or phase.get("continuation_passes")!=10
  or phase.get("additional_updates_per_arm")!=320 or phase.get("checkpoint_passes")!=[5,10]
  or phase.get("early_stopping") is not False or phase.get("VAL_influences_optimization") is not False
  or phase.get("later_all24_seals_before_any_VAL_truth") is not True
  or phase.get("selection")!=cfg["selection"]
  or phase.get("warmstart_completion_ref")!=refs["warmstart_completion"]
  or phase.get("witness_sidecar_manifest_ref")!=refs["witness_sidecar_manifest"]):
  raise ValueError("Explicit root split-phase fixed optimization/deferred original VAL selection required")


def run_v1(record):
 spec,cfg,registry,roles,bindings,reader,storage=inputs_v1(record)
 from .biological_candidate import configure_fit
 output=PilotOutputV9(spec["output_directory"],storage,reader)
 # Existing qualified title/device checks; realistic cap replaces the unrelated32MB allowance.
 budget_check_v9(storage,boundary=True,required_write_bytes=cfg["qualification_allowance_bytes"])
 base=reader.decode_json(cfg["base_fit_config"])
 device,runtime,lease=configure_fit("cuda",base,output.path)
 output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192
  for p in output.path.rglob("*") if p.is_file())
 if output.used>cfg["qualification_allowance_bytes"]:
  raise ValueError("Actual qualification output exceeds admitted2MB allowance")
 guard=lambda:original.lightweight_resources_v1(spec,lease)
 started=time.monotonic();helpers=fit.metric_helpers_v1()
 steps={a:0 for a in TRAIN_ARMS};checkpoints={};history=[];active_parent=None;active_pass=0
 receipt=dict(schema="directed_clade_exchange_train_stage_evidence_v1",hypothesis_id="H039",
  phase="train_only",status="running",arms=list(TRAIN_ARMS),seed=101,input_ref=record,
  config_ref=spec["refs"]["config"],phase_registration_ref=spec["refs"]["phase_registration"],
  source_sha256=spec["source_sha256"],runtime=runtime,TRAIN_parents=32,query_haplotypes=686,
  observed_haplotypes=952,VAL_bank_arrays_read=False,VAL_predictions_executed=False,VAL_truth_read=False,
  outer_truth_read=False,outer_predictions_executed=False,selection_deferred=True,
  optimizer_state_reset_for_all=True,extra_updates_per_arm=320,early_stopping=False)
 output.save("started.json",receipt)
 try:
  payload,warmstart=load_selected_warmstart_v1(spec["refs"]["warmstart_completion"],reader)
  if cfg["warmstart_checkpoint_ref"]!=warmstart["checkpoint_ref"]:
   raise ValueError("Exact actual selected aligned H038 checkpoint required")
  banks,parents,observation_seal,preparation=reuse_train_sidecars_v1(
   spec["refs"]["observation_cache"],spec["refs"]["witness_sidecar_manifest"],reader,
   registry=registry,entry_refs=spec["observation_entry_refs"],source_refs=spec["observation_cache_source_refs"],
   output=output,execution_guard=guard)
  if set(banks)!=set(TRAIN) or set(parents)!=set(TRAIN):raise ValueError("Exactly32TRAIN banks required")
  targets={}
  for p in TRAIN:
   guard();target=FullPanelExternalTargetV1(parents[p],bindings[p],"TRAIN_loss",reader)
   mass=h038.native_hold_train_mass_v1(parents[p],banks[p],target,helpers)
   targets[p]=torch.from_numpy(mass).pin_memory();del target,mass
  family,optimizers=build_family_v1(payload,cfg["model"],device=device,execution_guard=guard)
  del payload
  free,_=torch.cuda.mem_get_info(device);reserved=torch.cuda.memory_reserved(device)
  cap=min(int(free),spec["max_cuda_reserved_bytes"]-int(reserved))-cfg["workspace_reserve_bytes"]
  if cap<=0:raise ValueError("No admitted shared complete-parent residency headroom")
  residency=h038.FullParentResidencyV1(banks,targets,device,cap)
  all_bytes=sum(residency.needed(p) for p in TRAIN)
  if all_bytes<=cap:
   for p in TRAIN:residency.prefetch(p)
  output.save("initialization_and_admission.json",dict(warmstart=warmstart,observation_seal=observation_seal,
   preparation=preparation,shared_read_only_residency=True,all32_TRAIN_bank_and_mass_bytes=all_bytes,
   residency_cap_bytes=cap,all32_TRAIN_banks_resident=all_bytes<=cap,parameter_counts=COUNTS,
   optimizer_state_reset_for_all=True,VAL_targets_opened=False,VAL_predictions_executed=False,
   fixed_checkpoint_passes=[5,10],later_selection_uses_actual_frozen_checkpoints=True))
  for pass_index in range(1,11):
   active_pass=pass_index;order=list(TRAIN);random.Random(101+pass_index).shuffle(order)
   summaries={a:dict(mean_TRAIN_loss=0.,model_seconds=0.) for a in TRAIN_ARMS}
   pass_started=time.monotonic()
   for at,p in enumerate(order):
    active_parent=p;guard();bank,mass=residency.get(p)
    if at+1<len(order):residency.prefetch(order[at+1],protected=(p,))
    rows=train_shared_parent_v1(family,optimizers,bank,mass,execution_guard=guard)
    for arm,row in rows.items():
     steps[arm]+=1;summaries[arm]["mean_TRAIN_loss"]+=row["loss"]/32
     summaries[arm]["model_seconds"]+=row["update_seconds"]
    if pass_index==1 and at==0:
     print(json.dumps(dict(stage="H039_TRAIN_stage_first_actual_same_parent_updates",
      parent_id=p,steps=steps.copy(),arms=rows)),flush=True)
    if (at+1)%8==0:
     print(json.dumps(dict(stage="H039_TRAIN_stage_progress",pass_index=pass_index,
      completed_parents_in_pass=at+1,steps=steps.copy(),pass_elapsed_seconds=time.monotonic()-pass_started)),flush=True)
   active_parent=None
   if any(value!=32*pass_index for value in steps.values()):raise ValueError("Unequal fixed optimization exposure")
   row=dict(pass_index=pass_index,steps=steps.copy(),distinct_TRAIN_parents=32,
    complete_parent_updates_per_arm=32,arms=summaries,elapsed_seconds=time.monotonic()-pass_started,
    VAL_influences_optimization=False)
   history.append(row);output.save("TRAIN_pass_%02d.json"%pass_index,row)
   print(json.dumps(dict(stage="H039_TRAIN_stage_pass",**row)),flush=True)
   if pass_index in (5,10):
    group={}
    for arm in TRAIN_ARMS:
     guard();model=family[arm]
     group[arm]=output.save("%s.pass_%02d.pt"%(arm,pass_index),dict(
      model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
      optimizer_state=optimizers[arm].state_dict(),arm=arm,seed=101,pass_index=pass_index,
      additional_optimizer_steps=steps[arm],original_optimizer_steps=warmstart["original_optimizer_steps"],
      warmstart=warmstart,input_ref=record,train_config_ref=spec["refs"]["config"],
      phase_registration_ref=spec["refs"]["phase_registration"],model_kwargs=cfg["model"],
      parameter_count=COUNTS[arm],teacher_used=False,optimizer_state_reset=True,phase="train_only"),"torch")
    checkpoints[str(pass_index)]=group
    output.save("pass_%02d_all3_frozen_checkpoints.json"%pass_index,dict(
     pass_index=pass_index,steps=steps.copy(),checkpoints=group,all3_actual_states_retained=True,
     VAL_predictions_executed=False,VAL_truth_read=False))
    print(json.dumps(dict(stage="H039_TRAIN_stage_all3_checkpoints",pass_index=pass_index,
     steps=steps.copy(),checkpoints=group,owned_output_bytes=output.used)),flush=True)
  if (any(value!=320 for value in steps.values()) or set(checkpoints)!={"5","10"}
   or any(set(group)!=set(TRAIN_ARMS) for group in checkpoints.values())):
   raise ValueError("Exact3x320 updates and all6 actual pass5/pass10 checkpoints required")
  guard()
  receipt.update(status="completed_H039_TRAIN_stage_v1_all6_checkpoints_320_updates_per_arm",
   steps=steps,history=history,checkpoint_groups=checkpoints,checkpoint_count=6,warmstart=warmstart,
   observation_seal=observation_seal,preparation=preparation,selection_deferred=True,
   later_evaluation_passes=[5,10],original_selection_rule=cfg["selection"],
   retained_records={path:value[0] for path,value in output.owned.items()},
   elapsed_seconds=time.monotonic()-started,owned_output_bytes=output.used)
  output.save("completed.json",receipt);return receipt
 except Exception as error:
  receipt.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
   completed_same_parent_steps=steps.copy(),completed_passes=len(history),active_pass=active_pass,
   active_parent=active_parent,partial_active_parent_arm_updates_may_exist=active_parent is not None,
   checkpoint_groups=checkpoints,history=history,automatic_replay=False,elapsed_seconds=time.monotonic()-started)
  output.save("failed.json",receipt);raise


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
 p.add_argument("--input");p.add_argument("--expected-input-sha256");args=p.parse_args(argv)
 if not args.execute:
  print(json.dumps(dict(status="source_only_H039_TRAIN_stage_not_executed",source=SOURCE,config=CONFIG,
   arms=list(TRAIN_ARMS),phase="train_only",additional_updates_per_arm=320,checkpoint_passes=[5,10],
   checkpoint_count=6,output_cap_bytes=15000000,qualification_allowance_bytes=2000000,
   VAL_predictions_executed=False,VAL_truth_read=False)));return 0
 if not args.input or not args.expected_input_sha256:raise ValueError("Root hash-pinned TRAIN-phase activation required")
 run_v1({"path":args.input,"sha256":args.expected_input_sha256});return 0
if __name__=="__main__":raise SystemExit(main())
