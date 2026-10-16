"""Inactive guard-agnostic H056 CPU application of the frozen H045 checkpoint.

Root injects its existing admitted IO callbacks. No admission, guard, launcher,
scheduler, optimizer, calibration or payload read occurs at module import.
"""
from pathlib import Path
import datetime,hashlib,io,json,os,resource,signal,time
from research.reviews.H056_H36_implementation_20261016_v1.contract_v1 import (
    ROOT,HERE,H,R,Q,BEGIN,END,CHROMOSOMES,CLASSES,SUBJECTS,EARLIER,LATER,
    PROTOCOL_REF,ROSTER_REF,CHECKPOINT_REF,CHECKPOINT_DESCRIPTOR_REF,FEATURE_SEMANTICS,
    require,sha,encoded,validate_counts_v1,validate_source_manifest_v1)
SOURCE=HERE+"/apply_h36_v1.py"

def ref(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file(),"Canonical exact source reference")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p.read_bytes()))

def summarize_probabilities_v1(chromosome_results):
    """Fixed physical-span/haplotype/person/era order; all three are mandatory."""
    import numpy as np
    require(set(chromosome_results)==set(CHROMOSOMES),"All three chromosomes required, never substitute")
    means=[];weights=[];per_chrom=[]
    for chrom in CHROMOSOMES:
        item=chromosome_results[chrom]
        a=np.asarray(item["haplotype_proportions"],dtype=np.float64);bp=item["covered_bp"]
        require(a.shape==(Q,4) and np.isfinite(a).all() and np.all((a>=0)&(a<=1))
            and np.allclose(a.sum(-1),1.,atol=1e-6,rtol=0)
            and type(bp)in(int,float) and np.isfinite(bp) and bp>0,"Complete20 haplotype probability means and genuine bp")
        persons=a.reshape(10,2,4).mean(1)
        means.append(persons);weights.append(float(bp))
        per_chrom.append(dict(chromosome=chrom,covered_bp=float(bp),covered_absolute_bp0=item["covered_absolute_bp0"],
            person_probabilities=persons.tolist(),earlier_mean=persons[:5].mean(0).tolist(),
            later_mean=persons[5:].mean(0).tolist(),later_minus_earlier=(persons[5:].mean(0)-persons[:5].mean(0)).tolist(),
            diagnostics=item["diagnostics"]))
    weights=np.asarray(weights,dtype=np.float64)
    pooled=np.einsum("c,cpa->pa",weights,np.asarray(means))/weights.sum()
    earlier=pooled[:5].mean(0);later=pooled[5:].mean(0);delta=later-earlier
    deletions=[]
    for i,subject in enumerate(SUBJECTS[8:]):
        e=np.delete(pooled[:5],i,axis=0).mean(0) if i<5 else earlier
        l=np.delete(pooled[5:],i-5,axis=0).mean(0) if i>=5 else later
        deletions.append(dict(subject=subject,site_is_same_delete_unit=True,later_minus_earlier=(l-e).tolist()))
    leave_chrom=[]
    for c,chrom in enumerate(CHROMOSOMES):
        keep=[i for i in range(3) if i!=c];w=weights[keep]
        p=np.einsum("c,cpa->pa",w,np.asarray(means)[keep])/w.sum()
        leave_chrom.append(dict(deleted_chromosome=chrom,later_minus_earlier=(p[5:].mean(0)-p[:5].mean(0)).tolist()))
    primary={name:dict(later_minus_earlier=float(delta[CLASSES.index(name)]),
        positive_direction_observed=bool(delta[CLASSES.index(name)]>0),
        per_chromosome=[dict(chromosome=x["chromosome"],later_minus_earlier=x["later_minus_earlier"][CLASSES.index(name)]) for x in per_chrom])
        for name in ("EHG","CHG")}
    return dict(schema="H056_frozen_H045_Danish_descriptive_summary_v1",protocol_ref=PROTOCOL_REF,
        class_order=list(CLASSES),subjects=list(SUBJECTS[8:]),earlier=list(EARLIER),later=list(LATER),
        chromosome_covered_bp=weights.tolist(),pooled_person_probabilities=pooled.tolist(),
        earlier_mean=earlier.tolist(),later_mean=later.tolist(),later_minus_earlier=delta.tolist(),
        primary=primary,chromosomes=per_chrom,leave_subject_or_site_out=deletions,
        subject_site_units_identical=True,leave_chromosome_out=leave_chrom,
        probability_complement_max_abs_error=float(np.abs(pooled.sum(-1)-1).max()),
        estimation_order="span_bp_then_two_haplotypes_per_person_then_bp_pool_chromosomes_then_equal_people",
        optimization_steps=0,no_ancestry_accuracy_or_mixture_calibration=True,no_SNP_independent_pvalues=True,
        shared_imputation_and_proxies_not_independent_truth=True,
        pre_size_realdata_vs_post_size_training_domain_difference=True)

def execute_v1(ctx):
    """Same injected reader/writer/before_stage boundary as the old CPU application."""
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted")is True
        and callable(ctx["reader"].read) and callable(ctx["writer"]) and callable(ctx["before_stage"]),
        "Existing separately admitted root IO context")
    reader,writer,before=ctx["reader"],ctx["writer"],ctx["before_stage"]
    def raw(record,cap):
        require(isinstance(record,dict) and set(record)=={"path","sha256"} and type(cap)is int and cap>0,
            "Exact bounded root reference")
        data=reader.read(record)
        require(isinstance(data,bytes) and len(data)<=cap and sha(data)==record["sha256"],"Exact root-read bytes")
        return data
    def read(record):
        require(Path(record["path"]).suffix==".json","Bounded JSON metadata")
        return json.loads(raw(record,32*2**20))
    spec=read(ctx["input_ref"]);cfg=read(spec["config_ref"])
    require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["schema"]=="H056_H36_CPU_application_root_v1" and spec["root_authorizer"]=="root_agent"
        and spec["hypothesis_id"]=="H056" and spec["registered"]is True and spec["execution_authorized"]is True
        and spec["GPU_authorized"]is False and spec["optimization_authorized"]is False
        and spec["root_host_resource_enforcement_active"]is True and spec["protocol_ref"]==PROTOCOL_REF
        and spec["producer_ref"]==ctx["source_ref"]==ref(SOURCE)
        and ctx["config_ref"]==spec["config_ref"] and ctx["runtime_ref"]==spec["runtime_ref"] and ctx["cfg"]==cfg,
        "Exact original root CPU authority/source/config/runtime with no optimization")
    expiry=datetime.datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    require(expiry.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc)<expiry,"Current absolute application expiry")
    require(cfg["schema"]=="H056_H36_CPU_application_config_v1" and cfg["execution_enabled"]is True
        and cfg["protocol_ref"]==PROTOCOL_REF and cfg["checkpoint_ref"]==CHECKPOINT_REF
        and cfg["checkpoint_descriptor_ref"]==CHECKPOINT_DESCRIPTOR_REF
        and set(cfg["chromosomes"])==set(CHROMOSOMES) and cfg["CPU_threads"]==4
        and cfg["optimization_steps"]==0,"Frozen model and all three exact chromosome banks")
    keys=("output_cap_bytes","max_serialized_artifact_payload_bytes","bank_read_cap_bytes",
          "checkpoint_read_cap_bytes","episode_cap_bytes","max_host_RSS_bytes","wall_deadline_seconds")
    require(all(type(cfg[k])is int and cfg[k]>0 for k in keys)
        and cfg["max_serialized_artifact_payload_bytes"]<cfg["output_cap_bytes"]
        and cfg["checkpoint_read_cap_bytes"]<=1048576,"Explicit root count-derived finite caps; unchanged checkpoint read bound")
    before("validate_root_context",{k:cfg[k] for k in keys})
    before("serialize_complete_output_preflight",dict(required_write_bytes=cfg["max_serialized_artifact_payload_bytes"],output_cap_bytes=cfg["output_cap_bytes"]))
    protocol=read(PROTOCOL_REF);raw(ROSTER_REF,65536)
    require(protocol["hypothesis_id"]=="H056" and protocol["execution_authorized"]is False
        and tuple(protocol["subjects"])==SUBJECTS and tuple(protocol["earlier"])==EARLIER and tuple(protocol["later"])==LATER
        and protocol["frozen_predictor"]["checkpoint_ref"]==CHECKPOINT_REF,"Frozen registration is not runtime authority")
    manifest=read(cfg["source_manifest_ref"]);validate_source_manifest_v1(manifest);bindings=ctx["consumer_bindings"]
    required=[ctx["io_context_ref"],*ctx["resource_binding_refs"],spec["producer_ref"],spec["config_ref"],
        spec["runtime_ref"],cfg["source_manifest_ref"],PROTOCOL_REF,CHECKPOINT_DESCRIPTOR_REF,
        *manifest["required_metadata_refs"],*[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]]
    require(manifest["schema"]=="H056_H36_application_source_manifest_v1" and manifest["protocol_ref"]==PROTOCOL_REF
        and len(bindings)<=64 and all(Path(r["path"]).suffix in (".py",".sh",".json") and r in bindings
            and sha(raw(r,32*2**20))==r["sha256"] for r in required),"Exact source/resource/metadata closure within64 direct refs")
    runtime=read(spec["runtime_ref"])
    import importlib.metadata
    require(runtime["prefix"]==str(ROOT/".runtime/env") and runtime["CUDA_VISIBLE_DEVICES"]==""
        and runtime["packages"] and all(importlib.metadata.version(k)==v for k,v in runtime["packages"].items()),
        "Pinned existing CPU numerical runtime; no installation")
    descriptor=read(CHECKPOINT_DESCRIPTOR_REF)
    summary=read(descriptor["selection_summary_ref"]);train=read(descriptor["train_completion_ref"])
    traincfg=read(train["config_ref"])
    require(descriptor["checkpoint_ref"]==CHECKPOINT_REF==summary["arms"]["foreign_episode"]["checkpoint_ref"]
        and summary["arms"]["foreign_episode"]["selected_pass"]==30
        and train["checkpoint_groups"]["30"]["foreign_episode"]==CHECKPOINT_REF
        and train["status"]=="completed_fixed_30_pass_paired_TRAIN" and train["steps"]["foreign_episode"]==960,
        "Existing European development-selected pass30 without reselection")
    started=time.monotonic()
    old_alarm=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("H056 CPU wall deadline")))
    signal.alarm(cfg["wall_deadline_seconds"])
    def limits():
        require(os.environ.get("CUDA_VISIBLE_DEVICES")==""
            and datetime.datetime.now(datetime.timezone.utc)<expiry
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=cfg["max_host_RSS_bytes"]
            and time.monotonic()-started<=cfg["wall_deadline_seconds"],"Root CPU/expiry/RSS/deadline remains finite")
    used=0;artifacts={}
    out=ROOT/spec["output_directory"]
    require(not Path(spec["output_directory"]).is_absolute() and ".."not in Path(spec["output_directory"]).parts
        and out.resolve()==out and out.is_relative_to(ROOT),"Canonical admitted result scope")
    def write(name,data):
        nonlocal used
        require(Path(name).name==name and used+len(data)<=cfg["max_serialized_artifact_payload_bytes"],"Count-derived complete payload cap")
        limits();before("write_descriptive_artifact",dict(name=name,required_write_bytes=len(data)+8192))
        record=writer(name,data)
        require(record==dict(path=(out/name).relative_to(ROOT).as_posix(),sha256=sha(data)),"Exact exclusive original root writer result")
        used+=len(data);artifacts[name]=record;return record
    try:
        limits();before("before_frozen_checkpoint",dict(required_read_bytes=cfg["checkpoint_read_cap_bytes"]))
        import numpy as np,torch
        from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
        from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
        from research.reviews.H056_H36_implementation_20261016_v1.model_h36_v1 import H36EpisodeInterfaceV1,episode_inputs_v1
        from research.reviews.H056_H36_implementation_20261016_v1.preparation_h36_v1 import aligned_inputs_v1
        from research.reviews.H056_H36_implementation_20261016_v1.panel_h36_v1 import panel_metadata_v1
        from research.reviews.H056_H36_implementation_20261016_v1.reader_h36_v1 import bank_array_manifest_v1
        require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled()
            and not torch.is_autocast_enabled("cpu"),"Original FP32 CPU arithmetic, no AMP")
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cp=torch.load(io.BytesIO(raw(CHECKPOINT_REF,cfg["checkpoint_read_cap_bytes"])),map_location="cpu",weights_only=False)
        require(cp.get("schema")=="H045_episode_checkpoint_v1" and cp.get("hypothesis_id")=="H045" and cp.get("phase")=="train"
            and cp.get("arm")=="foreign_episode" and cp.get("seed")==101 and cp.get("pass_index")==30
            and cp.get("additional_optimizer_steps")==960 and cp.get("parameter_count")==31812
            and cp.get("model_kwargs")==traincfg["model"] and cp.get("teacher_used")is False and cp.get("optimizer_state_reset")is True
            and cp.get("scientific_refs")==train["scientific_refs"] and cp.get("input_ref")==train["input_ref"]
            and cp.get("config_ref")==train["config_ref"] and cp.get("observation_seal")==train["observations_seal"]
            and cp.get("warmstart_checkpoint_ref")==train["warmstart_checkpoint_ref"]
            and cp.get("initial_state_sha256")==train["initial_state_sha256"]["foreign_episode"]
            and cp.get("conditioned_on_H040_VAL")is False,"Exact unchanged selected checkpoint provenance")
        model=H36EpisodeInterfaceV1();expected=model.state_dict();state=cp["model_state"]
        require(set(state)==set(expected) and sum(v.numel() for v in state.values())==31812
            and all(isinstance(v,torch.Tensor) and v.device.type=="cpu" and v.dtype==torch.float32
                and v.shape==expected[k].shape and bool(torch.isfinite(v).all()) for k,v in state.items()),
            "Strict31812 finite FP32 original state values, no key/shape/weight transformation")
        model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
        del cp,state,expected
        chromosome_results={};bank_provenance=[]
        for chrom in CHROMOSOMES:
            limits();item=cfg["chromosomes"][chrom];counts=validate_counts_v1(item["counts"],graph=True)
            done=read(item["bank_completion_ref"]);wait=read(item["preparation_wait_ref"])
            require(done["schema"]=="H056_empirical_CPU_preparation_completed_v1"
                and done["status"]=="completed_observation_preparation" and done["protocol_ref"]==PROTOCOL_REF
                and done["chromosome"]==chrom and done["counts"]==counts and done["bank_ref"]==item["bank_ref"]
                and done["no_query_ancestry_truth"]is True and done["no_model_or_checkpoint_executed"]is True
                and done["no_GPU_used"]is True and done["metadata"]["feature_semantics"]==FEATURE_SEMANTICS
                and done["metadata"]["protocol_ref"]==PROTOCOL_REF
                and done["metadata"]["chromosome"]==chrom and done["metadata"]["expected_counts"]==counts
                and done["metadata"]["observation_refs"]["roster"]==item["roster_ref"]
                and done["metadata"]["no_model_or_checkpoint_executed"]is True
                and done["metadata"]["no_query_truth_read"]is True
                and wait["actual_exit_code"]==0 and wait["actual_direct_wait_observed"]is True
                and wait["process_group_drained"]is True and wait["completed_ref"]==item["bank_completion_ref"],
                "Genuine new H056 preparation completion/wait0/drain; never old recovery evidence")
            before("before_chromosome_bank",dict(chromosome=chrom,counts=counts,read_cap_bytes=cfg["bank_read_cap_bytes"]))
            saved=torch.load(io.BytesIO(raw(item["bank_ref"],cfg["bank_read_cap_bytes"])),map_location="cpu",weights_only=False)
            require(saved["schema"]=="H056_empirical_CPU_observation_bank_v1" and saved["protocol_ref"]==PROTOCOL_REF
                and saved["chromosome"]==chrom and saved["input_ref"]==done["input_ref"]
                and saved["metadata"]==done["metadata"],"Exact new observation-only bank and source metadata")
            bank=MutationRoutedWholeTreeInputsV1(**saved["bank_tensors"])
            report=validate_inputs_v1(bank,expected_haplotypes=H);t=counts["native_trees"];s=counts["sites"]
            require(bank.sample_index.shape==(t,H) and bank.genotype.shape==(s,H)
                and bank.parent.numel()==counts["vertex_occurrences"] and torch.equal(bank.query_column,torch.arange(R,H))
                and torch.equal(bank.readout_hold,torch.arange(t))
                and all(int((bank.reference_class==k).sum())==4 for k in range(4))
                and bool((bank.reference_class[R:]==-1).all())
                and bank_array_manifest_v1(bank)==done["metadata"]["model_input_tensor_manifest"],
                "Every sealed site/hold/vertex and exact H36/R16/Q20 tensor")
            roster=read(item["roster_ref"]);panel=panel_metadata_v1(roster)
            require(torch.equal(bank.reference_class,torch.tensor(panel["reference_class"],dtype=torch.long)),"Exact frozen donor-class vector, not only class counts")
            require(all(float(bank.sample_age[i])==float(torch.tensor(row["age_generations"],dtype=torch.float32))
                for i,row in enumerate(roster)),"Frozen source ages equal original FP32 input metadata")
            flags=AnonymousCladeFlagsV1.from_verified_arrays(saved["flag_node_flags"].numpy(),done["metadata"]["exact_clade_flags"])
            bank,flags=aligned_inputs_v1(dict(bank=bank,flags=flags,metadata=done["metadata"]),control="aligned")
            episodes,episode_meta=episode_inputs_v1(bank,known_flags=flags,max_payload_bytes=cfg["episode_cap_bytes"])
            before("before_scientific_forward",dict(chromosome=chrom,haplotypes=H,queries=Q,native_holds=t,optimization_steps=0))
            # Each call starts the original full bidirectional GRU once for this
            # chromosome; no artificial2Mb resets or cross-chromosome carry.
            with torch.inference_mode():
                probabilities=model(bank,episodes=episodes,control="aligned").softmax(-1).cpu().numpy()
            require(probabilities.shape==(t,Q,4) and probabilities.dtype==np.float32
                and np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1))
                and np.allclose(probabilities.sum(-1),1.,atol=1e-6,rtol=0),"All native-hold FP32 probabilities")
            coordinate=done["metadata"]["graph_projection"]["coordinate_view"]
            intervals=np.asarray(coordinate["intervals_regional_bp0"],dtype=np.float64);spans=np.diff(intervals,axis=1)[:,0]
            require(intervals.shape==(t,2) and np.isfinite(intervals).all() and np.all(spans>0)
                and np.array_equal(intervals[:-1,1],intervals[1:,0]) and intervals[0,0]>=0 and intervals[-1,1]<=END-BEGIN
                and np.array_equal(spans,bank.spans.numpy().astype(np.float64))
                and [float(intervals[0,0]),float(intervals[-1,1])]==item["genuine_regional_bp0"],
                "Exact new genuine intersected native support, no extrapolation")
            proportions=np.einsum("t,tqc->qc",spans,probabilities.astype(np.float64))/spans.sum()
            diagnostics=dict(holds=t,sites=s,constants=counts["constants"],vertex_occurrences=counts["vertex_occurrences"],
                max_probability_min=float(probabilities.max(-1).min()),max_probability_mean=float(probabilities.max(-1).mean()),
                simplex_max_abs_error=float(np.abs(probabilities.sum(-1)-1).max()),
                left_uncovered_bp=float(intervals[0,0]),right_uncovered_bp=float(END-BEGIN-intervals[-1,1]),
                all_qualified_sites_and_native_holds_retained=True,unaries_preserved=True)
            chromosome_results[chrom]=dict(haplotype_proportions=proportions,covered_bp=float(spans.sum()),
                covered_absolute_bp0=[float(intervals[0,0]+BEGIN),float(intervals[-1,1]+BEGIN)],diagnostics=diagnostics)
            stream=io.BytesIO();np.savez_compressed(stream,probabilities=probabilities,intervals_regional_bp0=intervals,
                intervals_absolute_bp0=intervals+BEGIN,query_columns=bank.query_column.numpy(),
                native_hold_ids=np.arange(t,dtype=np.int64),class_order=np.asarray(CLASSES),
                query_subject_ids=np.asarray([row["subject_id"] for row in roster[R:]]),
                query_haplotypes=np.asarray([row["haplotype"] for row in roster[R:]],dtype=np.int8))
            probability_ref=write("chr"+chrom+"_probabilities_all_native_holds.npz",stream.getvalue())
            bank_provenance.append(dict(chromosome=chrom,bank_ref=item["bank_ref"],bank_completion_ref=item["bank_completion_ref"],
                preparation_wait_ref=item["preparation_wait_ref"],roster_ref=item["roster_ref"],counts=counts,
                episode_metadata=episode_meta,probabilities_ref=probability_ref))
            del saved,bank,flags,episodes,probabilities,stream,intervals,spans
        result=summarize_probabilities_v1(chromosome_results)
        summary_ref=write("summary.json",encoded(result))
        provenance_ref=write("provenance.json",encoded(dict(schema="H056_frozen_H045_application_provenance_v1",
            protocol_ref=PROTOCOL_REF,input_ref=ctx["input_ref"],checkpoint_ref=CHECKPOINT_REF,
            checkpoint_descriptor_ref=CHECKPOINT_DESCRIPTOR_REF,source_manifest_ref=cfg["source_manifest_ref"],
            banks=bank_provenance,strict_full_state_load=True,parameter_count=31812,selected_arm="foreign_episode",
            selected_pass=30,optimization_steps=0,CPU_only=True,
            original_H045_learned_equations_and_normalization_preserved=True,
            new_H36_domain_not_runtime_qualified_by_source_review=True,
            pre_size_realdata_vs_post_size_training_domain_difference=True,
            no_query_ancestry_truth=True,accuracy_available=False,
            elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))
        completion=write("completed.json",encoded(dict(schema="H056_frozen_H045_application_completed_v1",
            status="completed_descriptive_fixed_cohort_H045_application_no_accuracy",protocol_ref=PROTOCOL_REF,
            checkpoint_ref=CHECKPOINT_REF,summary_ref=summary_ref,provenance_ref=provenance_ref,
            probability_refs=[x["probabilities_ref"] for x in bank_provenance],
            class_order=list(CLASSES),chromosomes=list(CHROMOSOMES),haplotypes=H,queries=Q,
            optimization_steps=0,query_ancestry_truth_read=False,accuracy_available=False,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
        limits();before("application_completed",dict(required_write_bytes=0))
        return completion
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)

if __name__=="__main__":
    raise SystemExit("Root must invoke execute_v1(ctx) via its existing separately admitted CPU IO route")
