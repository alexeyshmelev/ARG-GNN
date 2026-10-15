"""Guard-agnostic root-admitted descriptive H018/H045 empirical application.

Import does not load scientific payloads. Root injects its separately pinned
admitted IO context; original model arithmetic is unchanged.
"""
from pathlib import Path
import datetime,hashlib,html,io,json,os,resource,signal,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H018_H045_empirical_application_source_v1/inference_v2.py"
CAP,RSS,DEADLINE=20000000,2000000000,1800
CLASSES=("ANA","CHG","WHG","EHG")

def need(ok,message):
    if not ok:raise ValueError(message)

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p

def ref(value):
    p=local(value);return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def pca_v1(genotypes):
    """Descriptive centered per-site categorical-allele PCA on24 query haplotypes.

    Keep every original site and its original allele-code category; each site
    has four indicator columns. Missing calls use observed query allele means.
    No scaling, ancestry labels, filtering, phenotype or trained weights enter.
    """
    import numpy as np
    g=np.asarray(genotypes)
    need(g.ndim==2 and len(g)==24 and np.isin(g,(-1,0,1,2,3)).all(),"24 original integer-coded query haplotypes")
    called=g>=0;one=(g[...,None]==np.arange(4)).astype(np.float64)
    n=called.sum(0);mean=np.divide(one.sum(0),n[:,None],out=np.full((g.shape[1],4),.25),where=n[:,None]>0)
    features=np.where(called[...,None],one,mean[None]).reshape(len(g),-1)
    center=features.mean(0);u,s,vt=np.linalg.svd(features-center,full_matrices=False)
    for j in range(len(s)):
        pivot=int(np.argmax(np.abs(vt[j])))
        if vt[j,pivot]<0:vt[j]*=-1;u[:,j]*=-1
    variance=s*s/(len(g)-1);total=float(variance.sum())
    ratio=variance/total if total>0 else np.zeros_like(variance)
    return dict(scores=u[:,:2]*s[:2],components=vt[:2],feature_mean=center,
        explained_variance=variance,explained_variance_ratio=ratio,missing_query_calls=int((~called).sum()),
        original_sites=g.shape[1],retained_constant_indicator_columns=int(np.count_nonzero(np.ptp(features,axis=0)==0)))

def svg_v1(rows,proportions,pca):
    colors=("#3182bd","#e6550d","#31a354","#9467bd");esc=html.escape
    bars=['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="720" viewBox="0 0 920 720">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="17">Model ancestry probabilities over genuine covered bp</text>',
        '<text x="20" y="49" font-family="sans-serif" font-size="12">Descriptive prediction; no empirical ancestry truth or accuracy</text>']
    for k,c in enumerate(CLASSES):
        bars.append('<rect x="%d" y="64" width="12" height="12" fill="%s"/><text x="%d" y="75" font-size="12">%s</text>'%(20+140*k,colors[k],37+140*k,c))
    for i,(row,values) in enumerate(zip(rows,proportions)):
        y=95+i*24;x=200.
        bars.append('<text x="20" y="%s" font-family="sans-serif" font-size="11">%s/h%s</text>'%(y+12,esc(row["subject_id"]),row["haplotype"]))
        for c,value in zip(colors,values):
            width=float(value)*650
            bars.append('<rect x="%.4f" y="%d" width="%.4f" height="17" fill="%s"/>'%(x,y,width,c));x+=width
    bars.append('</svg>')
    points=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="650" viewBox="0 0 900 650">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="17">Observed-genotype PCA:24 query haplotypes /12 subjects</text>',
        '<text x="20" y="50" font-family="sans-serif" font-size="12">Centered allele indicators; descriptive only; no ancestry or demographic validation</text>']
    xy=pca["scores"];lo=xy.min(0);hi=xy.max(0)
    def coord(v,k,start,width):
        return start+width*(float(v)-float(lo[k]))/(float(hi[k]-lo[k]) or 1.)
    points+=['<path d="M 80 80 V 570 H 850" fill="none" stroke="#444"/>',
        '<text x="380" y="620" font-family="sans-serif">PC1 (%.2f%%)</text>'%(100*pca["explained_variance_ratio"][0]),
        '<text x="15" y="95" font-family="sans-serif">PC2 (%.2f%%)</text>'%(100*pca["explained_variance_ratio"][1])]
    for i,(row,xyrow) in enumerate(zip(rows,xy)):
        x=coord(xyrow[0],0,110,680);y=550-coord(xyrow[1],1,0,430)
        points.append('<circle cx="%.3f" cy="%.3f" r="4" fill="%s"/><text x="%.3f" y="%.3f" font-family="sans-serif" font-size="10">%s/h%s</text>'%(
            x,y,"#334155",x+6,y-4,esc(row["subject_id"]),row["haplotype"]))
    points.append('</svg>')
    return "\n".join(bars).encode(), "\n".join(points).encode()

def execute_v1(ctx):
    """Root injects reader.read(ref), writer(name,bytes), before_stage(stage,plan).

    The separately pinned IO context owns actual guard validation, exclusive
    fsynced charged writes and outer Popen2GB/1800s monitoring. This scientific
    entrypoint neither selects nor mutates a storage accounting policy.
    """
    need(ctx["root_authorizer"]=="root_agent" and ctx["admitted"] is True
        and callable(ctx["reader"].read) and callable(ctx["writer"]) and callable(ctx["before_stage"]),
        "Explicit admitted root IO context required")
    reader,writer,before=ctx["reader"],ctx["writer"],ctx["before_stage"]
    before("validate_root_context",dict(output_cap_bytes=CAP,max_host_RSS_bytes=RSS,wall_deadline_seconds=DEADLINE))
    def raw(record,cap):
        need(set(record)=={"path","sha256"},"Exact root-read reference")
        data=reader.read(record)
        need(isinstance(data,bytes) and len(data)<=cap and hashlib.sha256(data).hexdigest()==record["sha256"],
            "Exact bounded root-read bytes")
        return data
    def read(record):
        need(Path(record["path"]).suffix==".json","Only explicit JSON metadata")
        return json.loads(raw(record,1048576))
    def payload(record,cap):
        need(Path(record["path"]).suffix==".pt","Original bank/checkpoint payload only")
        return raw(record,cap)
    spec=read(ctx["input_ref"]);cfg=read(spec["config_ref"])
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["schema"]=="H018_H045_empirical_CPU_application_root_v1"
        and spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False and spec["optimization_authorized"] is False
        and spec["root_host_resource_enforcement_active"] is True and spec["producer_ref"]==ref(SOURCE)
        and cfg["execution_enabled"] is True and cfg["output_cap_bytes"]==CAP
        and cfg["max_host_RSS_bytes"]==RSS and cfg["wall_deadline_seconds"]==DEADLINE,
        "Explicit separate root CPU application authority and outer RSS/deadline monitoring")
    expiry=datetime.datetime.fromisoformat(spec["expires_at_utc"].replace("Z","+00:00"))
    need(expiry.tzinfo is not None and datetime.datetime.now(datetime.timezone.utc)<expiry,"Unexpired root application input")
    need(ctx["source_ref"]==spec["producer_ref"] and ctx["config_ref"]==spec["config_ref"]
        and ctx["runtime_ref"]==spec["runtime_ref"] and ctx["cfg"]==cfg,
        "Root IO context binds actual producer/config/runtime")
    manifest=read(cfg["source_manifest_ref"]);bindings=ctx["consumer_bindings"]
    required=[ctx["io_context_ref"],*ctx["resource_binding_refs"],spec["producer_ref"],spec["config_ref"],
        spec["runtime_ref"],cfg["source_manifest_ref"],*manifest["required_metadata_refs"],
        *[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]]
    need(len(bindings)<=64 and all(r in bindings and hashlib.sha256(raw(r,33554432)).hexdigest()==r["sha256"] for r in required),
        "Every actual IO/resource/source/runtime/checkpoint-descriptor and provenance ref directly bound")
    runtime=read(spec["runtime_ref"])
    import importlib.metadata
    def version(name):
        try:return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:return None
    need(runtime["prefix"]==str(ROOT/".runtime/env") and runtime["CUDA_VISIBLE_DEVICES"]==""
        and all(version(k)==v for k,v in runtime["packages"].items()),"Exact bound existing CPU runtime metadata")
    before("before_existing_payload_reads",dict(bank_read_cap_bytes=33554432,checkpoint_read_cap_bytes=1048576,
        output_cap_bytes=CAP,max_host_RSS_bytes=RSS,wall_deadline_seconds=DEADLINE))
    descriptor=read(cfg["checkpoint_descriptor_ref"]);done=read(cfg["bank_completion_ref"])
    summary=read(descriptor["selection_summary_ref"]);train=read(descriptor["train_completion_ref"])
    traincfg=read(train["config_ref"]);recovery=read(done["completion_only_recovery_authority_ref"])
    wait=read(recovery["original_failed_wait_ref"])
    need(done["schema"]=="H018_empirical_CPU_preparation_completed_v1" and done["status"]=="completed_with_root_metadata_recovery"
        and done["bank_ref"]==cfg["bank_ref"] and done["original_preparation_process_returncode"]==wait["returncode"]==1
        and wait["actual_direct_Popen_wait"] is True and done["no_query_ancestry_truth"] is True
        and done["no_model_or_checkpoint_executed"] is True and done["no_GPU_used"] is True
        and recovery["new_preparation_or_model_execution_authorized"] is False
        and done["metadata"]["original_strict_genotype_export_gate_stays_failed"] is True
        and done["metadata"]["feature_semantics"]["classes"]==list(CLASSES),"Original childexit1 and completion-only recovery preserved; native GT gate stays failed")
    checkpoint=summary["arms"]["foreign_episode"]["checkpoint_ref"]
    need(checkpoint==descriptor["checkpoint_ref"]==cfg["checkpoint_ref"]
        and summary["arms"]["foreign_episode"]["selected_pass"]==30
        and train["checkpoint_groups"]["30"]["foreign_episode"]==checkpoint
        and train["status"]=="completed_fixed_30_pass_paired_TRAIN"
        and train["steps"]["foreign_episode"]==960,"Exact currently selected H045 foreign_episode checkpoint")
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    old_alarm=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("CPU1800s wall deadline")))
    signal.alarm(DEADLINE);started=time.monotonic()
    def limits():
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RSS and time.monotonic()-started<=DEADLINE,"CPU application RSS/deadline")
    limits()
    import numpy as np,torch
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1,validate_inputs_v1
    from arggnn.anonymous_clade_flag_control_v1 import AnonymousCladeFlagsV1
    from research.reviews.H018_empirical_reader_source_v1.preparation_v1 import aligned_inputs_v1
    from research.reviews.H018_H045_empirical_application_source_v1.variable_panel_model_v1 import VariablePanelEpisodeInterfaceV1,episode_inputs_v1
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    saved=torch.load(io.BytesIO(payload(cfg["bank_ref"],33554432)),map_location="cpu",weights_only=False)
    need(saved["schema"]=="H018_empirical_CPU_observation_bank_v1" and saved["input_ref"]==done["input_ref"],"Exact existing prepared observation bank; no preparation rerun")
    bank=MutationRoutedWholeTreeInputsV1(**saved["bank_tensors"])
    report=validate_inputs_v1(bank,expected_haplotypes=40)
    need(bank.sample_index.shape==(355,40) and bank.genotype.shape==(2074,40)
        and torch.equal(bank.query_column,torch.arange(16,40)) and torch.equal(bank.readout_hold,torch.arange(355))
        and all(int((bank.reference_class==k).sum())==4 for k in range(4))
        and bool((bank.reference_class[16:]==-1).all()),"Every355 hold/2074 site/40 haplotype/24 original query preserved")
    flags=AnonymousCladeFlagsV1.from_verified_arrays(saved["flag_node_flags"].numpy(),done["metadata"]["exact_clade_flags"])
    bank,flags=aligned_inputs_v1(dict(bank=bank,flags=flags,metadata=done["metadata"]),control="aligned")
    episodes,episode_meta=episode_inputs_v1(bank,known_flags=flags,max_payload_bytes=1000000)
    limits();before("before_selected_checkpoint",dict(planning_allowance_bytes=1048576))
    cp=torch.load(io.BytesIO(payload(checkpoint,1048576)),map_location="cpu",weights_only=False)
    need(cp.get("schema")=="H045_episode_checkpoint_v1" and cp.get("hypothesis_id")=="H045" and cp.get("phase")=="train"
        and cp.get("arm")=="foreign_episode" and cp.get("seed")==101 and cp.get("pass_index")==30
        and cp.get("additional_optimizer_steps")==960 and cp.get("parameter_count")==31812
        and cp.get("model_kwargs")==traincfg["model"] and cp.get("teacher_used") is False and cp.get("optimizer_state_reset") is True
        and cp.get("scientific_refs")==train["scientific_refs"] and cp.get("input_ref")==train["input_ref"]
        and cp.get("config_ref")==train["config_ref"] and cp.get("observation_seal")==train["observations_seal"]
        and cp.get("warmstart_checkpoint_ref")==train["warmstart_checkpoint_ref"]
        and cp.get("initial_state_sha256")==train["initial_state_sha256"]["foreign_episode"]
        and cp.get("conditioned_on_H040_VAL") is False,"Original selected checkpoint state/provenance, no retraining or new selection")
    model=VariablePanelEpisodeInterfaceV1();expected=model.state_dict();state=cp["model_state"]
    need(set(state)==set(expected) and all(isinstance(v,torch.Tensor) and v.device.type=="cpu" and v.dtype==torch.float32
        and v.shape==expected[k].shape and bool(torch.isfinite(v).all()) for k,v in state.items()),"Strict complete31812 finite FP32 state, no key or weight transformation")
    model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
    before("before_scientific_forward",dict(haplotypes=40,queries=24,native_holds=355,optimization_steps=0))
    with torch.inference_mode():probabilities=model(bank,episodes=episodes,control="aligned").softmax(-1).cpu().numpy()
    need(probabilities.shape==(355,24,4) and np.isfinite(probabilities).all()
        and np.allclose(probabilities.sum(-1),1.,atol=1e-6,rtol=0),"Complete FP32 native-hold probabilities")
    limits()
    coordinate=done["metadata"]["graph_projection"]["coordinate_view"]
    intervals=np.asarray(coordinate["intervals_regional_bp0"],dtype=np.float64);spans=intervals[:,1]-intervals[:,0]
    need(intervals.shape==(355,2) and np.array_equal(intervals[:-1,1],intervals[1:,0])
        and np.array_equal(spans,bank.spans.numpy().astype(np.float64))
        and intervals[0,0]==427 and intervals[-1,1]==9997139,"Exact genuine9996712bp support, no extrapolation or2Mb resets")
    proportions=np.einsum("t,tqc->qc",spans,probabilities.astype(np.float64))/spans.sum()
    roster=read(cfg["roster_ref"]);rows=[roster[int(i)] for i in bank.query_column.tolist()]
    need(len(roster)==40 and all(row["role"]=="query" and row["observed_reference_class"] is None
        and row["query_ancestry_truth"] is None and row["sample_index"]==16+j for j,row in enumerate(rows)),"Original24 query haplotypes and observation-only IDs")
    pca=pca_v1(bank.genotype[:,bank.query_column].numpy().T)
    bar_svg,pca_svg=svg_v1(rows,proportions,pca)
    limits();out=local(spec["output_directory"])
    before("serialize_descriptive_outputs",dict(required_write_bytes=6000000,output_cap_bytes=CAP))
    used=0
    def write(name,raw):
        nonlocal used
        need(used+len(raw)<=5000000,"Compact complete empirical outputs within5MB payload/20MB allocation")
        before("write_descriptive_artifact",dict(name=name,required_write_bytes=len(raw)+8192))
        record=writer(name,raw)
        need(record==dict(path=(out/name).relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest()),
            "Actual root writer returned exact output path/SHA")
        used+=len(raw);return record
    def jwrite(name,value):return write(name,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    def npz(name,**arrays):
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);return write(name,stream.getvalue())
    prob_ref=npz("probabilities_all355_holds.npz",probabilities=probabilities,intervals_regional_bp0=intervals,
        intervals_absolute_bp0=intervals+20000000,query_columns=bank.query_column.numpy(),class_order=np.asarray(CLASSES),
        readout_positions_regional_bp0=intervals.mean(1),native_hold_ids=np.arange(355,dtype=np.int64))
    columns=["sample_index","subject_id","haplotype","age_calBP1950","ANA","CHG","WHG","EHG","PC1","PC2"]
    lines=["\t".join(columns)]
    for i,row in enumerate(rows):
        values=[row["sample_index"],row["subject_id"],row["haplotype"],row["age_mean_cal_BP"],*proportions[i],*pca["scores"][i]]
        lines.append("\t".join(str(v) if isinstance(v,(str,int)) else format(float(v),".10g") for v in values))
    table_ref=write("query_bpweighted_proportions_and_PCA.tsv",("\n".join(lines)+"\n").encode())
    pca_ref=npz("query_genotype_PCA.npz",scores=pca["scores"],components=pca["components"],feature_mean=pca["feature_mean"],
        explained_variance=pca["explained_variance"],explained_variance_ratio=pca["explained_variance_ratio"],
        query_columns=bank.query_column.numpy())
    bars_ref=write("bpweighted_ancestry_proportions.svg",bar_svg);plot_ref=write("query_genotype_PCA.svg",pca_svg)
    provenance=jwrite("provenance.json",dict(schema="H018_H045_empirical_application_provenance_v1",
        bank_completion_ref=cfg["bank_completion_ref"],bank_ref=cfg["bank_ref"],checkpoint_descriptor_ref=cfg["checkpoint_descriptor_ref"],
        checkpoint_ref=checkpoint,selection_summary_ref=descriptor["selection_summary_ref"],selected_arm="foreign_episode",selected_pass=30,
        training_updates=960,application_optimizer_updates=0,parameter_count=31812,strict_full_state_load=True,
        variable_panel_IO_adaptation=True,exact_original_fixed_panel_execution=False,
        original_H045_learned_arithmetic_state_keys_GRU_and_head_preserved=True,original_H018_aligned_interface_used=True,
        class_order=list(CLASSES),donor_semantics=done["metadata"]["feature_semantics"],
        bank_preparation_rerun=False,episode_membership_only_built_in_RAM_from_existing_bank=True,
        episode_metadata=episode_meta,original_preparation_child_returncode=1,
        original_preparation_wait_ref=recovery["original_failed_wait_ref"],completion_only_recovery_authority_ref=done["completion_only_recovery_authority_ref"],
        original_metadata_serialization_failure=done["original_metadata_serialization_failure"],
        original_native_GT_export_gate_stays_failed=True,original_VCF_feature_and_node_edge_only_Relate_semantics_preserved=True,
        covered_regional_bp0=[427,9997139],covered_absolute_bp0=[20000427,29997139],covered_bp=float(spans.sum()),
        no_full10Mb_fill_or_extrapolation=True,empirical355_hold_history_vs2Mb_training_domain_shift=True,
        PCA="centered_original_query_genotype_four_allele_indicators_all2074_sites; mean-impute_missing; unscaled_SVD; deterministic_loading_sign",
        PCA_missing_query_calls=pca["missing_query_calls"],PCA_unit="24 haplotypes from12 subjects; not24 independent subjects",
        descriptive_only=True,query_ancestry_truth_read=False,accuracy_available=False,calibration_or_demographic_validation=False,
        source_manifest_ref=cfg["source_manifest_ref"],io_context_ref=ctx["io_context_ref"],
        actual_resource_binding_refs=ctx["resource_binding_refs"],measured_peak_worker_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        elapsed_seconds=time.monotonic()-started))
    completion=jwrite("completed.json",dict(schema="H018_H045_empirical_descriptive_predictions_completed_v1",
        status="completed_descriptive_model_application_no_accuracy",probabilities_ref=prob_ref,probability_shape=[355,24,4],
        query_proportions_and_PCA_table_ref=table_ref,genotype_PCA_ref=pca_ref,proportions_plot_ref=bars_ref,PCA_plot_ref=plot_ref,
        provenance_ref=provenance,checkpoint_ref=checkpoint,bank_ref=cfg["bank_ref"],class_order=list(CLASSES),
        covered_bp=float(spans.sum()),query_ancestry_truth_read=False,accuracy_available=False,optimization_steps=0,
        variable_panel_IO_adaptation=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    before("application_completed",dict(required_write_bytes=0));signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
    return completion

if __name__=="__main__":
    raise SystemExit("Root must invoke execute_v1(ctx) through its separately pinned admitted IO context")
