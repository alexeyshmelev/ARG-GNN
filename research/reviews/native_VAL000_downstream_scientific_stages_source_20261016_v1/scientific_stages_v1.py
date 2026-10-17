"""Exact native downstream scientific consumers. No import-time execution.
ROOT supplies the existing reviewed GPU0/native or CPU supervisor context and a
separately admitted future output scope. No scheduler/reservation implementation.
"""
from pathlib import Path
import hashlib,importlib,importlib.util,json,os,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
AUTHOR_ROOT=ROOT/".runtime/author_sources/argformer_f7545be7bca0"
REFS={'author_manifest': {'path': 'research/reviews/American_C3_native_consumer_source_v1/author_source_manifest.json', 'sha256': 'bed570db438f2c4981ebf606460dc4ff72a97b148ba371e3873ab0141376736a'}, 'counts': {'path': 'experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json', 'sha256': '502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7'}, 'preparation': {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json', 'sha256': '40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b'}, 'preparation_config': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/admitted_native_paths_v1/config.json', 'sha256': 'c7c95390e12daa2216babc8ccf82d04227a2c96f4e9af8a04dfb626443fa3146'}, 'strict_native_helpers': {'path': 'research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py', 'sha256': 'f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e'}, 'cpu_native': {'path': 'src/arggnn/american_c3_native_concat_v1.py', 'sha256': 'a9de71c71baed134f829619b23ca11b910eefb63be92af24d8993e232b70c7bb'}, 'evaluator': {'path': 'src/arggnn/american_c3_evaluate_v1.py', 'sha256': '1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3'}, 'base_config': {'path': 'research/reviews/C3_bert_base_configuration_root_v1/config.json', 'sha256': '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20'}, 'contrastive_yaml': {'path': '.runtime/author_sources/argformer_f7545be7bca0/yamls/contrastive.yaml', 'sha256': '3b4f87e149a2b56d8a3efa2e35bd52ec94b61d2d3c768f15df67e82376ce4847'}, 'contrastive_author': {'path': '.runtime/author_sources/argformer_f7545be7bca0/sequence_contrastive.py', 'sha256': '7df3e0571aeb032b8ec745d49feb5d0335572a9a9594ee643603e961b4b98339'}, 'embedding_author': {'path': '.runtime/author_sources/argformer_f7545be7bca0/embeddings.py', 'sha256': 'f85c8c3b039b75ad1118d2a4904d3c77d5a253227032b9def551f09ab0144869'}, 'retrieve_author': {'path': '.runtime/author_sources/argformer_f7545be7bca0/retrieve.py', 'sha256': '520f1961c396432a875a40be253a01648e7debf1fad8c9a274a7522d3809e7ea'}}
ROW_COUNTS={"contrastive/train":8748479,"contrastive/val":972757,"query_paths/train":9003,"query_paths/val":997}
ORIGINAL_7GIB_ROOTS=("experiments/results/native_VAL000_full30mb_borrowed32_GPU0_v2","research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux")

def require(ok,message):
    if not ok:raise ValueError(message)

def full_mlm_metadata_prerequisite_v1(completion,wait,outer,*,completion_ref,wait_ref,checkpoint_ref):
    require(completion["stage"]=="mlm" and completion["parent_id"]=="american_c3_val_000"
        and completion["completed_epochs"]==5 and completion["actual_optimizer_updates"]==227990
        and completion["observed_counters_verified"] is True
        and completion["trainer_optimizer_counters"]["unique_step_values"]==[227990]
        and completion["checkpoint_optimizer_counters"]["unique_step_values"]==[227990]
        and completion["initializer_checkpoint_ref"] is None and completion["checkpoint_ref"]==checkpoint_ref
        and completion["preparation_ref"]==REFS["preparation"]
        and completion["author_source_manifest_ref"]==REFS["author_manifest"]
        and completion["query_ancestry_truth_opened"] is False,"Completed exact fresh full227990-update VAL000 MLM is mandatory")
    require(wait["returncode"]==0 and wait.get("actual_direct_worker_wait") is True
        and wait["all_owned_processes_and_groups_drained"] is True and wait["completed_ref"]==completion_ref
        and wait.get("operational_failure") is None and outer["actual_exit_code"]==0
        and outer["actual_terminal_consumed"] is True
        and outer.get("actual_wait_ref",outer.get("direct_wait_ref"))==wait_ref,
        "Actual MLM worker0, owned drain and separately consumed genuine outer0")
    return checkpoint_ref

def full_contrastive_metadata_prerequisite_v1(completion,wait,outer,*,completion_ref,wait_ref,checkpoint_ref,initializer_checkpoint_ref):
    require(completion["stage"]=="contrastive" and completion["parent_id"]=="american_c3_val_000"
        and completion["completed_epochs"]==1 and completion["actual_optimizer_updates"]==2135
        and completion["checkpoint_optimizer_counters"]["unique_step_values"]==[2135]
        and completion["initializer_checkpoint_ref"]==initializer_checkpoint_ref and completion["checkpoint_ref"]==checkpoint_ref
        and completion["author_source_manifest_ref"]==REFS["author_manifest"]
        and completion["query_ancestry_truth_opened"] is False,"Actual full same-parent learned-head contrastive checkpoint required")
    require(wait["returncode"]==0 and wait.get("actual_direct_worker_wait") is True
        and wait["all_owned_processes_and_groups_drained"] is True and wait["completed_ref"]==completion_ref
        and wait.get("operational_failure") is None and outer["actual_exit_code"]==0
        and outer["actual_terminal_consumed"] is True
        and outer.get("actual_wait_ref",outer.get("direct_wait_ref"))==wait_ref,
        "Actual full contrastive worker0/drain and genuine consumed outer0")
    return checkpoint_ref

def canonical_actual_wait_metadata_v1(raw,raw_ref):
    require(raw.get("returncode")==0 and raw.get("actual_direct_worker_wait") is True
        and raw.get("all_owned_processes_and_groups_drained") is True
        and raw.get("operational_failure") is None and raw.get("completed_ref") is not None
        and raw.get("group_observations") and all(r.get("absent") is True for r in raw["group_observations"]),
        "No unknown/missing wait/group observation can become a legacy success field")
    return dict(raw,actual_original_wait_ref=raw_ref,actual_direct_child_wait=True,process_group_ESRCH=True,
        child_completion_ref=raw["completed_ref"],child_input_ref=raw["input_ref"],
        metadata_aliases_only_no_new_OS_observation=True)

def _context_v1(ctx,request,phase,*,GPU):
    require(request.get("schema")=="native_VAL000_downstream_scientific_phase_v1"
        and request.get("execution_authorized") is True and request.get("phase")==phase
        and request.get("parent_id")=="american_c3_val_000"
        and request.get("separate_future_storage_admission_ref") is not None
        and request.get("original_MLM_7GiB_scope_extended") is False,
        "Exact active phase with separate future storage, never broaden original7GiB")
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and callable(ctx.get("validate_phase_authority_v1")) and callable(ctx.get("fast_guard"))
        and callable(ctx.get("reference")) and callable(getattr(ctx.get("writer"),"save",None)),
        "Actual ROOT existing reviewed supervisor/guard/writer context required")
    ctx["validate_phase_authority_v1"](request)
    require(request.get("ROOT_context_source_ref") is not None
        and ctx.get("context_source_ref")==request["ROOT_context_source_ref"],"Exact separately reviewed actual ROOT context source binding")
    directory=Path(ctx["output_directory"])
    require(directory.resolve()==directory and directory.is_relative_to(ROOT)
        and all(not directory.is_relative_to(ROOT/p) for p in ORIGINAL_7GIB_ROOTS),"Separate actual future output scope")
    require(directory.is_dir() and not (directory/"completed.json").exists()
        and Path(ctx["writer"].path).resolve()==directory,"Exact fresh existing supervisor-owned output and writer")
    if GPU:
        require(ctx.get("physical_gpu")==0 and callable(ctx.get("verify_native_imports_v1")),
            "Only current user-authorized GPU0, exact post-call native import validation")
    else:require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit CPU-only serialization/seal")
    ctx["fast_guard"]("before_native_downstream_stage")
    return directory

def _native_v1(ctx):
    require(callable(ctx.get("prepare_native_runtime_v1")),"Existing exact native inventory/title/lease/device preparation")
    # Existing ROOT context installs title/lease before scientific imports and
    # configures actual GPU0 allocator cap before its first CUDA tensor.
    prepared=ctx["prepare_native_runtime_v1"]()
    require(prepared["physical_gpu"]==0 and prepared["runtime_inventory_passed"] is True
        and prepared["one_actual_visible_CUDA_UUID"] is True,"Exact source-bound native runtime and actual device")
    path=ROOT/REFS["strict_native_helpers"]["path"]
    require(hashlib.sha256(path.read_bytes()).hexdigest()==REFS["strict_native_helpers"]["sha256"],"Unchanged strict native helper source")
    module_spec=importlib.util.spec_from_file_location("_VAL000_reviewed_native_scientific_helpers",path)
    helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    sys.path.insert(0,str(AUTHOR_ROOT))
    return prepared,helper

def contrastive_v1(ctx,request):
    directory=_context_v1(ctx,request,"contrastive",GPU=True)
    checkpoint=full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])
    prepared,helper=_native_v1(ctx);torch=prepared["torch"];OmegaConf=prepared["OmegaConf"]
    cfg=helper._io_config("contrastive",data_root=ROOT/"experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/contrastive",
        checkpoint=ROOT/checkpoint["path"],directory=directory,base_config=ROOT/REFS["base_config"]["path"],OmegaConf=OmegaConf,author_root=AUTHOR_ROOT)
    r=OmegaConf.to_container(cfg,resolve=True)
    require((r["max_duration"],r["global_train_batch_size"],r["device_train_microbatch_size"],r["precision"],r["seed"])==("1ep",4096,4096,"fp32",36)
        and r.get("train_subset_num_batches",-1)==-1 and r["max_seq_len"]==128
        and r["model"]["contrastive"]["temperature"]==.30
        and r["model"]["contrastive"]["normalize"] is True and r["model"]["contrastive"]["similarity"]=="dot"
        and r["model"]["contrastive"]["use_class_weights"] is True and "train_loader" not in r["model"]
        and r["save_interval"]=="20ba" and r["save_num_checkpoints_to_keep"]==-1,
        "Full unchanged author contrastive objective, uniform fallback, batch and retained-checkpoint policy")
    require(ctx["max_checkpoint_files"]>=107,"All106periodic and final author checkpoints retained")
    ctx["authenticate_native_inputs_once_v1"](REFS["preparation"],REFS["counts"])
    OmegaConf.save(cfg,directory/"resolved_io_config.yaml",resolve=True)
    module=importlib.import_module("sequence_contrastive")
    require(Path(module.__file__).resolve()==AUTHOR_ROOT/"sequence_contrastive.py","Exact author contrastive origin")
    helper._verify_author_import_origins(AUTHOR_ROOT)
    module.train(cfg)  # Unchanged author implementation, no custom optimizer/loss/head.
    final=helper._final_checkpoint(directory/"checkpoints",epochs=1,updates=2135,
        max_files=ctx["max_checkpoint_files"],max_bytes=ctx["max_checkpoint_bytes"],torch=torch,reference=ctx["reference"])
    ctx["verify_native_imports_v1"]()
    ctx["fast_guard"]("after_full_native_contrastive")
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage="contrastive",parent_id="american_c3_val_000",active_input_ref=ctx["input_ref"],child_input_ref=ctx["input_ref"],
        preparation_ref=REFS["preparation"],counts_ref=REFS["counts"],author_source_manifest_ref=REFS["author_manifest"],
        initializer_checkpoint_ref=checkpoint,resolved_config_ref=ctx["reference"](directory/"resolved_io_config.yaml"),
        uniform_class_weight_fallback_preserved=True,training_config_plumbing_correction=False,
        query_ancestry_truth_opened=False,author_code_modified=False,actual_saved_optimizer_updates=2135,
        all_author_interval_checkpoints_retained=True,learned_pooling_head_fitted_by_original_author_objective=True,**final))

def embedding_v1(ctx,request):
    directory=_context_v1(ctx,request,"embedding",GPU=True)
    view,split=request["view"],request["split"]
    require(view in ("contrastive","query_paths") and split in ("train","val"),"Exactly one full native view/split")
    mlm_checkpoint=full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])
    checkpoint=full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=mlm_checkpoint,
        **ctx["contrastive_prerequisite_metadata"])
    prepared,helper=_native_v1(ctx);OmegaConf=prepared["OmegaConf"]
    cfg=helper._io_config("embedding",data_root=ROOT/"experiments/results/American_C3_val000_full30mb_shared_v1/native_paths"/view,
        checkpoint=None,directory=directory,base_config=ROOT/REFS["base_config"]["path"],OmegaConf=OmegaConf,author_root=AUTHOR_ROOT)
    config_path=directory/"resolved_io_config.yaml";OmegaConf.save(cfg,config_path,resolve=True)
    module=importlib.import_module("embeddings");helper._verify_author_import_origins(AUTHOR_ROOT)
    model,loaded_cfg=module.load_checkpoint_model(str(config_path),str(ROOT/checkpoint["path"]))
    vectors=module.embed_arg_dataset(model,loaded_cfg,output_path=str(directory/"embeddings.pt"),batch_size=256,
        pooling_strategy="head_norm",layer_index=None,max_samples=0,is_classification=True,
        include_splits=split,use_split_labels=(view=="contrastive"))
    require(tuple(vectors.shape)==(ROW_COUNTS[view+"/"+split],768),"Every exact full path row, no extraction cap")
    ctx["verify_native_imports_v1"]()
    ctx["fast_guard"]("after_native_learned_head_embedding")
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage="embedding",parent_id="american_c3_val_000",active_input_ref=ctx["input_ref"],child_input_ref=ctx["input_ref"],
        view=view,split=split,embedding_ref=ctx["reference"](directory/"embeddings.pt"),checkpoint_ref=checkpoint,
        raw_dataset_ref=ctx["prepared_metadata"]["native_dataset_refs"][view+"/"+split],
        row_order_ref=ctx["prepared_metadata"]["full_path_id_order_refs"][view+"/"+split],
        pooling_strategy="head_norm",max_samples=0,no_random_split_interleaving=True,query_ancestry_truth_opened=False))

def retrieval_v1(ctx,request):
    directory=_context_v1(ctx,request,"retrieve",GPU=True)
    combined=ctx["validated_full_row_preserving_combination_v1"]()
    prepared,helper=_native_v1(ctx)
    corpus=combined["combined_embeddings_refs"]["contrastive"];queries=combined["combined_embeddings_refs"]["query_paths"]
    module=importlib.import_module("retrieve");helper._verify_author_import_origins(AUTHOR_ROOT)
    sys.argv=[str(AUTHOR_ROOT/"retrieve.py"),"--corpus",str(ROOT/corpus["path"]),"--queries",str(ROOT/queries["path"]),
        "--out",str(directory/"retrieval.pt"),"--topk","1","--q_batch","1024","--e_chunk","100000"]
    module.main()  # Original normalization, chunk order, native torch.topk/ties.
    ctx["verify_native_imports_v1"]()
    ctx["fast_guard"]("after_native_top1_retrieval")
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage="retrieve",parent_id="american_c3_val_000",active_input_ref=ctx["input_ref"],child_input_ref=ctx["input_ref"],
        retrieval_ref=ctx["reference"](directory/"retrieval.pt"),corpus_ref=corpus,queries_ref=queries,
        topk=1,q_batch=1024,e_chunk=100000,native_retrieval_normalization_unchanged=True,query_ancestry_truth_opened=False))

def reviewed_CPU_body_v1(ctx,request,spec,cfg):
    phase=request["phase"];_context_v1(ctx,request,phase,GPU=False)
    require(phase in ("concat","prediction_seal"),"Only original row-preserving concat or truth-free prediction seal")
    module=importlib.import_module("arggnn.american_c3_native_concat_v1")
    name="concatenate_embeddings_v1" if phase=="concat" else "prediction_seal_v1";original=getattr(module,name)
    ns=dict(original.__globals__)
    # Explicit metadata/context adaptation of the reviewed TRAIN000 consumer to
    # registered VAL000, not a modification of its tensor or identity operation.
    for key in ("preparation_v1","counts_v1","fit_completion_v1","metadata"):
        require(callable(ctx.get("validated_VAL000_"+key)),"ROOT must bind and review exact VAL000 lineage/wait metadata callback: "+key)
        ns[key]=ctx["validated_VAL000_"+key]
    adapted=types.FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__)
    return adapted(spec,cfg,ctx)
