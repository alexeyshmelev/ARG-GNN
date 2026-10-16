"""Separate native downstream I/O option. No execution or science import on load.
Keep author checkpoint events, rotate only new old interval files, retain final.
Re-use complete row-preserving concat in RAM and original retrieval arithmetic.
ROOT must supply a separately admitted existing supervisor/context, never MLM7GiB.
"""
from pathlib import Path
import ast,copy,hashlib,importlib,importlib.util,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE='research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/efficient_downstream_v1.py'
REFS={'original_stages': {'path': 'research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py', 'sha256': '455704b88e9247021c09566609801838f88a118420cde856e487aaef4cee30da'}, 'original_stages_handoff': {'path': 'research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/source_handoff.v1.json', 'sha256': 'e4056a2c500c4bfb0817252f89b95e7e9a941496d20c4d39bd7c9c323b7891fe'}, 'concat': {'path': 'src/arggnn/american_c3_native_concat_v1.py', 'sha256': 'a9de71c71baed134f829619b23ca11b910eefb63be92af24d8993e232b70c7bb'}, 'evaluator': {'path': 'src/arggnn/american_c3_evaluate_v1.py', 'sha256': '1ada0ab40fe9f198b89a1953bb7c0688e68f1dd44b34dc118600ac25eefc70d3'}, 'author_train': {'path': '.runtime/author_sources/argformer_f7545be7bca0/sequence_contrastive.py', 'sha256': '7df3e0571aeb032b8ec745d49feb5d0335572a9a9594ee643603e961b4b98339'}, 'author_embeddings': {'path': '.runtime/author_sources/argformer_f7545be7bca0/embeddings.py', 'sha256': 'f85c8c3b039b75ad1118d2a4904d3c77d5a253227032b9def551f09ab0144869'}, 'author_retrieval': {'path': '.runtime/author_sources/argformer_f7545be7bca0/retrieve.py', 'sha256': '520f1961c396432a875a40be253a01648e7debf1fad8c9a274a7522d3809e7ea'}, 'author_yaml': {'path': '.runtime/author_sources/argformer_f7545be7bca0/yamls/contrastive.yaml', 'sha256': '3b4f87e149a2b56d8a3efa2e35bd52ec94b61d2d3c768f15df67e82376ce4847'}, 'checkpoint_saver': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/callbacks/checkpoint_saver.py', 'sha256': '2795b5ca31d7d691cfedd486c81b9cd94df554142761328b86133f0440af8b2f'}, 'checkpoint_serialization': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/utils/checkpoint.py', 'sha256': 'f7acc3f1010e74bb8fc692ca95163c0d6e13e4bb8242fdd3581fe396eb688039'}, 'composer_state': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/core/state.py', 'sha256': '7ac71704e2d4c5d6680c4510d253e180aac64d7f740308260f6fa5a4fcf7b33d'}, 'RNG_getters': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/utils/reproducibility.py', 'sha256': '2f95c7e95734626e882a32f84ef606876ce0426f03bffd0ac6fbcd2d1c66d46c'}, 'interval_scheduler': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/utils/misc.py', 'sha256': 'e87c7e1bf9e9d94f8f2ac42525f78e3f82a7cbfc0cd7e46341de763bece95e69'}, 'composer_trainer': {'path': '.runtime/conda/envs/argformer_native_v1/lib/python3.11/site-packages/composer/trainer/trainer.py', 'sha256': 'd0c1d6cc233573bc6b9ce6256641426d8f06ca10988cf67eb9a71491ce1226fb'}, 'counts': {'path': 'experiments/results/American_C3_native_val000_full30mb_counts980_namespace_v1/completed.json', 'sha256': '502c111b443c46eea789dc03ee3ff8b6454546ce6e86e3f9fcddba25e82489a7'}, 'preparation': {'path': 'experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/completed.json', 'sha256': '40a3b588d0aa2b474c3f4aab492e34feb0a753a0f8cb73c20d9c0059d8a8c16b'}}
IO_POLICY_REF={'path': 'research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/io_policy.INACTIVE.v1.json', 'sha256': '76011c718b4c41f79643b81faf55cacf2d65f549744afb0fc6f3331c3a851b2d'}

def require(ok,message):
    if not ok:raise ValueError(message)

def source_bytes(record):
    require(type(record) is dict and set(record)=={"path","sha256"},"Exact immutable source/metadata reference")
    path=ROOT/record["path"]
    require(path.resolve()==path and path.is_relative_to(ROOT) and not path.is_symlink()
        and path.is_file() and path.stat().st_size<=32*2**20,"Bounded canonical source/compact metadata")
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact pinned source bytes")
    return raw

def _base_v1():
    for record in REFS.values():source_bytes(record)
    record=REFS["original_stages"]
    spec=importlib.util.spec_from_file_location("_VAL000_original_downstream_scientific_stages",ROOT/record["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def _policy_v1(ctx,request):
    require(request.get("operational_IO_policy_ref")==IO_POLICY_REF
        and ctx.get("operational_IO_policy_ref")==IO_POLICY_REF
        and request.get("operational_IO_source_ref")==ctx.get("operational_IO_source_ref")
        and request["operational_IO_source_ref"]["path"]==SOURCE,
        "Fresh ROOT context and request must explicitly authenticate/disclose this separate I/O option")
    source_bytes(request["operational_IO_source_ref"])
    policy=json.loads(source_bytes(IO_POLICY_REF))
    require(request.get("keep1_rotation_of_new_scope_authorized") is True
        and request.get("original_author_recipe_claimed_exact") is False,
        "No concealed retention change or authority to delete historical files")
    return policy

def contrastive_keep1_v1(ctx,request):
    policy=_policy_v1(ctx,request);base=_base_v1()
    directory=base._context_v1(ctx,request,"contrastive",GPU=True)
    checkpoint=base.full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])
    prepared,helper=base._native_v1(ctx);torch=prepared["torch"];OmegaConf=prepared["OmegaConf"]
    cfg=helper._io_config("contrastive",data_root=ROOT/"experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/contrastive",
        checkpoint=ROOT/checkpoint["path"],directory=directory,base_config=ROOT/base.REFS["base_config"]["path"],OmegaConf=OmegaConf,author_root=base.AUTHOR_ROOT)
    before=OmegaConf.to_container(cfg,resolve=True)
    require(before["save_interval"]=="20ba" and before["save_num_checkpoints_to_keep"]==-1,
        "Original author checkpoint events retained before explicit retention-only config adaptation")
    cfg.save_num_checkpoints_to_keep=1
    r=OmegaConf.to_container(cfg,resolve=True)
    require(r==dict(before,save_num_checkpoints_to_keep=1),"Only checkpoint retention differs from exact original I/O config")
    require((r["max_duration"],r["global_train_batch_size"],r["device_train_microbatch_size"],r["precision"],r["seed"])==("1ep",4096,4096,"fp32",36)
        and r.get("train_subset_num_batches",-1)==-1 and r["max_seq_len"]==128
        and r["model"]["contrastive"]["temperature"]==.30
        and r["model"]["contrastive"]["normalize"] is True and r["model"]["contrastive"]["similarity"]=="dot"
        and r["model"]["contrastive"]["use_class_weights"] is True and "train_loader" not in r["model"]
        and r["save_interval"]=="20ba" and r["save_num_checkpoints_to_keep"]==1,
        "Full unchanged author objective/fallback/batch/events, disclosed keep1 retention-only config adaptation")
    require(ctx["max_checkpoint_files"]==2 and ctx["max_checkpoint_bytes"]==policy["resource_failstop_proposal"]["max_single_checkpoint_bytes"],
        "Only the newly generated previous/current checkpoint pair may exist transiently")
    ctx["authenticate_native_inputs_once_v1"](base.REFS["preparation"],base.REFS["counts"])
    OmegaConf.save(cfg,directory/"resolved_io_config.yaml",resolve=True)
    module=importlib.import_module("sequence_contrastive")
    require(Path(module.__file__).resolve()==base.AUTHOR_ROOT/"sequence_contrastive.py","Exact author contrastive origin")
    helper._verify_author_import_origins(base.AUTHOR_ROOT)
    module.train(cfg)  # Unchanged author implementation, no custom optimizer/loss/head.
    final=helper._final_checkpoint(directory/"checkpoints",epochs=1,updates=2135,
        max_files=ctx["max_checkpoint_files"],max_bytes=ctx["max_checkpoint_bytes"],torch=torch,reference=ctx["reference"])
    ctx["verify_native_imports_v1"]()
    ctx["fast_guard"]("after_full_native_contrastive")
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage="contrastive",parent_id="american_c3_val_000",active_input_ref=ctx["input_ref"],child_input_ref=ctx["input_ref"],
        preparation_ref=base.REFS["preparation"],counts_ref=base.REFS["counts"],author_source_manifest_ref=base.REFS["author_manifest"],
        initializer_checkpoint_ref=checkpoint,resolved_config_ref=ctx["reference"](directory/"resolved_io_config.yaml"),
        uniform_class_weight_fallback_preserved=True,training_config_plumbing_correction=False,
        query_ancestry_truth_opened=False,author_code_modified=False,actual_saved_optimizer_updates=2135,
        all_author_interval_checkpoints_retained=False,
        all_author_checkpoint_save_events_preserved=True,original_checkpoint_retention=-1,actual_checkpoint_retention=1,
        operational_IO_policy_ref=IO_POLICY_REF,new_scope_checkpoint_rotation_only=True,
        final_checkpoint_selection_is_exact_epoch1_update2135=True,learned_pooling_head_fitted_by_original_author_objective=True,**final))

class RAMCombinationWriter:
    """Original concat writer with explicit virtual descriptors, never fake files."""
    def __init__(self,writer,embedding_completion,prepared_metadata):
        self.writer=writer;self.path=writer.path
        self.outputs=embedding_completion["outputs"];self.checkpoint=embedding_completion["checkpoint_ref"]
        self.prepared=prepared_metadata;self.values={};self.orders={};self.completion=None
    def descriptor(self,view,kind):
        rows=9721236 if view=="contrastive" else 10000
        return dict(schema="native_VAL000_RAM_row_preserving_descriptor_v1",storage_kind=kind,view=view,
            parent_id="american_c3_val_000",splits=["train","val"],rows=rows,
            source_embedding_refs=[self.outputs[view+"/"+s]["embedding_ref"] for s in ("train","val")],
            source_row_order_refs=[self.prepared["full_path_id_order_refs"][view+"/"+s] for s in ("train","val")],
            checkpoint_ref=self.checkpoint,actual_combined_payload_file_exists=False,
            no_sort_filter_cap_or_normalization=True)
    def save(self,name,value,kind=None):
        for view in ("query_paths","contrastive"):
            if name==view+"_train_then_val_embeddings.pt":
                require(kind=="torch" and view not in self.values,"Exactly one full RAM tensor concatenation per view")
                self.values[view]=value
                return self.descriptor(view,"RAM_tensors")
            if name==view+"_train_then_val_full_path_order.npz":
                require(kind=="npz" and view not in self.orders,"Exactly one full RAM order concatenation per view")
                self.orders[view]=value
                return self.descriptor(view,"RAM_full_path_order")
        require(name=="completed.json" and kind is None and self.completion is None
            and set(self.values)==set(self.orders)=={"query_paths","contrastive"}
            and value["status"]=="completed_C3_native_CPU_row_preserving_concat",
            "No unexpected virtual writes or incomplete RAM combination")
        self.completion=dict(value,status="completed_C3_native_RAM_row_preserving_concat",
            on_disk_combined_tensors_or_sidecars=False,operational_IO_policy_ref=IO_POLICY_REF)
        return self.writer.save("RAM_combination.json",self.completion)

def _original_concat_function_v1(ctx):
    source_bytes(REFS["concat"])
    module=importlib.import_module("arggnn.american_c3_native_concat_v1")
    require(Path(module.__file__).resolve()==ROOT/REFS["concat"]["path"],"Exact original concat implementation origin")
    original=module.concatenate_embeddings_v1;ns=dict(original.__globals__)
    for key in ("preparation_v1","counts_v1","fit_completion_v1","metadata"):
        callback=ctx.get("validated_VAL000_"+key)
        require(callable(callback),"Exact separately reviewed VAL000 lineage callback: "+key)
        ns[key]=callback
    return types.FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__)

def _original_RAM_seal_v1(ctx,ram):
    """The one I/O substitution replaces the existing nested embedding load only.
    Original row joins, labels, calls, focal order and saved score/index arrays
    stay bytecode-source equivalent after reversal of that one load expression.
    """
    source_bytes(REFS["evaluator"])
    module=importlib.import_module("arggnn.american_c3_evaluate_v1")
    require(Path(module.__file__).resolve()==ROOT/REFS["evaluator"]["path"],"Exact original seal implementation origin")
    tree=ast.parse(source_bytes(REFS["evaluator"]).decode("utf-8"))
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="native_top1_seal"))
    embedding=next(n for n in function.body if isinstance(n,ast.FunctionDef) and n.name=="embedding")
    assignment=embedding.body[0]
    expected=ast.parse("out=torch.load(bound(record),map_location='cpu',weights_only=True,mmap=True)").body[0]
    require(ast.dump(assignment,include_attributes=False)==ast.dump(expected,include_attributes=False),
        "Only the exact original nested embedding input-load expression can be adapted")
    assignment.value=ast.Call(func=ast.Name(id="_load_RAM_embedding_v1",ctx=ast.Load()),args=[ast.Name(id="record",ctx=ast.Load())],keywords=[])
    def load_RAM(record):
        for view in ("query_paths","contrastive"):
            if record==ram.descriptor(view,"RAM_tensors"):return ram.values[view]
        raise ValueError("Only the exact two already-validated complete RAM combinations are loadable")
    namespace=dict(module.native_top1_seal.__globals__,_load_RAM_embedding_v1=load_RAM)
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(ROOT/SOURCE)+":one_embedding_read_adapter","exec"),namespace)
    return namespace["native_top1_seal"]

class _RAMSealWriter:
    def __init__(self,writer):self.writer=writer
    def npz(self,name,**arrays):return self.writer.save(name,arrays,kind="npz")
    def json(self,name,value):
        require(name=="completed.json","One original native seal receipt")
        return self.writer.save("prediction_seal.json",dict(value,
            virtual_RAM_embedding_inputs_explicitly_disclosed=True,operational_IO_policy_ref=IO_POLICY_REF))

def RAM_retrieval_and_seal_v1(ctx,request,spec,cfg):
    policy=_policy_v1(ctx,request);base=_base_v1()
    directory=base._context_v1(ctx,request,"RAM_retrieve_and_seal",GPU=True)
    require(spec["parent_id"]=="american_c3_val_000" and spec["refs"]["preparation"]==base.REFS["preparation"]
        and spec["refs"]["counts"]==base.REFS["counts"]
        and cfg["max_embedding_input_bytes"]==policy["resource_failstop_proposal"]["maximum_single_embedding_input_bytes"],
        "Exact actual VAL000 full source metadata and bounded original embedding reads")
    mlm=base.full_mlm_metadata_prerequisite_v1(**ctx["MLM_prerequisite_metadata"])
    contrastive=base.full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=mlm,
        **ctx["contrastive_prerequisite_metadata"])
    prepared,helper=base._native_v1(ctx)
    require(prepared["torch"].cuda.is_available() and prepared["torch"].cuda.device_count()==1,
        "Actual held GPU0 retrieval, no CPU fallback")
    completion=ctx["validated_VAL000_metadata"](spec["refs"]["embedding_completion"])
    preparation=ctx["validated_VAL000_preparation_v1"](spec)[0]
    require(completion["checkpoint_ref"]==contrastive,"All four embeddings use the genuine full contrastive final checkpoint")
    ram=RAMCombinationWriter(ctx["writer"],completion,preparation)
    ramctx=dict(ctx,writer=ram)
    combination_ref=_original_concat_function_v1(ctx)(spec,cfg,ramctx)
    require(set(ram.values)==set(ram.orders)=={"query_paths","contrastive"},"Original full concat completed in RAM")
    ctx["fast_guard"]("after_original_full_row_preserving_RAM_concat")
    module=importlib.import_module("retrieve");helper._verify_author_import_origins(base.AUTHOR_ROOT)
    require(Path(module.__file__).resolve()==base.AUTHOR_ROOT/"retrieve.py","Exact original native retrieval origin")
    tokens={"ARGGNN_RAM_reference_train_then_val":ram.values["contrastive"],
        "ARGGNN_RAM_query_train_then_val":ram.values["query_paths"]}
    def load_RAM(path):
        require(path in tokens,"Only explicit already-validated RAM tensors, no alternative inputs")
        return tokens[path]
    namespace=dict(module.main.__globals__,load_embeddings=load_RAM)
    main=types.FunctionType(module.main.__code__,namespace,module.main.__name__,module.main.__defaults__,module.main.__closure__)
    retrieval_path=directory/"retrieval.pt"
    require(not retrieval_path.exists() and callable(ctx.get("boundary_guard")),"Fresh original retrieval output with admitted write boundary")
    ctx["boundary_guard"]("before_native_RAM_retrieval",required_write_bytes=policy["resource_failstop_proposal"]["retrieval_output_file_cap_bytes"])
    argv=sys.argv
    try:
        sys.argv=[str(base.AUTHOR_ROOT/"retrieve.py"),"--corpus","ARGGNN_RAM_reference_train_then_val",
            "--queries","ARGGNN_RAM_query_train_then_val","--out",str(retrieval_path),
            "--topk","1","--q_batch","1024","--e_chunk","100000"]
        main()  # Entire unchanged main code with a private I/O loader only.
    finally:sys.argv=argv
    require(retrieval_path.stat().st_size<=policy["resource_failstop_proposal"]["retrieval_output_file_cap_bytes"],
        "Actual native retrieval file fits separately admitted cap")
    retrieval_ref=ctx["reference"](retrieval_path)
    ctx["verify_native_imports_v1"]();ctx["fast_guard"]("after_native_RAM_top1")
    eval_module=importlib.import_module("arggnn.american_c3_evaluate_v1")
    config=dict(eval_module.DEFAULTS,mode="native_top1_seal",method="native_ARGformer_C3_release_per_ARG",
        preparation_ref=base.REFS["preparation"],focal_registry_ref=preparation["focals_ref"],
        native_query_embeddings_ref=ram.descriptor("query_paths","RAM_tensors"),
        native_corpus_embeddings_ref=ram.descriptor("contrastive","RAM_tensors"),
        native_retrieval_ref=retrieval_ref,external_truth_ref=None,prediction_ref=None)
    seal_ref=_original_RAM_seal_v1(ctx,ram)(config,_RAMSealWriter(ctx["writer"]),eval_module.registry(preparation["focals_ref"]))
    ctx["verify_native_imports_v1"]();ctx["fast_guard"]("after_unchanged_truth_free_RAM_seal")
    return ctx["writer"].save("completed.json",dict(schema="native_VAL000_RAM_retrieval_and_seal_completion_v1",
        status="completed_native_RAM_retrieval_and_truth_free_seal",parent_id="american_c3_val_000",
        source_ref=request["operational_IO_source_ref"],operational_IO_policy_ref=IO_POLICY_REF,
        actual_input_ref=ctx["input_ref"],preparation_ref=base.REFS["preparation"],counts_ref=base.REFS["counts"],
        original_four_embedding_outputs=completion["outputs"],checkpoint_ref=contrastive,
        RAM_combination_ref=combination_ref,retrieval_ref=retrieval_ref,prediction_seal_ref=seal_ref,
        query_ancestry_truth_opened=False,native_top1_arithmetic_or_ties_modified=False,
        full_reference_rows=9721236,full_query_rows=10000,combined_payloads_written_to_disk=False,
        all_actual_wait_drain_and_outer_terminal_prerequisites_still_required=True))
