"""Contrastive-only integration over original admitted-context keep1 stage.
No MLM entry, scheduler, guard mutation, admission or runtime access at import.
"""
from pathlib import Path
import importlib
from research.reviews.native_VAL000_contrastive_saved_MLM_integration_source_20261017_v1 import saved_mlm_integration_v1 as integration
ROOT=integration.ROOT
require=integration.require
IO_POLICY_REF={"path":"research/reviews/native_VAL000_downstream_IO_efficiency_source_20261016_v1/io_policy.INACTIVE.v1.json","sha256":"76011c718b4c41f79643b81faf55cacf2d65f549744afb0fc6f3331c3a851b2d"}

def contrastive_keep1_saved_MLM_v1(ctx,request):
    integration.bind_stage_v1(ctx,request)
    io=integration.load_source_v1("io")
    policy=io._policy_v1(ctx,request);base=io._base_v1()
    directory=base._context_v1(ctx,request,"contrastive",GPU=True)
    bridge,eligible,checkpoint=integration.saved_acceptance_v1(ctx,request)
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
    initialization=integration.run_original_keep1_contrastive_train_v1(cfg,eligible,author_train=module.train)
    initialization_ref=ctx["writer"].save("saved_MLM_initialization.json",initialization)
    final=integration.checkpoint_metadata_bridge_v1(helper)(directory/"checkpoints",epochs=1,updates=2135,
        max_files=ctx["max_checkpoint_files"],max_bytes=ctx["max_checkpoint_bytes"],torch=torch,reference=ctx["reference"])
    ctx["verify_native_imports_v1"]()
    ctx["fast_guard"]("after_full_native_contrastive")
    return ctx["writer"].save("completed.json",dict(schema="american_c3_native_child_completion_v1",status="completed_C3_native_child",
        stage="contrastive",parent_id="american_c3_val_000",active_input_ref=ctx["input_ref"],child_input_ref=ctx["input_ref"],
        preparation_ref=base.REFS["preparation"],counts_ref=base.REFS["counts"],author_source_manifest_ref=base.REFS["author_manifest"],
        saved_MLM_initialization_ref=initialization_ref,initializer_checkpoint_ref=checkpoint,resolved_config_ref=ctx["reference"](directory/"resolved_io_config.yaml"),
        uniform_class_weight_fallback_preserved=True,training_config_plumbing_correction=False,
        query_ancestry_truth_opened=False,author_code_modified=False,actual_saved_optimizer_updates=2135,
        all_author_interval_checkpoints_retained=False,
        all_author_checkpoint_save_events_preserved=True,original_checkpoint_retention=-1,actual_checkpoint_retention=1,
        operational_IO_policy_ref=IO_POLICY_REF,new_scope_checkpoint_rotation_only=True,
        final_checkpoint_selection_is_exact_epoch1_update2135=True,learned_pooling_head_fitted_by_original_author_objective=True,**final))
