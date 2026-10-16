"""Pure saved-state checks. Reader injection is for source fixtures; production creates TorchCPUReader."""
import hashlib,json,random
TRAIN=["american_c3_short_train_%03d"%i for i in range(20)]
EPOCHS=[5,10,15,20,25,30]
CLASSES=["EUR","EAS","AFR"]
CONDITION="fresh_C3_supervised_short_independent_complete_parents"
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def order(epoch):
    value=list(TRAIN);random.Random(70002+epoch).shuffle(value);return value
def original_history_v1():
    return [(e,j,p) for e in range(1,31) for j,p in enumerate(order(e))]
def history_v1(state,step):
    h=state["history"]
    require(type(step)is int and 0<step<=600 and len(h)==step,"Exact completed-parent history/cursor")
    for n,(row,expected) in enumerate(zip(h,original_history_v1()[:step])):
        require(row["global_step"]==n+1 and (row["epoch"],row["parent_index"],row["parent_id"])==expected
            and row["records"]==10000 and row["whole_parent_optimizer_updates"]==1
            and row["block_reset_or_truncated_BPTT"] is False,"Exact original600-update order and untruncated complete parents")
    require(state["focal_exposures"]==step*10000 and state["next_epoch"]==step//20+1
        and state["next_parent_index"]==step%20 and state["next_parent_order"]==(order(step//20+1) if step<600 else [])
        and state["scheduler"]==dict(name="constant_learning_rate",completed_parent_updates=step,lr=.001),
        "Original cursor/exposures/constant scheduler")
def train_provenance_v1(training,protocol,observation,targets,read_metadata,authenticate):
    require(observation["schema"]=="american_c3_all20_TRAIN_observation_barrier_v1"
        and observation["all20_ready"] is True and observation["observation_only"] is True
        and observation["truth_resolver_calls_before_barrier"]==0 and observation["parent_ids"]==TRAIN
        and set(observation["parents"])==set(TRAIN) and digest(observation)==training["observation_seal_sha256"],
        "All20 original observation seals before TRAIN truth, exact frozen digest")
    require(targets["schema"]=="american_c3_all20_external_TRAIN_target_joins_v1"
        and targets["all20_targets_ready"] is True and targets["labels_in_observation_bank"] is False
        and targets["purity_filter"] is False and set(targets["parents"])==set(TRAIN)
        and targets["observation_seal_sha256"]==training["observation_seal_sha256"]
        and digest(targets)==training["target_join_sha256"],"All20 exact external TRAIN joins, no label payload access")
    require(training["registry_ref"]==observation["registry_ref"]==targets["registry_ref"],"Same original scientific registry")
    authenticate(training["registry_ref"])
    sources=protocol["source_refs"]
    require(sources and all(set(r)=={"path","sha256"} for r in sources),"Complete frozen protocol source closure")
    for r in sources:authenticate(r)
    plans=protocol["parent_plan_refs"]
    require(set(plans)==set(TRAIN),"All20 frozen original TRAIN plans")
    for pid in TRAIN:
        seal=observation["parents"][pid];join=targets["parents"][pid]
        plan=read_metadata(seal["plan_ref"]);obs=read_metadata(seal["observation_ref"])
        require(seal["schema"]=="american_c3_observation_blocks_sealed_v1" and seal["parent_id"]==pid
            and seal["plan_ref"]==plans[pid] and seal["observation_only"] is True
            and seal["all_blocks_sealed_before_truth"] is True and seal["all_trees_and_samples_retained"] is True
            and seal["focal_records"]==10000 and seal["blocks"]==len(plan["blocks"])
            and len(seal["block_feature_sha256"])==seal["blocks"]
            and all(isinstance(x,str) and len(x)==64 for x in seal["block_feature_sha256"]),
            "Every original TRAIN parent/block seal, no resealing or cropped context")
        require(plan["schema"]=="american_c3_whole_tree_block_index_v1" and plan["parent_id"]==pid
            and plan["sequence_length_bp"]==2000000 and plan["haplotypes"]==1600 and plan["focal_records"]==10000
            and plan["targets_opened"] is False and plan["all_trees_and_samples_retained"] is True
            and plan["genomic_context"]=="two_pass_full_parent_state_carry" and plan["class_order"]==CLASSES
            and plan["observation_ref"]==seal["observation_ref"] and plan["focal_registry_ref"]==seal["focal_registry_ref"],
            "Frozen complete2Mb/1600/10000 TRAIN plan")
        blocks=plan["blocks"]
        require(blocks and blocks[0]["start_tree"]==0 and blocks[-1]["stop_tree"]==plan["trees"]
            and all(x["start_tree"]<x["stop_tree"] for x in blocks)
            and all(a["stop_tree"]==b["start_tree"] for a,b in zip(blocks,blocks[1:])),
            "Every original tree covered once in ordered blocks")
        require(obs["parent_id"]==pid and obs["split"]=="train" and obs["sequence_length_bp"]==2000000
            and obs["class_order"]==CLASSES and obs["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
            and obs["query_ancestry_truth_in_inputs"] is False and obs["sites_and_sample_identities_verified"] is True
            and obs["migrations_and_internal_population_labels_removed"] is True
            and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
            "Recorded inferred observation provenance, no true genealogy or query truth input")
        prep=read_metadata(plan["config"]["native_preparation_ref"])
        require(prep["parent_id"]==pid and prep["status"]=="completed_C3_exact_native_focal_registry_only_v1"
            and prep["observation_ref"]==seal["observation_ref"] and prep["focals_ref"]==seal["focal_registry_ref"]
            and prep["query_focal_count"]==10000 and prep["query_ancestry_truth_opened"] is False,
            "Exact label-blind native focal preparation")
        for key in ("operational_bridge_ref","original_native_focal_rule_source_ref","source_author_function_ref"):
            authenticate(prep[key])
        sim=read_metadata(join["actual_simulation_completion_ref"]);cfg=read_metadata(join["actual_simulation_config_ref"])
        require(join["schema"]=="C3_external_TRAIN_truth_focal_join_v1" and join["parent_id"]==pid
            and join["split"]=="train" and join["records"]==10000 and join["class_order"]==CLASSES
            and join["labels_in_observation_bank"] is False and join["purity_filter"] is False
            and join["target_equations_unchanged"] is True and join["original_scientific_registry_unchanged"] is True
            and join["nearest_or_majority_fallback"] is False
            and join["interval_rule"]=="half_open_searchsorted_right_original_sample_index"
            and join["focal_registry_ref"]==seal["focal_registry_ref"] and join["focal_identity_sha256"]==seal["focal_identity_sha256"]
            and sim["status"]=="completed_C3_simulation_reconstruction" and sim["parent_id"]==pid
            and sim["class_order"]==CLASSES and sim["observation_ref"]==obs["source_observation_ref"]
            and sim["private_truth_ref"]==join["truth_ref"]
            and cfg["parent_id"]==pid and cfg["split"]=="train" and cfg["sequence_length_bp"]==2000000
            and cfg["sample_counts"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000) and cfg["class_order"]==CLASSES,
            "Original20TRAIN source/simulation/inference/focal/target provenance joins without opening label payloads")
    return dict(parents=20,all_original_block_seals=True,source_refs_authenticated=len(sources),
        input_payloads_resealed=False,truth_payloads_opened=False,observation_digest=training["observation_seal_sha256"],
        target_join_digest=training["target_join_sha256"])
def audit_saved_v1(training,migration,observation,targets,reader,read_metadata,authenticate,fast=lambda *_:None):
    require(training["status"]=="completed_C3_checkpointed_complete_parent_training"
        and training["schema"]=="american_c3_checkpointed_complete_parent_training_v1"
        and training["complete_parent_updates"]==600 and training["epochs"]==30 and training["focal_exposures"]==6000000
        and training["trained_parent_ids"]==TRAIN and training["train_sequence_lengths_bp"]==[2000000]*20
        and training["VAL_used_for_optimization"] is False and training["TEST_deferred"] is True
        and training["H043_extension"] is None and training["purity_filter"] is False,
        "Exact original20x2Mb/600-update scientific completion metadata")
    require(training["resume_ref"]==migration["migrated_boundary_ref"] and migration["actual_global_step"]==100
        and migration["actual_AdamW_step"]==100 and migration["remaining_updates"]==500 and migration["passed"] is True
        and training["protocol_sha256"]==migration["successor_protocol_sha256"]
        and training["observation_seal_sha256"]==migration["observation_seal_sha256"]
        and training["target_join_sha256"]==migration["target_join_sha256"],"Exact full-state migrated prefix100 plus suffix500")
    candidates=training["checkpoint_candidates"]
    require([x["epoch"] for x in candidates]==EPOCHS and [x["global_step"] for x in candidates]==[20*e for e in EPOCHS]
        and len({x["checkpoint_ref"]["path"] for x in candidates})==6,"All six original candidates, never final-only")
    contract=dict(schema="american_c3_fresh_whole_tree_GNN_checkpoint_v1",class_order=CLASSES,hidden_dim=32,
        donor_cohorts=2,donor_channels=6,expected_haplotypes=1600,initialization="fresh_no_C4_loading",
        site_feature_width=15,observed_leaf_width=5,recurrent_contract="one_layer_bidirectional_equations_as_two_directional_GRUs",
        tree_backbone="unchanged_PyG_gated_upward_recipient_excluded_downward_cavity",whole_parent_training_required=True)
    rows=[];protocol=None
    for i,c in enumerate(candidates):
        fast("before_saved_checkpoint_read");value=reader.load(c["checkpoint_ref"]);state=value["trainer_state"];step=c["global_step"]
        require(value["schema"]==contract["schema"] and value["contract"]==contract and value["model_seed"]==70001
            and value["training_condition"]==CONDITION and value["C4_initialization_or_head_slicing"] is False
            and value["trained_parent_ids"]==TRAIN and value["train_sequence_lengths_bp"]==[2000000]*20
            and state["schema"]==training["schema"],"Original fresh29571parameter model family, no renamed/C4 weights")
        require(value["candidate"]==dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",completed_epoch=c["epoch"],global_step=step)
            and state["global_step"]==step,"Original scheduled checkpoint eligibility")
        history_v1(state,step)
        require(state["history"]==training["history"][:step] and state["scheduled_checkpoints"]==candidates[:i],
            "Actual completed history and prior scheduled hash chain")
        require(digest(state["protocol"])==state["protocol_sha256"]==training["protocol_sha256"]
            and state["observation_seal_sha256"]==training["observation_seal_sha256"]
            and state["target_join_sha256"]==training["target_join_sha256"],"Saved protocol and exact data-seal lineage")
        protocol=state["protocol"] if protocol is None else protocol
        require(state["protocol"]==protocol,"Same scientific/source protocol at all six candidates")
        require((i==0 and c["checkpoint_ref"]==migration["migrated_boundary_ref"]
            and c["checkpoint_ref"]==migration["migrated_checkpoint_receipts"][0]["migrated_ref"])
            or (i>0 and c["checkpoint_ref"]["path"]=="experiments/results/C3_GNN_efficient_resume_GPU0_train_v1/checkpoint_epoch_%03d.pt"%c["epoch"]),
            "Exact migrated epoch5 and original suffix namespace")
        model=value["model_state"];opt=value["optimizer_state"];states=opt["state"];groups=opt["param_groups"]
        require(len(model)==38 and len(groups)==1 and len(states)==38 and len(groups[0]["params"])==38
            and set(states)==set(groups[0]["params"]),"Every original parameter and AdamW state, no partial/reset optimizer")
        infos=[reader.tensor(t) for t in model.values()]
        require(sum(t["numel"] for t in infos)==29571 and all(t["CPU"] and t["finite"] and t["dtype"]=="torch.float32" for t in infos),
            "All29571model parameters finite original FP32 on CPU")
        steps=[];moment_rows=[]
        for pid,param_info in zip(groups[0]["params"],infos):
            item=states[pid];require(set(item)=={"step","exp_avg","exp_avg_sq"},"Full ordinary AdamW moments")
            info=[reader.tensor(item[k]) for k in ("step","exp_avg","exp_avg_sq")]
            count=reader.scalar_int(item["step"]);steps.append(count)
            require(count==step and all(t["CPU"] and t["finite"] for t in info)
                and info[1]["shape"]==info[2]["shape"]==param_info["shape"],"Actual finite saved AdamW counters/moments")
            moment_rows.append(info)
        g=groups[0]
        require(g["lr"]==.001 and g["weight_decay"]==.0001 and tuple(g["betas"])==(.9,.999) and g["eps"]==1e-8
            and all(g.get(k,False) is False for k in ("amsgrad","maximize","foreach","capturable","differentiable","fused")),
            "Original AdamW settings and constant schedule")
        rng=state["rng_state"]
        require(set(rng)=={"python","numpy","torch_CPU","torch_device","device_type"} and rng["device_type"]=="cuda"
            and rng["python"] and rng["numpy"]["algorithm"]=="MT19937" and len(rng["numpy"]["keys"])==624
            and reader.tensor(rng["torch_CPU"])["CPU"] and reader.tensor(rng["torch_device"])["CPU"],
            "Saved complete RNG state present, never restored by acceptance")
        reader.finish(c["checkpoint_ref"])
        rows.append(dict(epoch=c["epoch"],global_step=step,checkpoint_ref=c["checkpoint_ref"],model_parameters=29571,
            model_tensor_count=38,optimizer_parameter_states=38,actual_AdamW_step_min=min(steps),actual_AdamW_step_max=max(steps),
            model_tensor_manifest_sha256=digest(infos),optimizer_tensor_manifest_sha256=digest(moment_rows),
            history_sha256=digest(state["history"]),history_rows=step,all_checks_passed=True))
        del value,state,model,opt,states
        fast("after_saved_checkpoint_read")
    provenance=train_provenance_v1(training,protocol,observation,targets,read_metadata,authenticate)
    return dict(status="validated_C3_saved600_artifacts_after_failed_producer",checkpoints=rows,
        saved_effective_logical_updates=600,epochs=30,model_parameters=29571,provenance=provenance,
        protocol_sha256=training["protocol_sha256"],observation_seal_sha256=training["observation_seal_sha256"],
        target_join_sha256=training["target_join_sha256"],all_six_verified=True,physical_work_exactly600_claimed=False,
        original_producer_success_claimed=False,original_worker_drain_manifest_reconstructed=False,
        optimizer_updates_performed=0,model_forward_or_backward_performed=False,VAL_or_TEST_truth_opened=False,
        predictions_read_or_written=False,root_acceptance_still_required=True,storage_close_or_credit_authorized=False)
class TorchCPUReader:
    """Real future interface. Torch is supplied only after guarded CPU runtime validation."""
    def __init__(self,torch,root,expected):
        self.torch,self.root=torch,root
        self.expected={r["path"]:r for r in expected};self.identities={}
        require(len(self.expected)==6 and not torch.cuda.is_initialized(),"Six exact refs and no CUDA context")
    def _bound(self,record):
        import stat
        p=self.root/record["path"];s=p.lstat()
        require(record==self.expected.get(record["path"]) and p.resolve()==p and p.is_relative_to(self.root)
            and stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size<=2*2**20,"Exact bounded canonical saved checkpoint")
        require(hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],"Immutable checkpoint SHA")
        return p,(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def load(self,record):
        p,identity=self._bound(record);self.identities[record["path"]]=identity
        return self.torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    def tensor(self,value):
        t=self.torch
        require(t.is_tensor(value),"Real saved tensor required")
        require(value.device.type=="cpu","No CUDA checkpoint tensor")
        return dict(shape=list(value.shape),numel=value.numel(),dtype=str(value.dtype),CPU=True,
            finite=bool(t.isfinite(value).all()),sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
    def scalar_int(self,value):
        info=self.tensor(value);require(info["numel"]==1 and info["finite"],"One finite real saved step scalar")
        number=value.item();require(type(number)in(int,float) and int(number)==number,"Exact integer AdamW counter")
        return int(number)
    def finish(self,record):
        _,identity=self._bound(record)
        require(self.identities.pop(record["path"])==identity and not self.torch.cuda.is_initialized(),
            "Stable checkpoint identity/hash and no CUDA after load")
