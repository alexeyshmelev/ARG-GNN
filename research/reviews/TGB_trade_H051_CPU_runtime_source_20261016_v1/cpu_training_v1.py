"""Full frozen H051 CPU campaign. Inert stdlib import, explicit root call only."""
from pathlib import Path
import hashlib,importlib,io,json,os,time

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FIXED={"archive_source": {"path": "research/reviews/TGB_trade_H051_data_integration_root_20261016_v1/observed_archive_v1.py", "sha256": "b5a3c3587cb02ef688852e14d3e3ef2b60f687e935df7fbe7910e2bbe1836b5f"}, "handoff": {"path": "research/reviews/TGB_trade_H051_training_source_20261016_v1/source_handoff.v1.json", "sha256": "193d58c1fc29b9dbb28e8aa9c5056fb31ab769b770cdcd5aec5f2318a061f8db"}, "protocol": {"path": "research/reviews/TGB_trade_H051_training_source_20261016_v1/protocol.source_frozen.inactive.v1.json", "sha256": "a2253a9aec4d8405a14262d9dc69f9ee2bcff14ea36e0fbef466ba94424acb6e"}, "qualification": {"path": "research/reviews/TGB_trade_H051_data_integration_root_20261016_v1/actual_model_FP32_CPU_qualification.root_v1.json", "sha256": "ad2eaeff409f68b36a7137c542e1f98356bedb55242a32da15b253c66cd65f3c"}, "science_sources": {"path": "research/reviews/TGB_trade_H051_CPU_runtime_source_20261016_v1/science_sources.source_frozen.v1.json", "sha256": "8839419c9a483a43a621d2f314cdf3392ef68868900a75a42de88e38b68c55db"}}
SCIENCE="research.reviews.TGB_trade_H051_training_source_20261016_v1"
ARMS=("tree_only","first_order","conditional_triple")
CAP=12*2**20
STATE_CAP=640*2**10
JSON_CAP=128*2**10
NPY_CAP=5462100+8192
CONTROL_ALLOWANCE=2**20
FILE_GROWTH=65536
PLANNED_COMPLETE_BOUND=4*STATE_CAP+NPY_CAP+8*JSON_CAP+13*FILE_GROWTH+CONTROL_ALLOWANCE
_RAN=False

def need(ok,message):
    if not ok:raise ValueError(message)

def checked(record,limit=2**20):
    p=ROOT/record["path"]
    need(not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,
        "Canonical bounded source/metadata")
    raw=p.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==record["sha256"],"Exact immutable source/metadata")
    return raw

def run_training_v1(*,resource_check,guard,storage,input_ref,storage_ref):
    global _RAN
    need(not _RAN and callable(resource_check),"One root CPU full-campaign invocation")
    _RAN=True
    need(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and storage["reserved_new_bytes"]==CAP and PLANNED_COMPLETE_BOUND<=CAP,"Exact CPU-only12MiB scope")
    resource_check("before_frozen_science_authentication")
    closure=json.loads(checked(FIXED["science_sources"]))
    for r in closure["source_refs"].values():checked(r)
    protocol=json.loads(checked(FIXED["protocol"]))
    q=json.loads(checked(FIXED["qualification"]))
    need(q["schema"]=="TGB_H051_actual_model_FP32_CPU_qualification_v1" and q["status"]=="PASS"
        and q["actual_trainable_encoder_GRU_head_forward_and_backward"] is True
        and q["optimizer_steps"]==0 and set(q["arms"])==set(ARMS)
        and all(v["all_parameter_gradients_finite"] is True for v in q["arms"].values()),
        "Reuse actual existing CPU forward/backward qualification, never replay")
    need(protocol["primary_campaign"]["epochs"]==100 and protocol["primary_campaign"]["arms"]==list(ARMS)
        and protocol["primary_campaign"]["seeds"]==[17],"Original complete fixed scientific budget")
    import torch
    torch.set_num_threads(4)
    writer_module=importlib.import_module(SCIENCE+".artifact_writer_v1")
    train_module=importlib.import_module(SCIENCE+".trainer_v1")
    archive_module=importlib.import_module("research.reviews.TGB_trade_H051_data_integration_root_20261016_v1.observed_archive_v1")
    directory=ROOT/storage["output_directory"]
    states={"seed17.common_initial.pt",*("seed17.%s.final_epoch100.pt"%a for a in ARMS)}
    jsons={"protocol.executed.json","runtime_binding.json","seed17.common_initial.json","prediction_seal.json","fit_completed.json",
        *("seed17.%s.training.json"%a for a in ARMS)}
    expected=states|jsons|{"primary_all_heldout_predictions.npy"}
    written=set();forecast_sum=0
    def before(name,size,limit):
        nonlocal forecast_sum
        resource_check("before_original_TGB_artifact_"+name)
        need(name in expected and name not in written and type(size) is int and 0<=size<=limit,
            "Actual serialization fits exact finite per-file plan")
        forecast_sum+=size+FILE_GROWTH
        need(forecast_sum+CONTROL_ALLOWANCE<=CAP,"Complete count-derived artifact/control cap")
        guard.budget_check_v9(storage,boundary=True,required_write_bytes=size+FILE_GROWTH)
        written.add(name)
    def after():
        guard.budget_check_v9(storage,boundary=True,required_write_bytes=0)
        resource_check("after_original_TGB_artifact")
    class Writer(writer_module.DirectoryArtifactWriterV1):
        def _json(self,name,value):
            data=(json.dumps(value,indent=2,sort_keys=True)+"\n").encode()
            before(name,len(data),JSON_CAP)
            result=super()._json(name,value);after();return result
        def begin_run(self,value):
            need(value==protocol and not directory.exists(),"Original frozen protocol and fresh original writer")
            guard.budget_check_v9(storage,boundary=True,required_write_bytes=8192)
            result=super().begin_run(value)
            self._json("runtime_binding.json",dict(schema="TGB_H051_CPU_actual_runtime_binding_v1",input_ref=input_ref,
                storage_ref=storage_ref,frozen_protocol_ref=FIXED["protocol"],frozen_science_handoff_ref=FIXED["handoff"],
                reused_CPU_qualification_ref=FIXED["qualification"],device="cpu",physical_GPU=None,
                source_closure_ref=FIXED["science_sources"],whole_scope_cap_bytes=CAP,
                count_derived_scope_bound_bytes=PLANNED_COMPLETE_BOUND,full_frozen_scientific_protocol_unchanged=True))
            return result
        def save_initial(self,seed,state,metadata):
            buf=io.BytesIO();torch.save(state,buf);size=buf.tell();del buf
            before("seed%d.common_initial.pt"%seed,size,STATE_CAP)
            result=super().save_initial(seed,state,metadata);after();return result
        def save_final(self,seed,arm,state,metadata,history):
            need(seed==17 and arm in ARMS and len(history)==100 and metadata["optimizer_updates"]==100,
                "No partial budget or altered final selection")
            buf=io.BytesIO();torch.save(state,buf);size=buf.tell();del buf
            before("seed%d.%s.final_epoch100.pt"%(seed,arm),size,STATE_CAP)
            result=super().save_final(seed,arm,state,metadata,history);after();return result
        def seal_predictions(self,scores,metadata):
            import numpy as np
            need(tuple(scores.shape)==(1,3,7,255,255) and scores.dtype==torch.float32,"Every frozen arm/year/country score")
            buf=io.BytesIO();np.save(buf,scores.numpy(),allow_pickle=False);size=buf.tell();del buf
            before("primary_all_heldout_predictions.npy",size,NPY_CAP)
            result=super().seal_predictions(scores,metadata);after();return result
    writer=Writer(directory)
    resource_check("before_original_RAM_observation_archive")
    archive=archive_module.download_observations_v1()
    resource_check("after_original_RAM_observation_archive")
    dataset=archive.dataset_v1()
    started=time.monotonic()
    def monitored_check(stage):
        resource_check(stage)
        if stage.startswith("before_epoch_"):
            arm,epoch=stage[len("before_epoch_"):].rsplit("_",1)
            epoch=int(epoch)
            if epoch==1 or epoch%10==0:
                print(json.dumps(dict(stage="TGB_CPU_TRAIN_epoch_start",arm=arm,epoch=epoch,
                    fixed_epochs_per_arm=100,elapsed_seconds=time.monotonic()-started)),flush=True)
    bundle=train_module.run_primary_v1(dataset=dataset,load_train_target_year=archive.load_train_target_year_v1,
        writer=writer,runtime_check=monitored_check,device=torch.device("cpu"),protocol=protocol)
    need(written==expected and len(written)==13,"Every and only13 planned scientific artifacts")
    receipt=json.loads(Path(bundle.fit_receipt_ref["path"]).read_text())
    need(receipt["primary_full_budget_complete"] is True and receipt["total_models"]==3
        and receipt["total_optimizer_updates"]==300 and receipt["TRAIN_graph_forwards"]==6900
        and receipt["final_replay_graph_forwards"]==90 and receipt["VAL_or_TEST_metric_computed"] is False,
        "Genuine complete frozen campaign, no heldout scoring")
    resource_check("after_full_frozen_CPU_campaign")
    return dict(schema="TGB_H051_CPU_full_campaign_returned_v1",full_campaign_complete=True,device="cpu",physical_GPU=None,
        total_models=3,total_optimizer_updates=300,TRAIN_graph_forwards=6900,final_replay_graph_forwards=90,
        prediction_seal_ref=bundle.seal_ref,fit_receipt_ref=bundle.fit_receipt_ref,
        scientific_protocol_ref=FIXED["protocol"],source_manifest_ref=FIXED["science_sources"],
        reused_CPU_qualification_ref=FIXED["qualification"],new_qualification_runs=0,
        scientific_file_count=len(written),count_derived_maximum_scope_bytes=PLANNED_COMPLETE_BOUND,
        actual_scientific_serialization_plus_growth_bound_bytes=forecast_sum,
        model_selection_performed=False,VAL_metric_computed=False,TEST_metric_computed=False,
        actual_external_tool_terminal_consumed=False,actual_external_tool_exit_code=None,
        actual_ancestry_or_trade_score_claimed=False)
