"""Inactive CPU-only validator/evaluation worker. No scientific imports before guarded execute."""
from pathlib import Path
import argparse,importlib.util,json,os,resource,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE='research/reviews/C3_CPU_interleaved_guarded_integration_source_20261016_v1'
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/HERE)]
import acceptance_contract_v1 as c
class GuardedLoaderV1:
    """Operational checks only; every original block and its order passes unchanged."""
    def __init__(self,loader,check):self.loader,self.check=loader,check
    def __getattr__(self,name):return getattr(self.loader,name)
    def __iter__(self):
        self.check("before_original_loader_iteration")
        for block in self.loader:
            self.check("before_original_whole_tree_inference_block")
            yield block
            self.check("after_original_whole_tree_inference_block")
def validate_saved_v1(req,bundle,training,writer,fast):
    import torch
    import saved_state_core_v1 as core
    c.require(not torch.cuda.is_initialized(),"CPU acceptance never initializes CUDA")
    before=c.validate_old_owner_absence_v1()
    f=c.frozen_v1();migration=c.read(c.MIGRATION_REF)
    observation=c.read(training["observation_manifest_ref"]);targets=c.read(training["target_manifest_ref"])
    reader=core.TorchCPUReader(torch,ROOT,[x["checkpoint_ref"] for x in f["checkpoint_candidates"]])
    result=core.audit_saved_v1(training,migration,observation,targets,reader,c.read,
        lambda r:c.require(c.ref(r["path"])==r,"Exact recorded source/metadata provenance"),fast)
    after=c.validate_old_owner_absence_v1()
    result.update(training_completion_ref=bundle["training_completion_ref"],migration_manifest_ref=c.MIGRATION_REF,
        original_history=c.ORIGINAL_HISTORY,source_manifest_ref=c.ref(c.MANIFEST),
        dedicated_validator_CPU_runtime_checked=True,CUDA_initialized=False,
        old_owner_absence_before=before,old_owner_absence_after=after)
    return writer.save("saved600_validation_result.json",result)
def predict_v1(req,training_bundle,training,writer,fast):
    from research.reviews.C3_CPU_six_candidate_interleave_source_20261016_v1.prediction_bridge_v1 import predict_v1 as shared_predict
    return shared_predict(req,training_bundle,training,writer,fast,contract=c)

def matrix_v1(req,bundle,writer,fast):
    from arggnn.american_c3_gnn_predict_isolated_v2 import seal_validation_matrix_v1
    refs=[]
    for parent in c.VAL:
        env=c.read(req["VAL_predictions"][parent]["completion_ref"]);six=c.read(env["result_ref"])
        refs.extend(six["prediction_seal_refs"])
    fast("before_original30cell_prediction_barrier")
    return seal_validation_matrix_v1(refs,training_completion_ref=bundle["training_completion_ref"],writer=writer)

def score_v1(req,bundle,writer,fast):
    from arggnn.american_c3_gnn_selection_isolated_v2 import (
        validation_matrix_v1,score_validation_matrix_v1,select_original_val_checkpoint_v1)
    matrix_ref=c.read(req["matrix"]["completion_ref"])["result_ref"]
    # Authenticate every30-cell prediction and registry SHA before a resolver is built.
    matrix,training,cells=validation_matrix_v1(matrix_ref)
    c.require(matrix["training_completion_ref"]==bundle["training_completion_ref"],"Same complete trained family")
    truth={}
    for parent in c.VAL:
        plan=c.plan_ready_v1(req["plan_evidence_by_parent"][parent],parent)
        c.require(all(cell["focal_registry_ref"]==plan["focal_registry_ref"] for (epoch,p),cell in cells.items() if p==parent),
                  "Every checkpoint scored on exact same native focal identity/support registry")
        simref=req["VAL_truth_capability"][parent];sim=c.read(simref);obs=c.read(plan["observation_ref"])
        c.require(sim["status"]=="completed_C3_simulation_reconstruction" and sim["parent_id"]==parent
            and sim["class_order"]==["EUR","EAS","AFR"]
            and obs["source_observation_ref"]==sim["observation_ref"]
            and sim["private_truth_ref"]["path"]==str(Path(simref["path"]).parent/"truth/c3_pulse_ancestry.npz"),
            "Separate same-parent simulation truth, not an unrelated truth capability")
        truth[parent]=sim["private_truth_ref"]
        if parent=="american_c3_val_000":
            comparison=c.read(c.manifest_v1()["common_comparator_score_ref"])
            c.require(comparison["focal_registry_ref"]==plan["focal_registry_ref"]
                and comparison["external_truth_ref"]==truth[parent],"Exact existing FLARE common focal/truth support")
    def resolve(parent):
        c.require(parent in c.VAL,"Only original5VAL after complete30-cell barrier")
        return truth[parent]
    scored=score_validation_matrix_v1(matrix_ref,writer=writer,resolve_val_truth=resolve,runtime_guard=fast)
    fast("before_original_fixed_checkpoint_selection")
    selected=select_original_val_checkpoint_v1(scored,writer=writer)
    return writer.save("score_and_selection_completed.json",dict(status="completed_C3_original30_scores_and_selection",
        score_matrix_ref=scored,selected_checkpoint_ref=selected,validation_matrix_ref=matrix_ref,
        training_completion_ref=bundle["training_completion_ref"],original_metric_equations_unchanged=True,
        all30_seals_precede_VAL_truth=True,TEST_deferred=True,full_TreeLoom_history_pair_model_claimed=False))

def execute(record):
    spec,req,caps=c.validate_input_v1(record)
    c.require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact CPU-only repository execution")
    c.require(sys.prefix==str(ROOT/".runtime/env"),"Exact existing shared Torch/PyG interpreter")
    for key in ("TMPDIR","TMP","TEMP"):
        c.require(os.environ.get(key)==str(c.local(req["packet_directory"]+"/worker_tmp")),"Charged private runtime TMP")
    guard,_=c.consumer_guard();storage=c.read(spec["refs"]["storage"])
    guard.validate_storage_v9(spec,dict(output_cap_bytes=caps["reservation_cap_bytes"]),storage,
        source_ref=c.ref(c.ENTRY),config_ref=spec["refs"]["config"],runtime_ref=c.RUNTIME_REF)
    from arggnn.american_c3_common_v1 import runtime_report
    c.require(runtime_report()==c.read(c.RUNTIME_REF),"Exact prequalified shared CPU package/runtime metadata")
    import numpy as np,torch,torch_geometric
    c.require(str(torch.__version__)=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Original Torch/PyG versions")
    torch.set_num_threads(caps["CPU_threads"]);torch.set_num_interop_threads(1)
    c.require(not torch.cuda.is_initialized(),"No CUDA context in exact CPU option")
    from arggnn import census_tsinfer_fit_v1 as fit
    fit.np=np;fit.torch=torch
    raw=guard.PilotOutputV9(spec["output_directory"],storage,fit.ConsumerReaderV1());start=time.monotonic()
    def fast(event="C3_exact_CPU_evaluation"):
        c.active(spec);c.active(req)
        c.require(time.monotonic()-start<caps["operational_deadline_seconds"],"Actual full-stage deadline, no truncation")
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=caps["max_cpu_rss_bytes"],"Original owned main RSS failstop")
        c.require(not torch.cuda.is_initialized(),"CPU-only inference retains zero CUDA context")
    class Writer:
        path=raw.path
        tmp=req["packet_directory"]+"/worker_tmp"
        def save(self,name,value,kind="json"):
            fast("before_exact_guarded_product_write")
            result=raw.save(name,value,kind=kind)
            c.require(raw.used<=caps["phase_write_budget_bytes"],"Bounded complete per-stage product budget")
            return result
    writer=Writer()
    bundle,training=c.acceptance_inputs_v1(req) if req["phase"]==c.ACCEPT_PHASE else c.training_ready_v1(req["training_evidence_ref"])
    writer.save("started.json",dict(status="started_C3_exact_CPU_evaluation",stage=req["phase"],input_ref=record,
        architecture="AmericanC3WholeTreeGNNV1",training_completion_ref=bundle["training_completion_ref"],
        device="cpu",num_workers=0,full30Mb_fit_or_accuracy_claimed=False,TEST_deferred=True))
    fast()
    if req["phase"]==c.ACCEPT_PHASE:result=validate_saved_v1(req,bundle,training,writer,fast)
    elif req["phase"].startswith("predict_val_"):result=predict_v1(req,bundle,training,writer,fast)
    elif req["phase"]=="matrix":result=matrix_v1(req,bundle,writer,fast)
    else:result=score_v1(req,bundle,writer,fast)
    fast("completed_original_evaluation_stage")
    return writer.save("completed.json",dict(status=c.STATUSES[req["phase"]],stage=req["phase"],parent_id=c.PARENT,
        input_ref=record,root_driver_ref=c.ref(c.DRIVER),result_ref=result,device="cpu",
        native_MLM_dependency=False,TEST_deferred=True,training_completion_ref=bundle["training_completion_ref"]))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true")
    p.add_argument("--input");p.add_argument("--expected-input-sha256");a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",phases=list(c.PHASES),
            acceptance_or_training=False,accounting_invoked=False,CUDA_imported=False)));return 0
    c.require(a.input and a.expected_input_sha256,"Separate reviewed ROOT admission required")
    result=execute(dict(path=a.input,sha256=a.expected_input_sha256))
    print(json.dumps(dict(completed_ref=result)),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
