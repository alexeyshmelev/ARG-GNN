"""Separate original-VAL external scoring and prespecified checkpoint choice.

Prediction and training modules receive no VAL/TEST truth capability. This
callable delegates truth reads and all metric equations to the original frozen
C3 evaluator only after all30 candidate/parent predictions have been sealed.
Root supplies the writer, fast resource callback and original-VAL resolver.
"""
from __future__ import annotations
import math
from .american_c3_common_v1 import require,read,bound,CLASSES
from .american_c3_evaluate_v1 import DEFAULTS as EVALUATOR_DEFAULTS,execute as original_score,metrics
from .american_c3_gnn_targets_v1 import VAL
from .american_c3_gnn_train_v1 import CHECKPOINT_EPOCHS
from .american_c3_gnn_predict_v1 import completed_training_v1
METHOD="AmericanC3_typed_backbone_GNN_v1"

def validation_matrix_v1(record):
    matrix=read(record);training=completed_training_v1(matrix["training_completion_ref"])
    require(matrix.get("status")=="sealed_C3_all30_original_VAL_predictions"
        and matrix.get("protocol_sha256")==training["protocol_sha256"]
        and matrix.get("validation_parent_ids")==list(VAL)
        and matrix.get("checkpoint_epochs")==list(CHECKPOINT_EPOCHS)
        and matrix.get("all_predictions_sealed_before_VAL_truth") is True
        and matrix.get("metrics")=="original_C3_external_evaluator_unchanged" and len(matrix["cells"])==30,
        "Complete original5VAL x six-checkpoint prediction barrier before truth resolution")
    cells={}
    for cell in matrix["cells"]:
        key=(cell["epoch"],cell["parent_id"])
        require(key in {(e,p) for e in CHECKPOINT_EPOCHS for p in VAL} and key not in cells,
            "Unique complete original VAL cells")
        seal=read(cell["prediction_seal_ref"])
        require(seal["status"]=="sealed_C3_original_heldout_predictions" and seal["split"]=="val"
            and seal["epoch"]==key[0] and seal["parent_id"]==key[1]
            and seal["prediction_ref"]==cell["prediction_ref"] and seal["focal_registry_ref"]==cell["focal_registry_ref"]
            and seal["checkpoint_ref"]==cell["checkpoint_ref"] and seal["records"]==10000
            and seal["focal_identity_sha256"]==cell["focal_identity_sha256"]
            and seal["training_completion_ref"]==matrix["training_completion_ref"]
            and seal["protocol_sha256"]==training["protocol_sha256"] and seal["private_truth_opened"] is False
            and seal["purity_filter"] is False,"Exact sealed VAL prediction identity/provenance")
        candidate=next(c for c in training["checkpoint_candidates"] if c["epoch"]==key[0])
        require(cell["checkpoint_ref"]==candidate["checkpoint_ref"],"Frozen completed-run candidate")
        bound(cell["prediction_ref"]);bound(cell["focal_registry_ref"]);cells[key]=cell
    require(len(cells)==30,"All30 candidate/parent predictions precede every VAL resolver invocation")
    return matrix,training,cells

class _ExternalScoreWriter:
    def __init__(self,writer,*,cell,matrix_ref,training_ref):
        self.writer=writer;self.cell=cell;self.matrix_ref=matrix_ref;self.training_ref=training_ref
    def json(self,name,value):
        require(name=="completed.json","Original evaluator produces only its score completion here")
        # Provenance is added without changing result/support/probabilities.
        out=dict(value,validation_matrix_ref=self.matrix_ref,epoch=self.cell["epoch"],
            checkpoint_ref=self.cell["checkpoint_ref"],training_completion_ref=self.training_ref,
            original_evaluator_metric_equations_unchanged=True)
        return self.writer.save("VAL_epoch%03d_%s_score.json"%(self.cell["epoch"],self.cell["parent_id"]),out)

def score_validation_matrix_v1(matrix_ref,*,writer,resolve_val_truth,runtime_guard):
    """External score phase, never called by training or prediction.

    The resolver is invoked for only the five original VAL parent IDs after the
    complete SHA-bound30-cell barrier passes. Original evaluator verifies each
    prediction before opening its corresponding truth and computes unchanged
    C3 metrics. No optimizer or TEST capability is reachable here.
    """
    require(callable(resolve_val_truth) and callable(runtime_guard) and callable(getattr(writer,"save",None)),
        "Separate original-VAL resolver and admitted writer/resource callback")
    matrix,training,cells=validation_matrix_v1(matrix_ref)
    runtime_guard("before_external_original_VAL_scoring")
    truth_refs={p:resolve_val_truth(p) for p in VAL}
    require(all(isinstance(r,dict) and set(r)=={"path","sha256"} for r in truth_refs.values()),
        "Exact original-VAL private refs")
    scores=[]
    for key in sorted(cells):
        cell=cells[key];runtime_guard("external_original_VAL_score_cell")
        cfg=dict(EVALUATOR_DEFAULTS,execution_enabled=True,mode="score",method=METHOD,
            prediction_ref=cell["prediction_ref"],focal_registry_ref=cell["focal_registry_ref"],
            external_truth_ref=truth_refs[cell["parent_id"]])
        sink=_ExternalScoreWriter(writer,cell=cell,matrix_ref=matrix_ref,
            training_ref=matrix["training_completion_ref"])
        score_ref=original_score(cfg,sink)
        scores.append(dict(epoch=key[0],parent_id=key[1],score_ref=score_ref))
    return writer.save("VAL_all30_original_scores.json",dict(status="completed_C3_all30_original_VAL_scores",
        validation_matrix_ref=matrix_ref,training_completion_ref=matrix["training_completion_ref"],
        protocol_sha256=training["protocol_sha256"],cells=scores,original_evaluator_metrics_unchanged=True,
        TEST_truth_resolved=False,VAL_used_for_optimization=False))

def select_original_val_checkpoint_v1(score_matrix_ref,*,writer):
    """Greatest mean per-parent macro-F1 among six epochs; earlier exact tie."""
    scores=read(score_matrix_ref)
    matrix,training,cells=validation_matrix_v1(scores["validation_matrix_ref"])
    require(scores.get("status")=="completed_C3_all30_original_VAL_scores"
        and scores.get("training_completion_ref")==matrix["training_completion_ref"]
        and scores.get("protocol_sha256")==training["protocol_sha256"]
        and scores.get("original_evaluator_metrics_unchanged") is True
        and scores.get("TEST_truth_resolved") is False and scores.get("VAL_used_for_optimization") is False
        and len(scores["cells"])==30,"Completed original VAL score matrix only")
    values={};score_refs={}
    for item in scores["cells"]:
        key=(item["epoch"],item["parent_id"]);require(key in cells and key not in values,"Unique original VAL scores")
        score=read(item["score_ref"]);cell=cells[key]
        require(score.get("status")=="completed_external_C3_score" and score.get("method")==METHOD
            and score.get("support")==EVALUATOR_DEFAULTS["support"]
            and score.get("parent_id")==key[1] and score.get("epoch")==key[0]
            and score.get("class_order")==list(CLASSES) and score.get("records")==10000
            and score.get("prediction_ref")==cell["prediction_ref"]
            and score.get("focal_registry_ref")==cell["focal_registry_ref"]
            and score.get("checkpoint_ref")==cell["checkpoint_ref"]
            and score.get("validation_matrix_ref")==scores["validation_matrix_ref"]
            and score.get("training_completion_ref")==matrix["training_completion_ref"]
            and score.get("original_evaluator_metric_equations_unchanged") is True
            and score.get("probabilities_modified") is False and score.get("probability_floor") is False,
            "Original external metric receipt bound to exact pretruth prediction cell")
        result=score["result"];recomputed=metrics(result["confusion"])
        require(result==recomputed and sum(sum(row) for row in result["confusion"])==10000,
            "Original C3 metric equations/counts without pooling or changed support")
        f1=result["macro_F1"];require(type(f1) in (int,float) and math.isfinite(f1) and 0<=f1<=1,"Finite macro-F1")
        values[key]=float(f1);score_refs[key]=item["score_ref"]
    require(set(values)==set(cells),"All five independent VAL parents scored at each of six frozen epochs")
    table=[dict(epoch=e,mean_per_parent_macro_F1=sum(values[(e,p)] for p in VAL)/5,
        per_parent_macro_F1={p:values[(e,p)] for p in VAL},
        external_score_refs={p:score_refs[(e,p)] for p in VAL}) for e in CHECKPOINT_EPOCHS]
    winner=min(table,key=lambda r:(-r["mean_per_parent_macro_F1"],r["epoch"]))
    candidate=next(c for c in training["checkpoint_candidates"] if c["epoch"]==winner["epoch"])
    return writer.save("selected_original5VAL_checkpoint.json",dict(status="selected_C3_original5VAL_checkpoint",
        checkpoint_ref=candidate["checkpoint_ref"],epoch=winner["epoch"],
        training_completion_ref=matrix["training_completion_ref"],score_matrix_ref=score_matrix_ref,
        validation_matrix_ref=scores["validation_matrix_ref"],protocol_sha256=training["protocol_sha256"],
        validation_parent_ids=list(VAL),checkpoint_epochs=list(CHECKPOINT_EPOCHS),scores=table,
        criterion="greatest_mean_per_parent_original_macro_F1_ties_earlier_epoch",
        TEST_used_for_selection=False,VAL_used_for_optimization=False,purity_filter=False,
        uncertainty_unit="independent_parent",single_model_seed_uncertainty_not_established=True,
        training_2Mb_to_evaluation_30Mb_length_transfer=True))
