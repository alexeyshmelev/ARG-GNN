"""Complete common-support scoring with independent-parent summaries."""
from __future__ import annotations

import numpy as np
from . import paper_argmix_generator as g
from .paper_argmix_features import read_targets


def prediction_barrier(config_path, prediction_paths):
    config=g.metadata(config_path)
    expected={p["parent_id"] for p in config["parents"] if p["split"] in ("iid","misspec")}
    records=[]
    for path in prediction_paths:
        item=g.metadata(path)
        g.need(item["schema"]=="paper_argmix_predictions_v1" and item["config_ref"]==g.file_ref(config_path) and
               item["truth_read"] is False and item["examples"]==100000 and item["class_order"]==list(g.CLASSES),
               "Complete truth-free predictions required")
        g.verify_ref(item["probabilities_ref"])
        records.append(dict(parent_id=item["parent_id"],prediction_ref=g.file_ref(path)))
    g.need(len(records)==8 and {r["parent_id"] for r in records}==expected,"All three IID and five OOD predictions required")
    return dict(schema="paper_argmix_prediction_barrier_v1",config_ref=g.file_ref(config_path),
                predictions=records,complete=True,truth_read=False)


def verify_barrier(config_path, path):
    barrier=g.metadata(path)
    g.need(barrier["schema"]=="paper_argmix_prediction_barrier_v1" and barrier["complete"] is True and
           barrier["truth_read"] is False and barrier["config_ref"]==g.file_ref(config_path),"Complete prediction barrier required")
    regenerated=prediction_barrier(config_path,[g.verify_ref(row["prediction_ref"]) for row in barrier["predictions"]])
    g.need(regenerated==barrier,"Prediction barrier differs")
    return barrier


def score(config_path, barrier_path, label_paths, output, *, guard, comparator_path=None):
    barrier=verify_barrier(config_path,barrier_path)
    labels={g.metadata(path)["parent_id"]:path for path in label_paths}
    g.need(set(labels)=={r["parent_id"] for r in barrier["predictions"]},"Exact eight label parents required")
    results=[]
    for record in barrier["predictions"]:
        prediction=g.metadata(g.verify_ref(record["prediction_ref"]))
        label_path=labels[record["parent_id"]]
        label=g.metadata(label_path)
        g.need(label["config_ref"]==prediction["config_ref"] and label["class_order"]==prediction["class_order"] and
               label["support_manifest_ref"]==prediction["support_ref"] and label["examples"]==prediction["examples"],
               "Prediction/truth support differs")
        probabilities=np.load(g.verify_ref(prediction["probabilities_ref"]),allow_pickle=False)
        truth=np.fromfile(g.verify_ref(label["labels_ref"]),dtype=np.uint8)
        g.need(probabilities.shape==(100000,4) and truth.shape==(100000,) and np.all(truth<4) and
               np.all(np.isfinite(probabilities)) and np.all(probabilities>=0) and
               np.allclose(probabilities.sum(1),1,atol=1e-5),"Invalid complete predictions or labels")
        support=g.metadata(g.verify_ref(prediction["support_ref"]))
        rows=read_targets(g.verify_ref(support["targets_ref"]))
        g.need(len(rows)==100000,"Common-support row count differs")
        correct=probabilities.argmax(1)==truth
        cohorts=np.asarray([row["cohort"] for row in rows])
        per_cohort={name:dict(examples=int((cohorts==name).sum()),accuracy=float(correct[cohorts==name].mean())) for name in sorted(set(cohorts))}
        results.append(dict(parent_id=record["parent_id"],split=prediction["split"],examples=100000,
                            accuracy=float(correct.mean()),per_cohort=per_cohort,support_ref=prediction["support_ref"],
                            prediction_ref=record["prediction_ref"],label_ref=g.file_ref(label_path)))
        guard(0)
    summaries={}
    for split in ("iid","misspec"):
        values=np.asarray([r["accuracy"] for r in results if r["split"]==split])
        summaries[split]=dict(parents=len(values),mean_accuracy=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                              uncertainty_definition="Sample standard deviation across independent parents, ddof=1; not an inferred paper plus/minus definition")
    receipt=dict(schema="paper_argmix_complete_scores_v1",config_ref=g.file_ref(config_path),
                 prediction_barrier_ref=g.file_ref(barrier_path),source_ref=g.file_ref(__file__),
                 results=results,summaries=summaries,coverage="All 800000 frozen query/tree examples; no dropped rows")
    if comparator_path:
        comparator=g.metadata(comparator_path)
        previous={r["parent_id"]:r for r in comparator["results"]}
        g.need(comparator["config_ref"]==g.file_ref(config_path) and set(previous)=={r["parent_id"] for r in results},"Comparator parents differ")
        differences=[]
        for row in results:
            other=previous[row["parent_id"]]
            g.need(row["support_ref"]==other["support_ref"] and row["examples"]==other["examples"],"Comparator sampled rows differ")
            differences.append(dict(parent_id=row["parent_id"],split=row["split"],accuracy_difference=row["accuracy"]-other["accuracy"]))
        receipt["paired_comparator_ref"]=g.file_ref(comparator_path)
        receipt["paired_differences"]=differences
    output=g._fresh_directory(output,guard)
    g.Writes(4*1024*1024,guard).json(output/"completed.json",receipt)
    return receipt


def paired_barrier(config_path, first_path, second_path):
    first,second=verify_barrier(config_path,first_path),verify_barrier(config_path,second_path)
    left={row["parent_id"]:g.metadata(g.verify_ref(row["prediction_ref"])) for row in first["predictions"]}
    right={row["parent_id"]:g.metadata(g.verify_ref(row["prediction_ref"])) for row in second["predictions"]}
    g.need(set(left)==set(right),"Paired methods must predict identical parents")
    for parent in left:
        g.need(left[parent]["support_ref"]==right[parent]["support_ref"] and
               left[parent]["class_order"]==right[parent]["class_order"],"Paired prediction support differs")
    return dict(schema="paper_argmix_paired_prediction_barrier_v1",config_ref=g.file_ref(config_path),
                first_barrier_ref=g.file_ref(first_path),second_barrier_ref=g.file_ref(second_path),
                complete=True,truth_read=False,parent_ids=sorted(left),both_methods_all_eight_predictions_complete=True)


def verify_paired_barrier(config_path,path):
    record=g.metadata(path)
    g.need(record["schema"]=="paper_argmix_paired_prediction_barrier_v1","Both-method prediction barrier required before evaluation truth")
    reconstructed=paired_barrier(config_path,g.verify_ref(record["first_barrier_ref"]),g.verify_ref(record["second_barrier_ref"]))
    g.need(record==reconstructed,"Both-method barrier drift")
    return record
