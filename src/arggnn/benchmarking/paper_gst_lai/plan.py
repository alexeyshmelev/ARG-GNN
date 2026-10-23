"""Inert AP476 GST-LAI-8 paired plan with fixed starting hyperparameters."""
from pathlib import Path
from .workflow import make_call,SETTINGS
from .model import specification
from .. import european_native_models as io


def build(reuse_ref,output_root,*,treeloom_prediction_task="paper_ap_treeloom__all_test__predict"):
    from ..paper_ancestralpaths import CONDITION as AP_CONDITION
    from ..paper_ancestralpaths.workflow import make_call as ap_call
    reuse=io._read(reuse_ref,observed=True)
    io._need(reuse["schema"] == "paper_ancestralpaths_frozen476_reuse_v1",
             "Exact AP476 frozen observation/focal manifest required")
    interfaces=reuse["treeloom_observation_focal_interfaces"]
    train={p:v for p,v in interfaces.items() if p in reuse["training_products"]}
    test={p:v for p,v in interfaces.items() if p in reuse["test_features"]}
    root=io._path(output_root)
    io._need(root.is_relative_to(io.ROOT/"experiments/results"),"Fresh admitted results root required")
    python=".runtime/env/bin/python3.11"
    shared=dict(device="cuda",threads=8,max_sites=5000000,
                max_metadata_bytes=512*2**20,write_headroom_bytes=4*2**20)
    tasks=[]
    for identity,stage,args,deps,output_cap in [
        ("paper_gst_lai_8__all_train__fit","fit",
            dict(plan_ref=reuse["sampling_plan_ref"],train_inputs=train),[],64*2**20),
        ("paper_gst_lai_8__all_test__predict","predict",
            dict(fit_ref={"$task":"paper_gst_lai_8__all_train__fit"},heldout_inputs=test),
            ["paper_gst_lai_8__all_train__fit"],32*2**20)]:
        arguments=dict(shared,**args)
        call=make_call(stage,arguments,root/identity,python=python)
        tasks.append(dict(id=identity,kind="gpu",cpu_threads=8,dependencies=deps,python=python,
            output_directory=str(root/identity),max_seconds=86400,max_rss_bytes=128*2**30,
            max_output_bytes=output_cap,minimum_free_gpu_bytes=12*2**30,
            source_refs=call["source_refs"],
            call=dict(factory=dict(module="arggnn.benchmarking.paper_gst_lai.workflow",
                function="make_call",kwargs=dict(stage=stage,arguments=arguments,
                    output_directory=str(root/identity),python=python)))))
    completed_ref=io._ref(io.ROOT/"research/benchmark_rebuild_20261024/ancestralpaths/cnn_completed_products.json")
    completed=io._read(completed_ref,observed=True)
    predictions={AP_CONDITION:completed["predictions"],
        "ancestralpaths_dense_control":reuse["dense_control"]["test_predictions"],
        "treeloom":{p:{"$task":treeloom_prediction_task,"path":["predictions",p]} for p in test},
        "gst_lai_8_native_graphsetconv":{p:{"$task":"paper_gst_lai_8__all_test__predict",
            "path":["predictions",p]} for p in test}}
    score_id="paper_gst_lai_8__all_methods_all_test__score"
    score_args=dict(plan_ref=reuse["sampling_plan_ref"],predictions=predictions,
        test_targets=reuse["test_target_references"],expected_methods=list(predictions),
        max_truth_uncompressed_bytes=16*2**30,write_headroom_bytes=4*2**20)
    score_python=".runtime/conda/envs/ancestralpaths_cpu_v1/bin/python3.11"
    score_call=ap_call("score",score_args,root/score_id,python=score_python)
    tasks.append(dict(id=score_id,kind="cpu",cpu_threads=4,
        dependencies=["paper_gst_lai_8__all_test__predict",treeloom_prediction_task],
        python=score_python,output_directory=str(root/score_id),
        max_seconds=1800,max_rss_bytes=32*2**30,max_output_bytes=16*2**20,
        source_refs=score_call["source_refs"],
        call=dict(factory=dict(module="arggnn.benchmarking.paper_ancestralpaths.workflow",
            function="make_call",kwargs=dict(stage="score",arguments=score_args,
                output_directory=str(root/score_id),python=score_python)))))
    return dict(schema="paper_gst_lai_8_plan_v1",plan_only=True,admitted=False,
        model_spec=specification(),settings=SETTINGS,reuse_ref=reuse_ref,tasks=tasks,
        completed_CNN_products_ref=completed_ref,
        external_dependencies=[treeloom_prediction_task],
        scoring_interface="append gst_lai_8_native_graphsetconv parent predictions to predeclared AP native-six scorer methods",
        all_method_all_parent_barrier_before_TEST_truth=True,
        estimated_not_measured_production_resources=True,
        resource_notes="128GiB aggregate host cap includes eight spawn workers and replicated observed states. 12GiB initial GPU headroom is an admission estimate, not an allocator cap. Full-device policy applies. Production CUDA and full-fit duration not yet measured.",
        training_budget="30complete passes; one accumulated mean sampled loss/update per complete parent;90total updates",
        pipeline_comparison_not_isolated_layer_attribution=True,
        original_GST_synthetic_or_molecular_table_reproduction=False)
