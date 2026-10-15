"""Pure metadata and isolated-source checks. No guards, admission, launch or training."""
import ast,copy,hashlib,inspect,json
from pathlib import Path
import torch
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bridge_common_v1 as c
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import root_driver_v1 as d
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import context_factory_v1 as f
from research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1.root_bridge_v1 import bind_root_packet_v1 as b
SOURCE=c.HERE+"/metadata_check_v1.py"

def fake_ref(label,path=None):return dict(path=path or "handmade_metadata_only/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())

def forbidden(*args,**kwargs):raise AssertionError("Guard/admission/launch/resource operation forbidden in pure metadata check")

def main():
    checks=[]
    for module in (c,d,f):module.load_guard_v1=forbidden
    assert not torch.cuda.is_initialized()
    for name in ("bridge_common_v1.py","context_factory_v1.py","root_driver_v1.py","bind_root_packet_v1.py"):
        text=(c.ROOT/c.HERE/name).read_text();compile(text,name,"exec")
    original=d.isolated_original_supervisor_v1()
    original.load_guard=forbidden
    assert original.SOURCE==c.DRIVER and original.CANONICAL_DISPATCHER_REF==c.ref(c.worker.SOURCE)
    assert original.context() is f
    old_text=(c.ROOT/c.REVIEWED_DRIVER_REF["path"]).read_text()
    old_launch=next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name=="launch")
    assert inspect.getsource(original.launch).strip()==ast.get_source_segment(old_text,old_launch).strip()
    assert original.launch.__code__.co_filename==str(c.ROOT/c.REVIEWED_DRIVER_REF["path"])
    checks.append("exact_original_launch_body_private_bindings_no_launch_called")
    for path in (c.CONTEXT,c.DRIVER):
        text=(c.ROOT/path).read_text()
        assert "joint_three_arm" not in text and "all144" not in text
        assert "updates_total=2880" not in text and "checkpoint_files=18" not in text
        assert "framework.build_context_v1(" not in text and "framework.admit(" not in text
    checks.append("new_boundaries_do_not_invoke_old_H049_admit_context_or18_144_plan")
    cfg=json.loads((c.ROOT/c.INTEGRATION/"config_train_evaluate.proposed.inactive.json").read_text())
    cfg.update({k:10**13 for k in c.worker.RESOURCE_CAPS+c.EXTRA_CAPS})
    cfg.update(execution_enabled=True,root_resource_caps_reviewed=True,future_completion_not_asserted=False,
        expected_future_preparation_output="experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5",
        canonical_dispatcher_ref=c.ref(c.worker.SOURCE),root_driver_ref=c.ref(c.DRIVER),root_context_module_ref=c.ref(c.CONTEXT),
        metrics_source_config_ref=c.H045_CONFIG_REF,metrics=c.read(c.H045_CONFIG_REF)["metrics"],registration_ref=fake_ref("registration"))
    cfg["source_manifest_ref"]=fake_ref("source_manifest")
    for key in c.TERMINAL_FIELDS:cfg[key]=fake_ref(key)
    cfg["cut_preparation_completion_ref"]=fake_ref("prep_completion",cfg["expected_future_preparation_output"]+"/completed.json")
    values={cfg["registration_ref"]["path"]:dict(schema="H050_direct_readout_cut_memory_registration_v1",hypothesis_id="H050",
        status="registered_before_execution",scientific_protocol=c.worker.hooks.fixed_protocol_v1()),c.H045_CONFIG_REF["path"]:c.read(c.H045_CONFIG_REF)}
    class Reader:
        @staticmethod
        def decode_json(record):return values[record["path"]]
    reader=Reader();c.validate_config_v1(cfg,reader)
    def reject(fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,TypeError):return
        raise AssertionError("Invalid metadata accepted")
    reject(lambda:c.validate_config_v1(dict(cfg,execution_enabled=False),reader))
    reject(lambda:c.validate_config_v1(dict(cfg,metrics=dict(cfg["metrics"],boundary_tolerance_bp=1.)),reader))
    reject(lambda:c.validate_config_v1(dict(cfg,cut_preparation_registration_ref=cfg["registration_ref"]),reader))
    checks.append("inactive_config_changed_metrics_and_shared_registration_rejected")
    # Synthetic complete40 count shape; no observations or true count file is loaded.
    counts=dict(schema="root_H043_original40_observed_count_manifest_v1",counts_only_no_labels_or_model_metrics=True,
        outer_included=False,parents={p:dict(native_nodes=100+i,native_holds=5+i,haplotypes=952) for i,p in enumerate(c.worker.TRAIN+c.worker.VAL)})
    prep=dict(parents={p:dict(marker_tensor_bytes=1024,counts=dict(native_holds=row["native_holds"],haplotypes=952,queries=686,sites=10,scoring_centers=20000)) for p,row in counts["parents"].items()})
    cfg.update(checkpoint_file_cap_bytes=1000000,log_cap_bytes=1024,fixed_metadata_cap_bytes=1000000,
        qualification_allowance_bytes=1000000,root_auxiliary_metadata_cap_bytes=1000000)
    cfg["episode_python_scratch_allowance_bytes"]=4096
    for phase in ("train","evaluate"):
        plan=c.count_plan_v1(counts,phase,prep,cfg);c.validate_plan_caps_v1(plan,cfg,10**13)
        assert plan["scheduled_checkpoint_files"]==24 and plan["VAL_probability_seals_before_truth"]==192
        assert plan["all192_native_hold_probability_bytes"]==24*sum(counts["parents"][p]["native_holds"]*686*16 for p in c.worker.VAL)
        assert plan["checkpoint_files_written"]==(24 if phase=="train" else 0)
        reject(lambda:c.validate_plan_caps_v1(dict(plan,scheduled_checkpoint_files=18),cfg,10**13))
        reject(lambda:c.validate_plan_caps_v1(dict(plan,VAL_probability_seals_before_truth=144),cfg,10**13))
        reject(lambda:c.validate_plan_caps_v1(plan,dict(cfg,RAM_prediction_cap_bytes=1),10**13))
        reject(lambda:c.validate_plan_caps_v1(plan,dict(cfg,grid_preparation_cap_bytes=1),10**13))
    checks.append("synthetic4arm24file192native_RAM_count_and_old18_144_rejections")
    train_output="experiments/results/H050_GPU5_direct_readout_cut_memory_train_seed101_v1"
    input_ref=fake_ref("train_input",c.HERE+"/admitted_train/launch_input.json")
    train_config=fake_ref("train_config");runtime_ref=fake_ref("train_runtime")
    train_ref=fake_ref("train_completion",train_output+"/completed.json")
    launch_ref=fake_ref("train_launch",c.HERE+"/admitted_train/launch.json")
    wait_ref=fake_ref("train_wait",c.HERE+"/admitted_train/actual_wait.json")
    sources={"handmade_source_fixture_only":"b"*64}
    values[train_config["path"]]=copy.deepcopy(cfg)
    values[input_ref["path"]]=dict(schema="H050_direct_readout_root_input_v1",hypothesis_id="H050",arm="joint_four_arm",phase="train",
        root_authorizer="root_agent",status="active",physical_gpu=5,device="cuda",output_directory=train_output,job_id="handmade_job",
        root_driver_ref=c.ref(c.DRIVER),root_context_module_ref=c.ref(c.CONTEXT),producer_ref=c.ref(c.worker.SOURCE),
        canonical_dispatcher_ref=c.ref(c.worker.SOURCE),main_source_ref=c.ref(c.worker.SOURCE),worker_cli_ref=c.ref(c.worker.SOURCE),
        source_sha256=sources,refs=dict(config=train_config,runtime=runtime_ref),**c.terminal_v1(cfg))
    values[runtime_ref["path"]]=dict(job_id="handmade_job",device="cuda",physical_gpu=5,output_directory=train_output)
    train=dict(schema="H050_four_arm_train_completion_v1",hypothesis_id="H050",status="completed_fixed30pass_four_arm_TRAIN",
        arms=list(c.worker.ARMS),steps={a:960 for a in c.worker.ARMS},VAL_truth_read=False,outer_truth_read=False,
        common_initial_predictor_verified=True,scientific_protocol=c.worker.hooks.fixed_protocol_v1(),parameters_per_arm=c.worker.model.PARAMETERS,
        input_ref=input_ref,config_ref=train_config,producer_ref=c.ref(c.worker.SOURCE),
        history=[dict(pass_index=i,steps={a:32*i for a in c.worker.ARMS},full_parent_updates_per_arm=32,parent_order=list(c.worker.TRAIN)) for i in range(1,31)],
        first_step_qualification={a:dict(finite_FP32=True,part_of_continuous_960_updates=True,parent_id=c.worker.TRAIN[0],
            full_model_qualification=dict(parameters=c.worker.model.PARAMETERS[a],actual_optimizer_step=1,no_extra_updates=True,
                warmstart_checkpoint_ref=cfg["warmstart_checkpoint_ref"],gradient_checks={"body":dict(dtype="torch.float32",finite=True)},updated_parameters={"body":True})) for a in c.worker.ARMS},
        checkpoint_groups={str(at):{a:fake_ref(a+str(at),train_output+"/"+a+".pass_%02d.pt"%at) for a in c.worker.ARMS} for at in c.worker.PASSES})
    for key in ("registration_ref","warmstart_checkpoint_ref","warmstart_completion_ref","cut_preparation_completion_ref","cut_preparation_actual_wait_ref"):train[key]=cfg[key]
    launch=dict(schema=c.LEGACY_RECEIPT_SCHEMAS[0],launcher_pid=12345,launcher_start=123456,direct_child=True,input_ref=input_ref,phase="train",physical_gpu=5)
    wait=dict(schema=c.LEGACY_RECEIPT_SCHEMAS[1],launcher_pid=12345,launcher_start=123456,actual_direct_launcher_wait=True,returncode=0,
        owned_group_drained=True,completed_ref=train_ref,launch_ref=launch_ref,phase="train",physical_gpu=5,error=None)
    values.update({train_ref["path"]:train,launch_ref["path"]:launch,wait_ref["path"]:wait})
    spec=dict(phase="evaluate",train_completion_ref=train_ref,train_launch_ref=launch_ref,train_actual_wait_ref=wait_ref)
    exact=c.exact_sources_v1;c.exact_sources_v1=lambda *_:sources
    try:
        lineage=c.validate_train_lineage_v1(reader,spec,cfg)
        assert sum(map(len,lineage["train_checkpoint_groups"].values()))==24
        for label,mutate in [
            ("legacy_three_arm_schema",lambda x:x.update(schema="H049_three_arm_train_completion_v1")),
            ("missing_continuation_arm",lambda x:x["checkpoint_groups"]["5"].pop(c.worker.model.CONTINUATION)),
            ("old2880updates",lambda x:x["steps"].update({c.worker.model.CONTINUATION:0})),
            ("missing_first_step_qualification",lambda x:x["first_step_qualification"].pop(c.worker.model.CONTINUATION)),
            ("VAL_truth_before_train",lambda x:x.update(VAL_truth_read=True)),
            ("changed_warmstart",lambda x:x.update(warmstart_checkpoint_ref=fake_ref("wrong_warm")))]:
            value=copy.deepcopy(train);mutate(value);values[train_ref["path"]]=value
            reject(lambda:c.validate_train_lineage_v1(reader,spec,cfg));checks.append(label+"_rejected")
        values[train_ref["path"]]=train;values[wait_ref["path"]]=dict(wait,launcher_start=123457)
        reject(lambda:c.validate_train_lineage_v1(reader,spec,cfg));values[wait_ref["path"]]=wait
        values[wait_ref["path"]]=dict(wait,owned_group_drained=False)
        reject(lambda:c.validate_train_lineage_v1(reader,spec,cfg));values[wait_ref["path"]]=wait
        checks.append("same_launcher_start_identity_and_drain_required")
    finally:c.exact_sources_v1=exact
    assert not torch.cuda.is_initialized()
    result=dict(schema="H050_root_bridge_metadata_check_v1",status="PASS",checks=checks,
        guards_called=False,admission_called=False,launch_called=False,actual_preparation_bound=False,
        scientific_data_or_targets_or_checkpoint_payloads_read=False,optimizer_steps=0,CUDA_initialized=False,
        receipt_labels_are_legacy_operational_formats=list(c.LEGACY_RECEIPT_SCHEMAS))
    print(json.dumps(result,indent=2));return result

if __name__=="__main__":main()
