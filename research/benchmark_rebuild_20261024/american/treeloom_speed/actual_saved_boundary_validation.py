
"""Read-only validation of the actual live complete-parent checkpoint snapshot."""
import json
from arggnn.benchmarking import contracts as c
from research.benchmark_rebuild_20261024.american.treeloom_speed import training_v2 as training
from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
import torch
torch.set_num_threads(1)
base=c.ROOT/"research/benchmark_rebuild_20261024/american/treeloom_speed"
job=c.read_json(".runtime/benchmarking/workflow_20261022_gpu345_v2/jobs/AM_treeloom_fit.json")
original=c.read_ref(job["call"]["spec"]["job_ref"])["call"]["spec"]["training_spec"]
registry=fresh.FreshParentRegistry.from_ref(original["parent_registry_ref"])
runtime=fresh.FreshTreeLoom(registry);model=runtime.build_model()
native=c.contained_path(job["call"]["spec"]["output_directory"])
receipt=c.read_json(native/"latest_checkpoint.json")
checkpoint_ref=receipt["checkpoint_ref"]
assert training._hash(c.contained_path(checkpoint_ref["path"]))==checkpoint_ref
saved=torch.load(c.contained_path(checkpoint_ref["path"]),map_location="cpu",weights_only=True)
steps=training.validate_saved_state(saved,model,registry,expected_steps=receipt["complete_parent_updates"])
schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
assert saved["resume_schema"]=="treeloom_exact_complete_update_resume_v1"
assert saved["schedule_identity"]==training.schedule_identity(schedule)
assert saved["last_completed_parent"]==schedule[steps-1][1]
assert saved["next_schedule_index"]==steps
assert saved["observation_barrier_ref"]==job["call"]["spec"]["observation_barrier_ref"]
assert training._hash(c.contained_path(checkpoint_ref["path"]))==checkpoint_ref
result=dict(schema="TreeLoom_actual_saved_boundary_CPU_validation_v2",passed=True,
    read_only=True,CUDA_called=False,checkpoint_ref=checkpoint_ref,
    complete_parent_updates=steps,epoch=saved["epoch"],last_completed_parent=saved["last_completed_parent"],
    next_schedule_index=steps,next_parent=schedule[steps][1],actual_model_parameters=32132,
    all49_Adam_counters=steps,all49_moments_finite_and_shape_matched=True,
    original_schedule_identity=saved["schedule_identity"],parent_registry_ref=registry.reference,
    original_barrier_ref=saved["observation_barrier_ref"],actual_random_state_present="random_state" in saved,
    source_ref=c.file_ref(__file__),note="Live latest path may advance after this snapshot; ROOT must freeze an immutable copy after verified owned-process drain.")
with (base/"actual_saved_boundary_validation.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
print(json.dumps(result,indent=2))
