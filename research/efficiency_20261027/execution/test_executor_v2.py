"""Actual first/second execution methods on a CPU diagnostic context only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.efficiency_20261027.execution.executors_v2 import MinimalTransferLevelExecution
from research.efficiency_20261027.execution.level_sweep import LevelSweepPlan
from research.scalability_20261024.execution.test_layout_plan import sweep_bank
from research.scalability_20261024.execution.measure_representative import baseline_layout
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2 as frozen
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import _cp, NullProfileV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
from research.repairs.benchmark_recovery_20261020.runtime.cuts import prepare_cut_plan_v1

OUT=Path("research/efficiency_20261027/execution")
PACKET=Path("research/scalability_20261024/execution/representative_slice.npz")
CHECKPOINT=Path("research/preprocessing_optimization_20261026/integration/migration/update241.pt")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    meta=json.loads(PACKET.with_suffix(".json").read_text())
    assert hashlib.sha256(PACKET.read_bytes()).hexdigest()==meta["packet_ref"]["sha256"]
    data=np.load(PACKET,allow_pickle=False)
    arrays={}
    for field in fields(MutationRoutedWholeTreeInputsV1):
        name=field.name
        if name in meta["level_tuple_lengths"]:
            arrays[name]=tuple(data[name+"_"+str(i)] for i in range(meta["level_tuple_lengths"][name]))
        else:
            arrays[name]=data[name]
    bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    offsets=data["node_offsets"]
    isolated=tuple(i for i,(a,b) in enumerate(zip(offsets,offsets[1:])) if b-a==1600 and bool((bank.parent[int(a):int(b)]<0).all()))
    episodes,_=build_episodes(bank,max_payload_bytes=2**30,isolated_hold_indices=isolated)
    cut,_=prepare_cut_plan_v1(bank,expected_haplotypes=1600,max_nodes=400000,max_holds=128,isolated_hold_indices=isolated)
    counts=np.diff(data["node_offsets"]).tolist()
    layout=baseline_layout(bank,counts,episodes.member_hold,holds_per_block=4)
    z=layout[0]
    plan=LevelSweepPlan.build(sweep_bank(z))
    assert plan.transport is not None
    assert plan.to("cpu") is plan
    assert plan.transport.untyped_storage().nbytes()==plan.tensor_bytes()
    assert len({x.untyped_storage().data_ptr() for x in plan._buffers()})==1
    saved=torch.load(CHECKPOINT,map_location="cpu",weights_only=True,mmap=True)
    model=frozen.AmericanTreeLoomC3FullV1()
    model.load_state_dict(saved["model_state"],strict=True)
    candidate=copy.deepcopy(model)
    eligible=episodes.episode_size.index_select(0,episodes.member_episode.long())>=2
    torch.manual_seed(2701)
    initial=torch.randn(cut.group_offsets[4]-cut.group_offsets[0],8)
    def run(cls,selected):
        # CPU diagnostic calls actual methods; production constructor and CUDA
        # guards remain intact and are never monkeypatched or weakened.
        executor=object.__new__(cls)
        executor.model,executor.bank=selected,bank
        executor.bundle=SimpleNamespace(episodes=episodes,cut=cut)
        executor.device=torch.device("cpu")
        executor.tick=lambda stage:None
        executor.profile=NullProfileV1()
        executor.layout=()
        executor.sweep_plans={0:plan}
        selected.zero_grad(set_to_none=True)
        memory_value=initial.clone().requires_grad_(True)
        calls=[0,0]
        def first():
            calls[0]+=1
            return executor._first(z)
        def second(up,value):
            calls[1]+=1
            return executor._second(z,up,value,eligible)
        up,_=_cp(first)
        output=_cp(second,up,memory_value)
        (output*torch.linspace(.1,1.,output.shape[-1])).mean().backward()
        gradients={name:p.grad.detach().clone() for name,p in selected.named_parameters() if p.grad is not None}
        assert calls[0]>=2 and calls[1]>=2
        return output.detach(),gradients,memory_value.grad,calls
    original=run(IntegratedStagedExecution,model)
    result=run(MinimalTransferLevelExecution,candidate)
    torch.testing.assert_close(original[0],result[0],atol=1e-6,rtol=1e-5)
    assert original[1].keys()==result[1].keys()
    for key in original[1]:
        torch.testing.assert_close(original[1][key],result[1][key],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(original[2],result[2],atol=1e-6,rtol=1e-5)
    assert not torch.cuda.is_initialized()
    receipt=dict(schema="level_local_minimal_transfer_executor_CPU_wiring_v2",status="pass",actual_native_holds=4,
        all_haplotypes=1600,source_chunk_sites=256,model_checkpoint=str(CHECKPOINT),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        output_error=float((original[0]-result[0]).abs().max()),
        parameter_gradient_error=max(float((original[1][k]-result[1][k]).abs().max()) for k in original[1]),
        memory_input_gradient_error=float((original[2]-result[2]).abs().max()),
        connected_parameter_tensors=len(original[1]),original_checkpoint_calls=original[3],candidate_checkpoint_calls=result[3],
        packed_single_transfer_storage=True,CUDA_initialized=False,
        scope="Actual first/second methods with original site,clade,episode,cut injection. No chromosome history,GRU,optimizer,full-parent or GPU parity claim.")
    (OUT/"executor_v2_cpu_results.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
