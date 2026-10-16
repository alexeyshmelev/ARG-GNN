"""One deferred ordinary FP32 H050 CPU routing/forward qualification.
No numerical import, IO product, subprocess, optimizer or test execution on import.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib,json,math
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
MANIFEST_REF={'path': 'research/reviews/H050_one_FP32_CPU_forward_qualification_source_20261016_v1/scientific_source_manifest.v1.json', 'sha256': '7fa6426f46483633ff260877bafd618e3e2ca2831a3d6f43e1a7d86c50d8a641'}
_RAN=False

def require(ok,message):
    if not ok:raise ValueError(message)

def raw(record,limit=1048576):
    p=ROOT/record["path"]
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,"Exact bounded source/metadata")
    data=p.read_bytes()
    require(hashlib.sha256(data).hexdigest()==record["sha256"],"Authenticated unchanged source")
    return data

def function_source(record,name,namespace):
    tree=ast.parse(raw(record))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/record["path"]),"exec"),namespace)
    return namespace[name]

def original_bank(manifest,torch,m):
    # Only the existing bank-construction statements, not that old test campaign.
    tree=ast.parse(raw(manifest["existing_bank_fixture_ref"]));nodes=tree.body
    start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Tuple)
        and [x.id for x in n.targets[0].elts]==["H","T"])
    stop=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="bank")
    selected=nodes[start:stop+1]
    require(selected[0].lineno==manifest["bank_source_lines"][0]
        and selected[-1].end_lineno==manifest["bank_source_lines"][1],"Unchanged existing tiny bank segment")
    scope=dict(torch=torch,math=math,MutationRoutedWholeTreeInputsV1=m.MutationRoutedWholeTreeInputsV1)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(ROOT/manifest["existing_bank_fixture_ref"]["path"]),"exec"),scope)
    bank=scope["bank"]
    require(bank.sample_index.shape==(7,952) and bank.query_column.shape==(686,),"Original tiny seven-hold whole-tree fixture")
    return bank

def reference_forward(network,arm,bank,episodes,cut,m,torch):
    """Independent composition, no encode_trees/forward*/predict/transfer call.
    Original primitive tree sweeps, couplings, memory, GRU and classifier are
    deliberately shared. This checks route/composition, not primitive correctness.
    """
    base,ug,dg,degree=network.prepare_tree_base_v1(bank,control="aligned")
    upward,outside=network.sweep_tree_state_v1(bank,base,ug,dg,degree)
    injection=network.clade_coupling(upward,episodes.flags)
    injection=injection+network.episode_coupling(upward,episodes,arm="own_state")
    memory_max=0.
    if arm!=m.CONTINUATION:
        delta=network.memory(upward,outside,bank.spans,cut)
        memory_max=float(delta.abs().max())
        injection=injection.index_add(0,cut.representatives,delta)
    second_up,second_out=network.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
    context=network.leaf_context_v1(bank,second_up,second_out)
    t,h=bank.sample_index.shape
    # Scalar loop is separate from original vectorized geometry construction.
    rows=[]
    for at in range(t):
        previous=bank.spans.new_zeros(()) if at==0 else (bank.spans[at-1]+bank.spans[at])*.5
        following=bank.spans.new_zeros(()) if at+1==t else (bank.spans[at]+bank.spans[at+1])*.5
        rows.append(torch.stack((bank.spans[at].log1p(),previous.log1p(),following.log1p()))/math.log(2000001.))
    geometry=torch.stack(rows)
    sequence=torch.cat((context,geometry[:,None].expand(t,h,3)),-1)
    history,_=network.genomic_gru(sequence)
    all_logits=network.classifier(torch.cat((context,history),-1))
    return all_logits[:,bank.query_column,:],memory_max

def run_single_CPU_qualification_v1(*,resource_check):
    global _RAN
    require(not _RAN and callable(resource_check),"Exactly one actual root-guarded invocation")
    _RAN=True
    resource_check("before_H050_CPU_fixture_source_and_numerical_imports")
    manifest=json.loads(raw(MANIFEST_REF))
    for record in manifest["source_refs"]:raw(record)
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only visibility before numerical imports")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch,numpy as np
    from dataclasses import fields
    from arggnn.ordered_clade_episode_gnn_v1 import build_clade_episodes_v1
    m=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.direct_readout_cut_memory_v1")
    hooks=importlib.import_module("research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.integration_hooks_v1")
    require(torch.get_default_dtype()==torch.float32 and not torch.is_autocast_enabled("cpu")
        and not torch.cuda.is_initialized(),"Ordinary FP32 CPU only")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    prior=torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(101)
        bank=original_bank(manifest,torch,m)
        episodes,episode_metadata=build_clade_episodes_v1(bank)
        cut,cut_metadata=m.memory_core.prepare_cut_plan_v1(bank)
        arrays,recipe=m.codec.encode_v1(cut,cut_metadata)
        import io
        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
        bundle=m.bind_compact_recipe_v1(bank,episodes,episode_metadata,stream.getvalue(),recipe)
        bundle.validate()
        # CPU-route unit namespace contains only unchanged authenticated bodies.
        tensor_bytes=function_source(manifest["original_size_helper_ref"],"tensor_bytes_v1",dict(fields=fields))
        route=function_source(manifest["original_producer_ref"],"device_parent",
            dict(require=m.require,h45=SimpleNamespace(h038=SimpleNamespace(tensor_bytes_v1=tensor_bytes))))
        cfg=json.loads(raw(manifest["original_TRAIN_config_ref"]))
        checks=[0]
        def check():
            resource_check("H050_original_device_parent_CPU")
            checks[0]+=1
        ctx=dict(guard=check,cfg=cfg,device=torch.device("cpu"))
        transferred,mass=route(ctx,bundle)
        require(mass is None and checks[0]==1,"Actual no-target original CPU transfer route")
        for field in fields(bank):
            left,right=getattr(bank,field.name),getattr(transferred.bank,field.name)
            left=left if isinstance(left,tuple) else (left,)
            right=right if isinstance(right,tuple) else (right,)
            require(len(left)==len(right) and all(a.dtype==b.dtype and a.shape==b.shape
                and b.device.type=="cpu" and torch.equal(a,b) for a,b in zip(left,right)),
                "Original full-bank observation values/dtypes/devices preserved")
        resource_check("before_H050_four_arm_models")
        base=m.OrderedCladeEpisodeGNNV1(**m.MODEL_KWARGS,arm="own_state")
        with torch.no_grad():
            base.clade_coupling.O.weight.copy_(torch.linspace(-.015,.025,256).reshape(32,8))
            base.episode_coupling.O.weight.copy_(torch.linspace(-.02,.03,256).reshape(32,8))
        family=m.family_from_base_state_v1(base.state_dict(),checkpoint_memory=True)
        for arm in m.MEMORY_ARMS:
            with torch.no_grad():
                family[arm].memory.O.weight.copy_(torch.linspace(-.03,.04,256).reshape(32,8))
        rows={}
        for arm in m.ARMS:
            resource_check("H050_CPU_forward_arm_"+arm)
            candidate=family[arm].to("cpu").eval().requires_grad_(False)
            original=(m.OrderedCladeEpisodeGNNV1(**m.MODEL_KWARGS,arm="own_state")
                if arm==m.CONTINUATION else m.DirectReadoutCutMemoryGNNV1(memory_arm=arm))
            original.load_state_dict(candidate.state_dict(),strict=True)
            original=original.to("cpu").eval().requires_grad_(False)
            m.validate_arm_model_v1(arm,candidate);m.validate_arm_model_v1(arm,original)
            before={k:v.detach().clone() for k,v in candidate.state_dict().items()}
            with torch.no_grad(),torch.autocast(device_type="cpu",enabled=False):
                routed=hooks.predict_holds_v1(arm,candidate,transferred)
                expected,memory_max=reference_forward(original,arm,bank,episodes,bundle.cut,m,torch)
                probability=routed.softmax(-1);reference=expected.softmax(-1)
            require(routed.shape==expected.shape==probability.shape==(7,686,4)
                and routed.dtype==expected.dtype==probability.dtype==torch.float32
                and routed.device.type==expected.device.type=="cpu","All four original FP32 output shapes/devices")
            require(bool(torch.isfinite(routed).all()) and bool(torch.isfinite(expected).all())
                and bool(torch.isfinite(probability).all()) and bool(torch.isfinite(reference).all()),
                "Finite original/reference probabilities and logits")
            torch.testing.assert_close(routed,expected,atol=5e-5,rtol=5e-4)
            torch.testing.assert_close(probability,reference,atol=5e-5,rtol=5e-4)
            torch.testing.assert_close(probability.sum(-1),torch.ones((7,686)),atol=5e-5,rtol=5e-4)
            require(all(torch.equal(before[k],v) for k,v in candidate.state_dict().items())
                and all(p.grad is None for p in candidate.parameters())
                and all(p.grad is None for p in original.parameters()),"Zero optimization/backward and unchanged model state")
            require(arm==m.CONTINUATION or memory_max>0,"Synthetic memory branch genuinely active")
            rows[arm]=dict(logits_max_abs_error=float((routed-expected).abs().max()),
                probabilities_max_abs_error=float((probability-reference).abs().max()),
                probability_sum_max_abs_error=float((probability.sum(-1)-1).abs().max()),
                reference_memory_residual_max_abs=memory_max,
                parameter_count=sum(p.numel() for p in candidate.parameters()),
                FP32_CPU=True,passed_allclose=True,parameters_unchanged=True)
            del original
        bundle.validate();transferred.validate()
        require(set(rows)==set(m.ARMS) and not torch.cuda.is_initialized(),"All four CPU arms without CUDA")
        resource_check("after_H050_CPU_forward_qualification")
        result=dict(schema="H050_single_original_FP32_CPU_forward_comparison_v1",passed=True,
            precision="fp32",optimizer_steps=0,all_four_arms=True,all_probabilities_finite=True,
            passed_allclose=True,atol=5e-5,rtol=5e-4,seed=101,holds=7,haplotypes=952,queries=686,classes=4,
            source_fixture_reused=True,CPU_observation_transfer_preserved=True,
            independent_forward_composition=True,shared_primitive_kernel_correctness_claimed=False,
            CPU_runtime_ref=manifest["CPU_runtime_ref"],CPU_metadata_ref=manifest["CPU_metadata_ref"],
            original_producer_ref=manifest["original_producer_ref"],no_truth_read=True,no_optimization=True,
            actual_checkpoint_IO_or_full_evaluation_fit_qualified=False,arms=rows)
        return dict(schema="H050_single_CPU_qualification_callable_result_v1",
            fixture_result=result,root_terminal_consumption_not_asserted=True,
            admission_or_execution_authority_granted=False)
    finally:
        torch.random.set_rng_state(prior)
