"""One bounded FP32 observed-chunk comparison, missing case and slice forward."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import resource
import statistics
import time
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2
from research.scalability_20261024.categorical_site.categorical_site import (
    categorical_features, categorical_site_chunk, categorical_encode_sites)

OUT = Path("research/scalability_20261024/categorical_site")
PACKET = Path("research/scalability_20261024/execution/representative_slice.npz")
CHECKPOINT = Path("research/repairs/TreeLoom_resume94_RAM_headroom_v1/checkpoint.pt")
ATOL = RTOL = 1e-4


def ref(path):
    path = Path(path)
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def max_error(a,b):
    return float((a-b).abs().max()) if a.numel() else 0.


def current(model,*args):
    return model._site_chunk(*args)


def compare_outputs(a,b):
    for old,new in zip(a,b):
        torch.testing.assert_close(old,new,atol=ATOL,rtol=RTOL)
    assert torch.equal(a[2],b[2])
    return {"sum_max_abs_error":max_error(a[0],b[0]),
            "max_max_abs_error":max_error(a[1],b[1]),
            "counts_exact":True}


def gradient_probe(model,fn,args):
    model.zero_grad(set_to_none=True)
    started=time.perf_counter()
    sums,maxima,counts=fn(model,*args)
    means=sums/counts.clamp_min(1)
    channel_weights=torch.linspace(.1,1.,means.shape[-1]).reshape(1,1,-1)
    loss=(means*channel_weights).mean()+(maxima*channel_weights.flip(-1)).mean()
    loss.backward()
    elapsed=time.perf_counter()-started
    gradients={name:p.grad.detach().clone() for name,p in model.site_mlp.named_parameters()}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    return {"loss":float(loss.detach()),"forward_backward_seconds":elapsed},gradients


def compare_gradients(a,b):
    records={}
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=ATOL,rtol=RTOL)
        records[key]={"max_abs_error":max_error(a[key],b[key]),
            "reference_max_abs":float(a[key].abs().max()),
            "candidate_max_abs":float(b[key].abs().max())}
    return records


def make_args(bank,start,end):
    holds,inverse=torch.unique_consecutive(bank.site_hold[start:end],return_inverse=True)
    return (bank.genotype[start:end],bank.log_donor_allele_frequency[start:end],
            bank.donor_call_mask[start:end],bank.panel_allele_frequency[start:end],
            bank.site_entropy[start:end],inverse,len(holds))


def check_cpu():
    assert not torch.cuda.is_initialized()
    assert all(p.device.type=="cpu" for p in MODEL.parameters())
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 4*2**30


def main():
    global MODEL
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started=time.perf_counter()
    packet_meta=json.loads(PACKET.with_suffix(".json").read_text())
    assert ref(PACKET)==packet_meta["packet_ref"]
    assert packet_meta["native_hold_range"]==[0,128]
    assert packet_meta["site_global_chunk_alignment"]==0
    assert ref(CHECKPOINT)["sha256"]=="5ceebcd75ae978689f987a702ea5a157d6ddc2bd7d057e6aded56627bbf45276"
    with np.load(PACKET,allow_pickle=False) as z:
        fields=("sample_index","genotype","log_donor_allele_frequency",
                "donor_call_mask","panel_allele_frequency","site_entropy",
                "site_hold","spans")
        bank=SimpleNamespace(**{k:torch.from_numpy(z[k].copy()) for k in fields})
    saved=torch.load(CHECKPOINT,map_location="cpu",weights_only=True,mmap=True)
    assert saved["complete_parent_updates"]==94
    MODEL=static_model_v2.AmericanTreeLoomC3FullV1()
    MODEL.load_state_dict(saved["model_state"],strict=True)
    MODEL.eval()
    del saved
    candidate=copy.deepcopy(MODEL)
    args=make_args(bank,0,256)
    features,_,_=categorical_features(*args[:5])
    assert torch.isfinite(features).all()
    cold={}
    warm={"current":[],"categorical":[]}
    with torch.no_grad():
        tick=time.perf_counter();old=current(MODEL,*args);cold["current"]=time.perf_counter()-tick
        tick=time.perf_counter();new=categorical_site_chunk(candidate,*args);cold["categorical"]=time.perf_counter()-tick
        forward_comparison=compare_outputs(old,new)
        for _ in range(3):
            tick=time.perf_counter();old=current(MODEL,*args);warm["current"].append(time.perf_counter()-tick)
            tick=time.perf_counter();new=categorical_site_chunk(candidate,*args);warm["categorical"].append(time.perf_counter()-tick)
    old_gradient,old_grads=gradient_probe(MODEL,current,args)
    new_gradient,new_grads=gradient_probe(candidate,categorical_site_chunk,args)
    real_gradients=compare_gradients(old_grads,new_grads)
    check_cpu()
    print(json.dumps({"phase":"representative_chunk_passed","forward":forward_comparison,
                      "current_forward_backward_seconds":old_gradient["forward_backward_seconds"],
                      "categorical_forward_backward_seconds":new_gradient["forward_backward_seconds"]}),flush=True)
    # One tiny mixed-missing case, based on the actual observed feature rows.
    # Includes an all-missing hold, repeated identical site rows (max ties),
    # unavailable donor cohorts and an unused zero-probability allele slot.
    edge=[x[:8].clone() for x in args[:5]]
    edge[0]=edge[0][:,:32].clone()
    edge[0][0]=-1
    edge[0][1,::3]=-1
    for item in edge:
        item[3]=item[2]
    edge[2][4]=False
    edge[2][5,::2]=False
    edge[1]=torch.cat((edge[1],torch.zeros(8,1,edge[1].shape[2])),dim=1)
    edge[3]=torch.cat((edge[3],torch.zeros(8,1)),dim=1)
    edge_args=(*edge,torch.tensor([0,1,1,1,1,1,1,1],dtype=torch.long),2)
    with torch.no_grad():
        edge_comparison=compare_outputs(current(MODEL,*edge_args),
            categorical_site_chunk(candidate,*edge_args))
        assert torch.isfinite(categorical_features(*edge_args[:5])[0]).all()
    edge_old,edge_old_grads=gradient_probe(MODEL,current,edge_args)
    edge_new,edge_new_grads=gradient_probe(candidate,categorical_site_chunk,edge_args)
    edge_gradients=compare_gradients(edge_old_grads,edge_new_grads)
    check_cpu()
    # One forward per method for all681 sites, with unchanged global256 chunks,
    # chunk-crossing hold pooling and the trailing169-site chunk.
    with torch.no_grad():
        tick=time.perf_counter();old_slice=MODEL.encode_sites(bank);old_slice_seconds=time.perf_counter()-tick
        tick=time.perf_counter();new_slice=categorical_encode_sites(candidate,bank);new_slice_seconds=time.perf_counter()-tick
        torch.testing.assert_close(old_slice,new_slice,atol=ATOL,rtol=RTOL)
        slice_error=max_error(old_slice,new_slice)
    check_cpu()
    result={
        "schema":"TreeLoom_categorical_site_FP32_CPU_measurement_v1",
        "status":"passed","source_ref":ref(OUT/"categorical_site.py"),
        "measurement_source_ref":ref(__file__),"packet_ref":ref(PACKET),
        "checkpoint_ref":ref(CHECKPOINT),"checkpoint_updates":94,
        "current_site_source_ref":ref("src/arggnn/american_c3_gnn_model_v1.py"),
        "current_pool_source_ref":ref("src/arggnn/mutation_routed_whole_tree_gnn_v1.py"),
        "CPU_threads":4,"FP32":True,"atol":ATOL,"rtol":RTOL,
        "CUDA_initialized":torch.cuda.is_initialized(),"optimizer_created":False,
        "truth_read":False,"live_source_modified":False,
        "real_chunk":{"sites":256,"haplotypes":1600,"allele_axis":args[1].shape[1],
            "current_MLP_rows":256*1600,
            "categorical_MLP_rows":256*args[1].shape[1],
            "category_runtime_hashing":False,"forward":forward_comparison,
            "parameter_gradients":real_gradients,
            "current_gradient_probe":old_gradient,
            "categorical_gradient_probe":new_gradient},
        "mixed_missing_case":{"sites":8,"haplotypes":32,"allele_axis":edge_args[1].shape[1],
            "all_missing_hold":True,"max_ties_present":True,
            "unavailable_donor_masks":True,"unused_zero_panel_category":True,
            "forward":edge_comparison,"parameter_gradients":edge_gradients,
            "current_gradient_probe":edge_old,"categorical_gradient_probe":edge_new},
        "same_process_first_call_seconds":cold,
        "warm_forward_seconds":warm,
        "warm_forward_median_seconds":{k:statistics.median(v) for k,v in warm.items()},
        "real_chunk_warm_forward_speedup":statistics.median(warm["current"])/statistics.median(warm["categorical"]),
        "real_chunk_forward_backward_speedup":old_gradient["forward_backward_seconds"]/new_gradient["forward_backward_seconds"],
        "slice_forward":{"sites":len(bank.genotype),"native_holds":128,"haplotypes":1600,
            "global_chunk_size":256,"source_site_range":[0,len(bank.genotype)],
            "current_seconds":old_slice_seconds,"categorical_seconds":new_slice_seconds,
            "speedup":old_slice_seconds/new_slice_seconds,"max_abs_error":slice_error},
        "process_peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "elapsed_seconds":time.perf_counter()-started,
        "semantics":"Per-site allele table, finite inactive padding; gather back before unchanged sum/max/count. Current parameters recomputed every call. No learned state cache, inference simplification or changed chunk/source identities.",
        "limitations":["One observed TRAIN parent start slice and saved94 checkpoint.",
            "Practical FP32 closeness, not bitwise equality: GEMM and parameter-gradient accumulation shape changes.",
            "Gradient comparison is one mean-pooled weighted sum/max probe per real/missing case; no full-model training-gradient claim.",
            "CPU4-thread execution only; CUDA, allocator effects, checkpoint-recompute integration and whole-update speed unmeasured.",
            "Prototype pooling adapter is for diagnostic execution; existing supervised source remains untouched.",
            "The first681-site packet omits neighboring holds from the final shared global256chunk, as documented in its metadata."]}
    with (OUT/"measurement.json").open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status":"failed","error_type":type(error).__name__,
                          "error":str(error)}),flush=True)
        raise
