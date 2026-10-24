"""Bounded observed-only primitive census and saved94 first-sweep FP32 diagnostic."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math,os,resource,time
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as runtime
from research.benchmark_rebuild_20261024.american.treeloom_speed import training_v2 as validation

BASE=Path("research/scalability_20261024/model")
PACKET=Path("research/scalability_20261024/execution/representative_slice.npz")
CHECKPOINT=Path("research/repairs/TreeLoom_resume94_RAM_headroom_v1/checkpoint.pt")
GIB=2**30

def need(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

def rss():return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")

def intern(table,key):
    value=table.get(key)
    if value is None:value=len(table);table[key]=value
    return value

def raw_site_tokens(bank):
    genotype=bank.genotype;channels=6;called=genotype>=0;chosen=genotype.long().clamp_min(0)
    logfreq=bank.log_donor_allele_frequency.gather(1,chosen[...,None].expand(-1,-1,channels))
    available=bank.donor_call_mask[:,None,:].expand(-1,genotype.shape[1],-1)&called[...,None]
    logfreq=torch.where(available,logfreq,torch.zeros_like(logfreq))
    panel=bank.panel_allele_frequency.gather(1,chosen)
    panel_log=torch.where(called,panel,torch.ones_like(panel)).log()
    entropy=bank.site_entropy[:,None].expand_as(panel_log)
    features=torch.cat((logfreq,available.to(logfreq.dtype),panel_log[...,None],entropy[...,None],called[...,None].to(logfreq.dtype)),-1)
    features=torch.where(called[...,None],features,torch.zeros_like(features)).numpy()
    tokens=np.empty(genotype.shape,dtype=np.int64);table={};geno=genotype.numpy()
    for site,row in enumerate(geno):
        for allele in np.unique(row):
            selected=np.flatnonzero(row==allele)
            token=intern(table,features[site,selected[0]].tobytes())
            tokens[site,selected]=token
    return tokens,len(table)

def census(bank,z):
    n=len(bank.parent);h=bank.sample_index.shape[1];t=bank.sample_index.shape[0]
    parent=bank.parent.numpy();sample=bank.sample_index.numpy();sample_column=np.full(n,-1,dtype=np.int64)
    sample_column[sample.reshape(-1)]=np.tile(np.arange(h),t)
    node_age_bits=bank.node_time.numpy().view(np.uint32);age_bits=bank.sample_age.numpy().view(np.uint32)
    span_bits=bank.spans.numpy().view(np.uint32);tokens,feature_count=raw_site_tokens(bank)
    base_ids=np.empty(n,dtype=np.int64);base_table={}
    for node in range(n):
        if sample_column[node]<0:base_ids[node]=intern(base_table,("node",int(node_age_bits[node]),bool(parent[node]<0),False))
    hold=bank.site_hold.numpy();global_sites=z["site_global_index"]
    for tree in range(t):
        lo,hi=map(int,np.searchsorted(hold,[tree,tree+1]));segments=[]
        first=lo
        while first<hi:
            end=first+1
            while end<hi and global_sites[end]//256==global_sites[first]//256:end+=1
            segments.append((first,end));first=end
        for column,node in enumerate(sample[tree]):
            pool_key=(int(span_bits[tree]),tuple(tuple(map(int,tokens[a:b,column])) for a,b in segments))
            node_key=(int(node_age_bits[node]),bool(parent[node]<0),True)
            key=("sample",node_key,pool_key,int(bank.reference_class[column]),int(age_bits[column]))
            base_ids[node]=intern(base_table,key)
    gates=np.concatenate((bank.edge_geometry.numpy(),bank.branch_support_true.numpy()),axis=1).copy()
    _,gate_ids=np.unique(gates.view(np.dtype((np.void,gates.dtype.itemsize*9))).reshape(-1),return_inverse=True)
    children=[[] for _ in range(n)]
    for level in bank.upward_children:
        for child in level.numpy():children[int(parent[child])].append(int(child))
    up=np.empty(n,dtype=np.int64);shape=np.empty(n,dtype=np.int64);up_table={};shape_table={}
    for level in bank.upward_levels:
        for node in level.numpy():
            kids=children[node]
            shape[node]=intern(shape_table,(int(node_age_bits[node]),bool(parent[node]<0),int(sample_column[node]),tuple(int(shape[c]) for c in kids)))
            up[node]=intern(up_table,(int(base_ids[node]),max(1,len(kids)),tuple((int(up[c]),int(gate_ids[c])) for c in kids)))
    persistent=np.column_stack((z["native_node_ids"],shape))
    _,groups=np.unique(persistent,axis=0,return_inverse=True)
    geometry=bank.edge_geometry.numpy().copy()
    _,geometry_ids=np.unique(geometry.view(np.dtype((np.void,geometry.dtype.itemsize*4))).reshape(-1),return_inverse=True)
    nonroot=np.flatnonzero(parent>=0);degree=np.bincount(parent[nonroot],minlength=n)
    message_keys=np.column_stack((groups[nonroot],z["native_node_ids"][parent[nonroot]],geometry_ids[nonroot],degree[parent[nonroot]]))
    _,message_groups=np.unique(message_keys,axis=0,return_inverse=True)
    report=dict(policy="primitive_input_computation_identity_lower_bound",occurrences=n,
        primitive_site_MLP_inputs=feature_count,primitive_base_sources=len(base_table),primitive_gate_sources=int(gate_ids.max())+1,
        unique_upward_states=len(up_table),guaranteed_upward_reuse_fraction=1-len(up_table)/n,
        age_sample_ordered_shape_groups=len(shape_table),persistent_subtree_groups=int(groups.max())+1,
        source_inputs_not_hidden_values_used=True,full_history_and_second_sweep_not_censused=True,
        global_site_chunk_boundaries_preserved=True)
    return report,groups,nonroot,message_groups

def spectrum(values,group_ids,weights):
    ids=torch.from_numpy(group_ids.astype(np.int64));weights=weights.to(dtype=torch.float32)
    count=int(ids.max())+1;totals=values.new_zeros(count,values.shape[1]);mass=values.new_zeros(count)
    totals.index_add_(0,ids,values*weights[:,None]);mass.index_add_(0,ids,weights)
    means=totals/mass.clamp_min(1e-30)[:,None];residual=values-means[ids]
    gram=(residual.T@(residual*weights[:,None]))/weights.sum()
    eigen,basis=torch.linalg.eigh(gram);order=torch.argsort(eigen,descending=True)
    eigen=eigen[order].clamp_min(0);basis=basis[:,order]
    residual_energy=float((residual.square()*weights[:,None]).sum());total_energy=float((values.square()*weights[:,None]).sum())
    records={}
    for rank in (1,2,4,8,16):
        approximation=(residual@basis[:,:rank])@basis[:,:rank].T
        error=torch.linalg.vector_norm(residual-approximation,dim=1)
        relative=error/torch.linalg.vector_norm(values,dim=1).clamp_min(1e-8)
        records[str(rank)]=dict(centered_energy_fraction=float(eigen[:rank].sum()/eigen.sum().clamp_min(1e-30)),
            pointwise_error_norm_p99=float(torch.quantile(error,.99)),pointwise_error_norm_max=float(error.max()),
            relative_pointwise_error_p99=float(torch.quantile(relative,.99)),relative_pointwise_error_max=float(relative.max()))
    counts=np.bincount(group_ids)
    return dict(rows=len(values),groups=count,repeated_groups=int((counts>=2).sum()),rows_in_repeated_groups=int(counts[counts>=2].sum()),
        within_group_residual_fraction_of_total_energy=residual_energy/max(total_energy,1e-30),ranks=records,
        FP32_eigenvalues=[float(v) for v in eigen],basis_scope="one_shared_channel_basis_after_per_group_centering")

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);started=time.monotonic();initial_rss=rss()
    protocol=ref(BASE/"measurement_protocol.json")
    need(ref(CHECKPOINT)["sha256"]=="5ceebcd75ae978689f987a702ea5a157d6ddc2bd7d057e6aded56627bbf45276","Exact saved94 weights required")
    packet_meta=json.loads(PACKET.with_suffix(".json").read_text())
    need(ref(PACKET)==packet_meta["packet_ref"] and packet_meta["native_hold_range"]==[0,128]
         and packet_meta["site_global_chunk_alignment"]==0,"Exact first128 observed packet required")
    z=np.load(PACKET,allow_pickle=False)
    fields=("parent","node_time","sample_index","edge_geometry","branch_support_true","spans","site_hold","genotype", "log_donor_allele_frequency","donor_call_mask","panel_allele_frequency","site_entropy","reference_class","sample_age")
    bank=SimpleNamespace(**{name:torch.from_numpy(z[name]) for name in fields})
    for name in ("upward_levels","upward_children","upward_parent_rank","downward_levels"):
        keys=sorted((k for k in z.files if k.startswith(name+"_")),key=lambda k:int(k.rsplit("_",1)[1]))
        setattr(bank,name,tuple(torch.from_numpy(z[k]) for k in keys))
    saved=torch.load(CHECKPOINT,map_location="cpu",weights_only=True,mmap=True)
    registry=runtime.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
    model=runtime.FreshTreeLoom(registry).build_model();validation.validate_saved_state(saved,model,registry,expected_steps=94)
    model.load_state_dict(saved["model_state"],strict=True);model.eval();del saved
    timings={}
    def check(stage):
        need(rss()-initial_rss<4*GIB,"Bounded diagnostic extra RSS exceeded")
        need(not torch.cuda.is_initialized(),"CPU-only diagnostic")
    check("loaded")
    with torch.no_grad():
        tick=time.monotonic();base,ug,dg,degree=model.prepare_tree_base_v1(bank);timings["current_base_and_site_encoder_seconds"]=time.monotonic()-tick
        check("base")
        tick=time.monotonic();upward,outside=model.sweep_tree_state_v1(bank,base,ug,dg,degree);timings["current_first_cavity_sweep_seconds"]=time.monotonic()-tick
        check("sweep")
        del base,dg,outside
        tick=time.monotonic();canonical,groups,nonroot,message_groups=census(bank,z);timings["primitive_census_seconds"]=time.monotonic()-tick
        check("census")
        node_offsets=z["node_offsets"];tree_of_node=np.repeat(np.arange(128),np.diff(node_offsets))
        physical_weights=bank.spans[torch.from_numpy(tree_of_node)]
        up_results={mode:spectrum(upward,groups,weights) for mode,weights in
            (("equal_occurrence",torch.ones(len(upward))),("physical_span",physical_weights))}
        index=torch.from_numpy(nonroot)
        messages=model.up_message(upward[index])*ug[index]/degree[bank.parent[index],None]
        message_results={mode:spectrum(messages,message_groups,weights) for mode,weights in
            (("equal_occurrence",torch.ones(len(messages))),("physical_span",physical_weights[index]))}
        check("spectra")
    result=dict(schema="TreeLoom_saved94_observed_first128_variation_measurement_v1",protocol_ref=protocol,
        packet_ref=ref(PACKET),checkpoint_ref=ref(CHECKPOINT),source_ref=ref(__file__),
        observed_only=True,truth_opened=False,optimizer_created=False,CUDA_initialized=torch.cuda.is_initialized(),
        full_model_prediction_or_accuracy_claimed=False,current_saved_steps=94,torch_threads=4,
        measured_holds=128,samples=1600,sites=len(bank.site_hold),node_occurrences=len(bank.parent),
        strict_source_identity_census=packet_meta["strict_first_sweep_upward_census"],primitive_input_census=canonical,
        upward_state_variation=up_results,upward_edge_message_variation=message_results,
        timings=timings,seconds=time.monotonic()-started,initial_RSS_bytes=initial_rss,current_RSS_bytes=rss(),
        process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        limitations="One TRAIN parent start slice and one saved checkpoint; first sweep only. Partial terminal site chunk has no cross-hold pooling; no full-parent floating-kernel or end-to-end parity claim. Spectral energy and pointwise vector error are not ancestry/boundary accuracy or training-gradient guarantees.")
    path=BASE/"checkpoint_variation.json"
    with path.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    print(json.dumps(dict(result_ref=ref(path),primitive_census=canonical,seconds=result["seconds"],peak_RSS_bytes=result["process_peak_RSS_bytes"],
        upward_energy_ranks=up_results["equal_occurrence"]["ranks"],message_energy_ranks=message_results["equal_occurrence"]["ranks"]),indent=2))

if __name__=="__main__":main()
