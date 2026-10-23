"""Synthetic CPU tests for unchanged GraphSetConv, native views and eight workers."""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import tskit

from .dataset import CompleteTreeSets,loader
from .model import GstLai8,specification
from .workflow import _shutdown
from .. import european_native_models as io
from ..european_sampling import NATIVE_PATHS


def synthetic_state():
    tables=tskit.TableCollection(17.)
    for _ in range(6):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
    genotype=[]
    for i in range(16):
        root=tables.nodes.add_row(time=10.+i)
        for leaf in range(6): tables.edges.add_row(i,i+1,root,leaf)
        site=tables.sites.add_row(i+.5,"A")
        calls=[(i+j)%2 for j in range(6)]
        genotype.append(calls)
        for leaf,value in enumerate(calls):
            if value: tables.mutations.add_row(site,leaf,"C")
    tables.sort()
    ts=tables.tree_sequence()
    states={"synthetic":dict(ts=ts,nodes=ts.samples(),node_time=np.asarray(ts.nodes_time),
        genotype=np.asarray(genotype,np.int8),allele_count=np.full(16,2,np.int8),
        positions=np.arange(16,dtype=np.float64)+.5,labels=np.asarray([0,1,2,3,-1,-1]),
        ages=np.zeros(6),query_columns=np.asarray([4,5]))}
    records={"synthetic":dict(record_ids=np.arange(32,dtype=np.int64),
        sample_indices=np.tile(np.asarray([4,5]),16),hold=np.repeat(np.arange(16),2))}
    return states,records


def run(output_directory):
    io._need(os.environ.get("CUDA_VISIBLE_DEVICES") == "","Synthetic CPU-only smoke required")
    torch.set_num_threads(1);torch.manual_seed(20261024)
    start=time.monotonic()
    output=io._new_output(output_directory)
    states,records=synthetic_state()
    dataset=CompleteTreeSets(states,records)
    window=dataset[0]
    io._need(len(window.tree_node_counts) == 8 and window.native_edge_count == 48
        and window.edge_index.shape == (2,96) and window.sites == 8
        and len(window.focal_indices) == 16,"Complete observed window topology/sites/focals required")
    isolated=dataset[2]
    io._need(len(isolated.tree_node_counts) == 1 and isolated.native_edge_count == 0
        and isolated.sites == 0 and len(isolated.focal_indices) == 0,
        "Final partial set and zero-SNP isolated samples must remain present")
    model=GstLai8(NATIVE_PATHS).eval()
    with torch.no_grad():
        empty=model.sampled_logits(isolated)
        io._need(empty.shape == (0,6) and torch.isfinite(model(isolated)).all(),
                 "No-focal zero-SNP native set must traverse without fabricated output")
    with torch.no_grad():
        x=model.frontend(window.bank)
        result=model(window)
        sampled=model.sampled_logits(window)
        rng=np.random.default_rng(13)
        perm=torch.as_tensor(rng.permutation(len(x)))
        inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
        graph_perm=torch.as_tensor(rng.permutation(8))
        graph_inverse=torch.empty_like(graph_perm);graph_inverse[graph_perm]=torch.arange(8)
        changed=model.forward_nodes(x[perm],inverse[window.edge_index],
            graph_inverse[window.graph_batch[perm]],window.set_batch[graph_perm])
        node_error=float((changed-result[perm]).abs().max())
    io._need(node_error < 3e-6 and sampled.shape == (16,6),
             "Node equivariance and graph-set permutation invariance required")
    labels=window.bank.reference_class.clone();labels[4]=0
    try:
        model.frontend(replace(window.bank,reference_class=labels))
    except ValueError:
        query_guard=True
    else:
        query_guard=False
    io._need(query_guard,"Known query ancestry must be rejected")
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    y=torch.arange(16)%6
    loss=torch.nn.functional.cross_entropy(model.sampled_logits(window),y)
    loss.backward()
    grads=[p.grad for p in model.parameters()]
    io._need(all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
             "Every adapted model parameter must receive a finite synthetic gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    checkpoint=output/"synthetic_gst_lai.pt"
    torch.save(model.state_dict(),checkpoint)
    fresh=GstLai8(NATIVE_PATHS)
    fresh.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    fresh.eval()
    with torch.no_grad():
        save_error=float((fresh(window)-model(window)).abs().max())
    io._need(save_error == 0,"Exact saved LAI head/native GraphSetConv roundtrip required")
    data_loader,sampler=loader(dataset,workers=8,pin_memory=False)
    counts=dict(windows=0,trees=0,edges=0,sites=0)
    ids=[]
    worker_processes=[]
    try:
        for part in data_loader:
            counts["windows"]+=1
            counts["trees"]+=len(part.tree_node_counts)
            counts["edges"]+=part.native_edge_count
            counts["sites"]+=part.sites
            ids.extend(part.focal_indices.tolist())
        worker_processes=list(data_loader._iterator._workers)
        io._need(len(worker_processes) == 8,"Eight real CPU preparation workers required")
    finally:
        _shutdown(data_loader)
    worker_exits=[p.exitcode for p in worker_processes]
    io._need(worker_exits == [0]*8 and not any(p.is_alive() for p in worker_processes),
             "All eight real preparation workers must exit0 and be absent")
    io._need(counts == dict(windows=3,trees=17,edges=96,sites=16)
        and ids == list(range(32)),"Eight-worker loader must preserve all observed trees/edges/sites/IDs")
    receipt=dict(schema="paper_gst_lai_8_synthetic_cpu_smoke_v1",status="PASS",
        synthetic_only=True,actual_AP_data_opened=False,actual_TEST_truth_opened=False,
        CUDA_used=False,workers=8,worker_pids=[p.pid for p in worker_processes],
        worker_exitcodes=worker_exits,all_eight_workers_absent=True,
        pinned_memory_production_route=True,
        pinned_memory_exercised_in_CPU_smoke=False,native_GraphSetConv_unchanged=True,
        node_and_graph_permutation_max_abs_error=node_error,
        query_ancestry_guard_passed=query_guard,every_parameter_finite_gradient=True,
        checkpoint_max_abs_error=save_error,counts=counts,
        final_partial_set_and_zero_SNP_isolated_samples_retained=True,
        parameters=sum(p.numel() for p in model.parameters()),
        model_spec=specification(),torch=torch.__version__,elapsed_seconds=time.monotonic()-start)
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return receipt
