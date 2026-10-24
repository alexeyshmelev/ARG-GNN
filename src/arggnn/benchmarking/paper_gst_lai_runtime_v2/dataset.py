"""Streaming complete native tree sets using the actual observed-bank encoder."""
from __future__ import annotations
from dataclasses import dataclass,fields,replace
import math
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .. import european_native_models as io
from .. import european_treeloom as european
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_observations_v1 as observations


@dataclass(frozen=True)
class Window:
    parent_id: str
    start_hold: int
    bank: object
    graph_batch: object
    set_batch: object
    edge_index: object
    focal_indices: object
    focal_local_hold: object
    focal_sample_column: object
    tree_node_counts: tuple
    native_edge_count: int
    sites: int

    def to(self,device,non_blocking=True):
        values = {}
        for f in fields(self):
            value = getattr(self,f.name)
            if f.name == "bank":
                converted = {}
                for item in fields(value):
                    tensor = getattr(value,item.name)
                    converted[item.name] = (tuple(v.to(device,non_blocking=non_blocking) for v in tensor)
                        if isinstance(tensor,tuple) else tensor.to(device,non_blocking=non_blocking))
                values[f.name] = MutationRoutedWholeTreeInputsV1(**converted)
            elif torch.is_tensor(value):
                values[f.name] = value.to(device,non_blocking=non_blocking)
        return replace(self,**values)

    def pin_memory(self):
        values = {}
        for f in fields(self.bank):
            value = getattr(self.bank,f.name)
            values[f.name] = tuple(v.pin_memory() for v in value) if isinstance(value,tuple) else value.pin_memory()
        pinned = MutationRoutedWholeTreeInputsV1(**values)
        extra = {f.name:getattr(self,f.name).pin_memory() for f in fields(self)
                 if torch.is_tensor(getattr(self,f.name))}
        return replace(self,bank=pinned,**extra)


def from_native_trees(parent_id,start,trees,intervals,state,records):
    left,right = intervals[0][0],intervals[-1][1]
    lo,hi = np.searchsorted(state["positions"],[left,right])
    io._need(sum(len(tree.parent) for tree in trees) <= 262144 and int(hi-lo) <= 16384,
             "Complete eight-tree set exceeds explicit node/site bound; no truncation")
    bank,meta = observations.encode_block(trees,intervals,state["genotype"][lo:hi],
        state["allele_count"][lo:hi],state["positions"][lo:hi],state["labels"],state["ages"],
        num_classes=4,age_split=350.,query_columns=state["query_columns"])
    counts = tuple(len(t.parent) for t in trees)
    graph = torch.repeat_interleave(torch.arange(len(trees)),torch.tensor(counts))
    child = torch.nonzero(bank.parent >= 0,as_tuple=False).flatten()
    parent = bank.parent[child]
    io._need(torch.equal(graph[child],graph[parent]) and sum(counts) == bank.parent.numel(),
             "Every native edge must remain within its complete tree")
    edge = torch.stack((torch.cat((child,parent)),torch.cat((parent,child))))
    selected = np.flatnonzero((records["hold"] >= start) & (records["hold"] < start+len(trees)))
    return Window(parent_id,start,bank,graph,torch.zeros(len(trees),dtype=torch.long),edge,
        torch.as_tensor(records["record_ids"][selected].copy(),dtype=torch.long),
        torch.as_tensor(records["hold"][selected]-start,dtype=torch.long),
        torch.as_tensor(records["sample_indices"][selected].copy(),dtype=torch.long),
        counts,int(child.numel()),int(hi-lo))


class CompleteTreeSets(Dataset):
    """All native trees are covered once per pass, including sets without focals."""
    def __init__(self,states,records):
        self.states,self.records = states,records
        self.parent_order = tuple(states)
        self.windows = [(p,start) for p in self.parent_order
                        for start in range(0,states[p]["ts"].num_trees,8)]
        self.parent_windows = {p:[] for p in states}
        for i,(p,_) in enumerate(self.windows):
            self.parent_windows[p].append(i)
        self.workspaces = {}

    def __len__(self): return len(self.windows)

    def __getstate__(self):
        value = self.__dict__.copy()
        value["workspaces"] = {}
        return value

    def __getitem__(self,index):
        parent,start = self.windows[index]
        state = self.states[parent]
        ts = state["ts"]
        if parent not in self.workspaces:
            self.workspaces[parent] = observations.native_workspace_v1(ts)
        tree = ts.at_index(start)
        trees,intervals = [],[]
        for _ in range(min(8,ts.num_trees-start)):
            trees.append(observations.native_tree(tree,state["nodes"],
                node_time=state["node_time"],workspace=self.workspaces[parent]))
            intervals.append(tuple(tree.interval))
            tree.next()
        return from_native_trees(parent,start,trees,intervals,state,self.records[parent])


def worker_init(worker_id):
    torch.set_num_threads(1)
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")


class ParentOrderSampler:
    def __init__(self,dataset):
        self.dataset=dataset
        self.order=list(dataset.parent_order)
    def __iter__(self):
        for parent in self.order:
            yield from self.dataset.parent_windows[parent]
    def __len__(self): return len(self.dataset)


def loader(dataset,*,workers=4,pin_memory=True):
    io._need(workers == 4,"Declared four-worker memory-repair CPU route required")
    sampler = ParentOrderSampler(dataset)
    result = DataLoader(dataset,batch_size=None,sampler=sampler,num_workers=workers,
        pin_memory=pin_memory,persistent_workers=True,prefetch_factor=1,
        multiprocessing_context="spawn",worker_init_fn=worker_init,timeout=1800)
    return result,sampler


def load_parents(inputs,plan,*,split,protocol_id,max_sites,max_metadata_bytes,check):
    states,records,proof = {},{},{}
    expected = plan["train_parent_ids"] if split == "train" else plan["test_parent_ids"]
    io._need(set(inputs) == set(expected),"Every declared independent parent required")
    for parent in expected:
        item=inputs[parent]
        state=european.load_relate_observation(item["inference_ref"],item["observed_vcf_ref"],
            protocol_id=protocol_id,max_sites=max_sites,max_metadata_bytes=max_metadata_bytes,
            resource_check=check)
        r=european._sampled_records(item["sampled_registry_ref"],observation_ref=item["inference_ref"],
            parent_id=parent,split=split,max_metadata_bytes=max_metadata_bytes)
        ts=state["ts"]
        io._need(state["manifest"]["parent_id"] == parent and state["manifest"]["split"] == split
            and len(r["record_ids"]) == sum(plan["parents"][parent]["cohort_quotas"].values())
            and r["class_order"].tolist() == plan["native_path_names"],
            "Exact declared parent/split/native classes and all frozen focal rows required")
        local=r["positions_bp0"]-state["coordinate_offset_bp"]
        io._need(((local>=0)&(local<ts.sequence_length)).all()
            and np.array_equal(r["record_ids"],np.arange(len(r["record_ids"])))
            and np.isin(r["sample_indices"],state["query_columns"]).all()
            and np.array_equal(r["sample_nodes"],state["nodes"][r["sample_indices"]]),
            "Exact original canonical focal coordinates and observed query identities required")
        r["hold"]=np.searchsorted(np.asarray(list(ts.breakpoints())),local,side="right")-1
        # Discard the very large site-join Python metadata after its exact validation.
        state["manifest"]=dict(parent_id=parent,split=split,truth_or_oracle_read=False)
        state["node_time"]=np.asarray(ts.nodes_time,dtype=np.float64)
        states[parent],records[parent]=state,r
        proof[parent]=dict(input_refs=item,trees=ts.num_trees,native_unique_nodes=ts.num_nodes,
            sites=ts.num_sites,samples=ts.num_samples,focal_rows=len(r["record_ids"]),
            query_ancestry_truth_opened=False,all_observed_inputs_authenticated=True)
        check("GST_LAI_parent_observation_sealed")
    return states,records,proof
