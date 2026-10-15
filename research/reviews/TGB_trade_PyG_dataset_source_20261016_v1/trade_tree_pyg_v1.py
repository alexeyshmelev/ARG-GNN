"""Reusable causal whole-country-tree PyG Dataset and ordered loader.

Caller supplies the already authenticated catalogue and complete observed-year
loader. This module contains no archive/network/label-file reader or trainer.
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data,Dataset
from torch_geometric.loader import DataLoader
from .trade_tree_core_v1 import (CountryCatalogueV1,OFFICIAL_COUNTRIES,OFFICIAL_MAPPING_SHA256,
    FORECAST_YEARS,EDGE_YEARS,CUT_THRESHOLD,ObservedYearV1,require,snapshot_before_v1,
    complete_single_linkage_v1,snapshot_digest_v1)

NODE_INDEX_KEYS=frozenset(("leaf_node_index","root_node_index","cut_node_index"))

class TradeTreeDataV1(Data):
    def __inc__(self,key,value,*args,**kwargs):
        if key=="edge_index" or key in NODE_INDEX_KEYS:return self.num_nodes
        # Country IDs are stable witness metadata, never batched node pointers.
        if key=="country_witness_index":return 0
        return 0
    def __cat_dim__(self,key,value,*args,**kwargs):
        return 1 if key=="edge_index" else 0

def graph_from_snapshot_v1(snapshot,catalogue):
    n=len(catalogue.names);tree=complete_single_linkage_v1(snapshot.profile,snapshot.missing)
    count=len(tree.parent);child=np.flatnonzero(tree.parent>=0)
    tensor=lambda array:torch.from_numpy(np.asarray(array).copy())
    graph=TradeTreeDataV1(
        num_nodes=count,edge_index=tensor(np.stack((child,tree.parent[child]))),
        leaf_node_index=torch.arange(n,dtype=torch.long),
        country_witness_index=torch.arange(n,dtype=torch.long),
        root_node_index=torch.tensor([tree.root],dtype=torch.long),
        cut_node_index=tensor(tree.cut_node_index),
        leaf_profile=tensor(snapshot.profile),leaf_missing=tensor(snapshot.missing),
        leaf_zero_mass=tensor(snapshot.zero_mass),
        leaf_staleness_years=tensor(snapshot.staleness_years),
        leaf_last_observed_year=tensor(snapshot.last_observed_year),
        node_merge_height=tensor(tree.merge_height),
        node_leaf_count=tensor(tree.leaf_count),node_child_count=tensor(tree.child_count),
        node_is_leaf=torch.arange(count)<n,
        upward_level=tensor(tree.upward_level),downward_level=tensor(tree.downward_level),
        forecast_year=torch.tensor([snapshot.forecast_year],dtype=torch.long),
        observation_year=torch.tensor([snapshot.observation_year],dtype=torch.long),
        cut_threshold=torch.tensor([CUT_THRESHOLD],dtype=torch.float64),
        catalogue_sha256=catalogue.sha256,observation_sha256=snapshot_digest_v1(snapshot,tree),
        released_year_mask=torch.tensor([[y in snapshot.released_years for y in EDGE_YEARS]],dtype=torch.bool))
    return graph

class _TradeTreeDatasetCoreV1(Dataset):
    """Small lazy in-memory graph cache; constructor touches no year payload."""
    def __init__(self,catalogue,year_loader,*,available_edge_years=EDGE_YEARS,
            forecast_years=FORECAST_YEARS):
        require(isinstance(catalogue,CountryCatalogueV1) and callable(year_loader),"Catalogue and complete observed-year loader")
        years=tuple(forecast_years);available=tuple(available_edge_years)
        require(years and years==tuple(sorted(set(years))) and all(type(y) is int for y in years),
            "Ordered distinct integer forecast years")
        require(available==tuple(sorted(set(available))) and all(type(y) is int for y in available),
            "Ordered distinct available years; no payload loaded here")
        self.catalogue=catalogue;self.year_loader=year_loader;self.available_edge_years=available
        self.forecast_years=years;self._graphs={};self._observations={}
        super().__init__(root=None,transform=None,pre_transform=None,pre_filter=None,log=False)
    def _load_year(self,year):
        if year not in self._observations:
            value=self.year_loader(year)
            require(isinstance(value,ObservedYearV1) and value.year==year,"Requested complete observed year")
            self._observations[year]=ObservedYearV1(year,value.profile.copy(),value.present.copy(),value.zero_mass.copy())
        return self._observations[year]
    def len(self):return len(self.forecast_years)
    def get(self,index):
        require(type(index) is int and 0<=index<len(self.forecast_years),"Forecast dataset index")
        if index not in self._graphs:
            snapshot=snapshot_before_v1(self.catalogue,self.forecast_years[index],
                self.available_edge_years,self._load_year)
            self._graphs[index]=graph_from_snapshot_v1(snapshot,self.catalogue)
        # Callers may move or mutate their graph without poisoning retained cache.
        return self._graphs[index].clone()
    def clear_graph_cache(self):self._graphs.clear()

class CausalTradeTreeDatasetV1(_TradeTreeDatasetCoreV1):
    """Production interface: exactly the audited official 255-country catalogue."""
    def __init__(self,catalogue,year_loader,*,forecast_years=FORECAST_YEARS,
            available_edge_years=EDGE_YEARS):
        forecast_years=tuple(forecast_years);available_edge_years=tuple(available_edge_years)
        require(catalogue.official and len(catalogue.names)==OFFICIAL_COUNTRIES
            and catalogue.sha256==OFFICIAL_MAPPING_SHA256,"Official full-country catalogue")
        require(set(forecast_years)<=set(FORECAST_YEARS) and tuple(available_edge_years)==EDGE_YEARS,
            "H052 audited year domain")
        super().__init__(catalogue,year_loader,available_edge_years=available_edge_years,
            forecast_years=forecast_years)

def make_trade_tree_loader_v1(dataset,*,batch_size=1,num_workers=0):
    """Chronological, complete graphs; no shuffled/dropped forecast periods."""
    require(isinstance(dataset,_TradeTreeDatasetCoreV1) and type(batch_size) is int and batch_size>0
        and type(num_workers) is int and num_workers>=0,"Explicit loader sizes")
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=num_workers)

def aligned_country_cuts_v1(*graphs):
    """One country-aligned CPU membership vector per annual tree, for H051.

    Internal node indexes identify cells ONLY within their own annual graph.
    Country witnesses, rather than internal IDs, align a/b/c membership vectors.
    This helper does not create an edge-continuity witness.
    """
    require(graphs,"At least one graph")
    first=graphs[0];countries=first.country_witness_index
    require(countries.device.type=="cpu" and torch.equal(countries,torch.arange(len(countries))),
        "Single annual CPU tree in fixed catalogue leaf order")
    for graph in graphs:
        require(graph.catalogue_sha256==first.catalogue_sha256
            and torch.equal(graph.country_witness_index,countries)
            and graph.cut_node_index.shape==countries.shape,"Aligned complete catalogue witnesses")
    return tuple(graph.cut_node_index.clone() for graph in graphs)
