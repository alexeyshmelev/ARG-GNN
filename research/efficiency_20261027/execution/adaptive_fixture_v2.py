"""Original bounded two-hold CPU fixture; return independent inputs before preparation."""
import json
import numpy as np
import tskit
from arggnn.benchmarking import contracts as c
from research.efficiency_20261027.integration_candidate import training_v3 as training

def fixture(base,registry,runtime):
    parent=registry.ids("train")[0]
    ages=np.asarray(([0.]*100+[1000.]*100)*3+[0.]*1000)
    tables=tskit.TableCollection(sequence_length=30000000)
    current=[tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age) for age in ages]
    clock=5000.
    while len(current)>1:
        parents=[]
        for first in range(0,len(current),2):
            up=tables.nodes.add_row(time=clock)
            for child in current[first:first+2]:tables.edges.add_row(0,30000000,up,child)
            parents.append(up)
        current=parents;clock*=2
    edge_parents=tuple(tables.edges.parent);edge_children=tuple(tables.edges.child)
    tables.edges.clear()
    for left,right in ((0,500),(500,30000000)):
        for up,down in zip(edge_parents,edge_children):tables.edges.add_row(left,right,int(up),int(down))
    positions=np.asarray([10.,30.,15000060.,15000090.])
    for position in positions:tables.sites.add_row(position,"A")
    tables.sort();ts=tables.tree_sequence();nodes=np.asarray(ts.samples(),dtype=np.int64)
    focal_indices=600+np.arange(10000,dtype=np.int64)%1000
    focal_positions=np.concatenate((np.linspace(.05,499.95,5000),np.linspace(1000.,29999900.,5000)))
    focals=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=nodes[focal_indices],
        sample_indices=focal_indices,positions_bp0=focal_positions,parent_id=np.asarray(parent),
        class_order=np.asarray(training.fresh.CLASSES))
    np.savez(base/"focals.npz",**focals)
    manifest=dict(schema="american_c3_observed_v1",genealogy_kind=runtime.frozen.GENEALOGY,
        query_ancestry_truth_in_inputs=False,class_order=list(training.fresh.CLASSES),parent_id=parent,
        sequence_length_bp=30000000,synthetic_CPU_smoke=True)
    (base/"observation.json").write_text(json.dumps(manifest)+"\n")
    (base/"preparation.json").write_text(json.dumps(dict(synthetic_CPU_smoke=True))+"\n")
    refs=dict(observation_ref=c.file_ref(base/"observation.json"),
        focal_registry_ref=c.file_ref(base/"focals.npz"),native_preparation_ref=c.file_ref(base/"preparation.json"))
    state=dict(manifest=manifest,ts=ts,nodes=nodes,labels=np.asarray([2]*200+[0]*200+[1]*200+[-1]*1000),
        query_columns=np.arange(600,1600),focals=focals,
        genotype=np.random.default_rng(71).binomial(1,.2,size=(4,1600)).astype(np.int8),
        positions=positions,allele_count=np.full(4,2,dtype=np.int8),ages=ages,
        spans=np.asarray([500.,29999500.]),focal_tree=(focal_positions>=500).astype(np.int64))
    return state,refs
