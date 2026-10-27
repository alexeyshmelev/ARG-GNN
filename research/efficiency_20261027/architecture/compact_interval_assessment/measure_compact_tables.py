"""Authenticated compact TRAIN000 tables, interval-event counts and cut counts.

No expanded node/leaf bank, genotype payload, labels, checkpoint or GPU read.
Exact observed tree table hash is checked before tskit loads it. At most4GiB
address space and120s wall time; metadata JSON only is written in this scope.
"""
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
signal.alarm(120)
import numpy as np
import tskit

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
MANIFEST=dict(path='experiments/results/benchmark_campaign_20261018/fresh_american/argformer_american_train_000/inference/observed/manifest.json',sha256='83c64fbff2c1b4c1d728b756733020f9ba21a69b0698c51154a996897ef505e0')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    start=time.monotonic();path=ROOT/MANIFEST['path'];need(path.stat().st_size<2**20 and sha(path)==MANIFEST['sha256'],'Exact original TRAIN000 manifest')
    manifest=json.loads(path.read_text());need(manifest['split']=='train' and manifest['query_ancestry_truth_in_inputs'] is False,'Observed TRAIN only')
    tree_ref=manifest['tree_ref'];path=ROOT/tree_ref['path'];need(path.stat().st_size<128*2**20 and sha(path)==tree_ref['sha256'],'Authenticated bounded original compact tree table')
    ts=tskit.load(path);tables=ts.tables
    need(ts.num_samples==1600 and ts.num_trees==77472 and ts.num_sites==227474 and ts.sequence_length==30000000,'Exact original full TRAIN000 support')
    samples=ts.samples();sample_mask=np.zeros(ts.num_nodes,dtype=bool);sample_mask[samples]=True
    times=tables.nodes.time;edges=tables.edges
    need(np.all(np.isfinite(times)) and np.all(times[edges.parent]>=times[edges.child]),'Finite monotone observed ages')
    need(np.all((0<=edges.left)&(edges.left<edges.right)&(edges.right<=ts.sequence_length)),'Original interval endpoints')
    incident=np.zeros(ts.num_nodes,dtype=np.int32);active=ts.num_samples;crossing=0
    groups=[];expanded_nodes=0;root_total=0;direct_checks=[];above_samples=int(np.count_nonzero(times[samples]>350))
    events_in=events_out=0
    for i,(tree,diff) in enumerate(zip(ts.trees(),ts.edge_diffs())):
        interval,outgoing,incoming=diff;events_in+=len(incoming);events_out+=len(outgoing)
        for e in outgoing:
            crossing-=int(times[e.child]<=350<times[e.parent])
            for node in (e.parent,e.child):
                incident[node]-=1
                if incident[node]==0 and not sample_mask[node]:active-=1
        for e in incoming:
            crossing+=int(times[e.child]<=350<times[e.parent])
            for node in (e.parent,e.child):
                if incident[node]==0 and not sample_mask[node]:active+=1
                incident[node]+=1
        roots=tree.roots;root_total+=len(roots)
        count=crossing+sum(times[root]<=350 for root in roots)+above_samples
        need(0<count<=1600,'Complete positive cut-group partition')
        groups.append(int(count));expanded_nodes+=active
        if i in (0,1,2,77471):
            reps=[]
            for sample in samples:
                node=int(sample)
                while tree.parent(node)!=tskit.NULL and times[tree.parent(node)]<=350:node=tree.parent(node)
                reps.append(node)
            need(len(set(reps))==count,'Event-derived cut count matches explicit sample ancestor walk')
            direct_checks.append(dict(hold=i,groups=count))
    need(len(groups)==ts.num_trees and expanded_nodes==205369272,'Exact original expanded-node count reconstructed from compact edge events')
    g=np.asarray(groups,dtype=np.int64);span=edges.right-edges.left
    genotype=ROOT/manifest['genotypes_ref']['path']
    with genotype.open('rb') as handle:
        version=np.lib.format.read_magic(handle)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(handle)
        else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(handle)
    need(shape==(227474,1600) and dtype==np.dtype('int8'),'Original observed full genotype dimensions, header only')
    counts=dict(nodes=ts.num_nodes,edges=ts.num_edges,sites=ts.num_sites,mutations=ts.num_mutations,trees=ts.num_trees,
        samples=ts.num_samples,individuals=ts.num_individuals,sequence_length_bp=ts.sequence_length,
        expanded_node_occurrences=expanded_nodes,expanded_leaf_tree_cells=ts.num_trees*ts.num_samples,
        original_site_sample_cells=int(np.prod(shape)),cut_group_occurrences=int(g.sum()),
        maximum_groups_per_tree=int(g.max()),minimum_groups_per_tree=int(g.min()),
        maximum_groups_per512_hold_block=max(int(g[a:a+512].sum()) for a in range(0,len(g),512)),
        sample_age_counts={str(float(v)):int(np.count_nonzero(times[samples]==v)) for v in np.unique(times[samples])},
        edge_events_inserted=events_in,edge_events_removed_before_final_boundary=events_out,
        root_occurrences=root_total)
    report=dict(schema='TRAIN000_authenticated_compact_interval_counts_v1',passed=True,manifest_ref=MANIFEST,tree_ref=tree_ref,
        source_ref=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))),
        counts=counts,tree_file_bytes=path.stat().st_size,tskit_tables_nbytes=ts.nbytes,
        node_table_bytes=tables.nodes.nbytes,edge_table_bytes=tables.edges.nbytes,
        site_table_bytes=tables.sites.nbytes,mutation_table_bytes=tables.mutations.nbytes,
        edge_span_bp_quantiles={str(q):float(np.quantile(span,q)) for q in (0,.25,.5,.75,.9,.99,1)},
        direct_cut_count_checks=direct_checks,
        genotype_header=dict(ref=manifest['genotypes_ref'],shape=list(shape),dtype=str(dtype),fortran_order=fortran,
            logical_file_bytes=genotype.stat().st_size,payload_bytes=int(np.prod(shape)),payload_read=False,full_file_rehashed=False),
        site_payload_or_truth_read=False,expanded_bank_materialized=False,GPU_initialized=False,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-start,
        versions=dict(tskit=tskit.__version__,numpy=np.__version__))
    output=Path(__file__).with_name('TRAIN000_compact_counts_v1.json')
    with output.open('x') as handle:json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
