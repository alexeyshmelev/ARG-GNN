"""Bounded observed TRAIN topology traversal only; no labels, VCF or model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import tskit

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
TREE_REF=dict(path="experiments/results/benchmark_campaign_20261018_workflow/release_v8/inherited/AP__ancestralpaths_european_train_000__ap_infer/native/post_size.observed.trees",
    sha256="95b55b36b8f498a0fc0645eb2abc978486c171b41c003b1d6efd176d55f4dee6")
MANIFEST_REF=dict(path="experiments/results/benchmark_campaign_20261018_workflow/release_v8/inherited/AP__ancestralpaths_european_train_000__ap_infer/native/inference.json",
    sha256="1c8383e62f9b3bf914622ca6ac9f014d4142c6bb71870b8cd0508f49b8722e78")


def deadline(signum,frame):
    raise TimeoutError("Observed seek probe exceeded 180 second wall budget")


def authenticated(reference):
    path=ROOT/reference["path"]
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):digest.update(block)
    assert digest.hexdigest() == reference["sha256"]
    return path


def window(tree,start):
    assert tree.index == start
    digest=hashlib.sha256()
    counts=[]
    for offset in range(8):
        assert tree.index == start+offset
        nodes=np.asarray(list(tree.nodes()),dtype=np.int64)
        parents=tree.parent_array[nodes]
        digest.update(nodes.tobytes());digest.update(parents.tobytes())
        digest.update(np.asarray(tuple(tree.interval),dtype=np.float64).tobytes())
        counts.append(len(nodes))
        tree.next()
    return dict(start_hold=start,sha256=digest.hexdigest(),native_nodes=counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    resource.setrlimit(resource.RLIMIT_AS,(32*2**30,32*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
    output=Path(args.output)
    assert output.is_relative_to(ROOT) and not output.exists()
    overall_started=time.perf_counter()
    started=time.perf_counter()
    authenticated(MANIFEST_REF)
    path=authenticated(TREE_REF)
    authenticated_seconds=time.perf_counter()-started
    started=time.perf_counter();ts=tskit.load(path)
    load_seconds=time.perf_counter()-started
    start=(ts.num_trees//2//8)*8
    starts=[start+32*offset for offset in range(8)]
    assert starts[-1]+8 <= ts.num_trees
    reference=[];original_times=[]
    for start in starts:
        started=time.perf_counter();tree=ts.at_index(start)
        seek_seconds=time.perf_counter()-started
        started=time.perf_counter();reference.append(window(tree,start))
        original_times.append(dict(seek_seconds=seek_seconds,
            materialize_seconds=time.perf_counter()-started))
        del tree
    started=time.perf_counter();tree=ts.at_index(starts[0])
    initial_cursor_seek_seconds=time.perf_counter()-started
    candidate=[];cursor_times=[];advances=0
    for start in starts:
        started=time.perf_counter()
        while tree.index < start:
            assert tree.next();advances+=1
        advance_seconds=time.perf_counter()-started
        started=time.perf_counter();candidate.append(window(tree,start))
        cursor_times.append(dict(advance_seconds=advance_seconds,
            materialize_seconds=time.perf_counter()-started))
    assert candidate == reference
    result=dict(schema="GST_observed_TRAIN_interior_cursor_probe_v1",status="PASS",
        parent_id="ancestralpaths_european_train_000",inference_ref=MANIFEST_REF,tree_ref=TREE_REF,
        observed_topology_only=True,VCF_or_labels_or_truth_opened=False,CUDA_used=False,
        tskit=tskit.__version__,native_trees=ts.num_trees,native_unique_nodes=ts.num_nodes,
        sites=ts.num_sites,samples=ts.num_samples,file_bytes=path.stat().st_size,
        authenticated_seconds=authenticated_seconds,load_seconds=load_seconds,
        window_starts=starts,trees_per_window=8,
        exact_active_node_IDs_parents_intervals_for_all_64_trees=True,
        original_first_seek_seconds=original_times[0]["seek_seconds"],
        original_repeated_seek_seconds=sum(row["seek_seconds"] for row in original_times[1:]),
        cursor_initial_seek_seconds=initial_cursor_seek_seconds,
        cursor_repeated_advance_seconds=sum(row["advance_seconds"] for row in cursor_times[1:]),
        cursor_skipped_tree_advances=advances,original_per_window=original_times,
        cursor_per_window=cursor_times,window_proofs=reference,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        address_space_cap_bytes=32*2**30,wall_cap_seconds=180,CPU_cap_seconds=120,
        elapsed_seconds=time.perf_counter()-overall_started,
        caveat="One topology traversal microprobe. Full VCF/encoder/model/training and four-worker production throughput are not measured.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    signal.alarm(0)
    print(json.dumps({key:result[key] for key in ("status","native_trees","native_unique_nodes","load_seconds",
        "original_first_seek_seconds","original_repeated_seek_seconds","cursor_initial_seek_seconds",
        "cursor_repeated_advance_seconds","peak_RSS_bytes")},indent=2))


if __name__ == "__main__":
    main()
