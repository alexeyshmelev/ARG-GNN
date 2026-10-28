"""Extract exactly one observed TRAIN000 core tree plus two neighboring halos."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from research.redesign_20261028.data.window_data import TrainWindowDataset,COHORTS,CLASSES
HERE=Path('research/redesign_20261028/data')
if (HERE/'train000_first_window.npz').exists() or (HERE/'train000_first_window.json').exists():
    raise FileExistsError('Preserve the initial real fixture; use a new versioned probe')
start=time.perf_counter()
job=json.loads(Path('.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json').read_text())
parent='ancestralpaths_european_train_000'
refs=job['call']['spec']['scientific_spec']['train_inputs'][parent]
dataset=TrainWindowDataset(parent,refs,halo=2)
# Deterministic first position-sorted observed records, not a random performance sample.
selected=sorted(dataset.records,key=lambda i:(dataset.records[i]['position_local_bp'],i))[:4]
packet=dataset.batch(selected)
out=HERE/'train000_first_window.npz'
np.savez(out,**packet)
report=dict(schema='bounded_observed_TRAIN_window_probe_v1',parent_id=parent,
    cohort_order=COHORTS,class_order=CLASSES,source_evidence=dataset.source_evidence,
    selected_example_indices=selected,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in packet.items()},
    tree_intervals=packet['tree_interval_local_bp'].tolist(),
    roots_per_tree=[int(np.sum((packet['node_tree_index']==i)&(packet['parent']<0))) for i in range(len(packet['tree_core_mask']))],
    elapsed_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    VCF_bytes_read=dataset._vcf.bytes_read,output_bytes=out.stat().st_size,
    output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((HERE/'window_data.py').read_bytes()).hexdigest(),
    no_training=True,no_TEST_files=True,no_new_simulations=True,
    label_join='labels[example_index]; records are position-sorted',
    data_limit='Five complete inferred trees; all observed SNPs in these intervals; no chromosome-scale graph')
(HERE/'train000_first_window.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['selected_example_indices','tree_intervals','roots_per_tree','elapsed_seconds','peak_RSS_KiB','VCF_bytes_read','output_bytes']}))
print(json.dumps(report['arrays']))
