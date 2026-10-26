"""Seal real full-parent equality evidence after all three executions finish."""
from pathlib import Path
import json
from arggnn.benchmarking.paper_campaign import runtime as r

BASE = r.ROOT / 'research/preprocessing_optimization_20261026'

def main():
    paths = dict(reference=BASE/'integration/full_reference.json',
        serial=BASE/'integration/full_candidate.json',
        process4=BASE/'tree/full_process_parent.json')
    records = {name:json.loads(path.read_text()) for name,path in paths.items()}
    baseline = records['reference']
    for name,record in records.items():
        assert record['full_parent'] is True and record['truth_opened'] is False
        assert record['model_executed'] is False
        assert record['parent'] == baseline['parent']
        assert record['observed_inputs'] == baseline['observed_inputs']
        assert record['bundle'] == baseline['bundle'], name + ': complete tensor mismatch'
        assert record['metadata'] == baseline['metadata'], name + ': full metadata mismatch'
    frozen = json.loads((BASE/'tree/serial_source_manifest.json').read_text())
    for reference in frozen['files']:
        assert r.ref(reference['path']) == reference, 'Frozen serial source changed'
    for reference in records['process4']['source_refs']:
        assert r.ref(reference['path']) == reference, 'Process source changed'
    outcomes = {name:dict(prepare_seconds=value['prepare_seconds'],
        speedup_vs_current=baseline['prepare_seconds']/value['prepare_seconds'],
        peak_rss_kib=value['max_rss_kib']) for name,value in records.items()}
    result=dict(status='pass', full_parent=baseline['parent'],
        comparison='All preprocessing tensors/arrays, shapes, dtypes and metadata hashes match',
        excluded_fields='Only pointer/version stamps are excluded from semantic comparison',
        source_refs=frozen['files']+records['process4']['source_refs'],
        input_refs=baseline['observed_inputs'],record_refs={name:r.ref(path) for name,path in paths.items()},
        timings=outcomes,process_peak_rss_bytes=records['process4']['peak_rss_bytes'],
        timing_scope='One complete registered TRAIN000 parent. Runs overlap with production, timings are host observations rather than a controlled standalone throughput benchmark.',
        truth_opened=False,model_or_simulation_changed=False)
    target=BASE/'integration/full_parent_qualification.json'
    with target.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(outcomes,indent=2))

if __name__=='__main__':main()
