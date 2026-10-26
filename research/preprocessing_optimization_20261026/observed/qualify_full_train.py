"""Compare every retained real TRAIN SNP, with no duplicate scientific files."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = ROOT / 'research/preprocessing_optimization_20261026/observed'
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
from research.preprocessing_optimization_20261026.observed import parser, reference_reader
resource.setrlimit(resource.RLIMIT_AS, (24 * 2**30, 24 * 2**30))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def checked(ref):
    path = ROOT / ref['path']
    assert sha(path) == ref['sha256'], ref['path']
    return path


def main():
    t0 = time.perf_counter()
    original_source = (ROOT / 'src/arggnn/benchmarking/european_treeloom.py').read_text()
    begin = original_source.index('    h = len(rows)')
    end = original_source.index('    labels, queries, ages = [], [], []', begin)
    expected = original_source[begin:end].replace('ts.num_sites', 'n_sites')
    assert expected in (HERE / 'reference_reader.py').read_text(), 'Differential oracle must be an exact copy of the original loop'
    tasks = json.loads((ROOT / 'research/operations_resume_20261026/european_gst/replacement_tasks.json').read_text())
    item = tasks[1]['call']['factory']['kwargs']['arguments']['train_inputs']['ancestralpaths_european_train_000']
    receipt = json.loads(checked(item['inference_ref']).read_text())
    assert receipt['truth_or_oracle_read'] is False
    assert receipt['input_observed_sha256']['genotypes.vcf'] == item['observed_vcf_ref']['sha256']
    rows = json.loads(checked(receipt['observed_sample_manifest']).read_text())
    vcf_path = checked(item['observed_vcf_ref'])
    wanted = {j['VCF_position_1based']: j['native_site_id'] for j in receipt['post_five_iterations']['site_join']}
    assert len(wanted) == len(receipt['post_five_iterations']['site_join'])
    assert sorted(wanted.values()) == list(range(len(wanted)))
    del receipt
    gc.collect()
    result = dict(status='running', parent='ancestralpaths_european_train_000',
                  source_refs={k: item[k] for k in ['inference_ref', 'observed_vcf_ref']},
                  retained_sites=len(wanted), haplotypes=len(rows),
                  input_bytes=vcf_path.stat().st_size, initial_binding_seconds=time.perf_counter()-t0,
                  hard_address_space_cap_bytes=24*2**30,
                  truth_opened=False, original_oracle_source_verified=True,
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'parser.py', HERE/'adapter.py', HERE/'reference_reader.py', ROOT/'src/arggnn/benchmarking/european_treeloom.py']})
    progress_path = HERE / 'full_train_progress.json'
    print(json.dumps(result), flush=True)
    progress_path.write_text(json.dumps(result, indent=2)+'\n')
    observations = []
    for name, fn in [('reference', reference_reader.read_observed_calls), ('vectorized', parser.read_observed_calls)]:
        events, stage_start = [], time.perf_counter()
        def check(stage):
            events.append(stage)
            if len(events) == 1 or len(events) % 32 == 0:
                value = dict(stage=name, progress_retained_lower_bound=(len(events)-1)*4096,
                             elapsed_seconds=time.perf_counter()-stage_start,
                             max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                progress_path.write_text(json.dumps(value, indent=2)+'\n')
                print(json.dumps(value), flush=True)
        arrays = fn(vcf_path, rows, wanted, len(wanted), resource_check=check)
        elapsed = time.perf_counter()-stage_start
        hashes = [{'shape':list(a.shape), 'dtype':str(a.dtype), 'sha256':hashlib.sha256(memoryview(a)).hexdigest()} for a in arrays]
        record = dict(name=name, seconds=elapsed, arrays=hashes, resource_callbacks=events,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        observations.append(record)
        print(json.dumps({k:v for k,v in record.items() if k != 'resource_callbacks'}), flush=True)
        del arrays
        gc.collect()
    assert observations[0]['arrays'] == observations[1]['arrays']
    assert observations[0]['resource_callbacks'] == observations[1]['resource_callbacks']
    assert sha(vcf_path) == item['observed_vcf_ref']['sha256'], 'Input changed during comparison'
    result.update(status='pass', byte_identical_all_retained_calls=True,
                  callbacks_identical=True, observations=observations,
                  speedup=observations[0]['seconds']/observations[1]['seconds'],
                  elapsed_seconds=time.perf_counter()-t0,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/'full_train_qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    progress_path.write_text(json.dumps(dict(status='completed', speedup=result['speedup']), indent=2)+'\n')
    print(json.dumps(dict(status='pass', speedup=result['speedup'], elapsed_seconds=result['elapsed_seconds'])), flush=True)

if __name__ == '__main__':
    main()
