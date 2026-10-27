"""Bounded authorized TRAIN-only six-class coverage, no scientific input mutation."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import resource
import time
import zipfile
import numpy as np

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
CLASSES=('Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA')
JOB=dict(path='.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json',sha256='fecd670f6f26332ab165648f4d65ad51131650d195e86df2a39a5545f2b724c1')
BARRIER=dict(path='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_full_vram_v2/other/AP__all_train__treeloom_fit/native/observation_barrier.json',sha256='1c889198d6542da2a5193214318b1d8887df4d15e3a993b2812658f0824552ae')


def need(ok,message):
    if not ok:raise ValueError(message)


def read(ref,limit):
    p=ROOT/ref['path'];need(p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,'Bounded regular file')
    raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Exact pinned authorized TRAIN evidence')
    return raw


def main():
    started=time.monotonic();spec=json.loads(read(JOB,2**20))['call']['spec']['scientific_spec']
    plan=json.loads(read(spec['sampling_plan_ref'],2**20));barrier=json.loads(read(BARRIER,2**20))
    parents=sorted(spec['train_inputs']);need(parents==plan['train_parent_ids'] and len(parents)==3,'Original three TRAIN parents')
    need(barrier['all_original_TRAIN_observations_sealed_before_targets'] is True and barrier['private_truth_opened'] is False
         and barrier['sampling_plan_ref']==spec['sampling_plan_ref'] and sorted(barrier['parents'])==parents,
         'Existing all-three public observation barrier precedes label audit')
    records=[];totals=np.zeros(6,dtype=np.int64);cohort_totals={c:np.zeros(6,dtype=np.int64) for c in plan['parents'][parents[0]]['cohort_quotas']}
    pins=[JOB,BARRIER,spec['sampling_plan_ref']]
    for parent in parents:
        refs=spec['train_inputs'][parent];sealed=barrier['parents'][parent]
        need(sealed['parent_id']==parent and sealed['private_truth_opened'] is False
             and sealed['all_native_trees_nodes_sites_samples_retained'] is True
             and all(sealed[k]==refs[k] for k in ('inference_ref','observed_vcf_ref','sampled_registry_ref')),
             'Original parent/observation/registry barrier binding')
        reg=json.loads(read(refs['sampled_registry_ref'],32*2**20));meta=json.loads(read(refs['training_targets_ref'],2**20))
        need(reg['parent_id']==parent and reg['split']=='train' and reg['schema']=='ap_paper_sampled_registry_v1'
             and reg['observation_ref']==refs['inference_ref'] and reg['labels_present'] is False and reg['query_truth_read'] is False
             and reg['native_path_names']==list(CLASSES) and reg['cohort_quotas']==plan['parents'][parent]['cohort_quotas'],
             'Exact original TRAIN registry and sampling quotas')
        need(meta['schema']=='ap_paper_TRAIN_native_targets_v1' and meta['parent_id']==parent and meta['split']=='train'
             and meta['registry_ref']==refs['sampled_registry_ref'] and meta['no_label_dependent_selection'] is True
             and meta['native_path_names']==list(CLASSES),'Original bound TRAIN labels only')
        raw=read(meta['arrays_ref'],2*2**20)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            need(archive.namelist()==['native_labels_0based.npy'] and sum(i.file_size for i in archive.infolist())<=2*2**20,
                 'Only bounded canonical TRAIN native label vector')
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:labels=archive['native_labels_0based'].copy()
        canonical=sorted(reg['records'],key=lambda row:row['example_index']);n=len(canonical)
        need(labels.dtype==np.int64 and labels.shape==(n,) and meta['rows']==reg['examples']==n
             and [row['example_index'] for row in canonical]==list(range(n)) and bool(((labels>=0)&(labels<6)).all()),
             'Every canonical example has one actual native-six TRAIN target')
        cohort_array=np.asarray([row['cohort'] for row in canonical]);hist=np.bincount(labels,minlength=6);totals+=hist
        by_cohort={}
        for cohort,quota in reg['cohort_quotas'].items():
            mask=cohort_array==cohort;need(int(mask.sum())==quota,'Original cohort quotas retained')
            ch=np.bincount(labels[mask],minlength=6);cohort_totals[cohort]+=ch
            by_cohort[cohort]=dict(rows=quota,class_counts=ch.tolist())
        keys=[(row['position_original_bp'],row['sample_index']) for row in canonical]
        unique=len(set(keys));multiplicity=Counter(keys)
        records.append(dict(parent_id=parent,rows=n,class_counts=hist.tolist(),all_six_present=bool((hist>0).all()),
            by_cohort=by_cohort,unique_sample_position_coordinates=unique,repeated_coordinate_draws=n-unique,
            maximum_coordinate_multiplicity=max(multiplicity.values()),native_holds=sealed['native_holds']))
        pins += [refs['sampled_registry_ref'],refs['training_targets_ref'],meta['arrays_ref']]
    need(int(totals.sum())==50000 and all(int(h.sum())==10000 for h in cohort_totals.values()),'Original50000 rows and10000 per cohort')
    report=dict(schema='european_native6_TRAIN_class_coverage_v1',passed=True,class_order=list(CLASSES),
        inputs=pins,parents=records,total_rows=50000,total_class_counts=totals.tolist(),
        all_six_present_across_original_TRAIN=bool((totals>0).all()),
        by_cohort_total_class_counts={c:h.tolist() for c,h in cohort_totals.items()},
        split='train',TEST_opened=False,validation_opened=False,observations_used_as_targets=False,
        label_dependent_sampling_or_reweighting=False,model_inputs_or_original_support_changed=False,
        interpretation='Coverage only. Does not establish learnability, performance, or cause of previously reported scores.',
        wall_seconds=time.monotonic()-started,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    output=Path(__file__).with_name('TRAIN_coverage_v1.json')
    with output.open('x') as handle:json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','parents')},sort_keys=True))

if __name__=='__main__':main()
