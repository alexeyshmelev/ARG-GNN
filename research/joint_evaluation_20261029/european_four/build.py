"""Freeze direct4 scientific protocol and exact source, then seal successful CPU evidence."""
import argparse
import json
from pathlib import Path
from .common import BASE,ROOT,MODEL_ID,ref,read,write,require
from .dataset import MANIFEST,MANIFEST_SHA,PLAN_REF


def protocol():
    inherited=json.loads((ROOT/'research/joint_model_20261029/root/benchmark_protocol_v1.json').read_text())
    cnn=json.loads((ROOT/'research/joint_evaluation_20261029/queue/cnn4_v2_plan.json').read_text())
    prediction=json.loads((ROOT/'research/joint_evaluation_20261029/prediction/freeze_v2.json').read_text())
    refs={r['path']:dict(path=r['path'],sha256=r['sha256']) for r in inherited['source_refs']+cnn['tasks'][-1]['source_refs']+prediction['runtime_source_refs']+prediction['dependency_refs']}
    for name in ('__init__.py','dataset.py','common.py','train.py','predict.py','score.py','smoke.py','build.py','loader.py'):
        r=ref(BASE/name);refs[r['path']]=r
    for r in refs.values():require(ref(ROOT/r['path'])==r,'Inherited authenticated source')
    keys=('batch_sites','workers','model_seed','batch_seed','learning_rate','betas','epsilon','weight_decay','gradient_clip','worker_cache_bytes')
    p={key:inherited[key] for key in keys}
    p.update(schema='european_direct4_protocol_v1',condition=MODEL_ID,model=dict(width=64,layers=3,classes=4),mode='structure',epochs=50,
        training_sites=531,training_targets=40000,training_identity_sha256='64ea97057069216cec46ab6704529c0fe2fc40020f76b81e1ae4145354cf58c1',
        training_target_presentations=2000000,optimizer_updates=1700,source_plan_ref=PLAN_REF,source_refs=list(refs.values()),
        manifest_ref=dict(path=str(MANIFEST.relative_to(ROOT)),sha256=MANIFEST_SHA),
        original_training_protocol_ref=ref(ROOT/'research/joint_model_20261029/root/benchmark_protocol_v1.json'),
        prediction_metadata_ref=ref(BASE/'prediction_metadata_v1.json'),score_metadata_ref=ref(BASE/'score_metadata_v1.json'),
        output_cap_bytes=128*2**20,family_ram_cap_bytes=64*2**30,combined_evaluation_cap_bytes=2**30,
        target_selection='Observed cohort eligibility before reading TRAIN labels; exact original CNN4 non-BAA focal IDs.',
        context_policy='Reuse unchanged full native tree and genotype context including unknown-ancestry BAA query leaves. Only focal readout routing is filtered.',
        objective='M/(S*N)*sum CE with uniform shuffled retained sites and actual tail S; M=531,N=40000.',
        initialization='Fresh seed101, no six-class or qualification warmstart.',checkpoint_selection='Fixed final epoch50; no TEST selection.',
        output_classes=['Ana','CHG','WHG','EHG'],probability_merging=False,training_TEST_read=False,
        execution_chain=['TRAIN-only GPU capacity qualification','fresh fixed50 fit','all5TEST direct4 predictions','common paired CNN4 score after all10products sealed'],
        root_owns_GPU_admission_and_launch=True,single_writer_and_sequential_stages=True)
    return write(BASE/'protocol_v1.json',p)


def seal():
    p=read(ref(BASE/'protocol_v1.json'));smoke=read(ref(BASE/'smoke_result_v1.json'))
    require(smoke['passed'] and smoke['protocol_ref']==ref(BASE/'protocol_v1.json') and smoke['source_refs']==p['source_refs'],'Exact CPU-qualified source')
    for r in p['source_refs']:require(ref(ROOT/r['path'])==r,'Frozen source stable')
    return write(BASE/'freeze_v1.json',dict(schema='european_direct4_freeze_v1',condition=MODEL_ID,source_refs=p['source_refs'],protocol_ref=ref(BASE/'protocol_v1.json'),smoke_ref=ref(BASE/'smoke_result_v1.json'),
        checkpoint_policy='Fresh fixed final50 direct4, no six-class rerun or merged outputs',GPU_launch_owned_by_root=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('protocol','seal'));args=parser.parse_args()
    print(json.dumps(protocol() if args.stage=='protocol' else seal()),flush=True)
