"""Read-only real metadata/spec checks and strict release rejection contracts."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from arggnn.benchmarking import contracts as c
from . import training_v1 as training
from . import entrypoint_v1 as entry
from .configuration_v1 import configuration

ROOT=c.ROOT
BASE=ROOT/'research/efficiency_20261027/integration_candidate'
JOB='.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt5.json'


def need(ok,message):
    if not ok: raise AssertionError(message)


def failure(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError('Expected gate rejection')


def main():
    need(not torch.cuda.is_initialized(),'CPU-only metadata validation')
    job=c.read_json(JOB)
    original=job['call']['spec']['arguments']
    selections=[]
    for block in (128,512,1024):
        selected=configuration(dict(executor_kind='explicit_vjp',CPU_threads=8,holds_per_block=block,
            haplotypes_per_gru=64,history_holds_per_window=1024,context_residency='CPU',
            layout_cache_max_bytes=80*2**30,layout_cache_policy='actual_RSS_evict_unleased'))
        probe='experiments/results/paper_american_20261024/treeloom_fit/native/inert_candidate_spec_probe_'+str(block)
        arguments=dict(original,output_directory=probe,execution_configuration=selected)
        spec=training.build_spec(**arguments)
        training._validate_spec(spec)
        need(spec['settings']==training.SETTINGS and spec['requirements']['CPU_Torch_threads']==8
             and spec['requirements']['GNN_holds_per_block']==block
             and spec['memory_envelope']['max_rss_bytes']==original['max_rss_bytes'],
             'Truthful exact selected requirements and unchanged original memory/scientific contract')
        need(not (ROOT/probe).exists(),'Metadata validation must not create production output')
        selections.append(dict(block=block,source_refs=len(spec['requirements']['source_refs']),output_created=False))
    baseline=c.read_json('research/efficiency_20261027/root/runs/explicit_full_gpu5_v1/data/completed.json')
    run='research/efficiency_20261027/root/runs/explicit_full_gpu5_v1'
    full_ref=c.file_ref(run+'/data/completed.json')
    wait_ref=c.file_ref(run+'/actual_wait.json')
    launch_ref=c.file_ref(run+'/launch.json')
    pins=c.file_ref(BASE/'source_pins_v1.json')
    config=dict(selected,holds_per_block=128)
    review_ref={'path':'synthetic_independent_review','sha256':'0'*64}
    release_ref={'path':'synthetic_release','sha256':'0'*64}
    review=dict(status='pass',candidate_source_pins_ref=pins)
    release=dict(schema='TreeLoom_execution_successor_release_v1',status='independently_approved',
        configuration=config,source_pins_ref=pins,full_parent_qualification_ref=full_ref,
        independent_integration_review_ref=review_ref,full_parent_actual_wait_ref=wait_ref,
        full_parent_launch_ref=launch_ref)
    original_read=entry.read
    def fake_read(reference):
        if reference==review_ref:return copy.deepcopy(review)
        if reference==release_ref:return copy.deepcopy(release)
        return original_read(reference)
    with patch.object(entry,'read',side_effect=fake_read):
        entry.validate_release(release_ref)
        release['configuration']=dict(config,holds_per_block=512)
        failure(lambda:entry.validate_release(release_ref))
        release['configuration']=config
        review['candidate_source_pins_ref']={'path':'wrong','sha256':'0'*64}
        failure(lambda:entry.validate_release(release_ref))
        review['candidate_source_pins_ref']=pins
        review['status']='pending'
        failure(lambda:entry.validate_release(release_ref))
        review['status']='pass'
        release['status']='prepared'
        failure(lambda:entry.validate_release(release_ref))
    receipt=dict(status='passed',CPU_only=True,no_CUDA_initialization=not torch.cuda.is_initialized(),
        source_pins_ref=pins,test_ref=c.file_ref(Path(__file__).resolve()),selected_specs=selections,
        release_exact_full_configuration_required=True,independent_review_must_pin_candidate=True,
        unapproved_release_rejected=True,original_scientific_memory_contract_preserved=True,
        historical241_metadata_used_only_for_inert_validation=True,
        launchable_release_created=False,production_modified=False,jobs_started=0)
    with (BASE/'spec_release_contracts_v1.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
