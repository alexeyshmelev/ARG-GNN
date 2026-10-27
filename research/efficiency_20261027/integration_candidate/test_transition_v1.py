"""Synthetic transition files inside research only; never touches real outputs."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
from arggnn.benchmarking import contracts as c
from . import entrypoint_v1 as entry

BASE=c.ROOT/'research/efficiency_20261027/integration_candidate'
JOB='.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt5.json'


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream)
    return c.file_ref(path)


def fails(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected unsafe transition rejection')


def main():
    directory=BASE/'transition_cpu_fixture_v1'
    directory.mkdir(exist_ok=False)
    root=directory/'native'
    prior=root/'previous_attempt'
    prior.mkdir(parents=True)
    saved=prior/'checkpoints/latest.pt'
    saved.parent.mkdir()
    saved.write_bytes(b'synthetic checkpoint data; trainer validation is separately tested')
    copied=directory/'captured.pt'
    copied.write_bytes(saved.read_bytes())
    saved_ref,copied_ref=c.file_ref(saved),c.file_ref(copied)
    latest_ref=put(prior/'latest_checkpoint.json',dict(checkpoint_ref=saved_ref,complete_parent_updates=263))
    progress_ref=put(prior/'progress.jsonl',dict(event='synthetic retained history'))
    evidence_ref=put(directory/'evidence.json',dict(complete_parent_updates=263,checkpoint_ref=copied_ref))
    wait_ref=put(directory/'actual_wait.json',dict(actual_wait=True,owned_group_absent=True,
        pid=12345,process_start_ticks='987',actual_returncode=-15))
    output=str(root.relative_to(c.ROOT))
    transition=dict(schema='TreeLoom_existing_native_successor_boundary_v1',native_output_directory=output,
        predecessor_output_directory=str(prior.relative_to(c.ROOT)),warm_observation_cache_transfer=False,
        ROOT_confirmed_no_other_writer=True,predecessor_pid=12345,predecessor_start_ticks='987',
        predecessor_actual_wait_ref=wait_ref,checkpoint_ref=copied_ref,evidence_ref=evidence_ref,
        predecessor_job_ref=c.file_ref(JOB),preserved_history_refs=[saved_ref,latest_ref,progress_ref])
    transition_ref=put(directory/'transition.json',transition)
    arguments=copy.deepcopy(c.read_json(JOB)['call']['spec']['arguments'])
    arguments.update(output_directory=output,resume_checkpoint_ref=copied_ref,resume_evidence_ref=evidence_ref)
    with patch.object(entry,'OUTPUT',output),patch.object(entry,'identity',return_value=None),patch.object(entry,'owned_group_members',return_value=[]):
        entry.validate_transition(transition_ref,arguments)
        raised=dict(arguments,max_rss_bytes=arguments['max_rss_bytes']+1)
        fails(lambda:entry.validate_transition(transition_ref,raised))
        original_read=entry.read
        def missing_history(reference):
            return dict(transition,preserved_history_refs=[saved_ref,latest_ref]) if reference==transition_ref else original_read(reference)
        with patch.object(entry,'read',side_effect=missing_history):
            fails(lambda:entry.validate_transition(transition_ref,arguments))
        def older_evidence(reference):
            return dict(complete_parent_updates=262,checkpoint_ref=copied_ref) if reference==evidence_ref else original_read(reference)
        with patch.object(entry,'read',side_effect=older_evidence):
            fails(lambda:entry.validate_transition(transition_ref,arguments))
        copied.write_bytes(b'different stale or corrupt boundary')
        fails(lambda:entry.validate_transition(transition_ref,arguments))
    receipt=dict(status='passed',test_ref=c.file_ref(Path(__file__).resolve()),
        exact_latest_captured_digest_required=True,older_boundary_rejected=True,
        required_prior_history_required=True,RAM_increase_rejected=True,
        actual_production_files_modified=False,jobs_started=0,
        synthetic_checkpoint_bytes_not_a_training_qualification=True)
    put(BASE/'transition_contracts_v1.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
