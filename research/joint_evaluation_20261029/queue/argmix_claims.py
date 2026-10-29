"""Atomically claim one native output while the original queue is storage blocked."""
import json
import os
from pathlib import Path
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import claims

need = claims.need
MARKER = claims.MARKER
marker_value = claims.marker_value
permit_claimed_output = claims.permit_claimed_output


def verify_storage(plan):
    ledger = r.read(plan['storage_ledger_ref'])
    need(ledger['writer_closure_confirmed'] is True, 'Root writer closure required')
    for source in ledger['evidence_refs']:
        r.read(source)
    base = sum(row['reserved_bytes'] for row in ledger['base_allowances'])
    need(plan['reserved_existing_project_bytes'] == base + plan['other_queue_disk_reserved_bytes'], 'Combined queue baseline')
    own = next(t for t in plan['tasks'] if t['kind'] != 'external')
    total = plan['reserved_existing_project_bytes'] + own['max_output_bytes'] + sum(s['max_bytes'] for s in own['additional_mutable_scopes']) + plan['controller_metadata_allowance_bytes']
    need(total == plan['global_committed_bytes'] <= plan['storage_ceiling_bytes'] == ledger['ceiling_bytes'] == 1_200_000_000_000, 'Full combined disk reservations')


def verify_original(plan):
    own = next(t for t in plan['tasks'] if t['kind'] != 'external')
    view = dict(plan, tasks=[own])
    state = claims.verify_original(view)
    need([name for name, row in state['tasks'].items() if row['status'] == 'running'] == [plan['guard_task_id']], 'Only reviewed fixed-reservation original guard may be running')
    need(state['tasks'][own['id']].get('waiting_reason') == 'Storage reservation', 'Native task must still be storage blocked')
    dependency = own['dependencies'][0]
    source = next(t for t in plan['tasks'] if t['id'] == dependency)
    need(state['tasks'][dependency]['status'] == 'succeeded' and state['tasks'][dependency]['result_ref'] == source['result_ref'], 'Genuine completed original input dependency')
    growth = own['max_output_bytes'] + sum(max(0, s['max_bytes'] - r.scope_bytes(r.path(s['path']))) for s in own['additional_mutable_scopes'])
    need(state['project_reserved_bytes'] + growth > 995_000_000_000, 'Original fixed reservation must exclude duplicate native admission')
    need(not (r.path(own['additional_mutable_scopes'][0]['path']) / 'inferred').exists(), 'Fresh native inferred prefix required')
    return state, growth


def claim_all(plan):
    verify_storage(plan)
    state, growth = verify_original(plan)
    own = next(t for t in plan['tasks'] if t['kind'] != 'external')
    output = r.path(own['output_directory'])
    need(not output.exists(), 'Original native output must be absent')
    output.mkdir(exist_ok=False)
    with (output / MARKER).open('x') as stream:
        json.dump(marker_value(plan, own), stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    verify_original(plan)
    r.save(r.path(plan['records_directory']) / 'claims.json', dict(schema='native_exact_scope_claim_v1',
           claim_ref=r.ref(output / MARKER), original_reserved_bytes=state['project_reserved_bytes'],
           native_candidate_growth_bytes=growth, original_ceiling_bytes=995_000_000_000,
           original_state_unmodified=True, signals_sent=0))
