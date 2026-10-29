"""CPU baseline admission within the existing root-confirmed evaluation allowance."""
from arggnn.benchmarking.paper_campaign import runtime as r


def need(value, message):
    if not value:
        raise ValueError(message)


def within_shared_allowance(plan):
    limits = plan['combined_evaluation_allowance']
    return sum(r.scope_bytes(r.path(path)) for path in limits['paths']) <= limits['max_bytes']


def admit(plan):
    ledger = r.read(plan['storage_ledger_ref'])
    need(ledger['writer_closure_confirmed'] is True, 'Root writer closure required')
    allocation = r.read(plan['allocation_ref'])
    need(allocation['root_confirmed'] is True and allocation['cnn_reserved_bytes'] == 640*2**20, 'Root-confirmed 640 MiB allocation required')
    need(sum(t['max_output_bytes'] for t in plan['tasks']) == allocation['payload_bytes'] == 384*2**20, 'Exact original payload caps')
    need(plan['controller_metadata_allowance_bytes'] == allocation['metadata_bytes'] == 256*2**20, 'Exact metadata allowance')
    combined = r.read(plan['combined_native_queue_plan_ref'])
    need(plan['reserved_existing_project_bytes'] == combined['global_committed_bytes'], 'Entire other queue commitment charged')
    need(plan['reserved_existing_project_bytes'] + allocation['cnn_reserved_bytes'] <= plan['storage_ceiling_bytes'] == ledger['ceiling_bytes'] == 1_200_000_000_000, 'Owner project ceiling')
    need(plan['combined_evaluation_allowance']['max_bytes'] == 2**30, 'Existing 1 GiB evaluation allowance')
    other = r.scope_bytes(r.path('research/joint_evaluation_20261029'))
    need(other + allocation['cnn_reserved_bytes'] <= 2**30, 'Combined existing evaluation use and full CNN reservation')
    need(within_shared_allowance(plan), 'Combined evaluation scope cap')
    for task in plan['tasks']:
        need(not r.path(task['output_directory']).exists(), 'Fresh CNN output required: ' + task['id'])
    r.save(r.path(plan['records_directory'])/'admission.json', dict(schema='cnn_four_admission_v1',
           original_factory_plan_ref=plan['original_factory_plan_ref'], allocation_ref=plan['allocation_ref'],
           existing_evaluation_bytes=other, cnn_reserved_bytes=allocation['cnn_reserved_bytes'],
           scientific_plan_unchanged=True, claims_created=0, original_controllers_unmodified=True))
