"""Explicit checkpoint continuation after an owner-requested campaign stop.

The old controller state, successful outputs and failed-attempt evidence remain
immutable. Only a sealed continuation document may seed a fresh controller.
"""
from __future__ import annotations
import copy
from pathlib import Path
from . import runtime as r
from . import processes


def _need(value, message):
    if not value:
        raise ValueError(message)


def initial_rows(plan):
    reference = plan.get('continuation_ref')
    if reference is None:
        return None
    continuation = r.read(reference)
    _need(continuation['schema'] == 'paper_campaign_continuation_v1', 'Unknown continuation schema')
    source = r.read(continuation['source_state_ref'])
    prior_plan = r.read(continuation['effective_prior_plan_ref'])
    _need(source['plan_ref'] == continuation['original_plan_ref'], 'Source plan identity changed')
    _need(source.get('plan_revision', 0) == continuation['source_plan_revision'], 'Source revision changed')
    identity = continuation['prior_controller']
    stat = Path('/proc') / str(identity['pid']) / 'stat'
    if stat.exists():
        ticks = stat.read_text().rsplit(')', 1)[1].split()[19]
        _need(ticks != identity['start_ticks'], 'Prior controller is still alive')
    _need(not processes.session_members(identity['pid']), 'Prior controller session is not empty')
    old = source['tasks']
    tasks = {task['id']: task for task in plan['tasks']}
    originals = {task['id']: task for task in prior_plan['tasks']}
    additions = set(continuation.get('new_task_ids', []))
    _need(set(old) == set(originals) and set(tasks) == set(old) | additions,
          'Explicit identical task namespace plus declared new tasks required')
    _need(not additions & set(old), 'New task already exists in history')
    reset = set(continuation['reset_task_ids'])
    failed = {name for name, row in old.items() if row['status'] in ('failed', 'blocked')}
    _need(reset == failed, 'Every failed or blocked task must be explicitly accounted for')
    _need(not any(row['status'] not in ('pending', 'succeeded', 'failed', 'blocked')
                  for row in old.values()), 'Cannot adopt live or unknown tasks')
    replacements = set(continuation['replacement_task_ids'])
    _need(replacements <= set(old), 'Unknown replacement task')
    rows = {}
    for name, previous in old.items():
        if name not in replacements:
            _need(tasks[name] == originals[name], 'Undeclared task change: ' + name)
        if previous['status'] == 'succeeded':
            _need(name not in replacements, 'Completed task cannot be replaced')
            r.read(previous['result_ref'])
            rows[name] = copy.deepcopy(previous)
            rows[name]['continued_from_state_ref'] = continuation['source_state_ref']
        else:
            if previous['status'] == 'failed' and 'pid' in previous:
                _need(previous.get('actual_wait') and previous.get('owned_process_group_absent'),
                      'Failed process must have a genuine wait and drained session')
                stat = Path('/proc') / str(previous['pid']) / 'stat'
                if stat.exists():
                    ticks = stat.read_text().rsplit(')', 1)[1].split()[19]
                    _need(ticks != previous.get('process_start_ticks'), 'Historical worker is still alive')
                _need(not processes.session_members(previous['pid']), 'Historical worker session is not empty')
            rows[name] = dict(status='pending',
                              attempt=previous.get('attempt', 1) + (name in reset),
                              continued_from_state_ref=continuation['source_state_ref'])
    for name in additions:
        rows[name] = dict(status='pending')
    allow = continuation.get('existing_output_receipts', {})
    _need(set(allow) <= failed, 'Only stopped/failed tasks may retain an output directory')
    for name, receipt_ref in allow.items():
        receipt = r.read(receipt_ref)
        _need(receipt['task_id'] == name and receipt['output_directory'] == tasks[name]['output_directory']
              == originals[name]['output_directory'], 'Existing output identity changed')
        _need(receipt['old_wait_ref'] == r.ref(r.path(receipt['output_directory']) / 'actual_wait.json'),
              'Existing output wait record changed')
        r.read(receipt['preserved_wait_ref'])
        for item in receipt['preserved_metadata']:
            _need(r.ref(item['original_ref']['path']) == item['original_ref']
                  and r.ref(item['preserved_ref']['path']) == item['preserved_ref']
                  and item['original_ref']['sha256'] == item['preserved_ref']['sha256'],
                  'Interrupted root metadata not preserved')
        _need(r.read(receipt['old_wait_ref']) == r.read(receipt['preserved_wait_ref']),
              'Interrupted wait not preserved')
        for checkpoint in receipt['checkpoint_refs']:
            _need(r.ref(checkpoint['path']) == checkpoint, 'Resume checkpoint changed')
        _need(not (r.path(receipt['output_directory']) / 'result.json').exists(),
              'Existing output already has a result')
        _need(r.path(receipt['fresh_native_directory']).is_relative_to(r.path(receipt['output_directory'])),
              'Fresh native leaf must remain inside the accounted task scope')
        _need(not r.path(receipt['fresh_native_directory']).exists(), 'Resume leaf already exists')
        rows[name]['existing_output_receipt_ref'] = receipt_ref
    return rows


def permit_existing_output(task, row):
    """Called immediately before launch, never a general overwrite switch."""
    reference = row.get('existing_output_receipt_ref')
    if reference is None:
        return False
    receipt = r.read(reference)
    _need(receipt['task_id'] == task['id'] and receipt['output_directory'] == task['output_directory'],
          'Launch differs from reviewed continuation')
    _need(r.ref(r.path(task['output_directory']) / 'actual_wait.json') == receipt['old_wait_ref'],
          'Stopped task output changed before launch')
    _need(not (r.path(task['output_directory']) / 'result.json').exists(), 'Result already exists')
    _need(not r.path(receipt['fresh_native_directory']).exists(), 'Continuation leaf already exists')
    return True
