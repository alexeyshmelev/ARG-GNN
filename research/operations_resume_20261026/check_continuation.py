"""Small CPU-only checks for preserved successes and explicit retry admission."""
from pathlib import Path
import copy
import json
import os
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import continuation as c, runtime as r

base = r.path('research/operations_resume_20261026/fixture')
base.mkdir(exist_ok=True)

def save(name, value):
    return r.save(base / name, value)

success = save('result.json', {'fixture': 'completed stage, never rerun'})
tasks = [dict(id='completed'), dict(id='retry')]
prior = dict(tasks=tasks)
prior_ref = save('old_plan.json', prior)
old = dict(plan_ref=prior_ref, tasks={
    'completed': dict(status='succeeded', result_ref=success),
    'retry': dict(status='failed', attempt=3)}, plan_revision=6)
old_ref = save('old_state.json', old)
body = dict(schema='paper_campaign_continuation_v1', source_state_ref=old_ref,
            effective_prior_plan_ref=prior_ref, original_plan_ref=prior_ref,
            source_plan_revision=6, prior_controller=dict(pid=999999999, start_ticks='0'),
            reset_task_ids=['retry'], replacement_task_ids=[], archived_attempt_bytes=0)
plan = dict(tasks=tasks, continuation_ref=save('continuation.json', body))
rows = c.initial_rows(plan)
assert rows['completed']['result_ref'] == success and rows['completed']['status'] == 'succeeded'
assert rows['retry']['attempt'] == 4 and rows['retry']['status'] == 'pending'
checks = ['completed result preserved', 'failed attempt increments once']

def reject(label, mutant):
    try:
        c.initial_rows(mutant)
    except ValueError:
        checks.append(label)
    else:
        raise AssertionError(label)

mutant=copy.deepcopy(plan)
mutant['tasks'][0]['unexpected']='change'
reject('undeclared scientific task change rejected', mutant)
mutant_body=copy.deepcopy(body)
mutant_body['reset_task_ids']=[]
mutant=copy.deepcopy(plan)
mutant['continuation_ref']=save('missing_reset.json',mutant_body)
reject('unaccounted failed attempt rejected',mutant)
mutant_body=copy.deepcopy(body)
mutant_body['replacement_task_ids']=['completed']
mutant['continuation_ref']=save('replace_success.json',mutant_body)
reject('successful task replacement rejected',mutant)
mutant_old=copy.deepcopy(old)
mutant_old['tasks']['retry']['status']='running'
mutant_body=copy.deepcopy(body)
mutant_body['source_state_ref']=save('live_state.json',mutant_old)
mutant_body['reset_task_ids']=[]
mutant['continuation_ref']=save('live_continuation.json',mutant_body)
reject('live worker adoption rejected',mutant)
assert c.permit_existing_output(dict(id='retry'), dict()) is False
checks.append('unreviewed existing output remains forbidden')
# Explicit new score task is additive, leaving every old task unchanged.
mutant=copy.deepcopy(plan)
mutant['tasks'].append(dict(id='interim_score'))
mutant_body=copy.deepcopy(body)
mutant_body['new_task_ids']=['interim_score']
mutant['continuation_ref']=save('additive.json',mutant_body)
rows=c.initial_rows(mutant)
assert rows['interim_score']==dict(status='pending')
checks.append('explicit additive score leaves successes unchanged')
save('checks.json',dict(passed=checks,gpu_or_model_work=False))
print(json.dumps(dict(passed=len(checks),checks=checks),indent=2))
