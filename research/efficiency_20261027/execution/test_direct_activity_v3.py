"""CPU checks of direct API extraction against the preserved actual CUDA probe."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from .direct_activity_ledger_v3 import export_API

BASE=Path('research/efficiency_20261027/execution')
PROBE=Path('research/efficiency_20261027/root/runs/native_activity_probe_gpu5_v2/data')


class Event:
    def __init__(self, row): self.row=row
    def name(self): return self.row['name']
    def device_type(self): return self.row['device_type']
    def is_user_annotation(self): return self.row['is_user_annotation']
    def start_ns(self): return self.row['start_ns']
    def duration_ns(self): return self.row['duration_ns']
    def device_index(self): return self.row['device_index']
    def device_resource_id(self): return self.row['device_resource_id']
    def correlation_id(self): return self.row['correlation_id']


def main():
    output=BASE/'direct_activity_CPU_v3'
    output.mkdir(exist_ok=False)
    raw=json.loads((PROBE/'raw_CUDA_API.json').read_text())['rows']
    start=min(row['start_ns'] for row in raw)-100000
    root=dict(name='diagnostic_learned_update',device_type='DeviceType.CPU',is_user_annotation=True,
              start_ns=start,duration_ns=100000000,device_index=0,device_resource_id=1,correlation_id=0)
    events=[Event(row) for row in [root]+raw]
    profile=SimpleNamespace(profiler=SimpleNamespace(kineto_results=SimpleNamespace(events=lambda:events)))
    result=export_API(profile,output/'actual_probe_API')
    ledger=json.loads((output/'actual_probe_API_physical_ledger.json').read_text())
    native=json.loads((PROBE/'probe_physical_ledger.json').read_text())
    def physical(rows):
        return [(row[0],row[2],rows['names'][row[3]]) for row in rows['events']]
    assert physical(ledger)==physical(native)
    assert len(ledger['events'])==5 and result['excluded_category_counts']['gpu_user_annotation']==2
    assert len(ledger['CPU_annotation_envelopes_for_causality_only'])==1
    assert result['physical_CUDA_union']['busy_seconds']==0.000034816
    assert max(row['duration_us'] for row in result['physical_CUDA_union']['largest_gaps'])>30000
    # A copy-shaped name with an annotation flag must not enter physical counts.
    raw.append(dict(raw[0],name='Memcpy HtoD (annotation impostor)',is_user_annotation=True,duration_ns=100000000))
    events[:]=[Event(row) for row in [root]+raw]
    impostor=export_API(profile,output/'annotation_impostor')
    assert impostor['physical_event_counts']==result['physical_event_counts']
    for name in ('direct_activity_ledger_v3.py','profile_boundary_activity_v3.py','launch_boundary_activity_v3.py'):
        ast.parse((BASE/name).read_text())
    result=dict(status='passed',complete_actual_probe_API_matches_native_physical_events=True,
                annotation_impostor_excluded=True,CPU_annotation_causality_retained=True,
                learned_calls=0,GPU_calls=0,physical_events=5)
    with (BASE/'DIRECT_ACTIVITY_CPU_v3.json').open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
