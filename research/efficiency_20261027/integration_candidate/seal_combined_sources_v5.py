"""Seal the v5 source closure;terminal deployment release remains ROOT-owned."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).parent
ROUTER=ROOT/'research/efficiency_20261027/architecture/qualified_residency_router'

def ref(path):
    path=ROOT/Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    refs={}
    for manifest_path in (HERE/'source_pins_combined_v4.json',ROUTER/'source_pins_v1.json'):
        for row in json.loads(manifest_path.read_text())['source_refs']:
            assert ref(row['path'])==row,'Source closure changed: '+row['path']
            refs[row['path']]=row
        value=ref(manifest_path);refs[value['path']]=value
    paths=[*HERE.glob('*combined_v5.py'),HERE/'step_combined_v2.py',HERE/'runtime_combined_v2.py',
        HERE/'prepare_combined_sources_v5.py',HERE/'complete_router_binding_v5.py',HERE/'seal_combined_sources_v5.py',
        ROUTER/'CPU_RESULTS_v1.json',ROUTER/'HANDOFF_v1.json']
    for path in paths:
        value=ref(path);refs[value['path']]=value
    manifest=dict(schema='TreeLoom_combined_execution_sources_v5',source_refs=[refs[key] for key in sorted(refs)],
        prior_source_manifest_ref=ref(HERE/'source_pins_combined_v4.json'),qualified_router_source_manifest_ref=ref(ROUTER/'source_pins_v1.json'),
        original_scientific_math_and_schedule_unchanged=True,prepared_actual_nodes_and_live_capacity_before_forward=True,
        no_mid_update_fallback=True,production_modified=False)
    path=HERE/'source_pins_combined_v5.json'
    with path.open('x') as stream:json.dump(manifest,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(source_manifest_ref=ref(path),source_refs=len(refs))))

if __name__=='__main__':main()
