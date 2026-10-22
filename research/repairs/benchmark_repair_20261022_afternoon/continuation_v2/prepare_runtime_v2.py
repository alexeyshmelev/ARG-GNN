"""Prepare a static operational gateway that preserves the active v1 authority."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/repairs/benchmark_repair_20261022_afternoon'
OLD=BASE/'runtime'
NEW=BASE/'runtime_v2'
OLD_PACKAGE='research.repairs.benchmark_repair_20261022_afternoon.runtime'
NEW_PACKAGE='research.repairs.benchmark_repair_20261022_afternoon.runtime_v2'
NAMES=['contracts_recovery_v1.py','overlay_predecessor_v1.py','queue_execute_v1.py',
       'queue_gpu_v1.py','queue_native_admission_v1.py','queue_native_worker_v1.py',
       'queue_reporting_v1.py','readable_workflow_v1.py']


def ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare():
    validation=json.loads((OLD/'validation_v1.json').read_text())
    original_refs={item['path']:item for item in validation['operational_source_refs']}
    NEW.mkdir(exist_ok=False)
    lineage=[]
    for name in NAMES:
        source=OLD/name
        assert ref(source)==original_refs[str(source.relative_to(ROOT))]
        before=source.read_text()
        after=before.replace(OLD_PACKAGE,NEW_PACKAGE)
        after=after.replace('research/repairs/benchmark_repair_20261022_afternoon/runtime/',
                            'research/repairs/benchmark_repair_20261022_afternoon/runtime_v2/')
        if name in ['contracts_recovery_v1.py','queue_gpu_v1.py','readable_workflow_v1.py']:
            assert 'approved_recovery_workflow_ref' in after
            after=after.replace('approved_recovery_workflow_ref','approved_recovery_continuation_workflow_ref')
        if name=='contracts_recovery_v1.py':
            after=after.replace("'approved_recovery_continuation_workflow_ref')",
                                "'approved_recovery_workflow_ref','approved_recovery_continuation_workflow_ref')")
        if name=='readable_workflow_v1.py':
            assert 'recovery_workflow_drain_ref' in after
            after=after.replace('recovery_workflow_drain_ref','recovery_continuation_workflow_drain_ref')
        if name=='queue_reporting_v1.py':
            after=after.replace('readable-recovery-afternoon-queue:','readable-recovery-afternoon-v2-queue:')
            after=after.replace('benchmark_recovery_afternoon_current.json','benchmark_recovery_afternoon_v2_current.json')
        destination=NEW/name
        destination.write_text(after)
        diff=BASE/'continuation_v2'/(name+'.diff')
        diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(source),tofile=str(destination))))
        lineage.append(dict(original_ref=ref(source),successor_ref=ref(destination),diff_ref=ref(diff)))
    manifest=dict(schema='operational_recovery_gateway_v2',science_changed=False,
                  preserved_primary_approval_key='approved_recovery_workflow_ref',
                  new_approval_key='approved_recovery_continuation_workflow_ref',
                  source_lineage=lineage,source_preparer_ref=ref(Path(__file__)))
    path=BASE/'continuation_v2/runtime_manifest.json'
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(manifest_ref=ref(path),operational_files=len(lineage),launches=0),indent=2))


if __name__=='__main__':prepare()
