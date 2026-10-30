"""Publish unchanged scientific scores with the explicit ARGMix recovery state."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from research.joint_evaluation_20261029.root import publish_results as original
from arggnn.benchmarking.paper_campaign import runtime as r

BASE = Path('research/joint_evaluation_20261029')


def optional(path):
    path = r.path(path)
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest_ref = r.ref(args.manifest)
    manifest = r.read(manifest_ref)
    recovery = optional(manifest['effective_state_path'])
    gpu_chain = optional(manifest['GPU_chain_completed_path'])
    final_chain = optional(Path(manifest['records_directory']) / 'completed.json')
    # Keep all numerical tables and their existing result authentication unchanged.
    original.main()
    status_path = r.path(BASE / 'root/CURRENT_STATUS.json')
    status = json.loads(status_path.read_text())
    status['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    status['ARGMix_recovery_manifest_ref'] = manifest_ref
    status['ARGMix_previous_failed_attempt'] = optional(BASE / 'argmix/records_v2/joint_argmix.train_cache/actual_wait.json')
    scores_complete = any('/joint_argmix_20261029/scores/' in path for path in status['completed_comparison_artifacts'])
    status['ARGMix_operational_recovery'] = recovery or {'status': 'not_launched', 'manifest_ref': manifest_ref}
    if not scores_complete:
        status['new_ARGMix_joint_pipeline'] = status['ARGMix_operational_recovery']
    status['root_model_chain_receipts']['argmix_joint'] = gpu_chain
    status['ARGMix_recorded_chain_receipt'] = final_chain
    status['notes'].append('ARGMix live status uses the explicit memory recovery. The original failed attempt remains historical evidence.')
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    state = (recovery or {}).get('status', 'not_launched')
    reason = (recovery or {}).get('reason', 'Recovery controller has not started')
    if scores_complete:
        line = '- **ARGMix:** the paired ancestry comparison is complete and authenticated below. Operational recovery state is ' + state + '. The separate final validation diagnostic and independent audit retain their own completion records.'
    elif state == 'failed':
        line = '- **ARGMix:** the recovery reported a failure: ' + reason + '. The final paired comparison remains incomplete.'
    else:
        line = '- **ARGMix:** explicit memory recovery is ' + state + ': ' + reason + '. All five original TRAIN observations are ready. Final paired ancestry scores remain pending.'
    readme = r.ROOT / 'README.md'
    text = readme.read_text()
    start = text.index('<!-- joint-model-20261029:start -->')
    end = text.index('<!-- joint-model-20261029:end -->')
    lines = text[start:end].splitlines()
    for index, value in enumerate(lines):
        if value.startswith('- **ARGMix:**'):
            lines[index] = line
        if value.startswith('- **New American TreeLoom:**') and status['root_model_chain_receipts'].get('american_joint'):
            if not any('/joint_american_20261029/comparison_v1/' in path for path in status['completed_comparison_artifacts']):
                lines[index] = '- **New American TreeLoom:** the fixed 13-epoch fit and all fifteen held-out predictions are complete. Paired ancestry scoring awaits native comparator predictions.'
    readme.write_text(text[:start] + '\n'.join(lines) + '\n' + text[end:])
    print(json.dumps({'recovery_status_published': state, 'manifest_ref': manifest_ref, 'numerical_results_changed': False}))


if __name__ == '__main__':
    main()
