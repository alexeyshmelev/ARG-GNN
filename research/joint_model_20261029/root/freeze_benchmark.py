"""Freeze only the independently reviewed final-development selection."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    source=BASE/'root/development_protocol_v1.json'
    review_path=BASE/'review/COMPLETED_DEVELOPMENT_REVIEW.json'
    protocol=json.loads(source.read_text())
    review=json.loads(review_path.read_text())
    assert review['verdict']=='pass' and review['TEST_read'] is False
    assert review['protocol_sha256']==sha(source)
    assert review['trainer_sha256']==sha(BASE/'root/train.py')
    assert review['manifest_sha256']==sha(Path(protocol['manifest']))
    for ref in protocol['source_refs']:
        assert sha(ROOT/ref['path'])==ref['sha256']
    for mode,summary in review['modes'].items():
        run=BASE/'runs'/summary['run']
        assert summary['optimizer_updates']==2600
        assert summary['target_presentations']==1666700
        assert summary['checkpoint_finite_and_exact'] is True
        assert summary['loader_shutdown_passed'] is True
        assert sha(run/'result.json')==summary['result_sha256']
        assert sha(run/'final_state.pt')==summary['state_sha256']
        assert (BASE/'root'/f'development_{mode}_v1.exit').read_text().strip()=='0'
    selected=review['selected_arm']
    assert selected==review['ranking'][0]
    protocol.update(allowed_stages=['benchmark'],selected_arm=selected,
        benchmark_training_authorized=True,
        launch_scope='Fresh fixed-final 50-epoch fit on all 50000 original European six-class TRAIN targets. No TEST scoring or model selection in this process.',
        development_review_path=str(review_path.relative_to(ROOT)),
        development_review_sha256=sha(review_path),
        concurrent_runs='One new benchmark fit on GPU5. Other authorized native tasks are preserved.',
        worker_accounting='Four persistent fitting workers, each capped at12GiB immutable observation cache. No development loader during benchmark fit.',
        initialization_policy='Reinitialize from registered seed101. No warmstart from development checkpoints.',
        evaluation_policy='Freeze this model before eventual held-out benchmark scoring. Development scores do not establish superiority over historical TEST baselines.')
    out=BASE/'root/benchmark_protocol_v1.json'
    with out.open('x') as stream:
        json.dump(protocol,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(selected_arm=selected,protocol=str(out),sha256=sha(out))))

if __name__=='__main__':
    main()
