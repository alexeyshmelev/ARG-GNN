"""Small CPU-only failure-path checks for release, restore and binding contracts."""
from pathlib import Path
import copy
import importlib
import json
import shutil
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r
from research.preprocessing_optimization_20261026.integration import treeloom_resume as wrapper, hooks


def main():
    base = r.path('research/preprocessing_optimization_20261026/integration/resume_fixture')
    if base.exists():
        raise RuntimeError('Fixture scope already exists')
    base.mkdir()
    checks, counter = [], 0
    original_output = wrapper.OUTPUT
    wrapper.OUTPUT = str((base / 'restored').relative_to(r.ROOT))
    source = base / 'archive/native/checkpoints/latest.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'fixture historical checkpoint bytes, never loaded as model')

    def save(value):
        nonlocal counter
        counter += 1
        return r.save(base / ('case%d.json' % counter), value)

    def reject(label, function, exception=ValueError):
        try:
            function()
        except exception:
            checks.append(label)
        else:
            raise AssertionError('Expected rejection: ' + label)

    def restore(value, callback=lambda amount: None):
        return wrapper.restore_history(save(value), resource_check=callback)

    def assert_fresh():
        assert not (base / 'restored').exists()

    history = dict(schema='exact_preprocessing_preserved_history_v1', destination=wrapper.OUTPUT,
        files=[dict(relative_path='checkpoints/latest.pt', source_ref=r.ref(source), size_bytes=source.stat().st_size)],
        file_count=1, total_bytes=source.stat().st_size)
    try:
        calls = []
        target = restore(history, calls.append)
        assert (target / 'checkpoints/latest.pt').read_bytes() == source.read_bytes()
        assert calls and all(type(value) is int and value > 0 for value in calls)
        checks.append('byte-identical history with resource reservation before every copy block')
        reject('existing namespace rejected', lambda: restore(history))
        shutil.rmtree(target)
        mutations = [
            ('escaping path rejected before writes', lambda h: h['files'][0].update(relative_path='../escape')),
            ('noncanonical path rejected before writes', lambda h: h['files'][0].update(relative_path='checkpoints//latest.pt')),
            ('duplicate paths rejected before writes', lambda h: (h['files'].append(copy.deepcopy(h['files'][0])), h.update(file_count=2,total_bytes=2*h['total_bytes']))),
            ('overlapping file and directory paths rejected before writes', lambda h: (h['files'].append(dict(h['files'][0],relative_path='checkpoints')),h.update(file_count=2,total_bytes=2*h['total_bytes']))),
            ('byte cap rejected before writes', lambda h: h.update(total_bytes=wrapper.MAX_HISTORY_BYTES+1)),
            ('file-count cap rejected before writes', lambda h: h.update(file_count=wrapper.MAX_HISTORY_FILES+1)),
            ('inconsistent byte total rejected before writes', lambda h: h.update(total_bytes=h['total_bytes']+1)),
            ('incorrect source size rejected before writes', lambda h: h['files'][0].update(size_bytes=1)),
            ('completed training rejected before writes', lambda h: h['files'][0].update(relative_path='training_completed.json')),
            ('empty history rejected before writes', lambda h: h.update(files=[],file_count=0,total_bytes=0)),
        ]
        for label, change in mutations:
            bad = copy.deepcopy(history)
            change(bad)
            reject(label, lambda: restore(bad))
            assert_fresh()
        link = base / 'linked_archive'
        link.symlink_to(source.parent, target_is_directory=True)
        bad = copy.deepcopy(history)
        bad['files'][0]['source_ref']['path'] = str((link / source.name).relative_to(r.ROOT))
        reject('symlink source ancestor rejected before resolution', lambda: restore(bad))
        assert_fresh()
        link.unlink()
        destination_link = base / 'restored'
        destination_link.symlink_to(base / 'missing_destination', target_is_directory=True)
        reject('dangling destination symlink rejected', lambda: restore(history))
        destination_link.unlink()
        old = source.read_bytes()
        source.write_bytes(b'x' * len(old))
        reject('same-size altered archive rejected before writes', lambda: restore(history))
        source.write_bytes(old)
        assert_fresh()
        def refuse(amount):
            raise RuntimeError('fixture resource refusal')
        reject('resource refusal leaves destination absent', lambda: restore(history, refuse), RuntimeError)
        assert_fresh()
        calls = []
        def refuse_block(amount):
            calls.append(amount)
            if len(calls) == 3:
                raise RuntimeError('fixture block refusal')
        reject('mid-copy refusal retains partial evidence', lambda: restore(history, refuse_block), RuntimeError)
        assert (base / 'restored/checkpoints/latest.pt').exists()
        reject('partial output cannot be silently reused', lambda: restore(history))
        shutil.rmtree(base / 'restored')
        a = wrapper.attempt_directory(base, 300, 'attempt_3')
        b = wrapper.attempt_directory(base, 300, 'attempt_4')
        assert a != b
        a.mkdir()
        reject('same-boundary existing attempt is immutable', lambda: wrapper.attempt_directory(base,300,'attempt_3'))
        reject('unsafe attempt identifier rejected', lambda: wrapper.attempt_directory(base,300,'../attempt'))
        checks.append('distinct attempts at the same durable boundary have distinct leaves')

        evidence = save(dict(status='pass', fixture=True))
        def release(workers=1):
            roles = wrapper.PROOF_ROLES | (wrapper.PROCESS_PROOF_ROLES if workers == 4 else frozenset())
            return dict(schema='exact_preprocessing_release_v1',status='passed_release',native_competitor_changed=False,
                candidate_source_refs=hooks.source_refs(),encoder_workers=workers,
                proof_refs=[dict(role=role,reference=save(dict(schema='exact_preprocessing_proof_v1',
                    role=role,status='pass',evidence_refs=[evidence]))) for role in sorted(roles)])
        passed = release()
        wrapper.authenticate_release(save(passed))
        wrapper.authenticate_release(save(release(4)))
        checks.append('complete serial and process4 proof contracts accepted as fixture metadata')
        for label, change in [
            ('empty proof list rejected', lambda x:x.update(proof_refs=[])),
            ('missing proof role rejected', lambda x:x['proof_refs'].pop()),
            ('duplicate proof role rejected', lambda x:x['proof_refs'].__setitem__(0,x['proof_refs'][1])),
            ('unapproved release status rejected', lambda x:x.update(status='pending')),
            ('missing release schema rejected', lambda x:x.pop('schema')),
            ('unqualified worker count rejected', lambda x:x.update(encoder_workers=2)),
            ('process4 without additional proof roles rejected', lambda x:x.update(encoder_workers=4)),
        ]:
            bad = copy.deepcopy(passed); change(bad)
            reject(label, lambda:wrapper.authenticate_release(save(bad)))
        bad = copy.deepcopy(passed)
        next(ref for ref in bad['candidate_source_refs'] if ref['path'].endswith('/treeloom_resume.py'))['sha256'] = '0'*64
        reject('changed continuation wrapper pin rejected', lambda:wrapper.authenticate_release(save(bad)))
        for label, value in [('failed proof rejected','fail'), ('pending proof rejected','pending')]:
            bad = copy.deepcopy(passed)
            proof = r.read(bad['proof_refs'][0]['reference']); proof['status'] = value
            bad['proof_refs'][0]['reference'] = save(proof)
            reject(label, lambda:wrapper.authenticate_release(save(bad)))
        bad = copy.deepcopy(passed)
        proof = r.read(bad['proof_refs'][0]['reference']); proof['evidence_refs'] = [save(dict(status='fail'))]
        bad['proof_refs'][0]['reference'] = save(proof)
        reject('failed underlying evidence rejected', lambda:wrapper.authenticate_release(save(bad)))

        owners = [
            ('research.benchmark_rebuild_20261024.american.treeloom_speed.static_preparation_v2',
                ('encode_block','build_episodes','prepare_cut_plan_v1')),
            ('research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1',('encode_block',)),
            ('research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.episode_preparation_v1',('build_clade_episode_arrays_v1',)),
            ('research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.cut_preparation_v1',('prepare_cut_plan_v1',)),
        ]
        originals = [(importlib.import_module(name),attr,getattr(importlib.import_module(name),attr)) for name,attrs in owners for attr in attrs]
        def reset_bindings():
            for owner,attr,function in originals:setattr(owner,attr,function)
        try:
            owner,attr,function = originals[-1]
            setattr(owner,attr,lambda:None)
            reject('unexpected final binding rejects installation before first mutation', hooks.install_tree)
            assert all(getattr(owner,attr) is function for owner,attr,function in originals[:-1])
            reset_bindings()
            bindings = hooks.install_tree()
            assert len(bindings) == 6 and hooks.verify_bindings(bindings)
            checks.append('six exact serial callable bindings recorded and verified')
            owner,attr,function = originals[0]
            setattr(owner,attr,function)
            reject('binding substitution detected before completion',lambda:hooks.verify_bindings(bindings))
            reset_bindings()
            bindings = hooks.install_tree(4,max_rss_bytes=512*2**30)
            assert hooks.verify_bindings(bindings)
            assert bindings[0]['after']['partial_keywords'] == dict(max_rss_bytes=512*2**30)
            checks.append('process4 adapter and exact admitted task budget recorded')
            from research.preprocessing_optimization_20261026.tree import process_observations
            called=[]
            def fake_encoder(*args,**kwargs):called.append(kwargs);return 'fixture'
            with patch.object(process_observations,'encode_block',fake_encoder):
                assert originals[0][0].encode_block(resource_check=lambda stage:None) == 'fixture'
                assert called[0]['preparation_workers'] == 4 and called[0]['max_parallel_bytes'] == 8*2**30
                reject('process4 adapter refuses insufficient live aggregate reserve',
                    lambda:hooks._process4_encode_block(max_rss_bytes=1,resource_check=lambda stage:None))
            checks.append('fixed process4 admission and parameters checked without creating a worker')
        finally:
            reset_bindings()
    finally:
        wrapper.OUTPUT = original_output
        shutil.rmtree(base)
    report = dict(status='pass',checks=checks,model_executed=False,production_files_changed=False,
        process_workers_created=False,candidate_source_refs=hooks.source_refs(),test_source_ref=r.ref(__file__))
    r.save('research/preprocessing_optimization_20261026/integration/resume_contract.json',report)
    print(json.dumps(dict(status='pass',checks=len(checks),details=checks)))


if __name__ == '__main__':
    main()
