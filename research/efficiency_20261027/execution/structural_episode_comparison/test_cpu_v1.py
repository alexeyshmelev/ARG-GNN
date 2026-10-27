"""Tiny synthetic contracts; no real topology, genotype, truth or model payload."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import tskit
from . import comparator_v1 as c
from research.repairs.benchmark_recovery_20261020.runtime.observations import native_tree, native_workspace_v1
from research.preprocessing_optimization_20261026.tree.kernels import _layout_with_isolated_samples
from research.preprocessing_optimization_20261026.tree.episodes import build_clade_episode_arrays_v1 as original
from research.efficiency_20261027.independent_efficiency_challenge.episode_integer_vectorization.episodes_v1 import build_clade_episode_arrays_v1 as candidate
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import array_hash

torch.set_num_threads(1)
CAP = 8*2**30


def tiny_ts(site_in_isolated=False):
    tables = tskit.TableCollection(4.0)
    for _ in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for age in [1, 1, 2, 0.5, 1, 1, 2]:
        tables.nodes.add_row(time=age)
    for parent, child in [(4,0),(4,1),(5,2),(5,3),(6,4),(6,5)]:
        tables.edges.add_row(0,1.125,parent,child)
    for parent, child in [(7,0),(8,7),(8,2),(9,1),(9,3),(10,8),(10,9)]:
        tables.edges.add_row(1.125,3.75,parent,child)
    if site_in_isolated:
        tables.sites.add_row(3.8, ancestral_state='A')
    tables.sort()
    return tables.tree_sequence()


def exact_concat(ts, nodes):
    parents, samples, intervals, isolated = [], [], [], []
    offset = 0
    workspace = native_workspace_v1(ts)
    times = np.asarray(ts.nodes_time, dtype=np.float64)
    for hold, tree in enumerate(ts.trees()):
        native = native_tree(tree,nodes,node_time=times,workspace=workspace)
        p, clock, sn, sid, post, depth, height, descendant, duration = _layout_with_isolated_samples(native)
        global_parent = p.copy()
        global_parent[p>=0] += offset
        parents.append(global_parent)
        samples.append(sn+offset)
        intervals.append(tuple(tree.interval))
        if np.all(p == -1):
            isolated.append(hold)
        offset += len(p)
    intervals = np.asarray(intervals,dtype=np.float64)
    spans = torch.as_tensor(intervals[:,1]-intervals[:,0],dtype=torch.float32).numpy()
    return np.concatenate(parents),np.stack(samples),spans,tuple(isolated)


def synthetic_archive(parent,samples,spans,isolated):
    arrays, meta = original(parent,samples,spans,max_payload_bytes=CAP,isolated_hold_indices=isolated)
    values = {'bank.parent':parent,'bank.sample_index':samples,'bank.spans':spans}
    values.update({c.EPISODE_MAP[k]:v for k,v in arrays.items()})
    return dict(arrays={k:array_hash(v) for k,v in values.items()},
                array_metadata={k:dict(shape=list(v.shape),dtype=v.dtype.str,count=int(v.size)) for k,v in values.items()})


class Contracts(unittest.TestCase):
    def setUp(self):
        self.ts = tiny_ts()
        self.nodes = np.asarray(self.ts.samples(),dtype=np.int64)[[2,0,3,1]]
        self.parent,self.samples,self.spans,self.isolated = exact_concat(self.ts,self.nodes)
        self.archive = synthetic_archive(self.parent,self.samples,self.spans,self.isolated)

    def build(self, **kwargs):
        return c.structure_from_ts(self.ts,self.nodes,
            kwargs.pop('expected',(len(self.parent),len(self.spans),4)),
            check=lambda stage:None,native_tree=native_tree,workspace_factory=native_workspace_v1,
            layout=_layout_with_isolated_samples,**kwargs)

    def compare(self, builders=None, archived=None):
        return c.compare_builders(self.parent,self.samples,self.spans,self.isolated,
            self.archive if archived is None else archived,
            [('original',original),('candidate',candidate)] if builders is None else builders,
            check=lambda stage:None,payload_cap=CAP)

    def test_stream_matches_exact_native_encoder_concat(self):
        result=self.build()
        for a,b in zip(result[:3],(self.parent,self.samples,self.spans)):
            self.assertTrue(np.array_equal(a,b))
            self.assertEqual(a.dtype,b.dtype)
        self.assertEqual(result[3],self.isolated)
        self.assertEqual(self.isolated,(2,))
        self.assertGreater(self.ts.nbytes,0)

    def test_all_six_arrays_and_metadata_equal(self):
        result=self.compare()
        self.assertEqual(len(result['records']),2)
        self.assertEqual(set(result['records'][0]['arrays']),set(c.EPISODE_MAP))
        self.assertFalse(result['archived_metadata_equivalence_claim'])

    def test_input_failure_precedes_any_builder(self):
        calls=[]
        bad=copy.deepcopy(self.archive)
        bad['arrays']['bank.sample_index']='0'*64
        def never(*args,**kwargs):
            calls.append(True)
            raise AssertionError('Builder must not run')
        with self.assertRaisesRegex(ValueError,'sample_index'):
            self.compare([('original',never),('candidate',never)],bad)
        self.assertEqual(calls,[])

    def test_archive_dtype_shape_framing(self):
        record=c.array_record(self.parent)
        self.assertEqual(record['sha256'],array_hash(self.parent))
        self.assertNotEqual(record['sha256'],hashlib.sha256(self.parent.tobytes()).hexdigest())
        wrong=copy.deepcopy(self.archive)
        wrong['array_metadata']['bank.parent']['shape']=[1,len(self.parent)]
        with self.assertRaises(ValueError):
            c.authenticate_array(self.parent,'bank.parent',wrong)

    def test_candidate_array_change_rejected(self):
        def changed(*args,**kwargs):
            arrays,meta=candidate(*args,**kwargs)
            arrays['node_flags'][0,0] ^= 1
            return arrays,meta
        with self.assertRaisesRegex(ValueError,'flags.node_flags'):
            self.compare([('original',original),('candidate',changed)])

    def test_candidate_metadata_change_rejected(self):
        def changed(*args,**kwargs):
            arrays,meta=candidate(*args,**kwargs)
            meta['episode_policy']='wrong'
            return arrays,meta
        with self.assertRaisesRegex(ValueError,'full metadata'):
            self.compare([('original',original),('candidate',changed)])

    def test_baseline_arrays_released_before_candidate(self):
        refs=[]
        def baseline(*args,**kwargs):
            arrays,meta=original(*args,**kwargs)
            refs.extend(weakref.ref(a) for a in arrays.values())
            return arrays,meta
        def following(*args,**kwargs):
            gc.collect()
            self.assertTrue(refs and all(r() is None for r in refs))
            return candidate(*args,**kwargs)
        self.compare([('original',baseline),('candidate',following)])

    def test_global_node_count_bounds_reject_whole(self):
        for delta in [-1,1]:
            with self.assertRaises(ValueError):
                self.build(expected=(len(self.parent)+delta,len(self.spans),4))

    def test_native_tree_and_root_bounds_reject_whole(self):
        with self.assertRaises(ValueError):
            self.build(maximum_nodes_per_hold=3)
        with self.assertRaises(ValueError):
            self.build(maximum_roots=1)

    def test_nonempty_isolated_hold_rejected(self):
        self.ts=tiny_ts(site_in_isolated=True)
        with self.assertRaisesRegex(ValueError,'zero-site'):
            self.build()

    def test_CPU_RNG_unchanged(self):
        before=c.CPU_random_signature()
        self.build()
        self.compare()
        self.assertEqual(before,c.CPU_random_signature())

    def test_monitor_actual_join_without_children(self):
        with tempfile.TemporaryDirectory(dir=c.BASE) as directory:
            monitor=c.OwnWorkerMonitor(directory,operating_bytes=24*2**30,family_bytes=32*2**30,seconds=60)
            monitor.start()
            result=monitor.finish()
            self.assertFalse(monitor.thread.is_alive())
            self.assertEqual(result['own_children'],0)
            self.assertTrue(result['sampled_stop_not_OS_quota'])

    def test_monitor_stop_boundary(self):
        with tempfile.TemporaryDirectory(dir=c.BASE) as directory:
            monitor=c.OwnWorkerMonitor(directory,operating_bytes=1,family_bytes=2,seconds=60)
            with self.assertRaisesRegex(ValueError,'operating stop'):
                monitor.sample()

    def test_admission_freshness_and_exact_bytes(self):
        source_ref=dict(path='source.json',sha256='a'*64)
        spec_ref=dict(path='spec.json',sha256='b'*64)
        spec=dict(family_cap_bytes=32*2**30,operating_stop_bytes=24*2**30,
                  required_peer_directories=['shortVALv2','Europeanqualv2'])
        admission=dict(schema='TreeLoom_structural_episode_ROOT_admission_v1',
            observed_unix=1000.,approved=True,source_manifest_ref=source_ref,spec_ref=spec_ref,
            family_cap_bytes=spec['family_cap_bytes'],operating_stop_bytes=spec['operating_stop_bytes'],
            current_peers_checked=True,outer_guardian_required=True,
            peer_directories=spec['required_peer_directories'],retained_failed_v1_storage_charged=True)
        with tempfile.TemporaryDirectory(dir=c.BASE) as directory:
            path=Path(directory)/'admission.json'
            raw=json.dumps(admission).encode()
            path.write_bytes(raw)
            actual, reference=c.read_admission(path,source_ref=source_ref,spec_ref=spec_ref,spec=spec,now=1100.)
            self.assertEqual(actual,admission)
            self.assertEqual(reference['sha256'],hashlib.sha256(raw).hexdigest())
            for now in [994.,1120.001]:
                with self.assertRaisesRegex(ValueError,'120 seconds'):
                    c.read_admission(path,source_ref=source_ref,spec_ref=spec_ref,spec=spec,now=now)
            admission['peer_directories']=['shortVALv2']
            path.write_text(json.dumps(admission))
            self.assertNotEqual(c.file_ref(path),reference)
            with self.assertRaisesRegex(ValueError,'both European peers'):
                c.read_admission(path,source_ref=source_ref,spec_ref=spec_ref,spec=spec,now=1000.)

    def test_real_payload_readers_not_called(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')


if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='TreeLoom_structural_comparator_CPU_contracts_v1',passed=result.wasSuccessful(),
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        real_topology_loaded=False,genotype_truth_checkpoint_or_model_payload=False,
        CUDA_initialized=torch.cuda.is_initialized(),synthetic_complete_holds=3,
        baseline_arrays_released_before_candidate=True)
    c.write(c.BASE/'CONTRACTS_CPU_v2.json',receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
