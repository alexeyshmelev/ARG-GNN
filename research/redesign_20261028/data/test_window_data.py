"""CPU contract checks using hand-authored table/VCF fixtures, no simulation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tskit
from research.redesign_20261028.data.window_data import (
    ReplacementTreeReader,tree_from_tskit,generic_small_reader,SortedVCF,TrainWindowDataset,
    uniform_target_draws,node_budget_microbatches,uniform_coordinate_context,exact_vcf_position)
HERE=Path('research/redesign_20261028/data')


def fixture(persistent=False):
    tables=tskit.TableCollection(10)
    for _ in range(3):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0)
    for t in [1,2,3]:tables.nodes.add_row(time=t)
    if persistent:
        tables.edges.add_row(0,10,3,0)
        tables.edges.add_row(0,5,4,3);tables.edges.add_row(0,5,4,1)
        tables.edges.add_row(5,10,5,3);tables.edges.add_row(5,10,5,2)
    else:
        tables.edges.add_row(0,5,3,0)
        tables.edges.add_row(0,5,4,3);tables.edges.add_row(0,5,4,1)
        tables.edges.add_row(5,10,5,0);tables.edges.add_row(5,10,5,2)
    tables.sites.add_row(2,'A');tables.sites.add_row(3,'A')
    tables.sort();return tables.tree_sequence()


class WindowContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=HERE,prefix='fixture_')
        self.path=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_complete_replacement_equals_tskit_unary_forest_boundaries(self):
        ts=fixture();path=self.path/'replacement.trees';ts.dump(path)
        reader=ReplacementTreeReader(path,ts.samples())
        for pos in [0,4.999,5,9.999]:
            actual=reader.at(pos);expected=tree_from_tskit(ts.at(pos),ts.samples())
            for name in ['original_ids','parent','time','sample_index']:
                np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
            self.assertEqual((actual.left,actual.right),(expected.left,expected.right))
        first,core=reader.window(0,2);last,last_core=reader.window(9.999,2)
        self.assertEqual((len(first),core,len(last),last_core),(2,0,2,1))
        with self.assertRaises(ValueError):reader.at(10)
    def test_persistent_edges_rejected_and_small_generic_fallback_equivalent(self):
        ts=fixture(True);path=self.path/'persistent.trees';ts.dump(path)
        reader=ReplacementTreeReader(path,ts.samples())
        with self.assertRaisesRegex(ValueError,'Persistent edges'):reader.at(6)
        loaded=generic_small_reader(path,ts.samples())
        expected=tree_from_tskit(ts.at(6),ts.samples());actual=tree_from_tskit(loaded.at(6),ts.samples())
        np.testing.assert_array_equal(actual.parent,expected.parent)
        with self.assertRaisesRegex(ValueError,'size cap'):generic_small_reader(path,ts.samples(),max_bytes=1)
    def test_partial_future_insertion_and_previous_removal_rejected(self):
        for kind in ['future','previous']:
            tables=tskit.TableCollection(10)
            for _ in range(3):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0)
            for t in [1,2,3]:tables.nodes.add_row(time=t)
            if kind=='future':
                for child in [0,1]:tables.edges.add_row(0,10,3,child)
                for child in [3,2]:tables.edges.add_row(5,10,4,child)
                position=2;expected_bounds=(0,5)
            else:
                for child in [0,1]:tables.edges.add_row(0,5,3,child)
                tables.edges.add_row(2,10,5,2)
                position=6;expected_bounds=(5,10)
            tables.sort();ts=tables.tree_sequence();path=self.path/f'{kind}.trees';ts.dump(path)
            reader=ReplacementTreeReader(path,ts.samples())
            self.assertEqual(tuple(ts.at(position).interval),expected_bounds)
            with self.assertRaisesRegex(ValueError,'requires generic tskit fallback'):reader.at(position)
            fallback=generic_small_reader(path,ts.samples())
            actual=tree_from_tskit(fallback.at(position),ts.samples())
            self.assertEqual((actual.left,actual.right),expected_bounds)
    def test_vcf_seek_matches_every_exact_row_multiallelic_missing(self):
        path=self.path/'observed.vcf'
        panel=[dict(sample_id='a',haplotype=i) for i in [0,1]]
        lines=['##fileformat=VCFv4.2','#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta']
        for i in range(1,201):lines.append(f'22\t{i*17}\t{i}\tA\tC,G\t.\tPASS\t.\tGT\t0|2')
        lines[51]=lines[51].replace('0|2','.|1')
        path.write_text('\n'.join(lines)+'\n');reader=SortedVCF(path,panel)
        for i in range(1,201):
            calls,alleles=reader.at(i*17)
            self.assertEqual(alleles,['A','C','G'])
            self.assertEqual(calls.tolist(),[-1,1] if i==50 else [0,2])
        with self.assertRaisesRegex(ValueError,'not found'):reader.at(18)
        reader.close()
        with self.assertRaisesRegex(ValueError,'sample order'):SortedVCF(path,[dict(sample_id='b',haplotype=i) for i in [0,1]])
    def test_invalid_unphased_call_fails(self):
        path=self.path/'unphased.vcf';path.write_text('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\n22\t17\t0\tA\tC\t.\tPASS\t.\tGT\t0/1\n')
        reader=SortedVCF(path,[dict(sample_id='a',haplotype=i) for i in [0,1]])
        with self.assertRaisesRegex(ValueError,'Phased'):reader.at(17)
        reader.close()
    def test_sampling_math_and_microbatch_denominator_keep_duplicate_draws(self):
        class Enumerate:
            def integers(self,low,high,size):
                self_check=(low,high,size)
                assert self_check==(0,5,5)
                return np.arange(5)
        self.assertEqual(uniform_target_draws({'a':2,'b':3},5,Enumerate()),
                         [('a',0),('a',1),('b',0),('b',1),('b',2)])
        p=dict(tree_window_index=np.array([0,1]),node_tree_index=np.array([0,0,1,1,1]),
               focal_tree_index=np.array([0,1,0]),focal_example_index=np.array([7,9,7]))
        groups=node_budget_microbatches(p,3)
        self.assertEqual([g['target_indices'].tolist() for g in groups],[[0,2],[1]])
        self.assertEqual([g['loss_denominator'] for g in groups],[3,3])
        with self.assertRaisesRegex(ValueError,'exceeds'):node_budget_microbatches(p,2)
    def test_coordinate_draws_density_boundaries_and_integral_vcf(self):
        a=uniform_coordinate_context(np.random.default_rng(20261028),1,4,5,10)
        b=uniform_coordinate_context(np.random.default_rng(20261028),1,4,5,10)
        np.testing.assert_array_equal(a['coordinates'],b['coordinates'])
        self.assertEqual(a['domain'].tolist(),[0,5])
        self.assertEqual(a['coordinates'][0],1)
        np.testing.assert_array_equal(a['weights'],[0,.25,.25,.25,.25])
        np.testing.assert_array_equal(a['density'][1:],[.2]*4)
        self.assertEqual(a['density'][0],0)
        self.assertEqual(exact_vcf_position(434488+397),434886)
        with self.assertRaisesRegex(ValueError,'integral'):exact_vcf_position(3.5)
    def test_vcf_duplicate_position_and_mutation_rejected(self):
        path=self.path/'duplicate.vcf'
        path.write_text('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\n22\t17\t0\tA\tC\t.\tPASS\t.\tGT\t0|1\n22\t17\t1\tA\tG\t.\tPASS\t.\tGT\t0|1\n')
        reader=SortedVCF(path,[dict(sample_id='a',haplotype=i) for i in [0,1]])
        with self.assertRaisesRegex(ValueError,'Duplicate POS'):reader.at(17)
        reader.close()
        ts=fixture();treepath=self.path/'mutate.trees';ts.dump(treepath)
        tree_reader=ReplacementTreeReader(treepath,ts.samples())
        with treepath.open('ab') as stream:stream.write(b'x')
        with self.assertRaisesRegex(ValueError,'changed'):tree_reader.at(2)
    def test_coordinate_packet_retains_duplicate_draws_and_distinct_focal_contexts(self):
        ts=fixture();path=self.path/'context.trees';ts.dump(path)
        reader=ReplacementTreeReader(path,ts.samples())
        class ObservedCalls:
            def at(self,p):return np.array([0,1,0],dtype=np.int8),['A','C']
        ds=TrainWindowDataset.__new__(TrainWindowDataset)
        ds.sample_ids=ts.samples();ds.offset=397;ds.halo=2
        ds.panel=[dict(sample_id=f's{i}',haplotype=0,role='query',observed_cohort='present_day',
                       observed_reference_class=None,age_generations=0) for i in range(3)]
        ds.records={i:dict(example_index=i,sample_index=i,sample_id=f's{i}',haplotype=0,
                            query_node_id=i,cohort='present_day',position_local_bp=2+i,
                            position_original_bp=399+i,native_site_id=i) for i in range(2)}
        ds.labels=np.array([0,1]);ds.leaf_features=np.zeros((3,17),dtype=np.float32)
        ds._handles=lambda:(reader,ObservedCalls())
        specs={}
        for i,coordinates in enumerate(([2,7,7],[3,1,1])):
            specs[i]=dict(coordinates=np.array(coordinates),weights=np.array([0,.5,.5]),
                          density=np.array([0,.1,.1]),forced=np.array([1,0,0],dtype=bool),
                          domain=np.array([0,10]),kind=1)
        p=ds.batch([0,1,0],context_specs=specs)
        self.assertEqual(p['tree_window_index'].tolist(),[0,0,1])
        self.assertEqual(p['context_coordinate_local_bp'].tolist(),[2,7,7,3,1,1])
        self.assertEqual(p['context_tree_index'].tolist(),[0,1,1,2,2,2])
        self.assertEqual(p['focal_tree_index'].tolist(),[0,2,0])
        self.assertEqual(p['focal_position_original_bp'].tolist(),[399,400,399])
        self.assertEqual(p['labels'].tolist(),[0,1,0])
        self.assertTrue(np.all(p['context_proposal_density_per_bp']>=0))
        with self.assertRaisesRegex(ValueError,'node budget'):ds.batch([0,1],context_specs=specs,max_nodes=1)
    def test_real_packet_source_ids_metadata_and_all_sites(self):
        with np.load(HERE/'train000_first_window.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
        self.assertEqual(p['focal_example_index'].tolist(),[6667,6668,6669,6670])
        self.assertEqual(p['leaf_node_index'].shape,(5,952))
        self.assertEqual(p['site_genotypes'].shape,(39,952))
        self.assertTrue(np.all(p['reference_class'][p['query_mask']]==-1))
        self.assertEqual(len(np.unique(p['observed_cohort'])),9)
        self.assertEqual(int(p['query_mask'].sum()),686)
        for tid,leaves in enumerate(p['leaf_node_index']):
            np.testing.assert_array_equal(p['node_sample_index'][leaves],np.arange(952))
            self.assertTrue(np.all(p['node_tree_index'][leaves]==tid))
        for focal,sid in enumerate(p['focal_native_site_id']):
            candidates=np.flatnonzero((p['site_native_id']==sid)&(p['site_tree_index']==p['focal_tree_index'][focal]))
            self.assertEqual(len(candidates),1)
            np.testing.assert_array_equal(p['site_genotypes'][candidates[0]],p['focal_genotypes'][focal])

if __name__=='__main__':unittest.main(verbosity=2)
