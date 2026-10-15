#!/usr/bin/env python3
"""Small injected metadata cases; no msprime/numpy/tskit imports or simulations."""
import ast
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
import generation as g


def roster():
    return {'schema':'prospective1611_roster_v1','rows':[
        {'haplotype_id':c+'_h'+str(p),'diploid_id':c,'phase':p,'cohort':c,'age_generations':age,'node_id':None}
        for c,age in [('NF',180),('Ana',260)] for p in (0,1)]}


class Array(list):
    @property
    def shape(self):return (len(self),)
    def __lt__(self,other):return [x<other for x in self]
    def __getitem__(self,key):
        if isinstance(key,list):return Array(list.__getitem__(self,i) for i in key)
        return list.__getitem__(self,key)
    def tolist(self):return list(self)
    def tobytes(self):return bytes(self)


class FakeNP:
    int8='int8'
    def __init__(self):
        self.draws=iter([[.5]*4,[.0005,.001,.5,.9]])
        self.calls=[]
        outer=self
        class RNG:
            def random(self,n):outer.calls.append(n);return Array(next(outer.draws))
        self.random=types.SimpleNamespace(PCG64=lambda seed:('PCG64',seed),Generator=lambda kind:RNG())
    @staticmethod
    def asarray(x,dtype=None):return Array(x)
    @staticmethod
    def any(x):return any(x)
    @staticmethod
    def packbits(bits,bitorder):
        assert bitorder=='little'
        return Array([sum(int(v)<<i for i,v in enumerate(bits))])


class GenerationTests(unittest.TestCase):
    def test_frozen_science_and_metadata_roster(self):
        plan,parent,r,d=g.scientific_inputs('published_recon_train_000')
        self.assertEqual(len(r['rows']),3222)
        self.assertEqual(len(g.sample_specs(r)),1611)
        self.assertEqual(parent['new_seed_values']['genotype_error'],1031816016)
        self.assertEqual(d['census_time'],350)
        self.assertNotIn('BAA',d['populations_initial_size'])
        self.assertEqual([x['time'] for x in d['admixtures']],[136,186,161])
    def test_ood_requires_realized_parameters(self):
        with self.assertRaisesRegex(ValueError,'OOD'):g.scientific_inputs('published_recon_misspec_000')
    def test_unknown_parent_rejected(self):
        with self.assertRaises(ValueError):g.scientific_inputs('invented_parent')
    def test_sample_specs_explicit_phase_and_population(self):
        specs=g.sample_specs(roster())
        self.assertEqual(specs[0],{'diploid_id':'NF','population':'Neo','time':180,'ploidy':2,'num_samples':1,'haplotype_ids':['NF_h0','NF_h1']})
    def test_native_demography_arguments_without_loading_msprime(self):
        _,_,_,d=g.scientific_inputs('published_recon_iid_000')
        events=[]
        class Dem:
            def add_population(self,**kw):events.append(('population',kw))
            def add_admixture(self,**kw):events.append(('admixture',kw))
            def add_population_split(self,**kw):events.append(('split',kw))
            def add_census(self,**kw):events.append(('census',kw))
            def sort_events(self):events.append(('sort',{}))
        g.build_native_demography(types.SimpleNamespace(Demography=Dem),d)
        self.assertEqual(sum(k=='population' for k,v in events),10)
        self.assertEqual(sum(k=='admixture' for k,v in events),3)
        self.assertIn(('census',{'time':350}),events)
        self.assertNotIn('BAA',str(events))
    def test_exact_clipping_without_extrapolation(self):
        rows=g.clip_rebase([(100,0.,0.),(110,0.,1.),(120,0.,3.)],105,115)
        self.assertEqual(rows,[(0,100000.,0.),(5,200000.,.5),(10,0.,1.5)])
        self.assertEqual(g.rate_map_arrays(rows),([0,5,10],[.001,.002]))
        with self.assertRaises(ValueError):g.clip_rebase([(100,0.,0.),(110,0.,1.)],99,105)
    def test_no_guessed_or_fractional_interval(self):
        for a,b in [(None,10),(0,1.5),(True,10),(10,10)]:
            with self.assertRaises((ValueError,TypeError)):g.clip_rebase([(0,0.,0.),(10,0.,1.)],a,b)
    def test_authenticated_tiny_map_derivation_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=g.HERE) as directory:
            p=Path(directory);source=p/'source.txt';archive=p/'tiny_metadata.tar';rebased=p/'map.txt'
            header=['position','COMBINED_rate(cM/Mb)','Genetic_Map(cM)']
            source.write_text('Chromosome Position(bp) Rate(cM/Mb) Map(cM)\nchr22 100 100000 0\nchr22 110 200000 1\nchr22 120 0 3\n')
            with tarfile.open(archive,'w') as tar:tar.add(source,arcname='exact_chr22_member.txt')
            rebased.write_text(' '.join(header)+'\n0 100000 0\n5 200000 0.5\n10 0 1.5\n')
            doc={'schema':'prospective_chr22_GRCh37_map_derivation_receipt_v1','root_approved':True,'complete':True,
                 'chromosome':'22','genome_build':'GRCh37','source_url':'https://example.invalid/tiny_metadata_only','archive_path':str(archive),'archive_sha256':g.sha(archive),
                 'source_map_path':str(source),'source_map_sha256':g.sha(source),'source_member_name':'exact_chr22_member.txt',
                 'source_column_indices':{'chromosome':0,'position':1,'rate':2,'cumulative_cM':3},'source_header':['Chromosome','Position(bp)','Rate(cM/Mb)','Map(cM)'],
                 'rebased_column_indices':{'position':0,'rate':1,'cumulative_cM':2},
                 'source_interval_bp':[105,115],'length_bp':10,'rebased_map_path':str(rebased),'rebased_map_sha256':g.sha(rebased),
                 'derivation_algorithm':g.MAP_ALGORITHM,'simulation_rate_semantics':g.RATE_SEMANTICS,
                 'physical_coordinate_transform':'rebased_bp=source_bp-interval_start_bp; no source coordinate shift'}
            self.assertEqual(len(g.verify_map_document(doc,expected_archive_sha=g.sha(archive))),3)
            # Production always enforces the actual official archive digest.
            with self.assertRaises(ValueError):g.verify_map_document(doc)
            rebased.write_text(rebased.read_text().replace('0.5','0.6'));doc['rebased_map_sha256']=g.sha(rebased)
            with self.assertRaisesRegex(ValueError,'clipped source'):g.verify_map_document(doc,expected_archive_sha=g.sha(archive))
    def test_missing_map_binding_rejected(self):
        with self.assertRaises(ValueError):g.verify_map_document({})
    def test_identity_allocation_uses_actual_individuals_not_offsets(self):
        data={991:(8,2,180),17:(8,2,180),330:(4,8,260),114:(4,8,260)}
        class TS:
            def samples(self):return [991,17,330,114]
            def node(self,n):
                i,p,t=data[n];return types.SimpleNamespace(individual=i,population=p,time=t)
        mapping,allocation=g.bind_generated_samples(TS(),roster(),{2:'Neo',8:'Ana'})
        self.assertEqual([r['simulation_node_id'] for r in mapping['rows']],[991,17,330,114])
        self.assertEqual([r['phase'] for r in allocation],[0,1,0,1])
        self.assertEqual([r['simulation_individual_id'] for r in allocation],[8,8,4,4])
    def test_unpaired_generated_individual_rejected(self):
        class TS:
            def samples(self):return [901]
            def node(self,n):return types.SimpleNamespace(individual=9,population=2,time=180)
        with self.assertRaises(ValueError):g.bind_generated_samples(TS(),roster(),{2:'Neo'})
    def test_fixed_error_boundary_and_both_alleles(self):
        observed,flip=g.apply_haploid_error([0,1,0,1],[.000999,.00001,.001,.9])
        self.assertEqual(observed,[1,0,0,1]);self.assertEqual(flip,[True,True,False,False])
        for calls,draws in [([2],[.5]),([0],[1.]),([0],[]),([0],[float('nan')])]:
            with self.assertRaises(ValueError):g.apply_haploid_error(calls,draws)
    def test_common_phased_export_filtering_coordinate_and_error_draws(self):
        np=FakeNP();seen={}
        class TS:
            def variants(self,**kw):
                seen.update(kw)
                for pos,calls in [(0,[0,0,0,0]),(5,[0,1,1,0])]:
                    yield types.SimpleNamespace(site=types.SimpleNamespace(position=pos),alleles=('0','1'),genotypes=calls)
        mapping={'rows':[{'haplotype_id':r['haplotype_id'],'simulation_node_id':n} for r,n in zip(roster()['rows'],[103,901,87,23])]}
        with tempfile.TemporaryDirectory(dir=g.HERE) as directory:
            out=Path(directory);result=g.export_observed(TS(),roster(),mapping,out,100,10,99,np,g.OutputBudget(100000))
            self.assertEqual(seen,{'samples':[103,901,87,23],'isolated_as_missing':False})
            self.assertEqual(np.calls,[4,4])
            self.assertEqual(result['observed_sites'],1);self.assertEqual(result['posterror_invariant_sites_dropped'],1)
            self.assertEqual(result['all_site_haploid_error_flips'],1)
            vcf=(out/'observed.vcf').read_text();haps=(out/'observed.haps').read_text()
            self.assertIn('22\t6\tvar000000001\tA\tC\t.\tPASS\t.\tGT\t1|1\t1|0',vcf)
            self.assertIn('22 var000000001 6 A C 1 1 1 0',haps)
            self.assertIn('var000000001\t5\t6\t105\tA\tC\t3',(out/'variants.tsv').read_text())
            self.assertNotIn('error_flip',(out/'variants.tsv').read_text())
            self.assertNotIn('simulation_node',(out/'haplotypes.tsv').read_text())
    def test_inference_offset_and_normalized_span_must_be_explicit(self):
        observed={'map_binding':{'length_bp':10}}
        good={'sequence_length_bp':10,'coordinate_transform':{'export_position_encoding':'POS=rebased_bp+1','normalized_tree_coordinates':'rebased_bp_zero_based_half_open','map_relate_transform_receipt_sha256':'a'*64}}
        g.validate_inference_coordinates(good,observed)
        for key,value in [('export_position_encoding','POS=rebased_bp'),('normalized_tree_coordinates','encoded_one_based'),('map_relate_transform_receipt_sha256',None)]:
            bad=copy.deepcopy(good);bad['coordinate_transform'][key]=value
            with self.assertRaises(ValueError):g.validate_inference_coordinates(bad,observed)
        with self.assertRaises(ValueError):g.validate_inference_coordinates(dict(good,sequence_length_bp=11),observed)
    def test_output_budget_is_a_hard_stream_limit(self):
        b=g.OutputBudget(2);stream=io.StringIO();b.write(stream,'ab')
        with self.assertRaises(ValueError):b.write(stream,'x')
        self.assertEqual(stream.getvalue(),'ab')
    def test_no_numeric_imports_on_module_load(self):
        self.assertFalse({'msprime','numpy','tskit','torch','tensorflow','ancpath'}&set(sys.modules))
        tree=ast.parse((g.HERE/'generation.py').read_text())
        top=[]
        for n in tree.body:
            if isinstance(n,ast.Import):top.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):top.append(n.module)
        self.assertFalse({'numpy','msprime','tskit'}&set(top))
    def test_frozen_source_pins(self):self.assertEqual(g.verify_sources(),25)


if __name__=='__main__':unittest.main(verbosity=2)
