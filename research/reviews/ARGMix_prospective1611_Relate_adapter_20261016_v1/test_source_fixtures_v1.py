"""Focused stdlib-only fixtures: synthetic strings/stubs and mocked subprocess."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,subprocess,tempfile,unittest
from unittest.mock import patch
sp=importlib.util.spec_from_file_location("prospective_relate",Path(__file__).with_name("relate_adapter_v1.py"))
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

class EdgeTable:
    def clear(self):pass
    def set_columns(self,**v):self.values=v
class FakeRaw:
    num_samples=m.N;num_nodes=m.N+8;sequence_length=11
    def __init__(self):
        self.ids=list(range(7,m.N+7));self.tables=SimpleNamespace(sites=SimpleNamespace(position=[2.,10.]))
    def samples(self):return self.ids
    def node(self,n):return SimpleNamespace(time=0 if n in self.ids else 500)
    def edges(self):
        for n in self.ids:yield SimpleNamespace(parent=m.N+7,child=n,left=0.,right=11.,metadata=b"")
    def dump_tables(self):
        raw=self
        class Tables:
            def __init__(self):self.edges=EdgeTable();self.sites=SimpleNamespace(position=[2.,10.]);self.sequence_length=11
            def sort(self):pass
            def tree_sequence(self):
                tree=SimpleNamespace(num_roots=1,root=m.N+7,num_samples=lambda n:m.N,is_leaf=lambda n:True)
                return SimpleNamespace(samples=raw.samples,num_nodes=raw.num_nodes,sequence_length=self.sequence_length,
                    trees=lambda:iter([tree]),tables=self)
        return Tables()

class MockRootRunner:
    def run_stage_v1(self,**kw):
        result=subprocess.run(kw["argv"],cwd=kw["cwd"],capture_output=True)
        return dict(stage=kw["label"],argv=kw["argv"],cwd=kw["cwd"],returncode=result.returncode,
            actual_direct_native_wait=True,all_owned_processes_and_groups_drained=True,
            operational_failure=None,fixture_only=True)

class Fixtures(unittest.TestCase):
    def test_pair_sample(self):
        self.assertEqual(m.canonical_sample("ID_1 ID_2 missing\n0 0 0\na NA 0\nb b 0\n",["a","b"]),m.sample_text(["a","b"]))
        with self.assertRaises(ValueError):m.canonical_sample(m.sample_text(["b","a"]),["a","b"])
    def test_all_haplotype_ages(self):
        rows=[dict(diploid_id="d%d"%i,phase=k,age_generations=185 if i==158 else 0) for i in range(m.D) for k in (0,1)]
        ids,ages=m.roster_order(dict(rows=rows));self.assertEqual((len(ids),len(ages),ages[316:318]),(1611,3222,[185,185]))
        with self.assertRaises(ValueError):m.roster_order(dict(rows=rows[:-1]))
    def test_map_shift(self):
        result=m.shifted_map([(0,2.,0.),(50,0.,.0001),(100,0.,.0001)],100)
        self.assertEqual(result.splitlines()[1:],["1 2 0","51 0 0.0001","101 0 0.0001"])
        with self.assertRaises(ValueError):m.shifted_map([(0,1.,0.),(0,0.,1.)],100)
    def test_exact_full_panel_convert_join(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ids=["d%d"%i for i in range(m.D)];calls=["0","1"]+["0"]*(m.N-2)
            haps="22 v1 42 A C "+" ".join(calls)+"\n"
            for name in ("original.haps","native.haps"):(p/name).write_text(haps)
            genotypes=["|".join(calls[i:i+2]) for i in range(0,m.N,2)]
            (p/"input.vcf").write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(ids)+"\n22\t42\tv1\tA\tC\t.\tPASS\t.\tGT\t"+"\t".join(genotypes)+"\n")
            (p/"variants.tsv").write_text("variant_id\tposition_bp_zero_based\tvcf_position_one_based\tsource_position_bp\tREF\tALT\tobserved_alt_haplotype_count\nv1\t41\t42\t16051388\tA\tC\t1\n")
            got=m.validate_converted(p/"original.haps",p/"native.haps",p/"input.vcf",p/"variants.tsv",ids)
            self.assertEqual((got["variant_count"],got["last_POS1"]),(1,42))
            (p/"native.haps").write_text(haps.replace(" 42 "," 43 "))
            with self.assertRaises(ValueError):m.validate_converted(p/"original.haps",p/"native.haps",p/"input.vcf",p/"variants.tsv",ids)
    def test_supported_normalization_no_extension(self):
        raw=FakeRaw();rows=[dict(age_generations=0) for _ in range(m.N)]
        ts,s=m.normalize(raw,dict(rows=rows),dict(variant_count=2,last_POS1=10))
        self.assertEqual(s["supported_scoring_interval_bp"],[0,10]);self.assertEqual(s["unobserved_terminal_interval_bp"],[10,m.L])
        self.assertFalse(s["terminal_topology_extended"]);self.assertEqual(ts.samples(),raw.samples())
        self.assertEqual(ts.tables.sites.position,[1.,9.]);self.assertEqual(ts.tables.edges.values["left"][0],0.)
        rows[0]["age_generations"]=1
        with self.assertRaises(ValueError):m.normalize(raw,dict(rows=rows),dict(variant_count=2,last_POS1=10))
    @patch("subprocess.run")
    def test_four_exact_native_stages_mocked(self,run):
        run.return_value=SimpleNamespace(returncode=0)
        stages=m.commands(dict(Relate="/native/Relate",RelateFileFormats="/native/RelateFileFormats"),Path("/fixture/inputs"),Path("/fixture/work"),"published_recon_train_000",dict(observed_vcf_prefix="/fixture/observed"))
        self.assertEqual([x[0] for x in stages],["convert_from_vcf","annotate","infer_all","convert_to_trees"])
        self.assertIn("All",stages[2][1]);self.assertIn("12545298",stages[2][1]);self.assertNotIn("EstimatePopulationSize"," ".join(stages[2][1]))
        for label,argv in stages:m.invoke(MockRootRunner(),label,argv,Path("/fixture/work"),{},None,fixture=True)
        self.assertEqual(run.call_count,4)
        with self.assertRaises(ValueError):m.invoke(MockRootRunner(),*stages[0],Path("/fixture/work"),{},None)
        run.return_value=SimpleNamespace(returncode=1)
        with self.assertRaises(ValueError):m.invoke(MockRootRunner(),*stages[0],Path("/fixture/work"),{},None,fixture=True)

if __name__=="__main__":unittest.main()
