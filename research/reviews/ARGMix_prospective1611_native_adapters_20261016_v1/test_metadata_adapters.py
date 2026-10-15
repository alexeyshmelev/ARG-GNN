#!/usr/bin/env python3
"""Tiny deterministic metadata tests; never import Torch/TF/tskit/ancpath."""
import ast
import copy
import json
import types
import unittest
from pathlib import Path
from panel_adapter import *
from native_runtime import check_observed_mapping,validate_fit_support,finite_config,validate_rebased_map_rows
import tempfile
from truth_adapter import census_founder_label,simulation_sample_map,DEFINITION,construct_common_labels,validate_supervision_contract

def tiny(cohorts=("NF","Yam","Ana","Present")):
    rows=[]
    for ci,cohort in enumerate(cohorts):
        for phase in (0,1):
            # Mate IDs are intentionally not consecutive.
            rows.append({"haplotype_id":cohort+"_h"+str(phase),"diploid_id":cohort,
                         "phase":phase,"cohort":cohort,"age_generations":AGES[cohort],
                         "node_id":100*(ci+1)+(10 if phase==0 else 37)})
    return {"schema":"prospective1611_roster_v1","rows":rows}

class TinyTree:
    def __init__(self):
        self.parents={110:500,137:500,500:501,210:501,501:502,237:502,
                      502:504,310:503,337:503,503:504,504:506,
                      410:505,437:505,505:506,506:-1}
        self.times={500:190.9,501:200.9,502:220.7,503:275.5,504:300.9,505:100.1,506:400.9}
        self.children={}
        for n,p in self.parents.items():
            self.children.setdefault(p,[]).append(n)
    def parent(self,n):return self.parents[n]
    def time(self,n):return self.times[n]
    def leaves(self,n):
        if n not in self.children:return [n]
        out=[]
        for c in self.children[n]:out.extend(self.leaves(c))
        return out

def target(hid="NF_h0",**kw):
    row={"row_id":"r1","parent_id":"p","haplotype_id":hid,
         "tree_index":0,"left":0.0,"right":10.0,"position":5.0}
    row.update(kw);return row

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.doc=tiny();self.panel=Panel(self.doc,full=False)
    def test_full_counts_and_proposed_integer_dates(self):
        doc=make_roster();p=Panel(doc,require_bound=False)
        self.assertEqual(p.summary()["haplotypes"],3222)
        self.assertEqual(p.summary()["diploids"],1611)
        self.assertEqual(p.summary()["donor_candidates"],594)
        self.assertEqual(p.summary()["query_candidates"],2972)
        self.assertEqual(p.by_hap["NF_0158_h0"]["age_generations"],185)
    def test_arbitrary_exact_node_binding(self):
        doc=make_roster()
        mapping={"schema":"haplotype_node_mapping_v1","rows":[
            {"haplotype_id":r["haplotype_id"],"node_id":10000+7*(3221-i)}
            for i,r in enumerate(doc["rows"])]}
        bound=Panel(bind_roster(doc,mapping))
        self.assertEqual(bound.by_hap["Ana_0000_h0"]["node_id"],10000+7*3221)
        self.assertEqual(len(bound.donor_rows("NF_0000_h0")),592)
        self.assertEqual(len(bound.donor_rows("Yam_0000_h0")),592)
        self.assertEqual(len(bound.donor_rows("Present_0000_h0")),594)
    def test_missing_mapping_fails(self):
        with self.assertRaises(ValueError):
            bind_roster(make_roster(),{"schema":"haplotype_node_mapping_v1","rows":[]})
    def test_duplicate_node_fails(self):
        self.doc["rows"][1]["node_id"]=self.doc["rows"][0]["node_id"]
        with self.assertRaises(ValueError):Panel(self.doc,full=False)
    def test_unpaired_diploid_fails(self):
        self.doc["rows"].pop()
        with self.assertRaises(ValueError):Panel(self.doc,full=False)
    def test_fractional_age_fails(self):
        self.doc["rows"][0]["age_generations"]=180.5
        with self.assertRaises(ValueError):Panel(self.doc,full=False)
    def test_mate_cohort_disagreement_fails(self):
        self.doc["rows"][0]["cohort"]="Ana"
        with self.assertRaises(ValueError):Panel(self.doc,full=False)
    def test_dual_roles_and_identity_exclusion(self):
        nf={r["haplotype_id"] for r in self.panel.donor_rows("NF_h0")}
        yam={r["haplotype_id"] for r in self.panel.donor_rows("Yam_h0")}
        present={r["haplotype_id"] for r in self.panel.donor_rows("Present_h0")}
        self.assertFalse({"NF_h0","NF_h1"}&nf)
        self.assertTrue({"NF_h0","NF_h1"}<=yam)
        self.assertTrue({"Yam_h0","Yam_h1"}<=nf)
        self.assertEqual(present,{"NF_h0","NF_h1","Yam_h0","Yam_h1","Ana_h0","Ana_h1"})
    def test_native_constructor_exact_donor_ids_metadata_intact(self):
        captured={}
        class Capture:
            def __init__(self,*args):captured["args"]=args
        module=types.SimpleNamespace(SampleBatchIterator=Capture)
        req=argmix_request(self.panel,"NF_h0",num_refs=2)
        native_argmix_iterator(module,"observed.trees",req)
        a=captured["args"]
        self.assertEqual(a[7],[310,337,210,237])
        self.assertEqual(a[8],[110])
        self.assertEqual(a[2][110],("NF",180))
        self.assertNotIn(110,a[7]);self.assertNotIn(137,a[7])
        self.assertIsNone(a[4])
        self.assertEqual(a[5],0)
    def test_no_fallback_with_too_few_donors(self):
        with self.assertRaises(ValueError):argmix_request(self.panel,"NF_h0")
    def test_AP_literal_expected_NF_features(self):
        f=ap_features(TinyTree(),self.panel,"NF_h0")
        expected=[[0.,0.,1.,0.,0.,0.,0.,0.,200/1500],
                  [0.,0.,1.,0.,0.,0.,0.,0.,220/1500],
                  [0.,0.,0.,0.,0.,0.,1.,0.,300/1500],
                  [-15.]*9,[-15.]*9]
        self.assertEqual(f,expected)
    def test_AP_Yam_dual_role_features(self):
        f=ap_features(TinyTree(),self.panel,"Yam_h0")
        self.assertEqual(f[0],[0.,0.,0.,1.,0.,0.,0.,0.,200/1500])
        self.assertEqual(f[1],[0.,0.,0.,0.,0.,0.,1.,0.,300/1500])
        self.assertEqual(f[2:],[[ -15.]*9]*3)
    def test_AP_present_excludes_nonreference_cohorts(self):
        f=ap_features(TinyTree(),self.panel,"Present_h0")
        self.assertEqual(f[0],[0.,0.,1/3,1/3,0.,0.,1/3,0.,400/1500])
        self.assertEqual(f[1:],[[-15.]*9]*4)
    def test_AP_all_excluded_pads_missing_blocks(self):
        doc=tiny(("NF",));panel=Panel(doc,full=False)
        class T:
            def parent(self,n):return 500 if n!=500 else -1
            def leaves(self,n):return [110,137]
            def time(self,n):return 200.
        self.assertEqual(ap_features(T(),panel,"NF_h0"),[[-15.]*9]*5)
    def test_native_vocab_matches_pure_author_function(self):
        source=ast.parse((ARGMIX/"src/tree_sampler.py").read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="build_vocab_from_args")
        ns={"SPECIAL_TOKENS":list(SPECIAL_TOKENS)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"<author pure vocabulary>","exec"),ns)
        self.assertEqual(ns["build_vocab_from_args"](list(DONORS|QUERIES)),token_vocab())
    def test_native_flat_tokens_matrix_identity_no_labels(self):
        lookup=native_pop_token_lookup(self.panel)
        p=(list(range(9)),[0,3,4],[110],[(0.,10.)],1,3,["never forward this label"])
        rows=unpack_native_flat(p,lookup,expected_width=3)
        self.assertEqual(rows[0]["node_id"],110)
        self.assertEqual(rows[0]["tmrca"],[[0,1,2],[3,4,5],[6,7,8]])
        self.assertEqual(rows[0]["input_ids"][0],token_vocab()["<SAMPLE>"])
        self.assertNotIn("label",rows[0])
    def test_bad_native_flat_abi_fails(self):
        with self.assertRaises(ValueError):unpack_native_flat(([],),[0])
    def test_target_metadata_accepts_valid_midpoint(self):
        self.assertEqual(validate_targets([target()],self.panel)["target_rows"],1)
    def test_target_metadata_rejects_truth(self):
        with self.assertRaises(ValueError):validate_targets([target(label="Ana")],self.panel)
    def test_duplicate_target_rejected(self):
        with self.assertRaises(ValueError):validate_targets([target(),target(row_id="r2")],self.panel)
    def test_nonquery_target_rejected(self):
        with self.assertRaises(ValueError):validate_targets([target("Ana_h0")],self.panel)
    def test_midpoint_required(self):
        with self.assertRaises(ValueError):validate_targets([target(position=4.)],self.panel)
    def test_label_map_and_AP_probability_projection(self):
        self.assertEqual([encode_label(x) for x in LABELS],[0,1,2,3])
        self.assertEqual(project_ap_probabilities([.1,.2,.15,.25,.1,.2]),[.2,.4,.15,.25])
        with self.assertRaises(ValueError):encode_label("NF")
    def test_label_join_exact_once(self):
        t=[target()]
        self.assertEqual(validate_labels([{"row_id":"r1","ancestry":"CHG"}],t),{"r1":1})
        with self.assertRaises(ValueError):validate_labels([],t)
    def test_inferred_tree_interval_mapping_consistent_across_queries(self):
        with self.assertRaises(ValueError):
            validate_targets([target(),target("Yam_h0",row_id="r2",left=2.,position=6.)],self.panel)
    def test_map_metadata_exact_coverage(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            p=Path(directory)/"tiny_metadata_map.txt"
            p.write_text("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n0 1 0\n10 1 0.00001\n")
            self.assertEqual(validate_rebased_map_rows(p,10)["last_bp"],10)
            with self.assertRaises(ValueError):validate_rebased_map_rows(p,11)
    def test_map_metadata_forbids_nonmonotone_positions(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            p=Path(directory)/"tiny_metadata_map.txt"
            p.write_text("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n0 1 0\n10 1 0.00001\n10 1 0.00001\n")
            with self.assertRaises(ValueError):validate_rebased_map_rows(p,10)
    def test_paired_metadata_requires_same_realized_tree_and_targets(self):
        base={"schema":"prospective1611_native_features_v1","complete":True,
              "parent_id":"p","target_rows":1,"targets_sha256":"a"*64,
              "roster_sha256":"b"*64,"observed_trees_sha256":"c"*64,
              "source_pins_sha256":"d"*64,"fit_config_sha256":"e"*64}
        a=dict(base,method="argmix",feature_rows=10)
        b=dict(base,method="ap",feature_rows=1)
        self.assertTrue(compare_paired_metadata(a,b)["same_data_verified"])
        b["observed_trees_sha256"]="f"*64
        with self.assertRaises(ValueError):compare_paired_metadata(a,b)
    def test_deferred_runtime_has_no_top_level_numeric_imports(self):
        tree=ast.parse((HERE/"native_runtime.py").read_text())
        imports=[]
        for node in tree.body:
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        self.assertFalse(set(imports)&{"torch","tensorflow","numpy","tskit","ancpath"})
    def test_observed_node_age_and_interval_mapping(self):
        panel=self.panel
        class TS:
            def samples(self):return list(panel.by_node)
            def node(self,n):return types.SimpleNamespace(time=panel.by_node[n]["age_generations"])
            def at_index(self,i):return types.SimpleNamespace(interval=types.SimpleNamespace(left=0.,right=10.))
        check_observed_mapping(TS(),panel,[target()])
        with self.assertRaises(ValueError):
            check_observed_mapping(TS(),panel,[target(right=12.,position=6.)])
    def test_exact_census_population_is_truth_not_query_cohort(self):
        class TruthTree:
            def parent(self,n):return {991:880,880:770,770:-1}[n]
            def time(self,n):return {991:180.,880:350.,770:600.}[n]
            def population(self,n):return {880:42}[n]
        self.assertEqual(census_founder_label(TruthTree(),991,{42:"CHG"}),"CHG")
    def test_missing_exact_census_is_rejected(self):
        class TruthTree:
            def parent(self,n):return {991:880,880:-1}[n]
            def time(self,n):return {991:180.,880:350.9}[n]
            def population(self,n):return 42
        with self.assertRaises(ValueError):census_founder_label(TruthTree(),991,{42:"Ana"})
    def test_census_unknown_founder_rejected(self):
        class TruthTree:
            def parent(self,n):return {991:880,880:-1}[n]
            def time(self,n):return {991:180.,880:350.}[n]
            def population(self,n):return 42
        with self.assertRaises(ValueError):census_founder_label(TruthTree(),991,{42:"NF"})
    def test_simulation_mapping_is_separate_from_observed_ids(self):
        doc={"schema":"simulation_haplotype_node_mapping_v1","rows":[
            {"haplotype_id":hid,"simulation_node_id":9000+i*13}
            for i,hid in enumerate(self.panel.by_hap)]}
        mapping=simulation_sample_map(doc,self.panel)
        self.assertEqual(mapping["NF_h0"],9000)
        self.assertNotEqual(mapping["NF_h0"],self.panel.by_hap["NF_h0"]["node_id"])
    def test_paired_fit_signatures_include_exact_truth_identity(self):
        sig={"parent_id":"p","targets_sha256":"a"*64,"roster_sha256":"b"*64,
             "observed_trees_sha256":"c"*64,"labels_sha256":"d"*64,
             "supervision_contract_sha256":"e"*64,
             "truth_definition_sha256":canonical_sha(DEFINITION)}
        base={"schema":"prospective1611_native_fit_summary_v1",
              "train_shared_signatures":[sig],"val_shared_signatures":[sig],
              "logical_train_targets":1,"logical_val_targets":1,"fit_config_sha256":"f"*64}
        a=dict(base,method="argmix");b=copy.deepcopy(dict(base,method="ap"))
        self.assertTrue(compare_fit_summaries(a,b)["same_supervised_data_verified"])
        b["train_shared_signatures"][0]["labels_sha256"]="9"*64
        with self.assertRaises(ValueError):compare_fit_summaries(a,b)
    def test_common_truth_constructor_uses_separate_simulation_nodes(self):
        panel=self.panel
        sim={"schema":"simulation_haplotype_node_mapping_v1","rows":[
            {"haplotype_id":h,"simulation_node_id":9000+i*13}
            for i,h in enumerate(panel.by_hap)]}
        mapping={r["haplotype_id"]:r["simulation_node_id"] for r in sim["rows"]}
        inverse={n:panel.by_hap[h] for h,n in mapping.items()}
        names={1:"Neo",2:"Yam",3:"Ana",4:"present_bronze",5:"CHG"}
        sample_pops={"NF":1,"Yam":2,"Ana":3,"Present":4}
        class TruthTree:
            interval=types.SimpleNamespace(left=0.,right=10.)
            def parent(self,n):return n+100000 if n in inverse else -1
            def time(self,n):return inverse[n]["age_generations"] if n in inverse else 350.
            def population(self,n):return 5
        class TS:
            sequence_length=10.
            def samples(self):return list(inverse)
            def node(self,n):return types.SimpleNamespace(time=inverse[n]["age_generations"],population=sample_pops[inverse[n]["cohort"]])
            def population(self,p):return types.SimpleNamespace(metadata={"name":names[p]})
            def at(self,x):return TruthTree()
        pop={"schema":"simulation_population_mapping_v1","populations":{str(k):v for k,v in names.items()}}
        self.assertEqual(construct_common_labels(TS(),sim,pop,[target()],panel),
                         [{"row_id":"r1","ancestry":"CHG"}])
    def test_supervision_contract_binds_exact_labels_and_definition(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            d=Path(d);t=d/"targets.jsonl";r=d/"roster.json";l=d/"labels.jsonl"
            t.write_text(json.dumps(target())+"\n")
            r.write_text(json.dumps(self.doc))
            l.write_text(json.dumps({"row_id":"r1","ancestry":"CHG"})+"\n")
            dem={"census_time":350,"BAA_samples":0,
                 "populations_initial_size":{x:1000 for x in LABELS},
                 "admixtures":[{"time":180,"derived":"Neo","ancestral":["WHG","Ana"]}],
                 "splits":[{"time":600,"derived":["WHG","EHG"],"ancestral":"NE"}]}
            meta={"observed_trees_sha256":"c"*64,"targets_sha256":sha256(t),
                  "roster_sha256":sha256(r),"target_rows":1}
            c={"schema":"prospective1611_external_supervision_contract_v1","complete":True,
               "parent_id":"p","definition":DEFINITION,"truth_definition_sha256":canonical_sha(DEFINITION),
               "targets_sha256":sha256(t),"roster_sha256":sha256(r),"labels_sha256":sha256(l),
               "observed_trees_sha256":"c"*64,"target_rows":1,
               "constructor_source":{"path":str((HERE/"truth_adapter.py").relative_to(ROOT)),
                                     "sha256":sha256(HERE/"truth_adapter.py")},
               "truth_input_sha256":{x:"a"*64 for x in ("simulation_trees","simulation_sample_mapping","population_mapping","demography")},
               "demography":dem,"demography_semantic_sha256":canonical_sha(dem)}
            got=validate_supervision_contract(c,parent_id="p",targets_path=t,roster_path=r,
                                             labels_path=l,observed_metadata=meta)
            self.assertEqual(got["labels_sha256"],sha256(l))
            l.write_text(json.dumps({"row_id":"r1","ancestry":"Ana"})+"\n")
            with self.assertRaises(ValueError):
                validate_supervision_contract(c,parent_id="p",targets_path=t,roster_path=r,
                                               labels_path=l,observed_metadata=meta)
    def test_source_pins_and_finite_configs(self):
        self.assertGreater(verify_sources(),10)
        c=finite_config()
        self.assertEqual(c["argmix"]["model"]["num_refs"],15)
        self.assertNotIn("num_ancs",c["argmix"]["model"])
        self.assertEqual(c["argmix"]["model"]["num_classes"],4)
        self.assertEqual(c["ap"]["epochs"],5)
        registry=load_json(ROOT/"research/reviews/ARGMix_published_panel_next_source_20261015_v1/published_panel_plan.v1.json")
        self.assertEqual(c["argmix_seed"],registry["new_fit_seeds"]["ARGMix"])
        self.assertEqual(c["ap"]["seed"],registry["new_fit_seeds"]["AP"])
        self.assertEqual(c["ap"]["positive_target_indices"],[0,1,2,3])
        self.assertEqual(c["ap"]["no_positive_target_indices_native_BAA_paths"],[4,5])
        self.assertEqual(c["argmix"]["joint_model"]["train"]["num_epochs"],20)
    def test_tiny_metadata_is_not_full_published_panel(self):
        with self.assertRaises(ValueError):Panel(self.doc)
    def test_fit_rejects_tiny_parent_support(self):
        with self.assertRaises(ValueError):validate_fit_support([target()],[])

if __name__=="__main__":
    unittest.main(verbosity=2)
