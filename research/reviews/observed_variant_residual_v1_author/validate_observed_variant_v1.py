"""Bounded author validation: one hand-derived allele case and observed metadata joins."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from arggnn.observed_variant_residual_v1 import (
    categorical_log_evidence_v1, composite_window_fields_v1,
    center_observed_query_fields_v1, validate_observed_roster_v1)

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OUT = ROOT / "research/reviews/observed_variant_residual_v1_author"


def main():
    # Same hand case spans an artificial250kb boundary, includes categorical
    # multiallelic observations and one entirely uncalled query site.
    g = np.asarray([[1,0,0,0,1,1,0,0,1,1],
                    [2,0,0,0,2,2,0,0,2,2],
                    [-1,-1,0,0,1,1,0,0,1,1]], dtype=np.int8)
    k = np.asarray([2,3,2], dtype=np.int8)
    qi, ri = [0,1], list(range(2,10))
    labels = np.asarray([0,0,1,1,1,1,2,2], dtype=np.int64)
    ages = np.asarray([200,200,200,200,400,400,400,400], dtype=np.float64)
    score, valid, classes = categorical_log_evidence_v1(g,k,qi,ri,labels,ages)
    assert abs(np.exp(score[0,0,0]) - 1/6) < 1e-14
    assert abs(np.exp(score[0,0,1]) - 5/6) < 1e-14
    assert abs(np.exp(score[1,0,0]) - 1/7) < 1e-14
    assert abs(np.exp(score[1,0,1]) - 5/7) < 1e-14
    fields, counts, meta = composite_window_fields_v1(
        [249950.,250050.,270000.],score,valid,[250000.,270000.],[249000.,280000.])
    expected = np.asarray([[1/26,25/26,0,0,0,1/26,25/26,0],
                           [25/26,1/26,0,0,0,25/26,1/26,0]])
    assert np.allclose(fields[0],expected,rtol=0,atol=1e-14)
    assert np.array_equal(counts[0,:,0:2],np.full((2,2),2))
    assert np.array_equal(fields[1],np.zeros((2,8)))
    assert np.array_equal(fields[:,:,4],np.zeros((2,2)))
    centered = center_observed_query_fields_v1(fields,counts,[0.,0.]).numpy()
    assert np.allclose(centered[0,:,0],[-6/13,6/13],rtol=0,atol=3e-8)
    assert np.array_equal(centered[1],np.zeros((2,8),dtype=np.float32))
    assert np.array_equal(centered[:,:,4],np.zeros((2,2),dtype=np.float32))
    hand = {"case":"two_observed_SNPs_straddling250kb_boundary_plus_uncalled_site",
        "alpha":0.5,"first_site_query1_class_likelihoods":[1/6,5/6],
        "second_site_query1_class_likelihoods":[1/7,5/7],
        "combined_query1_young_class_field":[1/26,25/26,0,0],
        "combined_query1_old_class_field":[0,1/26,25/26,0],
        "centered_query1_young_first_class":-6/13,
        "exact_absent_older_ANA_zero":True,"empty_uncalled_window_exact_zero":True,
        "original_parent_window_crosses250kb_tile":True,"handcase_classes":classes.tolist(),
        "pass":True}
    variants = json.loads((OUT/"observed_variant_bindings.json").read_text())
    emitter = json.loads((ROOT/"research/reviews/persistence_trace_pilot_v1_root/emitter_bindings.json").read_text())
    records=[]
    for parent, refs in variants["parents"].items():
        folder=ROOT/f"data/biological_paths_execution_pilot_v1/{parent}/observed"
        original=json.loads((folder/"sample_manifest.json").read_text())
        summary=validate_observed_roster_v1(original)
        prepared_folder=ROOT/f"data/official_argformer_tsinfer_parent_v3/{parent}/prepared"
        entry=json.loads((prepared_folder/"observed_parent_entry.json").read_text())
        prepared=json.loads((prepared_folder/"prepared_manifest.json").read_text())
        inferred=json.loads((prepared_folder/"observed_roster.json").read_text())
        emitted=json.loads((ROOT/emitter["parents"][parent]["observed_roster"]["path"]).read_text())
        fields=("sample_index","sample_id","haplotype","role","observed_cohort","age_generations","observed_reference_class")
        canonical=[{k:r[k] for k in fields} for r in original]
        assert [{k:r[k] for k in fields} for r in inferred]==canonical
        assert [{k:r[k] for k in fields} for r in emitted]==canonical
        for key, filename in {"vcf":"genotypes.vcf","site_coordinates":"site_coordinate_manifest.json","source_roster":"sample_manifest.json"}.items():
            assert refs[key] == {"path":str(folder.relative_to(ROOT)/filename),
                                 "sha256":entry["input_observed_sha256"][filename]}
            if key!="vcf":
                assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==refs[key]["sha256"]
        coords=json.loads((folder/"site_coordinate_manifest.json").read_text())
        assert len(coords["rows"])==entry["native_full_sites"]==entry["sites_in_scored_support"]
        assert entry["no_input_variant_filtering"] is True
        assert prepared["coordinate_contract"]==entry["coordinate_contract"]
        coverage=entry["coordinate_contract"]["coverage_original_bp0"]
        indexes=entry["coordinate_contract"]["retained_fixed100bp_indexes"]
        expected=[i for i in range(20000) if coverage[0]<=50+100*i<coverage[1]]
        assert indexes==expected
        tile_counts=[sum(tile*250000<=50+100*i<(tile+1)*250000 for i in indexes) for tile in range(8)]
        assert all(tile_counts) and sum(tile_counts)==len(indexes)
        counts=[[sum(r["role"]=="pure_source_reference" and r["observed_reference_class"]==c and
                     (r["age_generations"]<350)==young for r in original) for c in range(4)] for young in (True,False)]
        assert sum(counts[0])==84 and sum(counts[1])==12 and counts[1][0]==0
        records.append({"parent_id":parent,"identity_role_date_class_joins_exact":True,
            "original_sites":len(coords["rows"]),"coverage_original_bp0":coverage,
            "retained_center_count":len(indexes),"tile_center_counts":tile_counts,
            "younger_class_counts":counts[0],"older_class_counts":counts[1],
            "VCF_hash_source":"copied_inference_metadata;payload_not_opened_or_hashed"})
    cfg=json.loads((ROOT/"configs/observed_variant_residual_v1.json").read_text())
    for ref in cfg["helper_refs"].values():
        assert hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest()==ref["sha256"]
    for name in ("src/arggnn/observed_variant_residual_v1.py","src/arggnn/train_observed_variant_residual_v1.py"):
        ast.parse((ROOT/name).read_text())
    # Source inspection establishes the preparation-before-target call ordering.
    runner=ast.parse((ROOT/"src/arggnn/train_observed_variant_residual_v1.py").read_text())
    prep=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=="prepare_v1")
    names={n.id for n in ast.walk(prep) if isinstance(n,ast.Name)}
    assert "bindings" not in names and "ExternalTargetV1" not in names and "targets" not in names
    run=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=="run_v1")
    feature_line=min(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id=="prepare_v1")
    target_line=min(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id=="train_targets_after_features_v1")
    assert target_line>feature_line
    result={"schema":"observed_variant_author_validation_v1","status":"passed_bounded_handcase_and_metadata_joins",
        "hand_derived_case":hand,"parents":records,"only_original_observed_metadata_and_source_read":True,
        "original_VCF_or_tree_payload_read":False,"models_or_query_targets_read":False,
        "GPU_used":False,"fits_run":False,"broad_test_suite_run":False}
    with (OUT/"validation.json").open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"status":result["status"],"handcase":True,"metadata_parents":len(records),
                      "payloads_opened":False,"GPU_used":False}))
if __name__=="__main__":
    main()
