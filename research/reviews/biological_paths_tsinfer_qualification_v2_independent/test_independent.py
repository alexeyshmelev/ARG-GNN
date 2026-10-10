"""Independent reduced qualification contract checks; no native execution."""
import ast, importlib.util, json
from pathlib import Path
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("qualification_author_helpers",ROOT/"research/reviews/biological_paths_tsinfer_native_failure_v1/test_qualification_v2_source_only.py")
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
h=helpers.h


def test_only_dating_delta_is_copy_isolated_and_biological1000_remains_pinned():
    base=h.load_producer();c=helpers.cfg()
    biological,effective,prior=h.validate_config(base,c)
    raw=(ROOT/c["biological_config"]["path"]).read_bytes()
    effective["dating"]["rescaling_intervals"]=999
    assert biological["dating"]["rescaling_intervals"]==1000
    assert c["qualification_dating"]["rescaling_intervals"]==1
    assert (ROOT/c["biological_config"]["path"]).read_bytes()==raw


def test_prior_scope_growth_is_counted_even_when_new_output_shrinks(monkeypatch):
    context,values,*_=helpers.storage_fixture(monkeypatch,new_bytes=500,prior_bytes=600)
    values["new"]=1
    values["oldtree"]=1000000000
    with pytest.raises(ValueError,match="combined preserved"):
        context.check()


def test_native_entry_reuses_fixture_pipeline_without_observation_loader_or_rebinding():
    tree=ast.parse(Path(h.__file__).read_text())
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert calls.count("base.handmade_assets")==1
    assert calls.count("base.pipeline")==1
    assert calls.count("base.qualification_summary")==1
    assert "base.load_assets" not in calls and "base.infer_parent" not in calls
    assert helpers.cfg()["kind"]!="biological_paths_tsinfer_v1_cpu_qualification"
    assert helpers.cfg()["certification"]=="reduced_single_interval_rescaling_only_not_exact1000_interval_biological_qualification"


def test_preserved_scope_unreadability_fails_before_acceptance(tmp_path,monkeypatch):
    base,prior,tree,log=helpers.prior_fixture(tmp_path,monkeypatch)
    def unreadable(path,*,followlinks,onerror):
        onerror(PermissionError("handmade inaccessible old scope"))
        return []
    monkeypatch.setattr(h.os,"walk",unreadable)
    with pytest.raises(ValueError,match="unreadable preserved"):
        h.verify_prior_scopes(base,prior)
