"""Independent sealed repair probes: source and fake metadata only."""
import importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("observed_author_v3_helpers",ROOT/"research/reviews/biological_paths_tsinfer_observed_v2_author/test_observed_v2_source_only.py")
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
h=t.h
PARENT="bio_paths_pilot_train_000"

def forbid_foreign_reads(monkeypatch, foreign):
    original=Path.open
    reads=[]
    def guarded(path,*args,**kwargs):
        if path.resolve()==foreign.resolve():
            reads.append(str(path))
            raise AssertionError("foreign artifact bytes were read")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    return reads

@pytest.mark.parametrize("artifact",["completion_provenance","inferred_tree","observed_sample_manifest"])
def test_foreign_primary_ref_rejected_before_hash_or_read(tmp_path,monkeypatch,artifact):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch)
    foreign=tmp_path/"foreign_artifact";foreign.write_text("foreign handmade bytes")
    native[artifact]={**base.ref(foreign),**({"condition":base.CONDITION} if artifact=="inferred_tree" else {})}
    if artifact!="completion_provenance":
        t.reseal_metadata(base,native,m,review,mp,rp,a,mp.parent/PARENT)
    reads=forbid_foreign_reads(monkeypatch,foreign)
    with pytest.raises(ValueError,match="expected phase/parent"):
        h.parent_adequacy(base,c,lock,native,expected_parent=PARENT,phase="canary")
    assert reads==[]

@pytest.mark.parametrize("artifact",["inference.json","observed.trees","observed_sample_manifest.json","native_verification.json","stages.json","modern_dated.trees","task_transfer_v2.json","manifest.json"])
def test_symlink_join_rejected_without_reading_foreign_bytes(tmp_path,monkeypatch,artifact):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch)
    work=mp.parent/PARENT
    path=mp if artifact=="manifest.json" else work/artifact
    foreign=tmp_path/("foreign_"+artifact)
    path.rename(foreign);path.symlink_to(foreign)
    reads=forbid_foreign_reads(monkeypatch,foreign)
    with pytest.raises(ValueError,match="canonical"):
        if artifact=="manifest.json":
            h.canary_gate(base,c,lock,a)
        elif artifact=="task_transfer_v2.json":
            h.wrapper_join(base,c,m["replicates"][0],expected_parent=PARENT,phase="canary")
        else:
            h.parent_adequacy(base,c,lock,native,expected_parent=PARENT,phase="canary")
    assert reads==[]

@pytest.mark.parametrize("parent",["bio_paths_pilot_train_001","bio_paths_pilot_val_000","bio_paths_pilot_val_001"])
def test_remaining_run_join_preserves_native_bytes_and_exact_call(tmp_path,monkeypatch,parent):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch,phase="remaining",parent=parent)
    work=mp.parent/parent
    completion_before=(work/"inference.json").read_bytes()
    (work/"task_transfer_v2.json").unlink()
    scientific={"dating":{"rescaling_intervals":1000}}
    checks=[];context=SimpleNamespace(check=lambda:checks.append("check"))
    calls=[]
    def handmade_call(biological,p,target,pin,passed_context):
        calls.append((biological,p,target,pin,passed_context))
        return native
    monkeypatch.setattr(base,"infer_parent",handmade_call)
    row=h.run_parent(base,c,scientific,lock,parent,mp.parent,context,phase="remaining")
    assert calls==[(scientific,parent,mp.parent,lock["parents"][parent],context)]
    assert checks==["check","check"]
    assert (work/"inference.json").read_bytes()==completion_before
    assert row["parent_id"]==parent and row["frozen_helper_result"]["parent_id"]==parent
    assert row["condition"]==h.CONDITION and row["source_helper_condition"]==base.CONDITION
    assert row["adequacy"]["phase"]=="remaining" and row["adequacy"]["expected_parent"]==parent
    assert scientific["dating"]["rescaling_intervals"]==row["rescaling_intervals"]==1000

def test_inner_parent_rejected_before_pin_selection_or_artifact_hash(tmp_path,monkeypatch):
    base,c,lock,native,*_=t.metadata_fixture(tmp_path,monkeypatch)
    native["parent_id"]="bio_paths_pilot_train_001"
    class NoPinSelection(dict):
        def __getitem__(self,key):
            raise AssertionError("pin selected before inner parent equality")
    lock["parents"]=NoPinSelection()
    monkeypatch.setattr(base,"checked_ref",lambda *a:(_ for _ in ()).throw(AssertionError("artifact hash before parent equality")))
    with pytest.raises(ValueError,match="inner native parent"):
        h.parent_adequacy(base,c,lock,native,expected_parent=PARENT,phase="canary")

@pytest.mark.parametrize("field",["condition","source_helper_condition"])
def test_outer_condition_rejected_before_native_completion_read(tmp_path,monkeypatch,field):
    base,c,lock,native,m,review,mp,rp,a=t.metadata_fixture(tmp_path,monkeypatch)
    m["replicates"][0][field]="foreign_condition"
    mp.write_text(json.dumps(m));a["canary_manifest"]=base.ref(mp)
    review["canary_manifest"]=a["canary_manifest"];rp.write_text(json.dumps(review));a["canary_adequacy"]=base.ref(rp)
    completion=mp.parent/PARENT/"inference.json"
    reads=forbid_foreign_reads(monkeypatch,completion)
    with pytest.raises(ValueError,match="outer parent/helper/phase"):
        h.canary_gate(base,c,lock,a)
    assert reads==[]
