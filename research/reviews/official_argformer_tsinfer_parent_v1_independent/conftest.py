"""All fixture writes and binary reads stay in this independent review directory."""
import builtins
import os
from pathlib import Path
import pytest

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/official_argformer_tsinfer_parent_v1_independent"
BINARY={".trees",".npz",".npy",".pkl",".pt",".vcf",".bcf",".samples"}

@pytest.fixture(autouse=True)
def independent_io_guard(monkeypatch):
    original_open=Path.open
    original_builtin=builtins.open
    original_os_open=os.open
    original_mkdir=Path.mkdir
    original_env=dict(os.environ)
    def permitted(value,mode="r",is_os=False):
        if isinstance(value,int):return
        p=Path(value)
        if not p.is_absolute():p=Path.cwd()/p
        p=p.absolute()
        inside=p==OWN or OWN in p.parents
        writable=bool(mode & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) if is_os else any(x in str(mode) for x in "wax+")
        if writable and not inside:raise AssertionError("review write outside ownership: "+str(p))
        if not writable and p.suffix in BINARY and not inside:raise AssertionError("actual binary/data read forbidden: "+str(p))
        if not writable and p in {ROOT/".runtime/gpup/storage_reservations.json",ROOT/".runtime/gpup/scheduler.lock"}:raise AssertionError("live ledger/mutex access forbidden")
    def path_open(p,mode="r",*args,**kwargs):
        permitted(p,mode);return original_open(p,mode,*args,**kwargs)
    def builtin_open(file,mode="r",*args,**kwargs):
        permitted(file,mode);return original_builtin(file,mode,*args,**kwargs)
    def os_open(file,flags,*args,**kwargs):
        permitted(file,flags,True);return original_os_open(file,flags,*args,**kwargs)
    def mkdir(p,*args,**kwargs):
        permitted(p,"w");return original_mkdir(p,*args,**kwargs)
    monkeypatch.setattr(Path,"open",path_open)
    monkeypatch.setattr(builtins,"open",builtin_open)
    monkeypatch.setattr(os,"open",os_open)
    monkeypatch.setattr(Path,"mkdir",mkdir)
    yield
    os.environ.clear();os.environ.update(original_env)
