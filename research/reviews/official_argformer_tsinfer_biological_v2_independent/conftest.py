import builtins
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def audit_no_real_arrays_or_native_models(monkeypatch):
    original_open=Path.open
    original_builtin=builtins.open
    original_import=builtins.__import__
    blocked={".trees",".npz",".npy",".pt",".pth",".pkl",".vcf",".bcf"}
    def path_open(path,*args,**kwargs):
        if path.suffix.lower() in blocked:
            raise AssertionError("Audit forbids real array/tree/model payload open: "+str(path))
        return original_open(path,*args,**kwargs)
    def builtin_open(file,*args,**kwargs):
        if isinstance(file,(str,bytes,Path)) and Path(file).suffix.lower() in blocked:
            raise AssertionError("Audit forbids payload open: "+str(file))
        return original_builtin(file,*args,**kwargs)
    def imports(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","tskit","tsinfer","tsdate","composer","transformers","omegaconf"}:
            raise AssertionError("Audit forbids native model/runtime import: "+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(Path,"open",path_open)
    monkeypatch.setattr(builtins,"open",builtin_open)
    monkeypatch.setattr(builtins,"__import__",imports)
