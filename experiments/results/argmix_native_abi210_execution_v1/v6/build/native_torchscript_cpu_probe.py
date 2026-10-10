#!/usr/bin/env python
"""Small CPU FFI/library probe; handmade fixture, not author-model or biological reproduction."""
import hashlib,json,os,pathlib,sys
if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
    raise RuntimeError("CPU-only admission requires empty CUDA visibility")
if "LIBTORCH_BYPASS_VERSION_CHECK" in os.environ:
    raise RuntimeError("Version-guard bypass is forbidden")
import torch
import ancpath
prefix=pathlib.Path(sys.prefix).resolve()
expected=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN/.runtime/official_argmix_abi210_v1/cu129/env")
if prefix!=expected or torch.__version__!="2.10.0+cu129":
    raise RuntimeError("Candidate runtime identity mismatch")
if not hasattr(ancpath,"RustTorchScriptInference"):
    raise RuntimeError("Actual default-feature native Torch class missing")
out=pathlib.Path(sys.argv[1]).resolve()
out.relative_to(pathlib.Path("/workspace-SR003.nfs2/ARG-GNN"))
out.mkdir(parents=True,exist_ok=False)
class Probe(torch.nn.Module):
    def forward(self,ids,tmrca,mask):
        slot=(ids[:,0]+tmrca[:,0,0]+mask.sum(1)).remainder(4)
        return torch.nn.functional.one_hot(slot,4).to(torch.float32)*3
tokens=[["A"],["B","A"],["missing"]]
matrices=[[[0,0],[0,0]],[[1,0],[0,0]],[[3,0],[0,0]]]
ids=torch.tensor([[2,0],[3,2],[1,0]],dtype=torch.int64)
tmrca=torch.tensor(matrices,dtype=torch.int64)
mask=ids.ne(0)
module=torch.jit.trace(Probe().eval(),(ids,tmrca,mask))
model=out/"handmade_ffi_probe.pt"
module.save(str(model))
vocab=out/"vocab.json"
vocab.write_text(json.dumps({"<PAD>":0,"<UNK>":1,"A":2,"B":3})+"\n")
native=ancpath.RustTorchScriptInference()
native.load_model(str(model),str(vocab),False)
actual=native.predict_batch(tokens,matrices)
expected_values=module(ids,tmrca,mask).argmax(-1).tolist()
if actual!=expected_values or native.get_device_info()!="Cpu":
    raise RuntimeError("Actual Python-to-Rust TorchScript CPU outputs/device differ")
paths=sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
              if "/" in line and any(s in line for s in ("libtorch","libc10","libhdf5","ancpath","libstdc++"))})
torchlib=(pathlib.Path(torch.__file__).resolve().parent/"lib").resolve()
torchpaths=[pathlib.Path(x).resolve() for x in paths if "libtorch" in x or "libc10" in x]
if not torchpaths or any(not x.is_relative_to(torchlib) for x in torchpaths):
    raise RuntimeError("Loaded Torch libraries are absent or outside the selected Python wheel")
extensions=sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                   if "ancpath" in line and ".so" in line})
if not extensions:
    raise RuntimeError("No actual native ancpath extension mapping")
receipt={"status":"passed","actual_CPU_Python_to_Rust_TorchScript_probe":True,
 "fixture":"three handmade observations; no targets or biological data",
 "Python_argmax":expected_values,"Rust_argmax":actual,"Rust_device":native.get_device_info(),
 "torch_version":torch.__version__,"ancpath_module":ancpath.__file__,
 "loaded_library_paths":paths,"native_extension_paths":extensions,
 "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
 "vocab_sha256":hashlib.sha256(vocab.read_bytes()).hexdigest(),
 "GPU_context_requested":False,"author_model_numerical_reproduction":False,
 "full_native_pipeline_admitted":False}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
