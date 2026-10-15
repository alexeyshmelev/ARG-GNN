"""One CPU child, same-process gate before the sealed numerical generator."""
from pathlib import Path
import argparse, hashlib, json, os, runpy, select, sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
GEN=ROOT/"research/reviews/ARGMix_prospective1611_generation_20261016_v1/generation.py"
GEN_SHA="8acad46c4fc1012ddd350431495ce039059f40141aa9b09382516d7a74216fd7"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate-fd",type=int,required=True)
    p.add_argument("--request",required=True);p.add_argument("--request-sha256",required=True)
    a=p.parse_args()
    if a.gate_fd<3:raise ValueError("Inherited dedicated gate fd required")
    try:
        ready,_,_=select.select([a.gate_fd],[],[],30)
        if not ready or os.read(a.gate_fd,4)!=b"GO\n":
            raise ValueError("Root ownership gate was not released; no numerical imports")
    finally:os.close(a.gate_fd)
    request=Path(a.request)
    if request.resolve()!=request or not request.is_relative_to(ROOT):
        raise ValueError("Exact canonical bound generation request")
    if hashlib.sha256(request.read_bytes()).hexdigest()!=a.request_sha256:
        raise ValueError("Generation request changed after admission")
    if hashlib.sha256(GEN.read_bytes()).hexdigest()!=GEN_SHA:
        raise ValueError("Sealed generator source changed")
    sys.dont_write_bytecode=True
    sys.argv=[str(GEN),"generate","--request",str(request)]
    runpy.run_path(str(GEN),run_name="__main__")
    return 0

if __name__=="__main__":raise SystemExit(main())
