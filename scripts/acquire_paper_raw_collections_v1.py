"""Acquire source-cited raw bytes without decoding or evaluating datasets."""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, hashlib, json, os, sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
CAP = 4_000_000_000

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def local(relative):
    p=ROOT/relative
    if not isinstance(relative,str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Invalid project path")
    p.resolve().relative_to(ROOT)
    if any(a.is_symlink() for a in (p,*p.parents) if a!=ROOT.parent):
        raise ValueError("Symlink path")
    return p

def write_new(path,obj):
    with path.open("x") as f:
        json.dump(obj,f,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def download(row,scope):
    p=scope/row["collection"]/row["name"]
    expected=int(row["bytes"])
    sha256=hashlib.sha256()
    md5=hashlib.md5()
    blob=hashlib.sha1(("blob "+str(expected)+"\0").encode())
    count=0
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    req=Request(row["url"],headers={"User-Agent":"ARG-GNN reproducible raw acquisition"})
    try:
        with urlopen(req,timeout=60) as response, p.open("xb") as f:
            if response.status!=200:raise ValueError("HTTP status")
            final=urlsplit(response.geturl())
            if final.scheme!="https":raise ValueError("Insecure redirect")
            for chunk in iter(lambda:response.read(1024*1024),b""):
                count+=len(chunk)
                if count>expected:raise ValueError("Object exceeds source metadata size")
                f.write(chunk)
                sha256.update(chunk);md5.update(chunk);blob.update(chunk)
            f.flush();os.fsync(f.fileno())
        if count!=expected:raise ValueError("Object size mismatch")
        if row.get("md5") and md5.hexdigest()!=row["md5"]:raise ValueError("Publisher MD5 mismatch")
        if row.get("git_blob_sha1") and blob.hexdigest()!=row["git_blob_sha1"]:raise ValueError("Pinned Git blob mismatch")
        result={**row,"status":"verified","actual_bytes":count,"sha256":sha256.hexdigest(),
                "actual_md5":md5.hexdigest(),"actual_git_blob_sha1":blob.hexdigest(),
                "path":str(p.relative_to(ROOT)),"started_utc":started,
                "finished_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
                "final_https_host":final.hostname,"decoded":False}
        write_new(scope/"receipts"/(row["collection"]+"__"+row["name"]+".json"),result)
        print(json.dumps({"file":row["name"],"status":"verified","bytes":count}),flush=True)
        return result
    except Exception as exc:
        result={**row,"status":"failed_preserved","actual_bytes_written":p.stat().st_size if p.exists() else 0,
                "started_utc":started,"error_type":type(exc).__name__,"error":str(exc),
                "partial_bytes_are_not_admitted":True}
        write_new(scope/"receipts"/(row["collection"]+"__"+row["name"]+".failed.json"),result)
        print(json.dumps({"file":row["name"],"status":"failed","error_type":type(exc).__name__}),flush=True)
        return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protocol",required=True);ap.add_argument("--protocol-sha256",required=True)
    ap.add_argument("--source-sha256",required=True);ap.add_argument("--ledger-sha256",required=True)
    args=ap.parse_args()
    if Path.cwd().resolve()!=ROOT:raise RuntimeError("Wrong project root")
    if sha(Path(__file__).resolve())!=args.source_sha256:raise ValueError("Source drift")
    protocol_path=local(args.protocol)
    if sha(protocol_path)!=args.protocol_sha256:raise ValueError("Protocol drift")
    spec=json.loads(protocol_path.read_bytes())
    for r in spec["input_refs"]:
        if sha(local(r["path"]))!=r["sha256"]:raise ValueError("Input metadata drift")
    ledger_path=local(".runtime/gpup/storage_reservations.json")
    if sha(ledger_path)!=args.ledger_sha256:raise ValueError("Fresh storage admission drift")
    ledger=json.loads(ledger_path.read_bytes())
    slot=ledger["jobs"][spec["ledger_job_key"]]
    scope=local(spec["output"])
    if slot["output"]!=str(scope) or slot["reserved_bytes"]!=CAP:raise ValueError("Wrong reservation")
    projected=ledger["baseline_bytes"]+ledger["external_reserve_bytes"]+sum(v["reserved_bytes"] for v in ledger["jobs"].values())+sum(v["reserved_bytes"] for v in ledger["controllers"].values())
    if projected>=450_000_000_000:raise ValueError("Storage warning threshold")
    files=spec["files"]
    if len(files)!=41 or sum(r["bytes"] for r in files)>=CAP:raise ValueError("Missing/full over-budget collection")
    seen=set()
    for r in files:
        if r["collection"] not in {"wohnsgenealogy5512994","gst_native_snapshots"}:raise ValueError("Collection")
        if Path(r["name"]).name!=r["name"] or r["name"] in {"",".",".."}:raise ValueError("Filename")
        if (r["collection"],r["name"]) in seen:raise ValueError("Duplicate object")
        seen.add((r["collection"],r["name"]))
        u=urlsplit(r["url"])
        if u.scheme!="https" or u.hostname not in {"zenodo.org","raw.githubusercontent.com"}:raise ValueError("Unpinned acquisition host")
        if not (r.get("md5") or r.get("git_blob_sha1")):raise ValueError("Missing upstream identity")
    scope.mkdir(exist_ok=False)
    for n in ["receipts","wohnsgenealogy5512994","gst_native_snapshots"]:(scope/n).mkdir()
    write_new(scope/"started.json",{"kind":"paper_native_raw_acquisition_v1","protocol_sha256":args.protocol_sha256,"source_sha256":args.source_sha256,"ledger_sha256":args.ledger_sha256,"projected_bytes_at_admission":projected,"pid":os.getpid(),"files":41,"decoding_or_scoring":False})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda row:download(row,scope),files))
    successful=all(r["status"]=="verified" for r in rows)
    result={"kind":"paper_native_raw_acquisition_v1","status":"all41_objects_verified" if successful else "partial_failure_preserved",
            "utc":dt.datetime.now(dt.timezone.utc).isoformat(),"protocol_sha256":args.protocol_sha256,"source_sha256":args.source_sha256,
            "raw_collection_bytes":sum(r.get("actual_bytes",r.get("actual_bytes_written",0)) for r in rows),"files":rows,
            "original_processed_paper_split_identity_proven":False,"original_ARGformer_subset_identity_proven":False,
            "GST_original_author_mutable_snapshot_proven":False,"data_decoded_or_extracted":False,"model_executed":False,"TEST_decoded_or_scored":False}
    write_new(scope/"completion.json",result)
    summary_dir=local(spec["summary_directory"])
    write_new(summary_dir/"completion.json",result)
    print(json.dumps({"status":result["status"],"verified":sum(r["status"]=="verified" for r in rows),"bytes":result["raw_collection_bytes"]}),flush=True)
    return 0 if successful else 1

if __name__=="__main__":
    raise SystemExit(main())
