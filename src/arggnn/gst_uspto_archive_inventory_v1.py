"""Metadata-only ZIP inventory for the pinned native GST USPTO archive.

No member payload is opened, extracted or parsed. Default CLI is source-only.
"""
from __future__ import annotations
import argparse, datetime, hashlib, io, json, os, stat, unicodedata, zipfile
from pathlib import Path, PurePosixPath

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
RAW=Path("data/paper_native_raw_acquisition_v1/gst_native_snapshots/USPTO15K_data.zip")
RAW_SHA="66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723"
RAW_BYTES=3446572
LIMITS={"members":512,"member_bytes":256000000,"total_bytes":1000000000,"compression_ratio":1000}
LOGICAL=("train.txt","valid.txt","test.txt")
def sha(raw): return hashlib.sha256(raw).hexdigest()
def inventory(raw: bytes):
    if not isinstance(raw,bytes):raise TypeError("Exact archive bytes required")
    with zipfile.ZipFile(io.BytesIO(raw),"r") as archive:
        members=archive.infolist()
        if len(members)>LIMITS["members"]:raise ValueError("Member cap exceeded")
        rows=[];seen={};total=0;split={}
        for rank,item in enumerate(members):
            name=item.orig_filename
            if not name or any(ord(c)<32 or ord(c)==127 for c in name):raise ValueError("Empty/control member name")
            if "\\" in name or name.startswith("/") or ":" in name:raise ValueError("Unsafe member path")
            directory=item.is_dir()
            stripped=name[:-1] if directory and name.endswith("/") else name
            parts=stripped.split("/")
            if any(part in {"",".",".."} for part in parts):raise ValueError("Unsafe path component")
            key=unicodedata.normalize("NFC",stripped).casefold()
            if key in seen:raise ValueError("Duplicate/normalized member collision")
            seen[key]=(directory,name)
            mode=item.external_attr>>16
            kind=stat.S_IFMT(mode)
            if kind not in {0,stat.S_IFREG,stat.S_IFDIR}:raise ValueError("Special/link member forbidden")
            if kind==stat.S_IFDIR and not directory or kind==stat.S_IFREG and directory:raise ValueError("Member type/name disagreement")
            if item.flag_bits&1 or item.flag_bits&0x40:raise ValueError("Encrypted member")
            if item.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}:raise ValueError("Unsupported compression")
            if item.file_size<0 or item.compress_size<0:raise ValueError("Negative member size")
            if directory and item.file_size:raise ValueError("Directory payload forbidden")
            if item.file_size>LIMITS["member_bytes"]:raise ValueError("Per-member cap exceeded")
            if item.file_size and item.file_size/max(1,item.compress_size)>LIMITS["compression_ratio"]:raise ValueError("Compression ratio cap exceeded")
            total+=item.file_size
            if total>LIMITS["total_bytes"]:raise ValueError("Total member cap exceeded")
            basename=parts[-1]
            if not directory and basename in LOGICAL:
                if basename in split:raise ValueError("Ambiguous logical split basename")
                split[basename]=name
            rows.append({"archive_order":rank,"name":name,"directory":directory,"create_system":item.create_system,
                "create_version":item.create_version,"extract_version":item.extract_version,"flags":item.flag_bits,
                "compression":item.compress_type,"compressed_bytes":item.compress_size,"uncompressed_bytes":item.file_size,
                "CRC32":f"{item.CRC:08x}","header_offset":item.header_offset,"external_attr":item.external_attr,
                "filename_encoding":"utf8" if item.flag_bits&0x800 else "cp437",
                "extra_bytes":len(item.extra),"comment_bytes":len(item.comment)})
        for directory,name in seen.values():
            parts=unicodedata.normalize("NFC",name).casefold().split("/")
            for end in range(1,len(parts)):
                prior=seen.get("/".join(parts[:end]))
                if prior and not prior[0]:raise ValueError("File/directory prefix collision")
        if set(split)!=set(LOGICAL):raise ValueError("Exactly three logical source split files required")
        return {"schema":"gst_uspto_archive_central_directory_inventory_v1","status":"metadata_only_complete",
            "archive_bytes":len(raw),"archive_sha256":sha(raw),"members":rows,"member_count":len(rows),
            "declared_total_uncompressed_bytes":total,"logical_split_member_paths":split,
            "archive_comment_bytes":len(archive.comment),"member_payload_opened":False,"data_extracted":False,
            "dataset_rows_parsed":False,"member_content_SHA256_known":False,"CRC_payload_verified":False,
            "native_model_run":False,"original_author_used_snapshot_proven":False}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true");ap.add_argument("--admission");ap.add_argument("--expected-admission-sha256")
    args=ap.parse_args()
    if not args.execute:
        print(json.dumps({"status":"source_only","raw":str(RAW),"raw_sha256":RAW_SHA,"limits":LIMITS,"archive_opened":False}))
        return
    if Path.cwd().resolve()!=ROOT or not args.admission or not args.expected_admission_sha256:raise ValueError("Root explicit bound metadata admission required")
    apath=ROOT/args.admission
    apath.resolve().relative_to(ROOT)
    if apath.is_symlink():raise ValueError("Admission symlink forbidden")
    ab=apath.read_bytes()
    if sha(ab)!=args.expected_admission_sha256:raise ValueError("Admission changed")
    authority=json.loads(ab)
    source=Path(__file__).resolve()
    required={"kind":"gst_uspto_archive_inventory_v1_root_admission","authorizer":"root_agent","active":True,
              "metadata_only":True,"payload_read_authorized":False,"native_model_authorized":False,
              "source_sha256":sha(source.read_bytes()),"raw":{"path":str(RAW),"sha256":RAW_SHA,"bytes":RAW_BYTES},
              "limits":LIMITS}
    if any(authority.get(k)!=v for k,v in required.items()):raise ValueError("Exact root metadata-only scope required")
    review=ROOT/authority["independent_review"]["path"]
    review.resolve().relative_to(ROOT)
    rb=review.read_bytes()
    if sha(rb)!=authority["independent_review"]["sha256"]:raise ValueError("Review changed")
    reviewed=json.loads(rb)
    if reviewed.get("verdict")!="PASS" or reviewed.get("source_sha256")!=required["source_sha256"]:raise ValueError("Exact independent source PASS required")
    out=ROOT/authority["output_directory"]
    out.resolve().relative_to(ROOT)
    if out.exists():raise ValueError("Fresh output required")
    rawpath=ROOT/RAW
    if rawpath.is_symlink() or rawpath.stat().st_size!=RAW_BYTES:raise ValueError("Raw size/type changed")
    rawpath.resolve().relative_to(ROOT)
    raw=rawpath.read_bytes()
    if len(raw)!=RAW_BYTES or sha(raw)!=RAW_SHA:raise ValueError("Raw opaque identity changed")
    # Root performs the legacy persistent reservation before launching this small
    # reader. Bind/check the exact owned slot at the first pre-payload boundary.
    import fcntl
    lock=ROOT/".runtime/locks/scheduler.lock"
    fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
    with os.fdopen(fd,"r+") as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        ledger=ROOT/".runtime/gpup/storage_reservations.json"
        lb=ledger.read_bytes()
        if sha(lb)!=authority["storage"]["ledger_sha256"]:raise ValueError("Admission ledger changed")
        state=json.loads(lb)
        charge=state["jobs"].get(authority["storage"]["reservation_key"])
        if charge!={"reserved_bytes":20000000,"output":str(out)}:raise ValueError("Own new20MB reservation required")
        projection=state["baseline_bytes"]+state["external_reserve_bytes"]+sum(v["reserved_bytes"] for cat in ("controllers","jobs") for v in state[cat].values())
        if projection>=state["limit_bytes"] or state["limit_bytes"]!=450000000000:raise ValueError("Legacy budget exceeded")
    out.mkdir(parents=True)
    try:
        result=inventory(raw)
    except Exception as error:
        result={"schema":"gst_uspto_archive_central_directory_inventory_v1","status":"failed_preserved",
                "error_type":type(error).__name__,"error":str(error),"member_payload_opened":False}
    result.update(admission_sha256=sha(ab),source_sha256=required["source_sha256"],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    payload=(json.dumps(result,indent=2,sort_keys=True)+"\n").encode()
    if len(payload)>1000000:raise ValueError("Metadata receipt cap exceeded")
    with (out/"inventory.json").open("xb") as f:f.write(payload)
    print(json.dumps({"status":result["status"],"receipt":str((out/"inventory.json").relative_to(ROOT)),"sha256":sha(payload)}))
    if result["status"]!="metadata_only_complete":raise SystemExit(1)
if __name__=="__main__":main()
