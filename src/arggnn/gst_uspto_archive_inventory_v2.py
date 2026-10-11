"""Inactive, root-admitted metadata-only inventory of the pinned GST USPTO ZIP.
No member payload API, extraction, dataset parser or model is executed here.
"""
from __future__ import annotations
import argparse,datetime,hashlib,io,json,os,stat,struct,unicodedata,zipfile,zlib
from pathlib import Path
from . import project_budget_v4 as budget
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE=ROOT/'src/arggnn/gst_uspto_archive_inventory_v2.py'
CONFIG=ROOT/'configs/gst_uspto_archive_inventory_v2.json'
CONFIG_SHA256='6714bef9d7964a89ddee337ba17b6ca4f01a74ef35686f582bcf1e268446c03f'
RAW=Path('data/paper_native_raw_acquisition_v1/gst_native_snapshots/USPTO15K_data.zip')
RAW_SHA='66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723'
RAW_BYTES=3446572
LIMITS={'members':512,'member_bytes':256000000,'total_bytes':1000000000,'compression_ratio':1000}
LOGICAL=('train.txt','valid.txt','test.txt')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def _extra_metadata(item,name):
    data=item.extra;offset=0;fields=[]
    while offset<len(data):
        if len(data)-offset<4:raise ValueError('Truncated central extra-field header')
        identity,length=struct.unpack_from('<HH',data,offset);offset+=4
        if identity in fields:raise ValueError('Duplicate central extra-field identity')
        if length>len(data)-offset:raise ValueError('Truncated central extra-field body')
        body=data[offset:offset+length];offset+=length;fields.append(identity)
        if identity==0x000d:
            if len(body)!=12:raise ValueError('PKWARE Unix link/device or malformed extra metadata')
        elif identity==0x756e:
            if len(body)!=14:raise ValueError('ASi Unix link or malformed extra metadata')
            checksum,mode=struct.unpack_from('<IH',body)
            if zlib.crc32(body[4:])&0xffffffff!=checksum:raise ValueError('ASi Unix metadata CRC differs')
            kind=stat.S_IFMT(mode)
            if kind not in (0,stat.S_IFREG,stat.S_IFDIR):raise ValueError('ASi Unix special/link type forbidden')
            if kind==stat.S_IFDIR and not item.is_dir() or kind==stat.S_IFREG and item.is_dir():raise ValueError('ASi Unix directory/name disagreement')
        elif identity==0x7075:
            if len(body)<5 or body[0]!=1:raise ValueError('Unsupported Unicode path extra metadata')
            original=name.encode('utf8' if item.flag_bits&0x800 else 'cp437')
            if struct.unpack_from('<I',body,1)[0]!=zlib.crc32(original)&0xffffffff:raise ValueError('Unicode path raw-name CRC differs')
            try:alternative=body[5:].decode('utf8',errors='strict')
            except UnicodeError as error:raise ValueError('Invalid Unicode path metadata') from error
            if alternative!=name:raise ValueError('Conflicting alternate decoded Unicode member path')
    return fields

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
            dos_directory=bool(item.external_attr&0x10)
            if dos_directory and not directory or item.create_system==0 and dos_directory!=directory:raise ValueError('DOS directory/name disagreement')
            extra_fields=_extra_metadata(item,name)
            mode=item.external_attr>>16
            kind=stat.S_IFMT(mode)
            if kind not in {0,stat.S_IFREG,stat.S_IFDIR}:raise ValueError("Special/link member forbidden")
            if kind==stat.S_IFDIR and not directory or kind==stat.S_IFREG and directory:raise ValueError("Member type/name disagreement")
            if item.flag_bits&(1|0x40|0x2000):raise ValueError("Encrypted member")
            if item.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}:raise ValueError("Unsupported compression")
            if item.file_size<0 or item.compress_size<0:raise ValueError("Negative member size")
            if directory and item.file_size:raise ValueError("Directory payload forbidden")
            if item.file_size>LIMITS["member_bytes"]:raise ValueError("Per-member cap exceeded")
            if item.file_size and item.file_size/max(1,item.compress_size)>LIMITS["compression_ratio"]:raise ValueError("Compression ratio cap exceeded")
            total+=item.file_size
            if total>LIMITS["total_bytes"]:raise ValueError("Total member cap exceeded")
            basename=parts[-1]
            logical=unicodedata.normalize('NFC',basename).casefold()
            if not directory and logical in LOGICAL:
                if basename!=logical:raise ValueError('Exact native logical split spelling required')
                if basename in split:raise ValueError("Ambiguous logical split basename")
                split[basename]=name
            rows.append({"archive_order":rank,"name":name,"directory":directory,"create_system":item.create_system,
                "create_version":item.create_version,"extract_version":item.extract_version,"flags":item.flag_bits,
                "compression":item.compress_type,"compressed_bytes":item.compress_size,"uncompressed_bytes":item.file_size,
                "CRC32":f"{item.CRC:08x}","header_offset":item.header_offset,"external_attr":item.external_attr,
                "filename_encoding":"utf8" if item.flag_bits&0x800 else "cp437",
                "extra_bytes":len(item.extra),"extra_field_ids":extra_fields,"comment_bytes":len(item.comment)})
        for directory,name in seen.values():
            parts=unicodedata.normalize("NFC",name).casefold().split("/")
            for end in range(1,len(parts)):
                prior=seen.get("/".join(parts[:end]))
                if prior and not prior[0]:raise ValueError("File/directory prefix collision")
        if set(split)!=set(LOGICAL):raise ValueError("Exactly three logical source split files required")
        return {"schema":"gst_uspto_archive_central_directory_inventory_v2","status":"metadata_only_complete",
            "archive_bytes":len(raw),"archive_sha256":sha(raw),"members":rows,"member_count":len(rows),
            "declared_total_uncompressed_bytes":total,"logical_split_member_paths":split,
            "archive_comment_bytes":len(archive.comment),"member_payload_opened":False,"data_extracted":False,
            "dataset_rows_parsed":False,"member_content_SHA256_known":False,"CRC_payload_verified":False,
            "native_model_run":False,"original_author_used_snapshot_proven":False}

def _same(value,expected,label):
    if budget.encoded(value)!=budget.encoded(expected):raise ValueError('Exact typed '+label+' required')
def _when(value):
    if not isinstance(value,str):raise ValueError('Typed UTC authority date required')
    result=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware authority date required')
    return result.astimezone(datetime.timezone.utc)
def _active(value):
    first=_when(value['created_utc']);last=_when(value['expires_utc']);now=datetime.datetime.now(datetime.timezone.utc)
    if not first<=now<last or (last-first).total_seconds()>86400:raise ValueError('Fresh bounded root metadata authority required')
def policy():
    cfg=budget.bound(ROOT,{'path':str(CONFIG.relative_to(ROOT)),'sha256':CONFIG_SHA256},True)
    _same(cfg['raw'],{'path':str(RAW),'sha256':RAW_SHA,'bytes':RAW_BYTES},'opaque raw identity')
    _same(cfg['limits'],LIMITS,'unchanged metadata limits')
    return cfg

def admit(record,*,api=budget):
    cfg=policy();source_ref=budget.reference(ROOT,SOURCE);config_ref={'path':str(CONFIG.relative_to(ROOT)),'sha256':CONFIG_SHA256}
    value=budget.bound(ROOT,record,True)
    fixed={'kind':'gst_uspto_archive_inventory_v2_root_admission','authorizer':'root_agent','active':True,'metadata_only':True,
      'payload_read_authorized':False,'native_model_authorized':False,'implementation':source_ref,'config':config_ref,'raw':cfg['raw'],'limits':LIMITS}
    if not isinstance(value,dict) or set(value)!=set(fixed)|{'created_utc','expires_utc','registration','independent_review','output_directory','reservation','initial_ledger_pins'}:raise ValueError('Exact metadata admission fields required')
    for k,v in fixed.items():_same(value[k],v,'metadata authority '+k)
    _active(value)
    for k in ('budget_source','budget_config','native_protocol'):budget.bound(ROOT,cfg[k])
    review=budget.bound(ROOT,value['independent_review'],True)
    required={'kind':'gst_uspto_archive_inventory_v2_independent_review','review_role':'independent_reviewer','verdict':'PASS',
      'implementation':source_ref,'config':config_ref,'source_handmade_metadata_verified':True,'actual_archive_opened':False,'actual_member_payload_opened':False,'fixture_mode':False,'actual_admission':False}
    for k,v in required.items():_same(review.get(k),v,'independent metadata review '+k)
    if not isinstance(review.get('tested_scope'),list) or not review['tested_scope'] or any(not isinstance(x,str) or not x for x in review['tested_scope']):raise ValueError('Independent tested metadata scope required')
    registration=budget.bound(ROOT,value['registration'],True)
    required={'kind':'gst_uspto_archive_inventory_v2_root_registration','authorizer':'root_agent','hypothesis_id':'H022',
      'archive_metadata_run_registered':True,'implementation':source_ref,'config':config_ref,'raw':cfg['raw'],'payload_read_authorized':False,'native_model_authorized':False}
    for k,v in required.items():_same(registration.get(k),v,'registered archive metadata run '+k)
    if not isinstance(registration.get('experiment_id'),str) or not registration['experiment_id']:raise ValueError('Root registered metadata experiment ID required')
    output=budget.canonical(ROOT,value['output_directory'])
    if os.path.lexists(output):raise ValueError('Exclusive fresh metadata output required')
    reservation=budget.bound(ROOT,value['reservation'],True)
    _same(reservation['output'],str(output),'owned output reservation')
    _same(reservation['reserved_bytes'],cfg['reservation_bytes'],'20MB metadata reservation')
    _same(reservation['auxiliary_scopes'],[],'no additional write scope')
    for ref in (source_ref,config_ref,value['independent_review']):
        if not any(budget.encoded(ref)==budget.encoded(r) for r in reservation['consumer_bindings']):raise ValueError('Reservation omits bound metadata consumer')
    pins=value['initial_ledger_pins']
    if not isinstance(pins,dict) or set(pins)!={'legacy_ledger_sha256','supplemental_ledger_sha256'} or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in pins.values()):raise ValueError('Exact first-read ledger SHA256 pins required')
    # The reviewed v4 code validates typed invariant-preserving legacy and complete supplemental charges under its canonical mutex.
    view=api.check_reservation(ROOT,reservation,expected_source_sha256=cfg['budget_source']['sha256'],expected_config_sha256=cfg['budget_config']['sha256'],initial_ledger_pins=pins,boundary=True,required_write_bytes=cfg['receipt_max_bytes']+8192)
    return value,cfg,reservation,output,view

def _opaque_bytes():
    p=budget.canonical(ROOT,RAW);before=p.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size!=RAW_BYTES:raise ValueError('Pinned opaque ZIP size/type changed')
    fields=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        if fields(os.fstat(stream.fileno()))!=fields(before):raise ValueError('Raw identity changed before opaque read')
        raw=stream.read(RAW_BYTES+1);after=os.fstat(stream.fileno())
    if fields(before)!=fields(after) or fields(after)!=fields(p.lstat()) or len(raw)!=RAW_BYTES or sha(raw)!=RAW_SHA:raise ValueError('Pinned opaque archive bytes changed')
    return raw

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--admission');ap.add_argument('--expected-admission-sha256')
    args=ap.parse_args(argv)
    if not args.execute:
        print(json.dumps({'kind':'gst_uspto_archive_inventory_v2_source_report','status':'source_only','implementation':budget.reference(ROOT,SOURCE),'config':budget.reference(ROOT,CONFIG),'raw':policy()['raw'],'limits':LIMITS,'archive_opened':False,'payload_opened':False,'live_budget_or_lock_access':False}));return 0
    if Path.cwd().resolve()!=ROOT or not args.admission or not args.expected_admission_sha256:raise ValueError('Root explicit bound metadata admission required')
    record={'path':args.admission,'sha256':args.expected_admission_sha256};admission,cfg,reservation,output,view=admit(record)
    # Registration/review/storage have all passed before any opaque raw ZIP read.
    output.mkdir(parents=True,exist_ok=False)
    try:
        result=inventory(_opaque_bytes())
    except Exception as error:
        result={'schema':'gst_uspto_archive_central_directory_inventory_v2','status':'failed_preserved','error_type':type(error).__name__,'error':str(error),'member_payload_opened':False,'data_extracted':False,'dataset_rows_parsed':False}
    result.update(admission_ref=record,source_sha256=sha(SOURCE.read_bytes()),config_ref=admission['config'],registration_ref=admission['registration'],storage_admission=view,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    payload=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(payload)>cfg['receipt_max_bytes']:raise ValueError('Metadata receipt cap exceeded')
    budget.check_reservation(ROOT,reservation,expected_source_sha256=cfg['budget_source']['sha256'],expected_config_sha256=cfg['budget_config']['sha256'],boundary=True,required_write_bytes=len(payload)+8192)
    budget.bound(ROOT,record)
    _active(admission)
    with (output/'inventory.json').open('xb') as f:f.write(payload)
    print(json.dumps({'status':result['status'],'receipt':str((output/'inventory.json').relative_to(ROOT)),'sha256':sha(payload)}))
    return 0 if result['status']=='metadata_only_complete' else 1
if __name__=='__main__':raise SystemExit(main())
