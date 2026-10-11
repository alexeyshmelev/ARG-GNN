"""Finite handmade ZIP byte/CRC/SHA/size and partial-write checks, stdlib only."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import uuid
import zipfile
from arggnn import gst_uspto_payload_copy_v2 as p
from arggnn import gst_uspto_archive_inventory_v2 as inv

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE / ('handmade-' + uuid.uuid4().hex)
CASE_ROOT.mkdir()
checks = []; cases = []
# Opening the actual archive in author fixtures is a hard error, including an
# indirect dependency call. Metadata inventory JSON remains explicitly allowed.
import sys
actual_raw = str(p.ROOT / inv.RAW)
forbidden_attempts = []
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path == actual_raw or path == str(inv.RAW):
            forbidden_attempts.append(path)
            raise AssertionError('Actual GST ZIP access forbidden in handmade suite')
sys.addaudithook(audit)


def check(name, value):
    checks.append({'check': name, 'passed': bool(value)})
    if not value: raise AssertionError(name)


def reject(name, fn):
    try: fn()
    except (ValueError, OSError, zipfile.BadZipFile): check(name, True)
    else: check(name, False)


def make_zip(method=zipfile.ZIP_DEFLATED, payload=None, ancillary='data/connor_original.tsv', extra=None):
    if payload is None: payload = bytes(range(256)) * 550 + b'\r\nline\n\x00last'
    contents = [('data/', b''), (ancillary, payload), ('data/test.txt', b'fake test\r\nkeep\n'), ('data/train.txt', b'fake train\n\nkeep\r\n'), ('data/valid.txt', b'fake valid\r\n')]
    if extra: contents.extend(extra)
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w') as archive:
        for name, raw in contents:
            item = zipfile.ZipInfo(name, date_time=(2020,1,2,3,4,6)); item.create_system=3
            item.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10 if name.endswith('/') else (stat.S_IFREG | 0o644) << 16
            item.compress_type = zipfile.ZIP_STORED if name.endswith('/') else method
            archive.writestr(item, raw)
    return target.getvalue(), contents


def run_case(name, raw, frozen=None, **kwargs):
    expected = inv.inventory(raw) if frozen is None else frozen
    destination = CASE_ROOT / name
    guards=[]
    result, receipt_sha = p.copy_payload_v2(raw, expected, destination, guard=lambda required: guards.append(required), **kwargs)
    manifest = json.loads((destination/'payload_manifest.json').read_text())
    check(name+' final receipt truthful and SHA-bound', manifest==result and receipt_sha==hashlib.sha256((destination/'payload_manifest.json').read_bytes()).hexdigest())
    check(name+' guarded fresh16MB scope', guards[0]==p.RESERVATION_BYTES and all(type(x) is int and 0<=x<=p.RESERVATION_BYTES for x in guards))
    cases.append({'name':name,'status':result['status'],'total_payload_bytes_written':result['total_payload_bytes_written'],'receipt_sha256':receipt_sha,'directory':str(destination.relative_to(p.ROOT))})
    return destination, result


source = p.source_report_v2()
check('default source report remains metadata-only', source['archive_opened'] is False and source['member_payload_opened'] is False and source['live_budget_or_lock_access'] is False)
check('actual inventory frozen payload cap', source['payload_bytes']==12710870 and source['reservation_bytes']==16000000)
for method, label in [(zipfile.ZIP_STORED,'stored'),(zipfile.ZIP_DEFLATED,'deflated')]:
    raw, contents=make_zip(method)
    destination,result=run_case(label,raw)
    check(label+' full member order and names preserved', [r['name'] for r in result['members']]==[name for name,_ in contents])
    check(label+' complete status without rows/model/scoring', result['status']=='complete_payload_bytes_verified_no_rows_parsed' and result['complete_payload_copy'] and not result['dataset_rows_parsed'] and not result['native_model_run'] and not result['scoring_performed'] and not result['dataset_accepted_for_parsing'])
    check(label+' exact member bytes including binary/newlines', all((destination/name).is_dir() if name.endswith('/') else (destination/name).read_bytes()==data for name,data in contents))
    check(label+' independently known content SHA and CRC verified', all(r['content_sha256']==hashlib.sha256(data).hexdigest() and r['CRC_payload_verified'] for r,(_,data) in zip(result['members'],contents)))
    check(label+' exact complete byte total',result['total_payload_bytes_written']==sum(len(data) for _,data in contents))
    check(label+' progress order equals archive order', [e['archive_order'] for e in map(json.loads,(destination/'progress.jsonl').read_text().splitlines()) if e['event']=='member_complete']==list(range(5)))
    reject(label+' existing output cannot overwrite', lambda: p.copy_payload_v2(raw,inv.inventory(raw),destination))
# Raw SHA and every frozen central member field/order are exact preflight pins.
raw,contents=make_zip()
frozen=inv.inventory(raw)
for label,mut in [
    ('raw_SHA_mismatch',lambda d:d.update(archive_sha256='0'*64)),
    ('archive_bytes_mismatch',lambda d:d.update(archive_bytes=len(raw)+1)),
    ('member_name_mismatch',lambda d:d['members'][1].update(name='data/other.tsv')),
    ('member_CRC_mismatch',lambda d:d['members'][1].update(CRC32='00000000')),
    ('member_size_mismatch',lambda d:d['members'][1].update(uncompressed_bytes=7)),
    ('member_order_mismatch',lambda d:d['members'].reverse()),
    ('member_missing',lambda d:d['members'].pop()),
]:
    changed=copy.deepcopy(frozen);mut(changed);destination=CASE_ROOT/label
    reject(label+' rejected before output/payload',lambda changed=changed,destination=destination:p.copy_payload_v2(raw,changed,destination))
    check(label+' no destination created',not destination.exists())
# CRC corruption with intact declared identity is tested under a handmade,
# newly inventoried raw SHA. Payload qualification must still reject it.
raw,contents=make_zip(zipfile.ZIP_STORED)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:member=archive.infolist()[1]
header=struct.unpack_from('<IHHHHHIIIHH',raw,member.header_offset)
start=member.header_offset+30+header[-2]+header[-1]
corrupt=bytearray(raw);corrupt[start+1]^=1
path,result=run_case('CRC_failure',bytes(corrupt))
check('CRC failure keeps exact named partial payload',result['status']=='failed_partial_preserved' and result['members'][1]['status']=='failed_partial' and not result['members'][1]['CRC_payload_verified'] and (path/contents[1][0]).exists())
check('CRC failure truthful complete prefix SHA',result['members'][1]['payload_prefix_sha256']==hashlib.sha256((path/contents[1][0]).read_bytes()).hexdigest() and result['members'][1]['payload_bytes_written']==len(contents[1][1]))
check('CRC failure retains no fabricated later hashes',all(r['status']=='unreached' and 'content_sha256' not in r for r in result['members'][2:]))
# Set both local+central declarations below/above the genuine deflate output.
# This counterexample distinguishes full-stream counting from ZipExtFile's
# declared-size truncation. No expected CRC is changed.
raw,contents=make_zip(zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:member=archive.infolist()[1]
central=raw.find(b'PK\x01\x02')
while True:
    name_length,extra_length,comment_length=struct.unpack_from('<HHH',raw,central+28)
    member_name=raw[central+46:central+46+name_length].decode('cp437')
    if member_name==member.filename:break
    central+=46+name_length+extra_length+comment_length
for label,size in [('size_overrun',100000),('size_deficit',len(contents[1][1])+1)]:
    changed=bytearray(raw);struct.pack_into('<I',changed,member.header_offset+22,size);struct.pack_into('<I',changed,central+24,size)
    path,result=run_case(label,bytes(changed))
    check(label+' independent full stream size rejection',result['status']=='failed_partial_preserved' and 'size' in result['error'].lower())
    actual=(path/contents[1][0]).read_bytes()
    check(label+' preserved prefix count/hash is exact',result['members'][1]['payload_bytes_written']==len(actual) and result['members'][1].get('payload_prefix_sha256')==hashlib.sha256(actual).hexdigest() and actual==contents[1][1][:len(actual)])
# Return an actual short OS write, then fail the next write. This checks bytes
# already written, not requested bytes or an invented recovered full member.
raw,contents=make_zip(zipfile.ZIP_STORED)
write_calls=[]
def short_then_fail(fd,data):
    if write_calls:raise OSError('handmade injected second payload write failure')
    wrote=os.write(fd,data[:5000]);write_calls.append(wrote);return wrote
path,result=run_case('partial_write',raw,payload_writer=short_then_fail)
check('short-write then failure preserves5000 real bytes',result['status']=='failed_partial_preserved' and result['total_payload_bytes_written']==5000 and (path/contents[1][0]).read_bytes()==contents[1][1][:5000])
check('partial extraction reported truthfully',result['data_extracted'] and result['partial_payload_preserved'] and not result['complete_payload_copy'])
check('partial prefix hash exact with later rows unreached',result['members'][1]['payload_prefix_sha256']==hashlib.sha256(contents[1][1][:5000]).hexdigest() and all(r['status']=='unreached' for r in result['members'][2:]))
# Unsafe member topology and receipt collisions stop before creation.
for label,kwargs in [('path_escape',{'ancillary':'../escape.tsv'}),('case_collision',{'extra':[('data/VALID.txt',b'other')]}),('receipt_collision',{'ancillary':'progress.jsonl'})]:
    raw,_=make_zip(**kwargs);destination=CASE_ROOT/label
    reject(label+' rejected before creation',lambda raw=raw,destination=destination:p.copy_payload_v2(raw,inv.inventory(raw),destination))
    check(label+' preserves no escaped output',not destination.exists())
# Stale authority at a boundary preserves earlier completed members and prefixes.
raw,contents=make_zip();destination=CASE_ROOT/'guard_failure'
guard_calls=[]
def stale_guard(required):
    guard_calls.append(required)
    if destination.exists() and (destination/'data/connor_original.tsv').exists():raise ValueError('handmade stale root authority')
result,receipt_sha=p.copy_payload_v2(raw,inv.inventory(raw),destination,guard=stale_guard)
check('boundary denial preserves exact already copied prefix', (destination/contents[1][0]).read_bytes()==contents[1][1][:result['total_payload_bytes_written']] and result['status']=='failed_partial_preserved' and result['authority_rejected'])
check('expired authority cannot write final receipt',receipt_sha is None and result['final_receipt_written'] is False and (destination/'progress.jsonl').is_file())
check('no actual archive access attempt',forbidden_attempts==[])
# Compact evidence contains hashes/status only. All fixture payloads are bounded.
fixture_bytes=sum(x.stat().st_size for x in CASE_ROOT.rglob('*') if x.is_file())
check('handmade disk evidence under10MB',fixture_bytes<10000000)
result={'kind':'gst_uspto_payload_copy_v2_handmade_checks','status':'passed','checks':checks,'check_count':len(checks),'cases':cases,'fixture_directory':str(CASE_ROOT.relative_to(p.ROOT)),'fixture_file_bytes':fixture_bytes,'fixture_file_cap_bytes':10000000,'forbidden_actual_archive_access_attempts':forbidden_attempts,'actual_archive_opened':False,'actual_member_payload_opened':False,'dataset_rows_parsed':False,'native_model_run':False,'GPU_used':False,'live_budget_operations':False}
(HERE/'handmade_checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'fixture_bytes':fixture_bytes,'actual_archive_opened':False},sort_keys=True))
