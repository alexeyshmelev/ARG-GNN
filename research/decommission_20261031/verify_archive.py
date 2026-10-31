"""Back up remaining compact predictions and verify every archived member."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import tarfile

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/decommission_20261031'
summary=json.loads((OUT/'archive_summary.json').read_text())
rows=json.loads((OUT/'additional_artifacts.json').read_text())
archives=summary['archives']
manifest=gzip.open(OUT/'additional_manifest.jsonl.gz','wt',encoding='utf-8')
archive=None
part_files=0
part_bytes=0
part_path=None

def finish():
    global archive
    if archive is None:return
    archive.close()
    data=part_path.read_bytes()
    assert len(data)<95*1024*1024
    archives.append({'path':str(part_path.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'files':part_files})
    print('ADDITIONAL',part_path.name,len(data),flush=True)
    archive=None

for row in rows:
    name=row['path']; p=ROOT/name
    assert not p.is_symlink() and ROOT in p.resolve().parents
    data=p.read_bytes()
    if archive is None or part_bytes+len(data)>64*1024*1024:
        finish()
        part_path=OUT/f'evidence_{len(archives)+1:03d}.tar.gz'
        archive=tarfile.open(part_path,'w:gz',compresslevel=3)
        part_bytes=0; part_files=0
    info=tarfile.TarInfo(name); info.size=len(data); info.mode=0o644; info.mtime=int(p.stat().st_mtime)
    archive.addfile(info,io.BytesIO(data))
    manifest.write(json.dumps({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'archive':part_path.name})+'\n')
    part_bytes+=len(data); part_files+=1
finish(); manifest.close()
expected={}
for filename in ['archive_manifest.jsonl.gz','additional_manifest.jsonl.gz']:
    with gzip.open(OUT/filename,'rt') as f:
        for line in f:
            item=json.loads(line)
            assert item['path'] not in expected
            expected[item['path']]=item
verified=set()
for row in archives:
    p=ROOT/row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    with tarfile.open(p,'r:gz') as tar:
        for member in tar:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            item=expected[member.name]
            assert item['archive']==p.name and item['bytes']==member.size
            with tar.extractfile(member) as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(f.read()).hexdigest()
            assert digest==item['sha256'], member.name
            verified.add(member.name)
    print('VERIFIED',p.name,flush=True)
assert verified==set(expected)
summary.update(status='local_archive_integrity_verified_remote_push_pending',files=len(expected),uncompressed_bytes=sum(r['bytes'] for r in expected.values()),compressed_bytes=sum(r['bytes'] for r in archives),model_and_prediction_files=summary['model_and_prediction_files']+len(rows),verified_utc=datetime.now(timezone.utc).isoformat())
(OUT/'archive_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('VERIFIED_ALL',len(expected),'FILES',len(archives),'ARCHIVES',summary['compressed_bytes'],'BYTES',flush=True)
