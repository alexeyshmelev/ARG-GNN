"""Freeze reviewed cleanup manifests and explicit ssh-agent exclusions."""
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/storage_cleanup_20261027'
assert Path.cwd()==ROOT and ROOT.resolve()==ROOT
exceptions=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        if proc.stat().st_uid!=os.getuid():continue
        raw=(proc/'cmdline').read_bytes()
        argv=[x.decode() for x in raw.rstrip(b'\0').split(b'\0')]
        if not argv or Path(argv[0]).name!='ssh-agent':continue
        start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        exceptions.append({'pid':int(proc.name),'start_ticks':start,'argv':argv,'cmdline_sha256':hashlib.sha256(raw).hexdigest()})
    except (OSError,UnicodeDecodeError):continue

def save(name,plan):
    plan['live_process_exceptions']=exceptions
    payload=(json.dumps(plan,indent=2)+'\n').encode()
    with (OUT/name).open('xb') as f:f.write(payload)
    print(name,hashlib.sha256(payload).hexdigest(),'files',len(plan['files']),'allocated',sum(r['allocated'] for r in plan['files']))

plan=json.loads((OUT/'phase1.json').read_text())
keep=set(json.loads((OUT/'fidelity_selected_evidence_v1.json').read_text())['keep_paths'])
retained=[]
for row in plan['files']:
    if row['path'] in keep:
        plan['retained_files'].append({'path':row['path'],'identity':[row['dev'],row['ino'],row['size'],row['mtime_ns']]})
    else:retained.append(row)
plan['files']=retained
save('phase1_reviewed.json',plan)

archive=json.loads((OUT/'execution_installer_archives_v1.json').read_text())
files=[]
for r in archive['archives']:
    if r['nlink']!=1:continue
    files.append({'path':r['path'],'size':r['size_bytes'],'allocated':r['allocated_bytes'],'dev':r['device'],'ino':r['inode'],'mtime_ns':r['mtime_ns'],'nlink':1,'reason':'Downloaded installer archive. Extracted packages and installed environments retained.'})
large=[json.loads(line) for line in (OUT/'large_files.jsonl').open()]
c4='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_multigpu/other/C4__ancestralpaths_european_train_000__release_archive/native/train.pt'
for r in large:
    if r['path']==c4:
        r['reason']='Superseded 952-person C4 ARGMix archive. Current benchmark uses the separate 1611-person campaign. All archive metadata retained.'
        files.append(r)
    elif r['path'].startswith('.runtime/cache/pip/') and r['path'].endswith('.body') and r['nlink']==1:
        r['reason']='Downloaded pip response cache. Installed environments retained.'
        files.append(r)
scopes=['.runtime/conda/pkgs','.runtime/gst_uspto_native_v1/conda_pkgs','.runtime/cache/pip',str(Path(c4).parent)]
for r in files:
    assert any(r['path'].startswith(scope+'/') for scope in scopes)
retained=[]
for p in (ROOT/Path(c4).parent).iterdir():
    if p.is_file() and not p.is_symlink() and str(p.relative_to(ROOT))!=c4:
        info=p.stat();retained.append({'path':str(p.relative_to(ROOT)),'identity':[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]})
plan={'root':str(ROOT),'schema':'explicit_cleanup_v1','approved_scopes':scopes,
      'protected_scopes':['.git','src','research','paper','checkpoints','configs','scripts','tests','.runtime/env','.runtime/envs','.runtime/conda/envs'],
      'files':sorted(files,key=lambda r:r['path']),'retained_files':retained,
      'authorization':'User requested deletion of unnecessary project data. Exact C4 scientific archive and installer-cache files only.'}
save('phase2_reviewed.json',plan)
