"""Select obsolete bulk payloads, preserve every other file, write a review plan."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/storage_cleanup_20261027'
assert ROOT == Path.cwd() and ROOT.resolve() == ROOT
scopes = [
    'data/biological_candidate_pilot_v1',
    'data/expanded_full_panel_v1',
    'data/expanded_native_argmix_prepared_v3',
    'data/ARGMix_prospective1611_Relate_accepted_train000_v4',
] + sorted(str(p.relative_to(ROOT)) for p in (ROOT/'results').iterdir() if p.is_dir() and not p.is_symlink())
allowed = {'.pt', '.npz', '.npy', '.trees', '.vcf', '.haps', '.hap', '.anc', '.mut', '.bin', '.annot', '.rpos', '.dist', '.bp', '.r', '.state'}
tracked = set(subprocess.check_output(['git','ls-files','-z'], cwd=ROOT).decode().split('\0'))
rows = [json.loads(line) for line in (OUT/'large_files.jsonl').open()]
selected = []
for row in rows:
    path = row['path']
    if not any(path.startswith(scope+'/') for scope in scopes): continue
    if Path(path).suffix not in allowed and not path.endswith(('.anc.gz','.mut.gz')): continue
    if row['nlink'] != 1 or path in tracked: continue
    row['reason'] = 'Superseded pilot or pre-rebuild data, no current benchmark dependency found. Source/configuration/metrics retained.'
    selected.append(row)
selected.sort(key=lambda r:r['path'])
selected_names = {r['path'] for r in selected}
retained = []
for scope in scopes:
    for directory, dirs, files in os.walk(ROOT/scope, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            path=Path(directory)/name
            relative=str(path.relative_to(ROOT))
            info=path.lstat()
            if relative not in selected_names and stat.S_ISREG(info.st_mode):
                retained.append({'path':relative,'identity':[info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns]})
plan={'root':str(ROOT), 'schema':'explicit_cleanup_v1', 'approved_scopes':scopes,
      'protected_scopes':['.git','src','research','paper','checkpoints','configs','scripts','tests','experiments'],
      'authorization':'User requested removing unnecessary data only inside this repository while preserving designs and metrics.',
      'files':selected, 'retained_files':retained}
payload=(json.dumps(plan,indent=2)+'\n').encode()
path=OUT/'phase1.json'
with path.open('xb') as f: f.write(payload)
print('MANIFEST',str(path.relative_to(ROOT)),'SHA256',hashlib.sha256(payload).hexdigest())
print('FILES',len(selected),'ALLOCATED',sum(r['allocated'] for r in selected),'RETAINED',len(retained))
for scope in scopes:
    subset=[r for r in selected if r['path'].startswith(scope+'/')]
    print(scope,len(subset),sum(r['allocated'] for r in subset))
