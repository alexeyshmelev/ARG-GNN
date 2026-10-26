"""Identify only the qualification by its exact owned stdout file."""
import json
import os
import subprocess
from pathlib import Path
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
LOG=ROOT/'research/preprocessing_optimization_20261026/tree/full_process_parent.log'
result=subprocess.run(['pgrep','-f','^AI Auto Research Pipeline Test'],capture_output=True,text=True)
owned=[]
for token in result.stdout.split():
    pid=int(token)
    try:
        if os.readlink(f'/proc/{pid}/fd/1')==str(LOG):
            status=Path(f'/proc/{pid}/status').read_text().splitlines()
            fields={row.split(':',1)[0]:row.split(':',1)[1].strip() for row in status if ':' in row}
            owned.append(dict(pid=pid,ppid=int(fields['PPid']),rss=fields.get('VmRSS'),highwater=fields.get('VmHWM'),threads=fields.get('Threads'),state=fields.get('State')))
    except (FileNotFoundError,PermissionError):pass
report=dict(owned_stdout=str(LOG.relative_to(ROOT)),processes=owned)
(ROOT/'research/preprocessing_optimization_20261026/tree/full_process_startup.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
