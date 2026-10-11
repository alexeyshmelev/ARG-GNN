"""One bounded synthetic test of scanner-FD lifetime during direct iteration."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib, json, sys, tempfile
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import project_budget_v9 as b
class Entries:
    def __init__(self,names): self.names=iter(names); self.live=True
    def __enter__(self): return self
    def __exit__(self,*args): self.live=False
    def __iter__(self): return self
    def __next__(self):
        try: return SimpleNamespace(name=next(self.names))
        except StopIteration: self.live=False; raise
with tempfile.TemporaryDirectory(prefix='observer-fd-',dir=Path(__file__).parent) as tmp:
    proc=Path(tmp).resolve(); p=proc/'101';p.mkdir()
    fields=['S']+['0']*18+['200']
    (p/'stat').write_text('101 (python) '+' '.join(fields))
    (p/'maps').write_text('')
    scanner=Entries(['5','6']);observed=[]
    def scandir(path):
        path=Path(path)
        if path==proc: return Entries(['101'])
        if path==p/'fd': return scanner
        raise AssertionError('unexpected scan')
    def readlink(path):
        path=Path(path)
        if path==p/'cwd': return '/outside'
        if path.parent==p/'fd':
            if path.name=='5' and not scanner.live:
                raise FileNotFoundError('synthetic scanner fd closes on exhaustion')
            observed.append((path.name,scanner.live))
            return '/outside'
        raise AssertionError('unexpected readlink')
    with patch.object(b.os,'scandir',side_effect=scandir),patch.object(b.os,'readlink',side_effect=readlink):
        result=b.no_live_writers([proc/'closed'],[901],proc_root=proc)
    assert observed==[('5',True),('6',True)]
    assert scanner.live is False and result['recorded_pids_absent'] is True
print(json.dumps({'kind':'project_budget_v9_author_observer_fd_validation',
 'case':'observer scanner descriptor inspected while iterator remains live',
 'result':'PASS','observed_descriptor_reads':observed,
 'actual_proc_inspected':False,'actual_process_inventory_performed':False,
 'actual_apply_performed':False,'inherited_matrix_rerun':False},indent=2,sort_keys=True))
