"""Bounded fixed90 progress across immutable reviewed continuation generations."""
from pathlib import Path
from .fit_gate_v2 import need,reference,read,contained
from .progress_v1 import summarize


def snapshot(output,checkpoint_cache):
    origin={};rows=[];checkpoints={}
    if (output/'fit_resume_origin.json').is_file():origin,_=read(output/'fit_resume_origin.json')
    inherited=origin.get('update_event_refs',[])
    need(len(inherited)<=90,'Bounded inherited update history')
    for ref in inherited:
        value,_=read(ref['path'],ref['sha256']);rows.append(value)
    names=sorted((output/'updates').glob('update_*.json')) if (output/'updates').exists() else []
    need(len(rows)+len(names)<=90,'Whole continuation90-update ceiling')
    for index,path in enumerate(names,len(rows)+1):
        need(path.name=='update_%03d.json'%index and path.stat().st_size<=64*2**10,'Exact next update record after restore')
        value,_=read(path);rows.append(value)
    def immutable(path,key):
        path=contained(path);info=path.stat()
        need(not path.is_symlink() and info.st_size<=8*2**20,'Bounded exact checkpoint')
        identity=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
        if key in checkpoint_cache:
            old,ref=checkpoint_cache[key];need(identity==old,'Immutable checkpoint changed')
        else:ref=reference(path);checkpoint_cache[key]=(identity,ref)
        return ref
    for epoch,ref in origin.get('checkpoint_event_refs',{}).items():
        need(immutable(ref['path'],('epoch',int(epoch)))==ref,'Exact inherited epoch checkpoint')
        checkpoints[int(epoch)]=ref
    for epoch in (5,10,15,20,25,30):
        path=output/'checkpoints'/('epoch_%02d.pt'%epoch)
        if path.exists():
            need(epoch not in checkpoints,'Epoch checkpoint cannot be silently replaced across runs')
            checkpoints[epoch]=immutable(path,('epoch',epoch))
    result=summarize(rows,checkpoints)
    refs=sorted((output/'resume').glob('update_*.pt')) if (output/'resume').exists() else []
    need(len(refs)<=90,'Bounded immutable per-update recovery artifacts')
    latest=origin.get('resume_checkpoint_ref')
    if refs:
        path=refs[-1];number=int(path.stem.split('_')[1])
        need(path.name=='update_%03d.pt'%number and 1<=number<=len(rows),'Recovery checkpoint follows completed update')
        latest=immutable(path,('resume',number))
    result.update(fresh_fit_only=False,resume_supported=True,checkpoint_contains_RNG=True,
        resume_checkpoint_contains_RNG=True,epoch_checkpoint_contains_RNG=False,
        latest_resume_checkpoint_ref=latest,resumed_from_ref=origin.get('resume_checkpoint_ref'),
        resume_requires_original_family_reaped=True,progress_is_completion_evidence_not_resume_authority=True)
    return result
