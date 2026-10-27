"""Bounded read-only FIT telemetry; never load weights or claim resumability."""
import json
from .fit_gate_v1 import need,reference,read


def summarize(rows,checkpoints):
    numbers=[row.get('optimizer_updates') for row in rows]
    need(numbers==list(range(1,len(rows)+1)) and len(rows)<=90,'Contiguous durable complete-parent updates, bounded at90')
    for index,row in enumerate(rows,1):
        need(row.get('epoch')==(index-1)//3+1 and type(row.get('parent_id')) is str
             and row.get('one_complete_parent_optimizer_update') is True,'Durable exact update ordinal and complete-parent proof')
    epochs=sorted(checkpoints)
    need(all(epoch in (5,10,15,20,25,30) and epoch*3<=len(rows) for epoch in epochs),
         'Only scheduled checkpoints following their completed updates')
    return dict(schema='European_cohort8_FIT_progress_v1',durable_optimizer_updates=len(rows),
        completed_epochs=len(rows)//3,total_optimizer_updates=90,total_epochs=30,
        latest_update=None if not rows else rows[-1],checkpoint_refs={str(epoch):checkpoints[epoch] for epoch in epochs},
        latest_checkpoint_epoch=None if not epochs else epochs[-1],fresh_fit_only=True,resume_supported=False,
        checkpoint_contains_model_and_optimizer=True,checkpoint_contains_RNG=False,
        progress_is_completion_evidence_not_resume_authority=True,TEST_truth_read=False)


def snapshot(output,checkpoint_cache):
    rows=[]
    update_directory=output/'updates'
    if update_directory.exists():
        names=sorted(update_directory.glob('update_*.json'))
        need(len(names)<=90,'Bounded durable update count')
        for index,path in enumerate(names,1):
            need(path.name=='update_%03d.json'%index and path.stat().st_size<=64*2**10,'Contiguous bounded update records')
            value,_=read(path);rows.append(value)
    checkpoints={}
    for epoch in (5,10,15,20,25,30):
        path=output/'checkpoints'/('epoch_%02d.pt'%epoch)
        if not path.exists():continue
        need(not path.is_symlink() and path.stat().st_size<=8*2**20 and epoch*3<=len(rows),'Bounded checkpoint after durable update')
        # Called synchronously by the same worker between driver writes. The
        # checkpoint is hashed once; a later size/mtime change is rejected.
        info=path.stat();identity=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
        if epoch in checkpoint_cache:
            previous,ref=checkpoint_cache[epoch]
            need(previous==identity,'Finished immutable checkpoint changed')
        else:
            ref=reference(path);checkpoint_cache[epoch]=(identity,ref)
        checkpoints[epoch]=ref
    return summarize(rows,checkpoints)
