"""Retire only new disposable native corpora after genuine final predictions."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
from . import contracts as c


def need(ok,message):
    if not ok:raise ValueError(message)


def factory(*,preparation_ref,embedding_ref,prediction_ref,owner_scopes,output_directory):
    spec=dict(schema='successful_native_scratch_retirement_v1',preparation_ref=preparation_ref,
        embedding_ref=embedding_ref,prediction_ref=prediction_ref,owner_scopes=owner_scopes,
        output_directory=output_directory)
    validate(spec)
    return dict(module='arggnn.benchmarking.retention',function='run',spec=spec,
        python='.runtime/env/bin/python',source_refs=[c.file_ref(__file__)])


def validate(spec):
    need(spec['schema']=='successful_native_scratch_retirement_v1','Explicit new-scope retirement required')
    prep=c.read_ref(spec['preparation_ref']);embed=c.read_ref(spec['embedding_ref']);pred=c.read_ref(spec['prediction_ref'])
    need(prep['schema']=='american_fresh_ARGformer_preparation_v1'
         and embed['stage']=='embeddings' and pred['stage']=='retrieval'
         and prep['parent_id']==embed['parent_id']==pred['parent_id']
         and pred['records']==10000 and pred['scoring_performed'] is False,
         'One complete actual native prediction chain required')
    need(embed['preparation_ref']==pred['preparation_ref']==spec['preparation_ref'],
         'Exact same full native preparation required')
    seal=c.read_ref(pred['prediction_seal_ref'])
    need(seal.get('query_ancestry_truth_opened') is False or seal.get('private_truth_opened') is False,
         'Prediction seal must precede truth access')
    dirs=[prep['disposable_scratch_directory'],embed['disposable_scratch_directory']]
    need(dirs==pred['disposable_scratch_directories'] and len(set(dirs))==2,'Exact declared scratch directories required')
    owners=[c.contained_path(x) for x in spec['owner_scopes']]
    need(len(owners)==3 and len(set(owners))==3,'Three actual producer scopes required')
    for owner,ref in zip(owners,(spec['preparation_ref'],spec['embedding_ref'],spec['prediction_ref'])):
        need(c.contained_path(ref['path']).is_relative_to(owner/'native'),'Producer metadata must belong to its own scope')
        wait=c.read_ref(c.file_ref(owner/'actual_wait.json'))
        need(wait.get('actual_direct_wait') is True and wait.get('actual_direct_returncode')==0
             and wait.get('owned_process_group_absent') is True and wait.get('supervisor_error') is None,
             'Every producer must have genuine successful wait and absent worker group')
    for value,owner in zip(dirs,owners[:2]):
        path=c.contained_path(value)
        need(path==owner/'native/scratch' and path.is_dir() and not path.is_symlink(),
             'Only exact successful new native scratch may be retired')
        need('benchmark_campaign_20261018_workflow' in path.parts,
             'No historical or externally launched scratch may be retired')
    return dirs


def run(spec,*,resource_check):
    from .workflow import save
    dirs=validate(spec)
    out=c.contained_path(spec['output_directory']);out.mkdir(parents=True,exist_ok=False)
    inventory=[]
    for value in dirs:
        root=c.contained_path(value)
        for base,subdirs,files in os.walk(root,followlinks=False):
            for name in subdirs+files:
                path=Path(base)/name
                need(not path.is_symlink(),'Disposable directory contains a symbolic link')
            for name in files:
                path=Path(base)/name;info=path.stat()
                need(path.is_file(),'Only regular scratch files allowed')
                inventory.append(dict(path=str(path.relative_to(c.ROOT)),bytes=info.st_size,mtime_ns=info.st_mtime_ns))
        resource_check('retention_inventory')
    receipt=dict(schema=spec['schema'],preparation_ref=spec['preparation_ref'],embedding_ref=spec['embedding_ref'],
                 prediction_ref=spec['prediction_ref'],directories=dirs,files=inventory,
                 reason='Successful reproducible native corpus and embeddings, final predictions/checkpoints retained')
    inventory_ref=save(out/'retirement_inventory.json',receipt,exclusive=True)
    # Validate all waits and exact directory identities again before deletion.
    need(validate(spec)==dirs,'Retirement evidence changed')
    for value in dirs:
        resource_check('before_successful_scratch_retirement')
        shutil.rmtree(c.contained_path(value))
    need(all(not c.contained_path(d).exists() for d in dirs),'All declared scratch must actually be absent')
    return save(out/'retirement_completed.json',dict(inventory_ref=inventory_ref,retired_directories=dirs,
        successful_new_scratch_only=True,original_sources_unchanged=True,original_failures_preserved=True),exclusive=True)
