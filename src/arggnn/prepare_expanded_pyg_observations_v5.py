"""CPU-only H038 preparation with parallel in-RAM encoding and guarded publication.

Pinned v4 operational helpers execute in a private namespace; scientific
full952 feature math and cache encoding stay unchanged. No action at import.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='src/arggnn/prepare_expanded_pyg_observations_v5.py'
BASE='src/arggnn/prepare_expanded_pyg_observations_v4.py'
BASE_SHA256='ee977bbc56f3c1b1fb71b74928ae3f065c1f7baca261fbf96fe5bdcefa3f7719'
FAST='src/arggnn/mutation_routed_full_panel_fast_reader_v1.py'
STORAGE='src/arggnn/project_storage_terminal_retired_scopes_v1.py'
MODE='journal_terminal_retired_scopes_credit_v1'
SCHEMA='expanded_pyg_observations_v5_root_admission'
_BASE=None


def _base():
 global _BASE
 raw=(ROOT/BASE).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=BASE_SHA256:raise ValueError('Exact frozen v4 preparation template required')
 if _BASE is None:
  text=raw.decode();before='"journal_terminal_scopes_credit_v1"'
  if text.count(before)!=1:raise ValueError('Exact original admission-mode predicate required')
  namespace=dict(__name__='_arggnn_private_v5_v4_helpers',__package__='arggnn',__file__=str(ROOT/BASE))
  exec(compile(text.replace(before,'"'+MODE+'"'),str(ROOT/BASE),'exec'),namespace)
  namespace.update(SOURCE=SOURCE,STORAGE=STORAGE,SCHEMA=SCHEMA)
  # Keep the original scientific cache provenance closure; record the new
  # execution/validation/accounting sources separately, without format drift.
  namespace['SOURCE_PATHS']=namespace['CACHE_SOURCES']+(SOURCE,FAST,STORAGE,namespace['BASE_STORAGE'])
  _BASE=namespace
 return _BASE


def _ref(path):return _base()['_ref'](path)
def _read(ref,expected=None):return _base()['_read'](ref,expected)
def _utc():return datetime.now(timezone.utc).isoformat()
def _peak_rss():return _base()['_peak_rss']()


def _worker_init(scratch,barrier):
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
 os.environ['TMPDIR']=scratch;tempfile.tempdir=scratch
 from . import mutation_routed_full_panel_fast_reader_v1 as fast
 fast._source_guard()  # All extra codec/PyG imports finish before the shared barrier.
 _base()['_worker_init'](scratch,barrier)


def _warmup():return os.getpid()
def _load(row,entry_ref,scratch):return _base()['_load'](row,entry_ref,scratch)


def _prepare_compressed(observed,bins):
 from . import mutation_routed_full_panel_observations_v1 as observations
 from . import mutation_routed_full_panel_fast_reader_v1 as fast
 from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
 started=time.monotonic()
 bank=observations._full_prepare(observed.trees,observed.intervals_bp,observed.positions_bp0,
  observed.genotype,observed.allele_count,observed.sample_ids,observed.reference_class,observed.sample_age,
  observed.query_ids,observed.readout_x_bp,sham_bin_edges=bins,fixture=False)
 inputs=MutationRoutedWholeTreeInputsV1.from_arrays(bank.model_arrays())
 item=dict(model_inputs=inputs,scoring=observed.scoring,
  observations_receipt=dict(observed.receipt,preprocessing=bank.metadata))
 fast.check_item_v1(observed.parent_id,item,observed.receipt['entry_ref'],{observed.parent_id:observed.split})
 prepared=time.monotonic();payload,record=fast.encode_parent_v1(item,expected_haplotypes=952)
 return (payload,record),dict(parent_id=observed.parent_id,prepare_worker_pid=os.getpid(),
  prepare_worker_peak_rss_bytes=_peak_rss(),prepare_seconds=prepared-started,
  compressed_encode_seconds=time.monotonic()-prepared,compressed_bytes=len(payload),
  model_tensor_IPC_used=False,filesystem_temp_writes=False)


def execute_v1(spec,input_ref):
 base=_base();config,registry,directory=base['_admit'](spec)
 import numpy as np
 import torch
 import multiprocessing.connection as connection
 from . import mutation_routed_full_panel_observations_v1 as observations
 from . import mutation_routed_tree_observations_v1 as tree_math
 from . import mutation_routed_full_panel_fast_reader_v1 as fast
 from . import mutation_routed_full_panel_cache_v1 as cache
 from . import project_storage_terminal_retired_scopes_v1 as adapter
 from .census_tsinfer_fit_v1 import ConsumerReaderV1
 from .mutation_routed_pyg_dataset_v1 import _worker_ipc_address_v1
 roles=observations.validate_registry_v1(registry);rows=[row for row in registry if row['split']!='outer']
 source_refs={path:{'path':path,'sha256':spec['source_sha256'][path]} for path in base['CACHE_SOURCES']}
 execution_refs={path:{'path':path,'sha256':spec['source_sha256'][path]} for path in (SOURCE,FAST,STORAGE)}
 if directory.exists():
  completed=directory/'completed.json'
  if not completed.is_file():raise FileExistsError('Incomplete output preserved; fresh admitted v5 output required')
  result=json.loads(completed.read_text())
  if (result.get('schema')!='expanded_pyg_observations_v5_completed' or result.get('input_ref')!=input_ref
   or result.get('execution_source_refs')!=execution_refs or result.get('targets_read') is not False
   or result.get('outer_included') is not False or result.get('CPU_workers')!=8):raise ValueError('Completed output belongs to another pinned admission')
  entries={}
  for row in rows:
   ready=base['_ready_entry'](row,spec,config)
   if ready is None:raise ValueError('Completed cache registered parent is no longer ready')
   entries[row['parent_id']]=ready[0]
  fast.read_full_cache_index_v1(result['cache_manifest_ref'],ConsumerReaderV1(),registry=registry,entry_refs=entries,source_refs=source_refs)
  return dict(result,reused_completed_cache=True)
 adapter.validate_storage_v9(spec,dict(config,output_cap_bytes=spec['output_cap_bytes']),spec['storage'],
  source_ref=spec['source_ref'],config_ref=spec['config_ref'],runtime_ref=spec['runtime_ref'])
 expiry=datetime.fromisoformat(spec['expires_at_utc'].replace('Z','+00:00'));guarded_bytes=0
 def guard(planned=0):
  nonlocal guarded_bytes
  if datetime.now(timezone.utc)>=expiry:raise ValueError('CPU preparation admission expired')
  value=adapter.budget_check_v9(spec['storage'],boundary=True,required_write_bytes=int(planned))
  guarded_bytes+=int(planned);return value
 output=base['_Output'](directory,guard)
 output.save('started.json',dict(schema='expanded_pyg_observations_v5_started',input_ref=input_ref,
  source_ref=spec['source_ref'],execution_source_refs=execution_refs,registry_sha256=base['_digest'](registry),started_utc=_utc(),CPU_workers=8))
 scratch=directory/'worker_tmp';guard(16*8192);scratch.mkdir();scratch_identity=(scratch.stat().st_dev,scratch.stat().st_ino)
 os.environ['TMPDIR']=str(scratch);tempfile.tempdir=str(scratch);torch.set_num_threads(1)
 connection.arbitrary_address=_worker_ipc_address_v1
 context=mp.get_context('spawn');barrier=context.Barrier(8)
 executor=ProcessPoolExecutor(max_workers=8,mp_context=context,initializer=_worker_init,initargs=(str(scratch),barrier))
 monitor=base['_Monitor'](executor,spec['max_cpu_rss_bytes']);started=time.monotonic();failed=True
 entries,loaded,stats,pending,submitted={},{},{},{},set();records={};publication_timings=[];cache_manifest=None
 def log_failure(error,kind='failed'):
  detail=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),submitted_parent_ids=sorted(submitted),
   main_peak_rss_bytes=_peak_rss(),owned_peak_rss_sum_bytes=monitor.peak_total)
  print(json.dumps(dict(utc=_utc(),event=kind,**detail)),file=sys.stderr,flush=True)
  try:
   output.event(kind,**detail)
   if not (directory/'failed.json').exists():output.save('failed.json',dict(schema='expanded_pyg_observations_v5_failure',input_ref=input_ref,**detail))
  except BaseException as logging_error:print('Failure receipt write rejected: '+str(logging_error),file=sys.stderr,flush=True)
 try:
  output.event('worker_warmup_started',CPU_workers=8)
  warmup=[executor.submit(_warmup) for _ in range(8)];warmup_deadline=time.monotonic()+60
  while warmup:
   done,_=wait(warmup,timeout=.5,return_when=FIRST_COMPLETED)
   for future in done:future.result();warmup.remove(future)
   if time.monotonic()>=warmup_deadline:raise TimeoutError('Eight-worker import barrier did not complete')
  output.event('worker_warmup_completed',CPU_workers=8)
  deadline=started+spec['readiness_timeout_seconds'];next_poll=0.
  while len(loaded)<40:
   if time.monotonic()>=deadline:raise TimeoutError('Registered inner observed parent readiness/load timeout')
   if time.monotonic()>=next_poll:
    for row in rows:
     p=row['parent_id']
     if p in submitted:continue
     ready=base['_ready_entry'](row,spec,config)
     if ready is None:continue
     if len(pending)>=8:break
     entries[p]=ready[0];guard(8192);pending[executor.submit(_load,row,ready[0],str(scratch))]=p;submitted.add(p)
     output.event('parent_load_submitted',parent_id=p,entry_ref=ready[0])
    next_poll=time.monotonic()+spec.get('poll_seconds',2.)
   done,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
   for future in done:
    p=pending.pop(future);observed,detail=future.result()
    if observed.parent_id!=p or p in loaded:raise ValueError('Parent load identity/exactly-once contract differs')
    loaded[p]=observed;stats[p]=detail;output.event('parent_loaded',loaded_parent_count=len(loaded),**detail)
   if not pending:time.sleep(.5)
  preload_seconds=time.monotonic()-started
  output.event('all_inner_parents_loaded',parent_count=40,train_count=32,val_count=8,targets_opened=False,preload_elapsed_seconds=preload_seconds)
  fit_started=time.monotonic();duration,count=[],[]
  for p in roles['train']:
   d,c=tree_math.observed_sham_geometry_v1(loaded[p].trees);duration.append(d);count.append(c)
  bins=tree_math.fit_sham_bin_edges_v1(np.concatenate(duration),np.concatenate(count),split='TRAIN',bins=4)
  del duration,count,d,c
  bin_seconds=time.monotonic()-fit_started
  output.event('legacy_bins_fitted',TRAIN_parent_ids=roles['train'],legacy_metadata_only=True,fit_seconds=bin_seconds,training_branch_count=bins.training_branch_count)
  prepare_started=time.monotonic();pending={};remaining=iter(row['parent_id'] for row in rows)
  cache_directory=directory/'cache';publication_seconds=0.
  def submit_prepare():
   p=next(remaining,None)
   if p is None:return False
   observed=loaded.pop(p);pending[executor.submit(_prepare_compressed,observed,bins)]=p;return True
  for _ in range(8):submit_prepare()
  while pending:
   done,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED)
   for future in done:
    p=pending.pop(future);(payload,record),detail=future.result()
    if (p in records or not isinstance(payload,bytes) or detail.get('parent_id')!=p
     or record['observations_receipt'].get('parent_id')!=p or record['observations_receipt'].get('entry_ref')!=entries[p]
     or record['observations_receipt'].get('targets_read') is not False or record['observations_receipt'].get('observations_only') is not True):
     raise ValueError('Exactly-once observed-only compressed worker record required')
    path=cache_directory/(p+'.npz');phase=time.monotonic()
    record.update(array_ref={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest()},compressed_bytes=len(payload))
    guard(len(payload)+8192);cache_directory.mkdir(exist_ok=True)
    cache.old._write_fresh_atomic(path,payload)  # Main is the sole scope publisher.
    elapsed=time.monotonic()-phase;publication_seconds+=elapsed
    records[p]=record;stats[p].update(detail);publication_timings.append(dict(parent_id=p,compressed_bytes=len(payload),publication_seconds=elapsed))
    output.event('parent_prepared_published',prepared_parent_count=len(records),**detail,publication_seconds=elapsed)
    del payload,record;submit_prepare()
  prepare_seconds=time.monotonic()-prepare_started
  if loaded or set(records)!=set(entries) or len(records)!=40 or len(submitted)!=40:raise ValueError('Complete exactly-once inner observation collection required')
  receipt=dict(schema='mutation_routed_full_panel_prepared_collection_v1',targets_read=False,observations_only=True,
   sham_bins_fit_from_parent_ids=roles['train'],original_native_holds_preserved=True,outer_included=False,
   elapsed_observed_preparation_seconds=time.monotonic()-started,placement_sham_computed=False,sham_bins_legacy_metadata_only=True,
   execution_source_refs=execution_refs,base_preparation_source_ref={'path':BASE,'sha256':BASE_SHA256},
   preparation_transport='worker_encoded_NPZ_bytes_and_metadata_only',worker_filesystem_temp_writes=False)
  metadata=dict(schema=cache.SCHEMA,registry=registry,entry_refs=entries,source_refs=source_refs,
   parents={row['parent_id']:records[row['parent_id']] for row in rows},sham_bin_edges=asdict(bins),receipt=receipt,
   targets_cached=False,observations_only=True,pickle_used=False)
  metadata['tuple_paths']=cache.old._tuple_paths(metadata);payload=cache.old._json_bytes(metadata)
  phase=time.monotonic();guard(len(payload));cache.old._write_fresh_atomic(cache_directory/'manifest.json',payload)
  cache_manifest={'path':str((cache_directory/'manifest.json').relative_to(ROOT)),'sha256':hashlib.sha256(payload).hexdigest()}
  publication_seconds+=time.monotonic()-phase
  output.event('cache_published',cache_manifest_ref=cache_manifest,parent_count=40,compressed_bytes=sum(row['compressed_bytes'] for row in publication_timings),
   publication_elapsed_seconds=publication_seconds,all40_observations_before_truth=True)
  failed=False
 except BaseException as error:log_failure(error);raise
 finally:
  monitor.close()
  try:
   base['_drain'](executor,failed);base['_path'](scratch)
   if (scratch.stat().st_dev,scratch.stat().st_ino)!=scratch_identity:raise ValueError('Owned scratch identity changed before cleanup')
   if not shutil.rmtree.avoids_symlink_attacks:raise RuntimeError('Safe owned scratch-tree deletion unavailable')
   shutil.rmtree(scratch)
  except BaseException as error:
   log_failure(error,'worker_cleanup_failed')
   if not failed:raise
 result=dict(schema='expanded_pyg_observations_v5_completed',input_ref=input_ref,source_ref=spec['source_ref'],execution_source_refs=execution_refs,
  config_ref=spec['config_ref'],registration_ref=spec['registration_ref'],source_refs=source_refs,runtime_ref=spec['runtime_ref'],
  registry_sha256=base['_digest'](registry),cache_manifest_ref=cache_manifest,completed_utc=_utc(),CPU_workers=8,parent_count=40,TRAIN_count=32,VAL_count=8,
  outer_included=False,targets_read=False,observations_only=True,inference_regenerated=False,GPU_used=False,owned_workers_drained=True,
  native_tree_snapshot='Linux_memfd_filelike',model_tensor_IPC_used=False,process_pool_results='compressed_NPZ_bytes_and_metadata_only',
  worker_filesystem_temp_writes=False,preload_seconds=preload_seconds,legacy_bin_fit_seconds=bin_seconds,prepare_seconds=prepare_seconds,
  publication_seconds=publication_seconds,total_elapsed_seconds=time.monotonic()-started,parent_stats=stats,publication_timings=publication_timings,
  compressed_bytes=sum(row['compressed_bytes'] for row in publication_timings),main_peak_rss_bytes=_peak_rss(),owned_peak_rss_sum_bytes=monitor.peak_total,
  RSS_sum_includes_shared_page_double_counting=True,guarded_planned_bytes=guarded_bytes,reused_completed_cache=False)
 output.save('completed.json',result);return result


def source_report_v1():
 base=_base()
 return dict(status='source_only_no_preparation',schema=SCHEMA,source_ref=_ref(SOURCE),source_sha256={path:_ref(path)['sha256'] for path in base['SOURCE_PATHS']},
  cache_source_refs={path:_ref(path) for path in base['CACHE_SOURCES']},execution_source_refs={path:_ref(path) for path in (SOURCE,FAST,STORAGE)},
  config_ref=_ref(base['CONFIG']),registration_ref=_ref(base['REGISTRATION']),producer_ref=_ref(base['PRODUCER']),
  original_producer_ref=_ref(base['ORIGINAL_PRODUCER']),storage_adapter_ref=_ref(STORAGE),base_preparation_source_ref={'path':BASE,'sha256':BASE_SHA256},
  CPU_workers=8,TRAIN_count=32,VAL_count=8,outer_included=False,targets_read=False,GPU_used=False)


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--input');parser.add_argument('--expected-input-sha256')
 args=parser.parse_args(argv)
 if not args.execute:print(json.dumps(source_report_v1(),indent=2,sort_keys=True));return
 if not args.input or not args.expected_input_sha256:raise ValueError('Pinned root CPU execution admission required')
 ref={'path':args.input,'sha256':args.expected_input_sha256};print(json.dumps(execute_v1(_read(ref),ref),indent=2))


if __name__=='__main__':main()
