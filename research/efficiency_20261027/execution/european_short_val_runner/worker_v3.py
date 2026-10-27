"""One ROOT-admitted short development parent; no execution on import."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from arggnn import prepare_ancestralpaths_full_panel_discovery_v1 as full
from research.efficiency_20261027.integration_candidate.european_short_development_val import generation_v3, native_relate_v3, sampling_v1
from research.efficiency_20261027.integration_candidate.european_short_development_val.aggregate_guard_v3 import AggregateOwner
from research.efficiency_20261027.integration_candidate.european_short_development_val.native_supervisor_v3 import NativeSupervisor, process_table, owned_family
from .contracts_v3 import (ROOT, AGGREGATE, HOST_CAP, MAX_SECONDS, HEADROOM, METADATA_CAP,
    need, ref, read, verify_sources, native_bindings, validate_release, validate_worker_records, plan,
    ROOT_STATUS_MAX_AGE_SECONDS, retained_attempt)
from .retained_allowance_v1 import configure_retained_allowance


def wait_for_launch(timeout=10):
    """No stage may run before the guardian publishes its exact child receipt."""
    until=time.monotonic()+timeout
    while not (AGGREGATE/'launch.json').exists():
        need(time.monotonic()<until,'ROOT launch receipt did not arrive before bounded startup gate')
        time.sleep(.05)
    return ref(AGGREGATE/'request.json'),ref(AGGREGATE/'launch.json')


def execution_checker(request_ref,launch_ref,release_ref,request,launch,argv):
    table=process_table();own=table.get(os.getpid());parent=table.get(os.getppid())
    need(own is not None and parent is not None and own['session']==own['pgrp']==own['pid'],
         'Actual new worker session and live guardian required')
    validate_worker_records(request,launch,argv,own['pid'],str(own['start']),parent['pid'],str(parent['start']))
    actual=[part.decode() for part in Path('/proc/self/cmdline').read_bytes().split(b'\0') if part]
    need(actual==request['command'],'Actual process command equals the visible ROOT argv')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(key in os.environ for key in
        ('ARGGNN_PHYSICAL_GPU','ARGGNN_READABLE_GPU_FD','ARGGNN_MAX_CUDA_RESERVED_BYTES','ARGGNN_GPU_MEMORY_POLICY')),
        'CUDA hidden and no GPU lease environment')
    deadline=request['deadline_unix'];created=request['created_unix']
    need(type(deadline) in (int,float) and type(created) in (int,float)
         and math.isfinite(deadline) and math.isfinite(created) and deadline-created==MAX_SECONDS,
         'Exact finite ROOT total wall deadline')
    remaining=deadline-time.time();need(0<remaining<=MAX_SECONDS,'Current remaining ROOT wall budget')
    monotonic_deadline=time.monotonic()+remaining;known={own['pid']:own}
    sampling_lock=threading.Lock();last_sample=[-math.inf]
    diagnostics=dict(max_status_age_seconds=0.,max_full_check_seconds=0.)
    def check():
        need(not (AGGREGATE/'execution_hold').exists(),'ROOT execution hold active')
        need(time.time()<deadline and time.monotonic()<monotonic_deadline,'ROOT six-hour total wall expired')
        # Hold and wall checks above are immediate,including each log callback.
        # Repeated metadata hashes and global process reads are sampled at most
        # once per second across both the monitor and stage threads. The ROOT
        # guardian independently samples the complete family and storage.
        with sampling_lock:
            if time.monotonic()-last_sample[0]<1:
                return
            sample_started=time.monotonic()
            need(ref(request_ref['path'])==request_ref and ref(launch_ref['path'])==launch_ref
                 and ref(release_ref['path'])==release_ref,'Exact ROOT request,launch and execution release remain current')
            table=process_table();current=table.get(own['pid']);guardian=table.get(parent['pid'])
            need(current is not None and guardian is not None
                 and current['start']==own['start'] and current['ppid']==parent['pid']
                 and guardian['start']==parent['start'] and guardian['state']!='Z',
                 'Exact ROOT guardian and worker generations remain alive')
            status_path=AGGREGATE/'status.json'
            need(status_path.resolve()==status_path and status_path.stat().st_size<=2**20,'Bounded ordinary ROOT live status')
            status=json.loads(status_path.read_text())
            age=time.time()-status['observed_unix']
            diagnostics.update(last_status_age_seconds=age,
                max_status_age_seconds=max(diagnostics['max_status_age_seconds'],age),
                guardian_admission_seconds=status.get('admission_seconds'))
            need(status.get('status') in ('admitted','running')
                 and 0<=age<=ROOT_STATUS_MAX_AGE_SECONDS,
                 'Fresh active ROOT admission status: age_seconds='+str(age)
                 +' maximum_seconds='+str(ROOT_STATUS_MAX_AGE_SECONDS))
            family=owned_family(table,own,known);known.update(family)
            rss=guardian['rss_bytes']+sum(row['rss_bytes'] for row in family.values() if row['state']!='Z')
            need(rss<HOST_CAP,'Complete guardian and recursive worker family64GiB RSS ceiling')
            last_sample[0]=time.monotonic()
            diagnostics['max_full_check_seconds']=max(diagnostics['max_full_check_seconds'],last_sample[0]-sample_started)
    check.diagnostics=diagnostics
    check()
    return check,remaining


def run_stages(owner,frozen,native_config,reviewed_tools):
    """Only the immutable registered generator,native pipeline and sampler."""
    parent={key:frozen['registration'][key] for key in ('parent_id','split','seeds')}
    recipe=frozen['recipe'];generation=owner.path(AGGREGATE/'generation')
    native_directory=owner.path(AGGREGATE/'native')
    owner(16384);generation.mkdir()
    generated=generation_v3.generate_parent(recipe,parent,generation,guard=owner,
        execution_check=owner.execution_check,write_headroom_bytes=HEADROOM)
    observed={}
    for name in native_relate_v3.OBSERVED_FILES:
        path=generation/'observed'/name
        binding=ref(path);record=generated['generation']['files']['observed/'+name]
        need(binding['sha256']==record['sha256'] and path.stat().st_size==record['bytes'],
             'Exact generated observed whitelist bytes')
        observed[name]=binding
    owner.check()
    native=NativeSupervisor(owner,reviewed_native_tools=reviewed_tools)
    inferred=native_relate_v3.infer_parent(recipe,parent,generation/'observed',native_directory,
        native_config=native_config,expected_input_refs=observed,native_step=native,guard=owner,
        execution_check=owner.execution_check,write_headroom_bytes=HEADROOM,max_metadata_bytes=METADATA_CAP)
    owner.check()
    inference_ref=ref(native_directory/'inference.json')
    need(read(inference_ref,METADATA_CAP)==inferred,'Exact complete observed inference receipt')
    expected_rows=ref(native_directory/'observed_sample_manifest.json')
    need(inferred['observed_sample_manifest']==expected_rows,'Observed sample mapping remains in exact native directory')
    rows=read(expected_rows,METADATA_CAP)
    registry=sampling_v1.build_validation_registry(frozen['sampling_plan'],parent['parent_id'],rows,
        inferred['post_five_iterations'],observation_ref=inference_ref)
    need(registry['examples']==10000 and registry['labels_present'] is False
         and registry['query_truth_read'] is False,'Exact10000 observed sampled rows with no labels')
    registry_path=owner.path(AGGREGATE/'sampled_registry.json')
    full.json_write(registry_path,registry,owner)
    owner.check()
    return dict(parent_id=parent['parent_id'],split='validation',sequence_length_bp=10000000,
        inference_ref=inference_ref,inferred_tree_ref=inferred['inferred_tree'],
        observed_sample_manifest_ref=expected_rows,native_commands_ref=inferred['commands'],
        sampled_registry_ref=ref(registry_path),observed_input_refs=observed,
        sampled_examples=10000,development_evidence_only=True,independent_chromosome_replication=False,
        generation_truth_retained_and_charged=True,downstream_truth_or_oracle_read=False,
        model_fit_prediction_or_scoring_performed=False,original_TRAIN_TEST_unchanged=True)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(__doc__)
    for name in ('source-pins-sha256','source-review-path','source-review-sha256',
                 'execution-release-path','execution-release-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);need(Path.cwd()==ROOT,'Explicit project root')
    review_ref=dict(path=args.source_review_path,sha256=args.source_review_sha256)
    manifest_ref,frozen=verify_sources(args.source_pins_sha256,review_ref)
    release_ref=dict(path=args.execution_release_path,sha256=args.execution_release_sha256)
    release=read(release_ref)
    validate_release(release,manifest_ref,review_ref,release.get('peer_directories'))
    config,tools,chain_ref=native_bindings()
    retained_ref,retained_bytes=retained_attempt()
    request_ref,launch_ref=wait_for_launch();request=read(request_ref);launch=read(launch_ref)
    need(request.get('source_manifest_ref')==manifest_ref and request.get('independent_source_review_ref')==review_ref
         and request.get('execution_release_ref')==release_ref and request.get('peer_directories')==release['peer_directories']
         and request.get('other_project_reserved_bytes')==release['other_project_reserved_bytes']
          and request.get('retained_attempt_ref')==release['retained_attempt_ref']==retained_ref
          and request.get('retained_charged_bytes')==retained_bytes,
         'Worker matches exact ROOT source and global reservation binding')
    check,remaining=execution_checker(request_ref,launch_ref,release_ref,request,launch,argv)
    allowance=configure_retained_allowance(retained_bytes)
    owner=None
    def cancelled(signum,frame):
        raise KeyboardInterrupt('ROOT shortVAL worker received signal '+str(signum))
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    try:
        owner=AggregateOwner(AGGREGATE,execution_check=check,max_family_rss_bytes=HOST_CAP,max_seconds=remaining)
        owner.start_monitor()
        # All temporary/cache paths are charged; the generation directory itself
        # remains empty until the immutable generator receives it.
        temporary=owner.path(AGGREGATE/'tmp');cache=owner.path(AGGREGATE/'cache')
        owner(16384);temporary.mkdir();cache.mkdir()
        os.environ.update(TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary),
            XDG_CACHE_HOME=str(cache),NUMBA_CACHE_DIR=str(cache),MPLCONFIGDIR=str(cache))
        result=run_stages(owner,frozen,config,tools)
        owner.write_json(AGGREGATE/'observed_artifacts.json',dict(
            schema='European_short_VAL_observed_artifacts_v1',status='prepared',
            source_manifest_ref=manifest_ref,source_review_ref=review_ref,plan_ref=plan.PLAN_REF,
            execution_release_ref=release_ref,native_chain_review_ref=chain_ref,
            storage_metrics=dict(owner.metrics),retained_allowance=allowance,
            execution_check_metrics=dict(check.diagnostics),finished_unix=time.time(),**result))
        owner.check()
    except BaseException as error:
        if owner is not None:
            owner.write_json(AGGREGATE/'worker_failure.json',dict(error=repr(error),
                failed_evidence_retained=True,retained_allowance=allowance,
                execution_check_metrics=dict(check.diagnostics),observed_unix=time.time()),emergency=True)
        raise
    finally:
        if owner is not None:owner.close()

if __name__=='__main__':main()
