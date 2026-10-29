"""Predict one canonical TEST parent from a prospectively sealed checkpoint."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback
import numpy as np
import torch
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel, collate_prepared
from research.redesign_20261028.data.window_data import ROOT, path_for, require
from .common import BASE, MODEL_ID, TEST_PARENTS, now, ref, write_json, canonical_output
from .freeze import GPU_UUIDS, validate_freeze, check_inputs
from .observations import TestObservationSource, make_site_plan
from .workers import prepared_stream


def check_gpu(index):
    uuid = GPU_UUIDS[index]
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'Expose exactly the registered GPU UUID')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    mapping = {int(line.split(',')[0]): line.split(',')[1].strip() for line in result.stdout.splitlines()}
    require(mapping.get(index) == uuid, 'Physical GPU index/UUID mismatch')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one assigned CUDA device')
    torch.cuda.set_device(0)
    return uuid


def run(args, output, started):
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    frozen = validate_freeze(args.freeze)
    freeze_ref = ref(args.freeze)
    gpu_uuid = check_gpu(args.gpu)
    check_inputs(frozen, args.parent)
    model = JointAncestryModel(**frozen['model'])
    model.load_state_dict(torch.load(path_for(frozen['checkpoint_ref']), weights_only=True, map_location='cpu'), strict=True)
    model.eval().to('cuda:0')
    item = frozen['parents'][args.parent]
    write_json(output / 'started.json', dict(started_utc=now(), pid=os.getpid(), parent_id=args.parent,
               model_id=MODEL_ID, freeze_ref=freeze_ref, gpu=args.gpu, gpu_uuid=gpu_uuid,
               process_title='AI Auto Research Pipeline Test', workers=4, batch_sites=16,
               query_ancestry_truth_opened=False))
    source = TestObservationSource(args.parent, item['binding'])
    plan = make_site_plan(source, item['parent_ordinal'])
    plan_ref = write_json(output / 'site_plan.json', plan)
    sites = plan['sites']
    all_logits, all_ids, diagnostics = [], [], []
    batch, batch_ids = [], []
    inference_seconds = 0.
    with (output / 'progress.jsonl').open('x', encoding='utf-8') as progress:
        for index, prepared, metadata in prepared_stream(args.parent, item['binding'], sites):
            require(index == len(diagnostics), 'Prepared sites preserve deterministic plan order')
            site = sites[index]
            require(prepared.target_count == len(site['example_indices']), 'Prepared focal target count')
            require(np.array_equal(prepared.focal_sample.numpy(), np.asarray(site['sample_indices'], dtype=np.int64)),
                    'Prepared focal sample identities equal canonical site rows')
            batch.append(prepared)
            batch_ids.extend(site['example_indices'])
            diagnostics.append(metadata)
            if len(batch) == 16 or index + 1 == len(sites):
                before = time.monotonic()
                assembled = collate_prepared(batch).pin_memory().to('cuda:0', non_blocking=True)
                with torch.inference_mode():
                    predicted = model(assembled, mode='structure').detach().cpu().numpy()
                require(predicted.shape == (len(batch_ids), 6), 'Exactly one six-logit row per focal target')
                all_logits.append(predicted)
                all_ids.extend(batch_ids)
                batch.clear()
                batch_ids.clear()
                del assembled, prepared
                inference_seconds += time.monotonic() - before
                record = dict(utc=now(), prepared_sites=index + 1, total_sites=len(sites),
                              canonical_targets=len(all_ids), elapsed_seconds=time.monotonic() - started)
                progress.write(json.dumps(record) + '\n')
                progress.flush()
                print(json.dumps(record), flush=True)
    values = canonical_output(np.concatenate(all_logits), np.asarray(all_ids, dtype=np.int64), source.records, 10000)
    check_inputs(frozen, args.parent)
    validate_freeze(args.freeze)
    require(ref(args.freeze) == freeze_ref, 'Prospective freeze unchanged during prediction')
    with (output / 'predictions.npz').open('xb') as stream:
        np.savez(stream, **values)
    predictions_ref = ref(output / 'predictions.npz')
    diagnostics_ref = write_json(output / 'diagnostics.json', dict(sites=diagnostics,
          elapsed_seconds=time.monotonic() - started, inference_seconds=inference_seconds,
          parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), workers_exited=True))
    require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) < 64 * 2**20,
            'Small per-parent prediction output cap')
    product = dict(schema='joint_model_native6_prediction_v1', model_id=MODEL_ID, parent_id=args.parent,
          split='test', rows=10000, native_path_names=frozen['native_path_names'], mode='structure',
          registry_ref=item['binding']['refs']['sampled_registry_ref'], predictions_ref=predictions_ref,
          freeze_ref=freeze_ref, checkpoint_ref=frozen['checkpoint_ref'], frozen_checkpoint_ref=frozen['checkpoint_ref'],
          runtime_source_refs=frozen['runtime_source_refs'], dependency_refs=frozen['dependency_refs'],
          site_plan_ref=plan_ref, diagnostics_ref=diagnostics_ref, query_ancestry_truth_opened=False,
          probability_convention='raw native-six softmax computed in float64 from stored float32 logits, stored float32; no cohort postprocessing',
          completed_utc=now(), complete=True)
    product_ref = write_json(output / 'product.json', product)
    print(json.dumps(dict(complete=True, product_ref=product_ref)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', choices=TEST_PARENTS, required=True)
    parser.add_argument('--gpu', type=int, choices=sorted(GPU_UUIDS), required=True)
    parser.add_argument('--freeze', default=str(BASE / 'freeze.json'))
    parser.add_argument('--output', help='New output directory under prediction/outputs; never overwrite a prior attempt')
    args = parser.parse_args()
    output = (Path(args.output) if args.output else BASE / 'outputs' / args.parent).resolve()
    require(output.is_relative_to(BASE / 'outputs') and output != BASE / 'outputs', 'Parent output directory containment')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        run(args, output, started)
    except BaseException as error:
        write_json(output / 'failure.json', dict(complete=False, failed_utc=now(), parent_id=args.parent,
                   error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                   elapsed_seconds=time.monotonic() - started, query_ancestry_truth_opened=False))
        raise


if __name__ == '__main__':
    main()
