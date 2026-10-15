# C3 observation v3: SampleData handle correction and explicit eight-thread runtime

Root may now review and run one fresh actual100kb observation qualification, then proceed to full30Mb if it succeeds. No inference, acquisition, data rewrite, GPU, admission, guard configuration, installation or Git action was performed by this source task.

## Confirmed cause

The failed v2 actual wait exited1 after74.9s at `tsinfer.load(observed.samples)`. The retained73,728-byte sample store is valid: tsinfer-sample-data format5.1, finalised true,64 samples,244 sites,100,000bp. Its SHA256 before and after read-only diagnosis is `c5a1cf9478d3427f72568a8e11bcac524b9522effba8d866cf7b8026ee8ae821`.

Installed tsinfer0.5.1 `DataContainer.__exit__` finalises a BUILD-mode SampleData object. `finalise` closes its writer then calls `_open_readonly`, leaving a live read-only LMDB environment held by the still-referenced `data` object. The original worker immediately calls `tsinfer.load` for the same path. With lmdb3.0.0, the duplicate environment raises `FileFormatError: The environment ... is already open in this process`. Generic tsinfer.load catches that specific exception, also tries AncestorData, then reports only Unrecognised file format.

Read-only access to the existing artifact demonstrated the exact failure: first SampleData.load succeeds; a second simultaneous SampleData.load raises the duplicate-environment error; close first and reload succeeds. No records were rewritten or inference attempted. `sample_store_diagnosis.json` pins the failed wait/log, existing artifact, installed source files and relevant source-line ranges.

## Minimal new worker

`src/arggnn/american_c3_observe_v2.py` preserves the original v1 module. It explicitly calls `data.close()` after context finalisation and before the retained-store finish receipt, then opens a single typed reader:

    data.close()
    w.finish(sample_path,cfg['max_sample_store_bytes'])
    with tsinfer.SampleData.load(str(sample_path)) as samples:
        inferred=tsinfer.infer(samples,num_threads=cfg['num_threads'],record_provenance=True)

The typed reader closes when inference returns or raises. No inferred arrays are created by this source task. Original sample ordering, genotype/allele/site preservation, sanitization, mutation rate, tsdate defaults, failure preservation and qualification checks remain unchanged.

Root explicitly requested num_threads8. Installed tsinfer.infer accepts that keyword, defaults to0 (sequential), and forwards it to generate_ancestors, match_ancestors and match_samples. Its engine remains the installed default constants.C_ENGINE. Both inactive configs now require the integer num_threads8, with no fallback. This is a declared reconstruction runtime setting; it does not establish equivalence to paper settings or bitwise equality across thread schedules. Existing BLAS/OMP/NUMBA environment limits remain1; tsinfer's explicit worker count is8. Output manifests and completion record the setting.

`observation_worker_fix.diff` contains the complete v1→v2 source correction and disclosure changes.

## Fresh root driver and interface

Source is `research/reviews/American_C3_observation_root_execution_20261015_v3/root_observation_driver.py`. It preserves root's v2 worker_aux scope correction and all current capacity-v2 marker-repair closure requirements. Compact root metadata stays outside worker_aux; runtime caches/temp/logs are in the reserved fresh worker_aux scope.

From `/workspace-SR003.nfs2/ARG-GNN` root invokes:

    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v3/root_observation_driver.py --preflight qualification100kb
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v3/root_observation_driver.py --admit qualification100kb
    .runtime/env/bin/python -B research/reviews/American_C3_observation_root_execution_20261015_v3/root_observation_driver.py --launch qualification100kb

APIs are preflight_v3(stage), admit_v3(stage), launch_v3(stage). The worker's direct child module is arggnn.american_c3_observe_v2. It uses the same common root admission/schema and the new exact source/config/runtime refs. No-flag/--source-report and --preflight modes are read-only. Root must coordinate admissions and refresh initial pins to avoid a concurrent-ledger race.

Fresh packets are admitted_qualification100kb_v3 and admitted_full30mb_v3. Fresh output directories are:

- experiments/results/American_C3_observed_qualification100kb_20261015_v3
- experiments/results/American_C3_observed_full30mb_20261015_v3

The failed v1 output and v2 root admission/wait/log remain preserved. Inputs are the exact original244-site/64-haplotype100kb observation and226,185-site/1,600-haplotype30Mb observation. No source roster, simulation, true-tree payload or private truth changes occur.

Qualification has the same1,000,000,000-byte cap and952,375,168-byte total plan. Full has the same12,500,000,000-byte cap and8,202,375,168-byte total plan. Store/table/count caps,600s qualification /75,600s full walltimes and22h admissions remain unchanged. These are operational bounds, not measured inference requirements; oversize/time failures preserve partial output.

After actual qualification direct exit0, verified all-sites/all-samples manifest and explicit recorded eight-thread setting, root may use the same --preflight/--admit/--launch sequence for full30mb. No broad test campaign or automatic fallback is introduced.

## Source checks and limits

Both new sources compiled. Bounded read-only preflights verified the exact four observation input-file hashes and NPY header metadata. The existing sample store diagnosis succeeded with no hash change. Full preflight correctly remains gated on the new actual qualification. No new SampleData file or test dataset was written.

Installed runtime is unchanged: Python3.11.17, tsinfer0.5.1, tsdate0.2.7, tskit1.0.3, numpy2.4.6, zarr2.18.7, lmdb3.0.0 and numcodecs0.15.1. The correction addresses the observed handle failure only. Successful inference/dating and operational full30Mb performance remain for root's actual runs to establish.
