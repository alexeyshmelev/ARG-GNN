# Hash-pinned graph-GP training/validation runner

The CPU runner in src/arggnn/train_graph_gp.py fits and reports all three
independent graph-OU GP variants on the existing Chickenpox protocol.
It is a validation-only pilot. It has no final-test scoring option,
no variant selection, and no model/seed/budget overrides on the CLI.

The GP family and its limitations are documented in
[graph_gp.md](graph_gp.md). This runner does not strengthen its
family-level attribution into an exact Nikitin et al. reproduction claim.

## Frozen study schema

The root agent owns creation/freezing of a separate study JSON. Required
fields are:

~~~json
{
  "protocol_config": "configs/chickenpox_common_input_pilot_v1.json",
  "protocol_config_sha256": "<SHA256 of exact common-config bytes>",
  "source_archive": "data/raw/chickenpox/hungarian-chickenpox-cases.zip",
  "source_archive_sha256": "dc306b1845c86616d2a72c202deeefac8fd1fcf0a34b3067cf6721e377d9da44",
  "source_manifest": "data/raw/chickenpox/source_manifest.json",
  "source_manifest_sha256": "<SHA256 of exact source-manifest bytes>",
  "variants": ["nonseparable", "separable", "temporal"],
  "seed": 101,
  "restarts": 3,
  "maxiter": 100
}
~~~

An optional nonempty descriptive stage string is accepted and preserved
in results. Root's frozen stage graph_gp_common_input_train_validation_pilot_v1
uses this field. All paths, including inputs and outputs, must resolve inside
the project.
The CLI requires the expected SHA256 of the study JSON itself. Every
hash is checked before creating an output directory or fitting a model.

The common protocol must retain history 52, train_end 365,
validation_end 443, observations 522, the pinned source archive hash,
and final_test_evaluated=false. The source manifest must match the
archive hash/byte length, 522 observations, CC BY 4.0 license and
loaded edge-ID county order. The archive is independently checked
against its known benchmark hash by the existing bounded source loader.

The variants, order and optimizer fields must match the exact values
above. The runner deliberately refuses a subset, reordered variants,
different seed, fewer/more starts or a changed iteration budget.
Any future budget study requires a separately reviewed runner/protocol.

## Invocation

On anogena inside /workspace-SR003.nfs2/ARG-GNN:

~~~bash
source scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$ARGGNN_PYTHON" -m arggnn.train_graph_gp \
  --config configs/<frozen-gp-study>.json \
  --expected-config-sha256 <frozen-gp-study-sha256> \
  --output results/<fresh-versioned-gp-pilot>
~~~

The only arguments are --config, --expected-config-sha256 and --output.
An explicitly empty CUDA_VISIBLE_DEVICES is required. The runner sets
the project research process title, applies repository storage guards
and limits numerical thread pools to one. It uses NumPy/SciPy CPU code
and never creates a CUDA context. It refuses an existing output
directory rather than overwriting or resuming a run.

The root agent must launch actual fitting. Implementer qualification
tests use synthetic stubs and do not run an actual GP training job or
score public validation/test labels.

## Data access and comparison contract

The archive loader verifies the complete 522-week source as a source
integrity operation. After verification, the runner creates a restricted
data object with every raw count from index 443 onward replaced by NaN.
Only this restricted object is passed downstream. Source verification
therefore reads the archive containing the held-out segment, but model
fitting and forecasting do not receive its final-test values.

Each variant fits on all 365 raw training observations per county,
using training-only scaling and full stationary marginal likelihood.
Those observations include the first 52 warmup weeks. Supervised
forecasters under the common protocol have 313 training forecast
targets, indices [52,365). Both have access to the same 365 raw training
observations. The differing fitting objectives and observation counts
must be disclosed in comparisons.

For each of the 78 validation targets [365,443), the runner supplies
exactly counts[t−52:t]. Earlier observed validation weeks may enter a
later validation forecast's history. Models keep their scaler and
hyperparameters fixed, reset to a stationary prior on each call and
predict one week beyond the history. Final-test target indices are
never requested. Raw case predictions have no clipping.

All three variants are reported. Validation does not choose a GP
variant, optimizer start, covariance bound or search budget. Start
selection uses training marginal likelihood inside GraphOUForecaster.
Optimizer termination failures, incomplete convergence and bound
contacts must be retained and assessed when interpreting the run.

## Saved artifacts

Each fresh run directory contains:

- study_config.json, common_protocol_config.json and source_manifest.json:
  byte-for-byte snapshots of the pinned inputs.
- progress.json: atomic progress updates, completed-model records and,
  on failure, exception type/message with status=failed.
- graph_gp_<variant>_fit.json: complete metadata from the fit, including
  every optimizer attempt/status/message, selected candidate and bound
  contacts. This is written immediately after fitting, before scoring.
- graph_gp_<variant>_validation_predictions.npz: raw prediction [78,20],
  validation target indices/dates and county names in edge-ID order.
  No ground-truth or final-test target array is saved in this file.
- graph_gp_<variant>.json: fit metadata, raw-case aggregate MAE/RMSE,
  per-county MAE/RMSE, 78 per-week metrics and exact history ranges,
  prediction-file path/SHA256 and fit/forecast wall times.
- results.json: complete three-variant results, written only after
  successful completion.

Run-level metadata additionally records the current Git commit,
SHA256 hashes of the runner, GP module, Chickenpox loader and resource
guard, NumPy/SciPy/Python versions, effective numerical thread-pool
information, all input pins, county order, training observation counts,
fixed validation indices/dates, and final_test_evaluated=false.
Stored fit metadata preserves the distinction between a finite selected
candidate and successful optimizer convergence.

A failed run retains available evidence in its fresh directory. It
cannot be restarted into that same directory. The root agent should
investigate the recorded failure, freeze any required revised protocol,
and use a new versioned directory for a subsequent run.

## Verification

tests/test_train_graph_gp.py exercises the runner with synthetic fitted
objects and synthetic counts, including NaN sentinels throughout the
final-test segment. It verifies all three variants, the exact 78×52
window contract, the 365/52/313 training-count disclosure, SHA256 pins,
all optimizer-status preservation, per-week/per-county metrics,
prediction-file hashes and contents, failed-run records, repository-only
paths, CPU-mask enforcement and fresh-output protection. The CLI is
checked to expose no final-test, device, model or optimizer override.

On 10 October 2026, the finalized 23 runner tests and 27 GP numerical
tests passed together (50 tests in 3.44 seconds), with CPU library threads
set to one. The GP module's separate numerical
oracle tests remain the evidence for exact covariance, likelihood,
gradients and conditional prediction. Runner test metrics are synthetic
pipeline checks and are not empirical Chickenpox accuracy results.

