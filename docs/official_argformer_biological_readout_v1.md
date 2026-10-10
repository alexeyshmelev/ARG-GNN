# Native ARGformer external biological VAL readout v1

This source-only adapter scores the completed native ARGformer pilot's direct
C4 downstream readout on the same exact physical support as AP/candidate scoring.
Native class order is [ANA, CHG, WHG, EHG], corresponding to external class indexes
0/1/2/3. It never applies a native-path collapse. Original FP32 probabilities and
their zeros are preserved without a floor, clipping, renormalization or calibration.

The native fitting regime is within each observed ARG: released MLM pretraining
followed by pure-reference contrastive training, with independent vocabulary and
checkpoints per parent. The local downstream adapter uses the trained author mean
embedding head and class mean exp cosine readout at the existing temperature0.3;
the unused native phantom classifier is not an ancestry readout. Native supervision
differs from candidate/AP supervision. Common external metrics establish a common
scoring domain only; this is not an original paper-task reproduction, identical
supervision comparison, or final holdout evaluation.

Root's four-parent fit sequence has declared completion path
experiments/results/official_argformer_biological_fitting_v1_sequence/completion.json.
This document does not assert that those fits have completed. The actual full4096
memory probe supplied by root passed on TRAIN0 only; this adapter requires no
per-parent memory probe and makes no additional memory-probe claim.

## Frozen inputs and completed fits

Review/freeze this script and its policy first. After all four fits complete,
root writes a separate inputs JSON and records its SHA256 before scoring:

    {
      "kind": "official_argformer_biological_readout_inputs_v1",
      "observed_manifest": {
        "path": "data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json",
        "sha256": "<root-reviewed completed observed manifest SHA256>"
      },
      "parents": {
        "<each exact parent ID from the policy>": {
          "prepared_manifest": {
            "path": "<canonical native prepared directory>/prepared_manifest.json",
            "sha256": "<root-reviewed SHA256>"
          },
          "fit_receipt": {
            "path": "<canonical completed native fit directory>/receipt.json",
            "sha256": "<root-reviewed SHA256>"
          }
        }
      }
    }

The parents object must contain exactly TRAIN000, TRAIN001, VAL000 and VAL001 using
their full policy IDs. No missing/pending placeholder is accepted at runtime.
This format lets fit receipts remain immutable while the readout policy is frozen
before their values exist.

The adapter verifies all four observed manifests and completed fit receipts:
passed actual fit status, native source/config/prepared hash identities, genuine
changed-state optimizer updates, MLM5/contrastive1 completed epochs, full_batch
execution, trained-encoder initialization, native direct-C4 readout identity and
unused classifier. Phase receipt files are also bound to the main receipt.
Checkpoint and optimizer fingerprints are provenance fields; this adapter loads
no models or Torch checkpoint. Only the two VAL predictions are scored.

The frozen AP observed-contract helper checks completed Relate source/provenance,
all280 explicit sample-to-inferred-node identities, source and derived rosters,
fixed100bp centers, original positions, local offset, retained indexes and clipped
coverage. Native prepared roster/parent/coordinate joins must match those exact
observations. Subject IDs, diploid haplotypes, raw truth query-node IDs, inferred
query-node IDs, query/reference path joins, reference classes and class order are
checked and retained in the new sealed scoring exports.

## Sealing, raw metrics and auxiliary log-NLL

The runtime first loads only existing native prediction NPZs, checks their
receipt-bound file and tensor hashes, and verifies every grid/query/reference
join. Shape must be [retained positions,120,4], with finite FP32 logits and raw
FP32 probabilities. The native probabilities must normalize within1e-6; they are
never repaired. Each new export preserves raw native values and adds explicit
original physical bin edges, coverage and subject/query metadata.

Both VAL exports, their individual seals and an aggregate two-export seal are
written before the external target manifest or truth arrays are opened.
External truth must cover the complete original [0,2000000] domain with contiguous
positive spans and match the same ordered120 raw query-node IDs. Scoring uses every
exact retained-center clipped-bin/full-truth intersection; uncovered tails and
partial edge bins without retained centers are excluded. Coordinates retain
original physical bp/2000000 units.

Primary present-day64, secondary ancient56 and all120 are reported. The unchanged
frozen full-span C4 scorer provides raw NLL, Brier, descriptive15-bin ECE, accuracy,
confusion, per-class metrics, diploid dosage, tract and boundary metrics using exact
query-bp weights and denominators. If positive truth mass intersects raw probability
zero, raw nll is null with nll_is_infinite=true and infinite_nll_query_bp retained.
This explicitly represents infinite raw NLL without invalid JSON.

A separately labeled auxiliary C4 log-NLL uses float64 shifted logsumexp of the
unchanged native FP32 C4 logits and takes true-class logp directly over the same
intersections. Its exponentiated probabilities must agree with native raw probabilities
within1e-6. No floor, fitted temperature or calibration is introduced. Underflow of
exp(logp) does not invalidate finite logp. Numerical error, zero counts and argmax
disagreements are disclosed, while all accuracy/Brier/ECE/tract metrics use raw
probabilities. Raw and auxiliary denominators must be exactly identical.

## Root execution

From /workspace-SR003.nfs2/ARG-GNN, source scripts/project_env.sh and use the existing
CPU interpreter. Default mode performs binding/observed-schema preflight only:
it opens no prediction arrays, external truth manifest, truth arrays or score files.

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python scripts/score_official_argformer_biological_v1.py --expected-config-sha256 <reviewed-policy-sha256> --bindings <root-inputs.json> --expected-bindings-sha256 <reviewed-inputs-sha256>

Actual scoring additionally requires --execute and a fresh separate output:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python scripts/score_official_argformer_biological_v1.py --expected-config-sha256 <reviewed-policy-sha256> --bindings <root-inputs.json> --expected-bindings-sha256 <reviewed-inputs-sha256> --execute --output experiments/results/official_argformer_biological_readout_v1

Root retains outer stdout/stderr/exit status. The adapter creates exclusive new
exports/seals/metrics/receipt files and preserves a failed stage. Original native
outputs, fit receipts and prepared inputs are reverified after successful scoring
and are never overwritten. No inference, training, installation or GPU work occurs.
The preparing agent performed source-only inspection, AST parsing and source/config
hash checks without trained-output, VAL truth or VAL score-array access.
