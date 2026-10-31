# Observed census-lineage memory v1

One three-arm biological experiment reuses the current whole-tree Census model.
It adds no encoder, decoder, optimizer, loss or learned parameters. This is a
domain-specific retention variant, with no novelty or accuracy claim.

The hypothesis is that observed changes in the edge crossing census 350 identify
useful ancestry-memory resets, including cases where query-clade membership
stays unchanged. Compare current Jaccard retention, genuine edge changes and
matched sham changes on the existing same-tsinfer/tsdate observations.

prepare_lineage_memory_v1(observation, expected_certificate_sha256,
seed=seed, experiment_config_sha256=config_hash) accepts the existing
CertifiedCensusTsinferObservationV1. From the consumer loader, pass
ConsumerCertificateV1.outer. The caller must independently pin the exact
configs/census_lineage_memory_v1.json bytes. The adapter checks those bytes,
the configured seed, and the original full inner and canonical outer
fingerprints without invoking AST capsules. Complete parent-wide native maps
come from certificate provenance. IDs are fixed joins, never learned features
or ancestry labels.

A key is the actual (lower_native_node, upper_native_node) edge with
lower_time <=350 <upper_time, matching the original cut convention. Missing maps
or query joins raise. Dynamics leaves at/older than350 raise under the existing
require_younger policy. Older pure references remain allowed by the original
model. A root reached without a crossing edge yields None/no_crossing_edge.
If either boundary endpoint has no key, genuine/sham retention is 1. No root or
lineage identity is invented, and no query or grid point is dropped.

The numerical bank is unchanged for current_jaccard. For genuine_edge/sham_edge,
dataclasses.replace changes only boundary_rho: a change uses exp(-1), while no
change/unknown uses 1. The original rho^eta action therefore gives
exp(-eta*edge_changed) in both existing genomic scans. Within-clade exchange,
physical-span integration, gates and the C4 emissions head remain unchanged.

Every arm, including current_jaccard, receives a fresh inner and outer
certificate. The inner certificate fingerprints the complete derived bank,
including boundary_rho, with copied provenance extended by source/config
path-hash bindings, original inner/outer hashes, arm, seed, census and retention,
missing-key and sham policies. The unchanged full inner validator checks the
fresh fingerprint. The outer certificate uses the existing canonical schema
and binds that new inner hash and extended lineage. The original observation,
bank and provenance remain unchanged; its old hashes cannot certify a derived
arm. certificate_for_arm(arm) returns the selected fresh outer certificate.
bank_for_arm(arm) is only a convenience accessor for its certified inner bank.

Sham indicators permute only known-edge events within each parent/query/tile and
fixed right-tree physical-span bin: [0,100), [100,1000), [1000,10000),
[10000,infinity) bp. Exact counts are checked in every stratum, thereby preserving
every parent/query/span-bin count. Local seeded RNG leaves model RNG untouched.
Degenerate strata remain unchanged. summary() reports eligibility, change counts,
actual reassignment and fresh outer hashes by arm. Cross-tile memory remains
reset as in the original fit.

The config copies the existing model, seeds 101/202/303, and 30 epochs/480 steps
recipe. Root's predictive driver should share initialization, optimizer,
TRAIN-only actual120 loss, unlabeled mixed64 context, full grid, serialization
and VAL selection. Genuine-edge NLL must beat both Jaccard and sham; membership-
stable but edge-changed boundaries are the mechanism-specific subgroup.

The adapter supplies certified observations, not a trainer. Prepare it once per
tile and matched seed, select certificate_for_arm(arm), then wrap that outer
certificate in the existing ConsumerCertificateV1. Pass the selected fresh
outer certificate_sha256 as expected_hash to the existing validator, forward,
train_step_v1 and raw_prediction_v1. Use control="proposed" for all three
experimental arm labels, so the original model/action and optimizer/loss/scoring
functions are reused through explicit calls in the future root-owned driver.
Do not pass a changed bank under the original observation's expected hash.

Pinned modules are unchanged. No monkeypatch, source AST compilation, mock
package, numerical qualification or resource tooling is included. No actual
data, model or GPU entry point is provided.

Matching inferred edge IDs are an observed continuity proxy. They do not prove
true ancestral-lineage continuity.
