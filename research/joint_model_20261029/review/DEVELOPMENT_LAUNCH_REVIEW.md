# Independent development-launch review, 29 October 2026

Verdict: the complete model, registered data, optimized no-cross view, trainer,
and measured batch-16 launch envelope pass review for the three registered
50-epoch internal-development fits. Root must freeze a development-only launch
protocol binding the exact sources and selected batch size. This conclusion
does not authorize the final benchmark refit before completed-development review.

The reviewer ran CPU forward/VJP tests, source inspections, data/metadata audits,
and saved-result checks only. No reviewer GPU run, optimizer update, TEST read,
job change, competitor modification or simulation change occurred.

## Exact scope and pins

- Core: `model/joint_model_v1.py`, SHA256
  `ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4`.
- Trainer: `root/train.py`, SHA256
  `1904191767320ceb738a10be417716d2497941bdeaadea45a0a0c8eb5985671c`.
- Prepared manifest: `data/cache_v1/manifest.json`, SHA256
  `a9a2839b450ee7bf12f90bc269a257bb759b79fdd6e8a93fa8e7eb4c0737e253`.
- Runtime loader: `data/site_dataset.py`, SHA256
  `eb29f0d96909846eb8e85f6c2aaaab89bbcdbb918e198f2efe3fd2af372be9b1`.
- No-cross view: `model/no_cross_view.py`, SHA256
  `0227f5cacbe4dc9a9fbd0bcca56a2f8582cd0d4a2c990b48a054e1a221f2b3e7`.
- Data policy: `82d42d8e597ac385bccadfba720350ff689b50be8e37c8ebcec819a88717f387`.
- Site plan: `bfebc589dc0fb5a2f5a658f488d28dda9fd7b3baa2eb6fb03da76ed7f14c3a11`.

Architecture: width64, three synchronous joint layers, six classes, identical
parameter skeleton for no_cross/leaf_only/structure. Descendant initialization
and post-joint all-ancestor focal readout retain complete native tree support.
Branch overlap is log-age mass times descendant Jaccard specificity. This is a
new complete architecture, not numerical equivalence to the old cavity model.
The stable interval formula fixes the documented positive-duration cancellation
bug; its original finding is preserved and closed in the additive receipt.

## Evidence supporting the launch

`SHARED_SITE_MODEL_REVIEW.json` independently verifies shared multiquery outputs
and full input/parameter VJPs against separate gathers, independent batching,
and cross-window isolation. The author's pinned complete-model qualification
covers 24 checks, including supported edge cases, sparse/scatter parity, and a
foreign-ancestor dependency beyond 161 native hops. This is a dependency witness,
not evidence of learned biological advantage.

`REGISTRY_AND_WEIGHTING_REVIEW.json`, `SITE_PLAN_REVIEW.json`, and
`EXPORTED_OBSERVATIONS_REVIEW.json` independently authenticate all registered
rows, joins, contexts, raw observation packets and separate labels. The stable
haplotype identity is parent/sample_id/haplotype; sample_id alone is not unique.
The complete support is 50,000 targets at 1,245 sites. Fit uses TRAIN000/001
(33,334 targets, 830 sites); entire TRAIN002 is internal development
(16,666 targets, 415 sites). These are original TRAIN chromosomes, not TEST.

`FINAL_NUMERICAL_DATA_REVIEW.json` binds all prepared metadata and archived
sources, current runtime provenance, exact support, and three independent raw
packet-to-prepared recomputations including the 384-target site. The author
verified exact all-tensor/label/target-order equality for a full 830-site epoch
with zero and four workers; the reviewer inspected that source and authenticated
the receipt without repeating the full 9 GB readback. Cache mutation rejection,
LRU eviction, label alignment and focal-only dataset integration passed.

`GPU_QUALIFICATION_REVIEW.json` independently recomputes saved full-model
CPU/GPU differences for 399 targets in all three modes. Worst initial logit
error is 3.1293e-7, full-gradient relative L2 3.5311e-7, and post-AdamW logit
error 8.7917e-7; all are finite and no argmax changes occur. Gradient presence
for this comparison is supported by the pinned runtime harness and saved
boolean; its per-parameter presence arrays were not retained. Post-update
comparison concerns logits, not parameter or long-run optimizer parity.

The focal-only no-cross GPU comparison covers 398 targets. Independently
recomputed gradient relative L2 is 1.34694e-7 and maximum logit error 1.78814e-7.
All 55 saved gradient-presence entries match, with 40 active tensors, and no
argmax changes occur. The optimization retains the complete focal tree and all
of its observations; removed nonfocal trees cannot influence no-cross outputs.

Trainer source review confirms M/(S*N) times summed target loss for uniform
site batches, including the actual final batch size; exact canonical evaluation
sets; finite loss/gradient checks; explicit AdamW/FP32 settings; fixed final
checkpoint policy; failure identities; source/manifest hashes; stage gates;
and full 830-site profile epochs. Runtime mode and the 12 GiB worker cache cap
are passed explicitly. Checkpoint reload and completed result verification will
be audited on saved development outputs before benchmark selection.

## Measured launch envelope

`PROFILE_REVIEW.json` authenticates three complete two-epoch structure profiles
on GPU5. Each presents 66,668 targets with no missing sites, finite progress,
identical initialization, no retained profile weights, no activity errors, and
no foreign compute PID in sampled activity. Each candidate has one cold epoch
and one measured repeat epoch; execution order was 16, 8, 32 sites per batch.

| Sites/batch | Repeat epoch seconds | Targets/second | Peak allocated GB | Peak reserved GB |
| --- | ---: | ---: | ---: | ---: |
| 8 | 5.498 | 6,063 | 1.650 | 8.307 |
| 16 | 4.528 | 7,362 | 3.041 | 14.194 |
| 32 | 4.505 | 7,399 | 5.860 | 21.928 |

The registered smallest-batch-within-90%-of-maximum rule selects **16**. Cold
epochs take 29.57-30.52 seconds. Four persistent workers each have a 12 GiB
immutable-cache cap; parent RSS does not include worker memory. Offline export
and preparation cost 38.47 and 306.36 seconds respectively; prepared cache is
9,011,935,132 bytes. Root remains responsible for current device availability,
source freeze and the isolated 16 GiB output/storage envelope.

These measurements support a feasible launch. One warmed epoch per candidate
in fixed order does not establish statistical speedup. Online losses differ
with update count and were not used for batch selection. Device utilization
mixes cold and warm work and is not a learning-quality measure.

## Scientific and final-fit limits

All 50 epochs, original targets and final checkpoints must be retained for each
mode's result review. Select by final development target accuracy, then lower
cross-entropy for an exact accuracy tie. Verify common initialization, target
order, 2,600 optimizer updates and 1,666,700 target presentations per arm, class
outcomes, finite states and final checkpoint reload before freezing selection.
The final benchmark fit must initialize afresh and train all 50,000 targets for
50 epochs under its own authorized protocol. No new TEST inspection is needed.

Jaccard-weighted branch support is not ancestor identity or confidence. Fixed
log-age scale100 is a prior; full-network age-scale or unary/subdivision
invariance is not claimed. Complete native SNP pools can extend outside the
nominal coordinate domain, as declared and audited. Equal parameter skeletons
do not equalize active capacity. One internal-development chromosome cannot
establish population-level superiority or uncertainty. A mechanism claim still
requires suitable alignment-destruction/capacity controls; a biological claim
requires independent held-out evidence and relevant tract/boundary diagnostics.
