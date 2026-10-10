# AncestralPaths biological integration v1: corrected audit

Status: **no concrete static integration blockers** for runner SHA256
`826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056`.
Config SHA is `2c87400c94092b91a38b85d8915cf12f385bf26b37ced98c24df45ff91737941`;
doc SHA is `c5ab47089cd9b348afed4f4d80bc9ad1f4d5751248bbb604597f99803503bd82`.
The initial audit remains preserved; this review supersedes its two blockers for
these exact corrected bytes.

## Resolved joins and physical support

Runner137-249 now consumes exact frozen producer kind/status/fields and binds
post-size tree, all280 sample/node joins, source and derived roster hashes
separately, completion-receipt equality and commands. Every derived row preserves
all observed attributes through the specified ID-field transformation.

Runner265-365 requires local length=coverage_right-coverage_left, shifts actual
tree interval bounds by the declared offset to match physical centers, and adds
the offset to local sites before verifying the physical site join and later
TRAIN truth lookup. It retains exactly declared original center indexes.
Retained bins are contiguous and their outer edges use
`max(original_left,coverage_left)` and `min(original_right,coverage_right)`.
Partial edge bins with unretained centers are excluded, and no tails are filled.

## Metric weighting and disclosure

Intersections494-509 use exact positive physical bin/truth overlap and require
span sum to equal represented support length. Scoring545-675 weights confusion,
accuracy,NLL,Brier and calibration by those spans. Denominator is
`sum(spans)*selected_query_count`; dosage denominator is
`scored_query_bp/2*4` after complete ordered diploid verification.

Partial-support accuracy is explicitly labeled conditional in `score_scope`.
The metric record separately reports exact `denominator_query_bp`, full original
parent query-bp,2Mb normalization,coverage,physical coverage fraction,represented
support/fraction,indexes,and covered span omitted for lack of a retained center.
It does not silently normalize partial confusion to2Mb. Predicted and visible
truth tracts are clipped to represented support; support endpoints are excluded
from switches. Original full truth length governs<=10kb selection, while visible
correct overlap and support censoring are retained.

All17 pinned author files still match. Native TRAIN stride/budgets/path caps/
replacement, Dense model,5epochs,and final checkpoint remain unchanged.
VAL probabilities are written and hashed before its truth capability; no VAL
checkpoint/stopping/calibration selection exists. C4 sums probabilities before
argmax, and no probability floor hides infinite-NLL span. Exact preserved source
rosters keep the previously verified96pure/64mixed/120query role alignment and
null query classes; native ancient focal self-inclusion remains disclosed.

This was source/config/doc review and file hashing only. No actual truth,
inferred trees,caches,or model outputs were opened; no runner/author execution,
fit,install,GPU,Git,or runner edits occurred. Runtime manifest admission,feature
cache equivalence,quota filling,optimizer fit and metric execution remain root
checks when the completed Relate manifest is ready. No candidate bridge work was
started.
