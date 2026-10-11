# H026 reference-census innovation: discovery design v1

H026 tests whether observed reference context improves the frozen selected H025
smoother beyond its Present64 accuracy 0.859397 and NLL 0.389136. It is a diagnostic
reference-kernel experiment, **not a new general GNN or a novelty claim**. H025's
sample, genuine and sham trace arms were nearly identical. Its common residual
head had a direct affine baseline-logit path, so those gains do not establish
ancestral persistence. H026 removes that path.

## Biological and observation contract

All actual Q120 samples are younger than census 350: 64 at age 0, 16 at 135, 8 at 155,
16 at 170 and 16 at 200 generations. Observed pure-reference labels are sampling
cohort/source-class observations. The panel has the following haplotype counts.

| Family | ANA | CHG | WHG | EHG | Total |
|---|---:|---:|---:|---:|---:|
| Reference age <350 |24|16|22|22|84|
| Reference age >350 |0|8|2|2|12|

There are no references exactly at 350. Reference ages range 176–473. The source
simulation's four pure populations persist through 350, with source mergers at 600
and 800 and no migration. Thus a true younger pure reference has a defensible
class interpretation at the true census. Inferred topology/dates can violate the
corresponding observed-cut grouping, so the model uses **soft cohort affinity**,
not hard census assignments or an assertion that an inferred clade is pure.

A reference sampled before the census in calendar history (age>350) has no
sample-to-ancestor path crossing the later 350 date. Its observed cohort label
stays at its actual sampling node and age in a separate older-reference channel.
Do not invent its descendant, census label, or age350 cut. No true population,
ancestor, census flag, query label or migration annotation enters context.

Frozen TRAIN skip predictions remain in-sample, not out-of-fold.

The complete independently inferred tsinfer/tsdate sample-ancestor union is
retained, including unlabeled mixed64 context. MRCA times and metric paths are
computed from those complete trees. Cuts are analytic points on original edges,
not deletion or contraction of trees. The frozen native ARGMix Relate emitter
and its selected H025 smoother remain unchanged, so this is explicitly a hybrid
observation experiment, not a native ARGMix reproduction.

## Exact fixed FP32 context and four fitted arms

Let `T(q,r)` be the observed MRCA time and `a_q,a_r` observed sampling ages.
For each age family `g` and reference class `c`, let `n[g,c]` be its observed
haplotype count. Use scores `-distance/scale - log(n[g,class_r])`, stable FP32
softmax across references in that family, then sum the weights by observed class.
This balances represented classes without inventing an absent older ANA source.
The resulting context has four younger and four older channels. Fixed scales
are 350 generations for younger references and 1400 for older references in ALL
arms. These are declared discovery settings, not validated biological constants.

| Arm | Younger-reference distance/score geometry | Older channel |
|---|---|---|
| `native_reference` | `2*T-a_q-a_r` | Common dated cohort-affinity field |
| `age_free_reference` | `2*T` | Same common field |
| `census_reference` | `2*max(T-350,0)` | Same common field |
| `census_sham` | Census field, but query-context association deranged | Same field with that query association deranged |

The common older field uses score `-(2*T-a_r)/1400-log(n[old,class_r])`.
It is an observed-cohort affinity feature, not a census posterior. All four arms
receive identical true reference classes, ages and whole-tree geometry. The
sham uses a fixed parent-wide derangement within exact observed query-age
**diploid blocks**, preserving both haplotypes and their order. It changes only
which query receives a contextual vector. Frozen query logits/identities and
reference label-to-node assignments remain unchanged.

Two cancellations determine the interpretation. For older references, moving
the query readout from its sample to350 changes native distance by `a_q-350`,
a reference-independent constant. Row normalization cancels it exactly. There
is therefore no new older-reference census information in this kernel. For
younger references, the native score is, up to a query-only constant,
`-2*T/350+a_r/350-log(n)`. The age-free arm removes donor-age weighting. Whenever
ALL younger-reference MRCA times are at least 350, the census score equals the age-free
score plus 2, so their normalized fields are identical. Census versus age-free
isolates the observed pre350 MRCA saturation; census versus native alone would
confound it with removal of donor-age weighting.

At each original physical readout, use its containing-tree context. Preserve
all original 100 bp readouts, coverage, clipping and span loss. Do not blur these
context changes by interpolation between tree centers. The frozen smoother's
full-grid skip already supplies genomic context and permits ancestry changes
inside a long inferred-tree hold.

## Context-only correction and matched experiment

For each position and actual query-age group, subtract that group's mean
8-channel context. Use only a shared **bias-free 8→4 linear head**, initialized at
zero, on the centered context. Add it to frozen selected-smoother log probabilities.
There is no baseline-logit head input, class bias, age embedding, temperature,
query-ID embedding or free learned pointwise recalibrator. The per-class logit
correction sums to zero within Present64 and every ancient age group.

This centering is transductive unlabeled query context. It uses the FULL actual
Q roster of each observed age at each coordinate, not targets, scored subsets or
minibatch means. Predictions depend on that roster composition, which must stay
fixed in all arms. Constant reference-marginal subtraction would cancel under
this centering and need not be implemented twice.

Run seed 101, 10 epochs and 160 updates per fitted arm, with identical head
initialization, parameters, optimizer and selection budget. Keep the existing
TRAIN supervision, original parent split and minimum equal-parent VAL Present64
NLL selection rule. Frozen smoother is the reference, not another fitted arm.
No class-specific change is selected from current VAL confusions.

Interpretation is predeclared: `age_free-native` tests donor-age weighting;
`census-age_free` tests the350 saturation; `census-sham` tests query-specific
reference association; every arm versus the frozen smoother tests practical
incremental value. A gain only over the plain emitter is insufficient. Preserve
per-parent accuracy/NLL, all120/ancient56, class recall/confusion, calibration,
short-tract measures and boundary F1 within 1 kb. H025's VAL boundary F1 values were
only 0.02349/0.05021, so aggregate accuracy cannot certify correct tract changes.
These repeatedly used four development parents provide pilot evidence, not
independent confirmation.

## Current novelty assessment and deferred ordered mechanism

The kernel, class-weighted reference votes, transductive centering and residual
head are established mechanisms. Zhou et al.'s *Learning with Local and Global
Consistency* already spreads labels through normalized graph/diffusion kernels.
The inspected current ARGMix source even supplies `TmrcaKernelWeighted`, with
exponential TMRCA-weighted reference one-hot votes, besides TMRCA attention and a
bidirectional genomic GRU. ARGformer uses shared ancestral-node path tokens and
reference retrieval. Ancestral Paths uses ancestor-shell cohort proportions and
node ages. H026 is a biologically specified discovery comparison within this
existing methodological neighborhood. It has a pairwise-MRCA representation;
it does not demonstrate a new learned whole-tree propagation layer.

An explicitly **unimplemented** follow-up is reference-driven, edit-resolved
innovation: attribute a changing whole-tree reference response to actual
ancestral edge edits and cut-source relocation, then retain those signed,
edge-indexed responses over physical genomic distance with partial qualified
correspondence. Shared edges would index observed response calculations, never
assert preserved ancestry. The resolvent identity used for attribution is known.
The unresolved contribution is whether edit-resolved interactions retain useful
information beyond static reference context plus an equally tuned generic
filter of context differences. Temporal GNNs and GraphSSM already process events,
structure and persistent-node memory. A useful H026 result would justify this
next question; it would not establish its novelty or accuracy.

The accompanying packet contains checked source references, the cancellation
algebra and the current classification. No fitting or GPU run was performed by
the science author.
