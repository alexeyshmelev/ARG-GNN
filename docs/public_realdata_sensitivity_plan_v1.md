# Public real-data sensitivity plan v1

Status: source-only proposal for root review, registration and implementation.
No condition below has been generated, inferred, fitted or scored. Existing H018
handoffs and biological protocols remain unchanged. Passing this plan can support
a limited, proxy-conditional stability interpretation. It cannot establish true
local ancestry accuracy on the public genomes.

## Evidence, repeat reasons and scope

Evidence anchors: [H018 handoff](../experiments/results/public_realdata_metadata_join_v1/handoff_manifest.json),
[date contract](../experiments/results/public_realdata_metadata_join_v1/age_domain_contract.json),
[H020 sampling config](../configs/biological_paths_execution_pilot_v1.json) and
[decision ledger](../research/experiment_ledger.md). The receipt records source hashes.

H018 froze eight reference subjects/R16 and twelve ancient query subjects/Q24 for
chr22 GRCh37 absolute [20,000,000,22,000,000), plus coherent HS draws 0-7 and the
median diploid-query JS threshold 0.05 nats. No real ARG or accuracy result exists.
The geographic groups do not establish reference purity. Its R16/Q24/M0 roster is
a different condition from H020's R96/Q120/M64, not merely a smaller R96 panel.

| Prior entry | Evidence to reuse and new condition |
| --- | --- |
| H018 |Reuse public_realdata_metadata_join_v1 handoff/endpoint/age tables. Panel, date and observation sensitivities are explicitly unfinished. Do not repeat the metadata download/join. |
| H020 |Reuse completed native source/runtime admission and later sealed selected checkpoints. The new target-shaped cohort and perturbations are new development conditions; no new TEST. Existing biological fits must finish first. |
| H014 |Retain completed Dense seeds 101/202/303 and raw-zero/stable-log distinction. Do not refit it or invent the unspecified paper CNN. |
| H012/H013 |Native ARGMix c2481248 and ARGformer f7545be7 remain main competitors. Current-release and within-ARG protocols retain their reproduction/supervision distinctions. Existing admission is not an R16 runtime result. |
| H006/H017 |Small Q64 age observations and workflow engineering checks do not certify biological dating robustness or real ancestry accuracy. Reuse checks unless the relevant adapter changes. |

Root should register four new local condition families under H018/H020, with the
repeat reasons above, before implementation/execution. S1-S4 below are proposed
family names, not already registered hypothesis IDs. The old H020 R96 scores are
context, not a target-shaped control. An optional paired H020 R16-thinning bridge
may reuse existing R96 outputs, but is not needed to generate the core matrix and
must never be counted as independent confirmation.

## Paired development parents and method freeze

Use four new independent 2 Mb DEVELOPMENT-VAL genetic parents, initially two and
then two more after admission. No new TRAIN or TEST is needed for the fixed-weight
transfer screen. Derive seeds as first 8 SHA256 bytes, big-endian, modulo(2**32-1)+1 from
arggnn|public_realdata_sensitivity_v1|master=20261010|parent={000..003}|
purpose={ancestry,mutations,obs_phase,obs_flip,proxy_blocks}|draw={none,0..7}|
subject={none,synthetic_ID}. Use draw/subject=none for ancestry/mutations and
parent/draw/subject for observation streams shared across methods. Collision-audit
against registered metadata seeds. The seed manifest and all family membership are frozen before generation; do not choose
parents, queries, variants or stress levels by results. Parent is the statistical
unit; SNPs, windows, twelve subjects and eight draws are not independent parents.

For each new parent, first generate one master observation with R96 plus Q24 and
no M64 block: 120 unique haplotypes. Use the fixed source eleven-population events,
census 350, JC69/discrete genome, mutation 1.25e-8 and recombination 1e-8. Each proxy
class has twelve independently generated diploid references: six at each of its
two H018 reference-date prototypes. R16 retains the first frozen synthetic subject
at each prototype, hence two diploids/four haplotypes per class. This repeats ages,
not genomes or reference leaves. Keep the same twelve synthetic query diploids in
both panels. Downsample observed genotypes and re-infer R16's complete 40-leaf ARG;
do not relabel the discarded eighty pure-reference leaves as latent queries.

Generate true sample times once at the twenty metadata means divided by 28.
The target-shaped synthetic query scenario samples Q24 from the declared model
Neo population at those dates. This is an explicit simulation scenario, not a
claim that Denmark_Neolithic subjects have that genetic ancestry. H020's diverse
query results remain separate context; this Neo scenario cannot establish
four-source superiority for real subjects. Preserve raw query identities across
thinning and all observation variants with explicit original-to-child mappings.

Before any simulation, root must qualify population activity and pulse/census
legality against the frozen full demography. The means place synthetic queries
below 259 generations; stress bounds/year conversions also require explicit
source validation. Do not move pulses, census or merge times, clamp ages, substitute
another source population or resample to repair an illegal case. Freeze the
model-zero/cal.BP alignment, provisionally 1950 BP with no automatic 2026 offset,
before using these dates. Fractional generations are retained.

Freeze completed H020 age-D2, plain-D2, matched-GRU and native ARGMix selected
checkpoints, and H014 Dense checkpoints, before inspecting these new parents.
Evaluate each existing seed separately; do not select a seed on this matrix or
real data. They receive no new query-label training or calibration. ARGMix retains
its current-release 15-reference native model and source seed 50, including the
declared nearest-reference policy. R16 must supply the required unique leaves
even under reference-selection augmentation; missing rows are an unavailable
cell, not a reduced denominator.

ARGformer needs a separate protocol arm because its node vocabulary/checkpoint
is within each observed ARG. Keep native MLM 5 epochs/seed 17, contrastive 1 epoch/seed 36, path
split 42 and logical batch 4096; require enough unique pure-reference paths.
On the first new R16 parent, preregister one same-input/same-seed technical repeat
of this protocol to measure fit variation. Median diploid JS > 0.005 nats blocks
an input-only stability attribution; report protocol sensitivity separately.
This is a new-condition technical control, not a repeat of answered H020 fitting.
Preserve the unused three-label classifier and direct C4 readout. Per-variant
reference-label contrastive fits are not fixed-checkpoint inference and are not
wholly unsupervised. A new vocabulary/OOV or real adaptation policy needs root
admission and an explicit reconciliation with H018's no ancestry-target adaptation
rule. Do not silently rename proxy-label fitting as unsupervised or apply an
unqualified simulation node vocabulary. Failure of source 4096 admission is an
unsupported cell; no reference duplication, new batch size or hidden fallback.

## Minimum matrix

All rows use the same master query genomes and truth within a parent. Only the
R96 versus R16 observation subset, declared dates, proxy labels or measured
haplotypes change. True generative dates and demographic events remain fixed.
Every family retains all four declared classes and the same query identities.

| Family/hypothesis | Frozen conditions and pairing | Cases per parent |
| --- | --- | ---: |
| S1: reference count materially affects transfer |Correct-label R96 and nested R16 at mean/28 dates. Same raw variants/query genomes. Counts24 versus4 haplotypes per class; compare after independently inferred whole ARGs. |2 |
| S2: metadata dating conventions materially affect inferred geometry/posteriors |R16, bounds{lower95,mean,upper95} x years{25,28,29,30}. Apply BP/g to all observed tip ages and rerun inference/age features. Mean/28 is S1's R16 control. These are stress cases, not a calibrated dating posterior. |11 additional |
| S3: proxy assignment/impurity/imbalance materially determine paintings |On R16 mean/28, swap both haplotypes of one frozen Ana/CHG subject pair, then a second cell also swaps one WHG/EHG subject pair. Exactly 25% and 50% of reference subjects are misassigned, with four observed reference haplotypes/class preserved. Select subjects by frozen IDs, never by truth/score. Also copy four seeded 100 kb blocks (20% of 2 Mb) from a next-class donor into one proxy diploid/class, keeping observed labels/dates, as a mosaic-contamination diagnostic; and replace one EHG diploid by another independent Ana master-panel diploid, giving haplotype counts [6,4,4,2]. These isolate observed proxy mismatch/imbalance without changing source pulses. Chronogeographic input remains provenance; proxy_class is the downstream assignment. |4 |
| S4: genotype/phase uncertainty changes posterior stability |R16 mean/28: phase-only Poisson switches 0.1 per cM per diploid; genotype-only allele flips 0.001 per called allele; and combined stress switches 0.5 per cM plus flips 0.001. Eight fixed draws0-7 per cell, using independent purpose seeds shared across methods. |24 additional |

Total 41 observation variants per parent, 82 in the first stage and 164 if four
parents are admitted. S3 may reuse the same inferred ARG only if the qualification
proves that the changed proxy_class is absent from Relate inputs; otherwise the
full inference chain is rerun and recorded. The proxy-mosaic cell is an observation-contamination stress, not a new historical
admixture model or proof of biological proxy impurity. Donor blocks/subjects are
fixed without query ancestry truth. Imbalance admission must retain unique actual
references/landmarks; an unsupported consumer is reported, never repaired by
reference duplication, omitted classes or a changed neural interface. The composition
replacement changes its age mixture as well; label it a joint panel-composition
stress, not an isolated count effect.
Preserve all full trees and retained
centers; no tree thinning, cohort-specific SNP selection or compressed operator
substitute is permitted.

For S4, phase switches toggle the two haplotypes from each event onward along the
same individual, using the frozen synthetic recombination map. Never redraw phase
independently at each SNP. Genotype flips and switch seeds are independent of
ancestry labels. Reuse each draw's genotype-flip mask in genotype-only and combined
cells; thin the 0.5/cM event process with probability 0.2 for the 0.1/cM phase cell. Keep the same SNP/QC mask in all paired runs, record monomorphic
attrition, and hide the phase-permutation trace in the external scoring capability.
Diploid truth dosage is unchanged; any haplotype-level score must use the explicit
hidden permutation correspondence. These synthetic stresses are not GLIMPSE HS
posterior draws or empirically calibrated ancient error rates.

The future public endpoint remains exactly phased-GT versus coherent HS 0-7,
NMAIN 15, one posterior iteration per individual across the entire ligated region.
HS can change genotype as well as phase. None of these simulated noise operations
changes that frozen real endpoint or fabricates GP/HS/INFO fields.

## Endpoints and support

Before truth is opened, seal every prediction required for a paired parent/family,
its method/checkpoint/protocol identity and the observed-support intersection.
Use original region coordinates/2 Mb, retained 100 bp center bins clipped to inferred
coverage, 250 kb tiles and complete physical-truth intersections. Primary paired
support is the common observed bin set across the compared conditions/methods,
chosen from observation metadata only. Report each original support and the paired
denominator. A missing prediction on admitted support fails the cell; never drop
its rows. Require at least 90% of the original 2 Mb and 1,000 retained observed SNPs
for target-shaped admission, without tuning the region/QC to rescue failures.
Freeze a common biallelic A/C/G/T transversion mask without target-cohort AF.
The synthetic study does not manufacture public RAF/GP/HS/INFO fields or claim
that it reproduces imputation ascertainment; real H018 QC remains a separate gate.

Simulation primary endpoint is paired change in exact C4 Brier averaged equally
over independent parents and their 24 query haplotypes. The co-primary safeguard is
diploid C4 dosage MSE. Report per-parent/per-subject results, ancestry error, raw
NLL with positive-truth zero-probability mass explicitly infinite, ECE 15, confusion,
short tracts <= 10 kb and boundaries at 1 kb tolerance. AP native probabilities collapse
to C4 before scoring; native ARGMix/ARGformer/candidate are C4 directly. No floor,
renormalization or calibration is added. Any stable-log NLL is separately labeled.

The common observable endpoint is diploid JS: average each subject's two haplotype
C4 vectors, integrate natural-log JS against the baseline over identical physical
bins, divide by represented subject-bp, average across eight S4 draws, and take
the median of twelve subjects. Use 0log0=0. S1-S3 have one comparison per condition.
Report each method/checkpoint separately. ARGformer protocol sensitivity is labeled
separately from fixed-checkpoint sensitivity. Four parents give development paired
variation only; no SNP-bootstrap, independent superiority or powered holdout claim.

## Promotion and resources

First require new R16/Q24/source-role runtime and coordinate qualification, faithful
full-field bank roundtrips, unchanged model parameter interfaces, exact class/query
joins, zero AP leakage and source 4096 admission. Existing R96 hard-coded producers
are not executable R16 admissions. Root must version/freeze the new adapter/config,
all inputs, method bindings and scientific claims before the first run.

Reject a trivially stable but uninformative model first: freeze uniform C4,
observed-reference class-prior and H020 TRAIN-Neo exact-span constant priors before
new parents are generated, using TRAIN labels only for that last prior. On R16
mean/28, require mean Brier at least 0.02 below both the uniform/reference-prior and
TRAIN-Neo prior, with improvement on every parent. No evaluation truth selects
these baselines or checkpoints. These are proposed development adequacy margins,
not powered noninferiority or real-accuracy certificates.

Proposed moderate-condition development gate, to be reviewed/frozen before results:
for S1, all S2 cells, S3 except the 50% swap, and the first two S4 cells, mean paired
Brier increase <= 0.05, no parent increase > 0.10, mean dosage-MSE increase <= 0.05 and
median diploid JS <= 0.05 nats. Report failures and the worst parent. The 50% swap and
combined fivefold-switch stress diagnose limits; they are not selected away or
required to show accuracy invariance to arbitrary labels. Failure of a moderate
gate blocks an unqualified robust-transfer interpretation. Passing does not prove
real reference purity, correct dates, phase or ancestry.

Finish two screening parents first; a frozen failure stops promotion while all
partial/negative records remain. Add two independent parents only if runtime,
moderate gates and storage admission permit. The final four-parent screen remains
development; a powered independent simulation confirmation needs a separate freeze.
No real checkpoint/threshold tuning follows this screen.

Budget is a bounded planning reservation, not measured R16 performance:

| Item | First 2 parents | Full 4 parents |
| --- | ---: | ---: |
| Target observations/inference cases |<=82 |<=164 |
| Native ARGformer protocol fits, including one technical repeat |<=83 |<=165 |
| Native fit reserve at 2 GPU-hours/case |166 GPU-hours |330 GPU-hours |
| Fixed-weight inference/CPU-bank overhead reserve |10 GPU-hours +200 core-hours |20 GPU-hours +400 core-hours |
| Incremental storage reservation |130 GB |240 GB |

Full storage arithmetic: four R96 control banks <= 2 GB each plus 160 R16 banks <= 0.25 GB
each = 48 GB; native fit artifacts <= 0.75 GB/case = 123.75 GB; observed products/corpora 16.4 GB;
probabilities/seals/metrics 25 GB; transient/rejected working space 20 GB, rounded
up to240 GB. These are admission caps. The first full-field R16/R96 preparation and
native fit must measure actual size/time; if a cap fails, stop and propose a new
budget before expanding. Do not satisfy it by dropping matrices, native checkpoint
evidence, failure records or predictions. This plan requires no new neural fits using query labels,
installers or unchanged source qualification suite repetition.

Before each stage root supplies a fresh complete project-root storage snapshot
and accounts for ongoing writes. Require used+planned_increment+50 GB concurrent
reserve <= 450 GB, with the 500 GB hard maximum. Thus first-stage admission needs
used <= 270 GB and full retained matrix needs used <= 160 GB. The older H020 66.238 GB
snapshot is not present headroom. At most two free eligible physical GPUs 2-5 may
run, using the existing launcher, UUID lease, exact AI Auto Research Pipeline Test
title and actual NVTOP evidence; GPUs 0/1/6 remain excluded. With two admitted GPUs,
reserve roughly 4-5 days for screening and 8-9 days total; one GPU roughly doubles
native-fit wall time. Stop at the reserved budget and record incomplete cells,
rather than deleting prior evidence or changing scientific settings.

Only after these gates may root consider the unchanged H018 real endpoint:
actual region/QC/phased GT/HS/map/polarization and publisher-integrity limitations
still need admission, and the 12-subject median JS must be <= 0.05 nats for the stated
observation-stability claim. Every individual/method/support is reported even on
failure. Published AP remains excluded from input/reference/QC/tuning and, if later
consulted, enters an external comparator namespace only after predictions are
sealed. Concordance, low JS or a passed simulation stress screen never establishes
unverifiable real ancestry accuracy.
