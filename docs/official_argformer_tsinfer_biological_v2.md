# Native ARGformer on observed tsinfer/tsdate development data v2

This is an inactive source and metadata bridge for genuine native ARGformer. It
uses the current four-parent European development family and the separately
completed current tsinfer/tsdate heterochronous observation condition. It is not
a reproduction of the paper's numerical task. The original Relate fit/readout,
author archive, dependency recipes, runtime guards and GPU pool allowlist remain
unchanged. No prepared corpus, native model, fit or new prediction is produced by
this source handoff.

## Source authority and observation lineage

Native model authority is AI-sandbox/ARGformer commit
f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596. All 60 archived author files retain tree
SHA256 2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b.
The archive SHA256 is
44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c.
The source-only bridge imports standard-library definitions, verifies the frozen
archive and project sources, and compiles genuine scientific functions in an
isolated namespace. It never imports Torch, Composer, tskit or the author model
during default inspection. It does not vendor or edit author files.

The existing native fit authority is
src/arggnn/train_official_argformer_biological.py SHA256
edc290b9fbce093f4b844f7d07c4963c23bf24257f43c7d502c43a2753170b94.
The pure full-corpus retrieval kernel authority is
scripts/score_official_argformer_retrieval_biological_v1.py SHA256
2f4fc18efa5e83e7c1c3c72ce01cd3783fa03c83dc62225a52b99733c0d56f37.
The former name official_argformer_biological.py does not exist in this checkout.

The root-sealed input is
data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v2/combined_observed_manifest.json,
SHA256 cbcabd3346802ad87a5b4aa0ea584b1ace8cfcf5ec06a04da4c827f51c3fb44c.
The canary and remaining phase references remain explicit. Original wrapper rows
and frozen_helper_result are retained, with exact source/config/runtime/input
pins. Native condition is
official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1.
Outer condition is
current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2.
The combined input is never relabelled as Relate.

Each parent contains 280 observed haplotypes, 96 pure references, 64 mixed
reference-only haplotypes and 120 queries. Four reference classes are ANA, CHG,
WHG and EHG, with 24 pure haplotypes each. The exact complete source roster row
must match the native row after only the recorded source-to-inferred node-ID
join. This preserves sample ID, diploid haplotype, cohort, ages, roles and observed
reference class. Only pure-reference population labels supervise contrastive
learning. No query ancestry or census truth trains any model, including TRAIN
query labels. Observed query paths may enter prespecified unsupervised MLM.

The tsinfer helper preserves native node IDs and times during sanitation.
Preparation uses those native IDs exactly. A namespace is parent_id followed by
a colon and the observed-tree SHA256. Node IDs are never equated across unrelated
parents, invented from equal local numbers/clades, simplified, or relabelled to
create sharing. Root connected-ID diagnostics support input representation only,
not correct genealogy or ancestry accuracy.

The preparer tail, starting at explicit tree/roster provenance joins, is compiled
unchanged from the frozen fit module. Its one mechanical metadata bridge adds
observed_sample_manifest.source_manifest from the genuine
original_observed_sample_manifest in a copied preparation entry. Original
native/helper/wrapper bytes remain immutable. Full leaf-to-root extraction,
deduplication before extant exclusion/truncation, vocabulary construction, seeded
path split, source-compatible pickles and all physical occurrence joins stay
unchanged. Actual tree flags, dates, sample mapping, coordinate grid and complete
common support still require root CPU execution.

## Native model, schedule and checkpoint contract

The 20 selected native fit functions retain exact original ASTs in a separate
namespace. Config and runner path bindings refer to this new bridge only. Native factories remain real imports from the pinned author
archive when an authorized later stage executes. The model stays FlexBERT with
12 layers, 768 hidden width, 3072 intermediate width, 12 attention heads, RoPE,
RMSNorm, released per-phase attention/unpadding differences and the pinned FA2
backend. Native ARGDataset and Transformers MLM collator retain length128,
skip_extant_tokens=true and source special tokens/masking behavior.

The existing disclosed development schedule is fixed:

| Item | MLM | Contrastive |
| --- | --- | --- |
| Epochs and seed | 5 and17 | 1 and36 |
| Logical batch | 512 | 4096 |
| Microbatch | 64 | 4096 |
| Precision | amp_bf16 | fp32 |
| Learning rate | 5e-4 | 2e-5 |
| Optimizer betas | .9/.98 | .9/.999 |
| Weight decay | 1e-5 | 1e-6 |
| Source duration | 5ep | 1ep |
| Training masking | .30 | none |
| Validation masking | .15 | none |
| Temperature | MLM objective | .3 |
| Train/VAL paths | seeded90/10, seed42 | pure-reference rows in that same split |

MLM microbatch64 differs from released microbatch512 and is disclosed.
Contrastive stays full4096. Small-batch loss accumulation is prohibited.
GradCache is a separate prespecified optional route requiring a real same-corpus
full4096 optimizer-memory attempt and 16-path/chunk4 native loss/gradient parity.
Non-memory failures are preserved and cannot trigger cache or parameter fallback.
The new condition does not inherit memory/parity receipts from Relate.

Source drop-last requires at least4096 pure-reference training paths.
Actual corpus counts and completed update budgets are presently unknown. The
metadata budget function requires exact unique-full-path, MLM train/VAL, pure
reference train/VAL, and all-scored-query-path counts. It computes five times
floor(MLM_train/512) MLM updates and floor(pure_reference_train/4096)
contrastive updates. It refuses insufficient corpora and excessive exhaustive
retrieval cost rather than reducing the baseline. No actual counts are invented.

Each observed ARG has independent vocabulary, MLM checkpoint, contrastive
checkpoint and embeddings. Contrastive initialization requires exact equality of
every loaded bert encoder key with the trained MLM Composer checkpoint, retaining
the source head-overlap/missing/extra-key report. Genuine changed-state optimizer
updates are required. The source uniform class-weight fallback and unused
three-output phantom classifier remain. No cross-ARG checkpoint reuse is claimed.

## Full reference retrieval and postprocessing

The six retrieval/support kernels retain their exact reviewed ASTs. Primary
output is source-default cosine nearest-neighbor k1, with prespecified k5
sensitivity. Corpus eligibility is every unique full path belonging to any of
the96 pure reference haplotypes in the same ARG across native path train and VAL.
Queries are every unique full query path needed by all retained common query120
centers. Mixed and query paths cannot enter the reference corpus.

F.normalize float32 vectors with epsilon1e-12, exhaustive dot products and all
corpus rows are preserved. Distinct full-path rows with identical embeddings
remain. No span weighting, class balancing, target eligibility, class-mean
substitute, smoothing, calibration, probability floor or support deletion is
allowed. Predictions are unweighted raw class counts/k. Hard-vote zero
probabilities remain, including infinite raw NLL if external truth later assigns
positive mass to a zero. Existing local class-mean exp-cosine output may remain
as a labelled auxiliary legacy export from the unchanged fit function. It cannot
replace primary full-corpus retrieval.

CPU streaming uses query blocks128 and corpus blocks8192, with four BLAS/Torch
threads and one interop thread. Released retrieve.py defaults1024/100000.
This is a disclosed resource adaptation. Scores are float32 and exact computed
ties resolve by ascending original full-path ID; class ties use first fixed C4
index. Near-tie FP32 ordering can differ across providers/devices. The source
simulated-LAI k, published corpus curation and original tie ordering remain
unresolved. k5 is stated explicitly for paper real-data retrieval and is a
sensitivity here. Published population enrichment/log-lift is a different output,
not a C4 probability repair.

A later full-corpus retrieval stage must bind completed native fit/checkpoint and
embedding tensor identities, check complete occurrence/leaf/support joins and
admit the actual query-by-reference pair count before loading embeddings.
It must seal all query/k outputs before any external truth. Native fitting and
this bridge have no truth reader. External scoring remains a separate root-owned,
predeclared operation.

## Strict paper comparison boundary

The current family is 140 diploids, four European source classes and independent
2-Mb development parents, including real sampled heterochronous dates. The paper
simulation is AmericanAdmixture_4B18 with1600 samples, including1000 admixed and200
each European/African/East Asian, on30Mb with flat recombination1e-8 and
mutation1.25e-8, tsinfer+tsdate, and no phasing/genotype error. Its three-source
ontology, preprocessing, original inferred archives/checkpoints, evaluated paths
and scoring task are not supplied by this development manifest.

The released YAML is source authority for this fixed schedule, not proof that all
paper experiments used those counts/budgets. Current execution, current native
fit success, common labels/support or shared ancestry tokens cannot establish
paper numerical reproduction. A comparison eventually must bind each paper's
actual data, task, preprocessing and score, with every change named. Missing
original archives stay missing. The current development condition cannot be
ranked as an exact paper benchmark.

## Inactive state and concrete first root stage

Default command from the dedicated repository:

    . scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" src/arggnn/official_argformer_tsinfer_biological_v2.py

Optional JSON-only binding verification uses --metadata. Both modes open source
and authorized metadata only. --execute, preparation and native capsule
side-effect functions fail while execution_enabled=false. Native preparation
and fit functions are genuine source definitions, but this handoff deliberately
supplies no active CLI stage dispatcher. Activation is a separately reviewed,
root-admitted revision, not a private flag or in-memory policy override.

The first execution stage is parent-scoped CPU path preparation and actual
representation qualification on TRAIN000 using the sealed combined manifest.
Before it, root must freeze a stage dispatcher/admission binding exact new source,
config and combined metadata, route original observed rosters, reserve all new
prepared/cache/log output, and verify the actual tree/input bindings. It then uses
preparation_entry and the unchanged preparation tail, retaining native IDs,
source dates, every complete unique path and physical occurrence. Record exact
path/vocabulary/corpus roles, sharing, path lengths/truncation, missing support
and logical update budgets. Continue only if original source4096 remains viable.

No existing20GB observed-inference slot is borrowed. Fresh storage projection,
reservation and sampled caps must cover new prepared corpora, checkpoints,
embeddings, both readouts, logs, caches and short repository-local multiprocessing
temporary paths. The suggested new20GB is not an admission or available capacity.
Root must count all active jobs/controllers and130GB external reserve against
the450GB strict ceiling and500GB outer guard. Caps are stage-boundary sampled,
not OS quotas or transient-peak guarantees.

After source/representation/corpus admission, root must schedule fresh real
same-corpus GPU memory/runtime qualification through a separately admitted
shared pool route. This bridge does not change the current allowlist or dependencies.
The fixed isolated native runtime, inherited UUID lease, process title
AI Auto Research Pipeline Test, at-most-two physical GPUs and eligible free
GPUs2/3/4/5 stay mandatory. Actual CPU preparation, GPU fitting, full-corpus
retrieval and scoring are not certified by source/handmade test success.
