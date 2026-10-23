# Paper-based benchmark rebuild

The user authorized this rebuild on 24 October 2026. The objective is a fair comparison using reference implementations and paper simulation recipes, with explicit reasonable reconstructions where details are absent. Exact reproduction of published numbers is not implied. Earlier results remain development evidence and are reused only after condition and artifact verification.

## Completion criteria for this work session

- Every included method has an executable data, fit, prediction and scoring route, with explicit dependencies.
- Shared observations, reference/query identities, independent simulation parents and held-out evaluation support are fixed before fitting.
- Necessary paper/source conflicts are resolved with a named condition and a written rationale, preserving the unchanged release control.
- The complete queue passes dependency and interface checks. First substantive jobs demonstrate increasing optimizer counters and healthy throughput.
- GPU jobs use physical GPUs 3, 4 and 5 only. Project processes have the requested title. Other users' processes remain untouched.
- Existing evidence, source snapshots, failures and valid ongoing work are preserved. New code and concise receipts are pushed.

## Parallel ownership

| Workstream | Owner | Deliverable | State |
|---|---|---|---|
| ARGMix | ARGMix worker | 1,611-person recipe, 29-node extraction, paper attention/raw-distance corrections, training and scoring | Implementing |
| AncestralPaths | AP worker | Historical CNN, 476-person observations/support reuse, fit and scoring | Implementing |
| ARGformer / FLARE | ARGformer worker | Paper-oriented masking, pooling, weights, PCA10 evaluation, paired scoring | Implementing |
| GST and TreeLoom integration | Root | Original-task adapter, shared inputs and scientifically appropriate set-mode comparison | Implementing |
| Queue, performance, storage, publication | Root | Readable executable DAG, reservations, progress evidence, README and Git | Implementing |

## Issue ledger

| ID | Issue | Resolution / remaining work | Status |
|---|---|---|---|
| B01 | Release experiments were conflated with paper comparisons | Preserve as named controls, build separate paper-oriented arms | In progress |
| B02 | ARGMix release representation differs | Use 25 donors, three ancestors, query, raw capped distances and paper attention equation in an explicit corrected copy | Worker implementing |
| B03 | ARGMix data scenario differs | 1,611-person chr22 recipe, BAA excluded, 0.001 allele-error reconstruction. Reuse authenticated TRAIN0 observations | Worker implementing |
| B04 | AncestralPaths current Dense network differs from paper CNN | Recover historical architecture including exact flatten order and author training budget | Worker implementing |
| B05 | ARGformer training defaults differ from paper | Fresh named arm with paper temperature, literal masking, extant token pooling and effective class weights | Worker implementing |
| B06 | ARGformer ancestry classifier unreleased | Explicit PCA10 reconstruction with training reference labels only | Worker implementing |
| B07 | GST omitted from active comparison | Prepare valid unordered-set candidate and author-task comparisons without invented ancestry identity | Root implementing |
| B08 | Old TreeLoom steps are very expensive | Inspect actual stages and optimize batching/loading without silently changing frozen mathematics | Root investigating |
| B09 | Storage near conservative projected capacity | Reuse data, calculate remaining reservations, keep total under 1 TB, no repeated broad scan | Root investigating |
| B10 | Queue coverage and current scores hard to inspect | One campaign plan, explicit dependencies, status/metric receipts in README | Root implementing |
| B11 | Historical instructions contain superseded GPU assignments | Current user authority is GPUs 3, 4, 5 | Resolved |
| B12 | Benchmark outputs not yet complete | Publish only measured held-out metrics with replicate SD, do not turn training loss into accuracy | Open |

## Reuse decisions

- American observations, native focal registry and FLARE outputs are candidates for reuse after identity checks. ARGformer release checkpoints are not reused as paper-oriented checkpoints because masking and pooling change.
- ARGMix TRAIN0 completed observations are reusable candidates. Older top-level partial files and unfinished Relate outputs are not completed inference.
- European 476-person simulations and sampled supports require manifest checks before CNN reuse.
- Prior scores remain historical development results until matched conditions are demonstrated.

## Execution sequence

1. Inspect active jobs and preserve healthy compatible work.
2. Implement disjoint method routes and freeze reconstruction choices.
3. Independently review adapters and test synthetic shape, label, split and metric contracts.
4. Construct the full executable campaign with storage/resource accounting and real full budgets.
5. Launch eligible stages in parallel on GPUs 3, 4, 5 and CPU preprocessing workers.
6. Verify actual progress, record any failure, repair within the same declared condition, and publish receipts.

The queue being prepared and launched is different from all benchmarks being completed. Final comparisons remain pending until real predictions and scoring finish.
