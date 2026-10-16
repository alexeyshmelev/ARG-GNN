# Prospective 1,611-diploid IID producer and common-support bridge

This packet implements the next concrete missing generation route. It is source only, and its proposed contract creates no execution authority. No numerical/project module was imported or executed during authoring. No simulation, Relate command, model, scientific array, GPU/process probe or Git operation was run. Only this dedicated review folder was written. Root must adopt the disclosed choices, bind these exact sources and apply its existing admission and actual supervision.

## Implemented scope

`prospective1611_iid_v1.py` exposes six bounded root-callable stages. Every producer requires an actual `guard(required_write_bytes)` callback. There is no default permissive stub, admission system or monitor. Calls allocate finite outputs and fail without redraw, shortening, site subsampling or changed targets if a cap or scientific assumption fails.

1. `generate_parent` generates one exact registered TRAIN, VAL or IID parent with 1,611 diploids and 3,222 phased haplotypes. It uses all 45,329 authenticated chr22 map knots on 35,178,458 bp, with genetic length 74.109562 cM. The ancestry model is the retained AP population-size/event graph with the unused BAA branch deleted and the explicitly new integer-age roster. The binary finite-sites mutation process is followed by one independent Bernoulli 0.001 allele flip across all cohorts at all simulated mutation sites. Segregating-site filtering follows the single error realization. VCF, packed states, ordered variants, IDs, ages and roles are exported. Only the segregated truth directory receives the census ancestry tree.
2. `prepare_relate` opens only the observed whitelist, writes count-generic native `.haps`, `.sample`, map and ages, and freezes exact pinned Relate 1.2.4 `--mode All` and `ConvertToTreeSequence` argv. These are format adapters. Author inference algorithms are unchanged. Binary0 and1 are explicitly encoded as A and C. VCF, `.haps` and the same empirical map all use rebased position+1. No subprocess is launched by this function.
3. Root executes both native commands under its existing CPU supervisor. This prospective recipe uses N30000 and mu1.25e-8, with no additional population-size iterations, branch re-estimation or annotation. It is not the older development panel's five-iteration recipe or an exact paper reproduction. Relate native defaults within `--mode All` remain in force and are pinned by executable hashes.
4. `finalize_relate` requires root's actual successful-wait receipt and verifies unchanged commands, every original SNP position, sample count, ages and sample order. It clips conversion coordinates to observed SNP support without extrapolating end trees, retains the native node IDs and stores local-to-rebased offset. It strips metadata and forbids converted mutation tables as genotype features. The original packed observed genotypes remain authoritative.
5. `sample_targets` freezes exactly 100,000 distinct inferred-tree/query pairs uniformly without replacement. TRAIN uses zero-based tree index modulo10, VAL modulo100 and IID all positive-span inferred trees. Edge trees are eligible. It uses the existing sampler seed with CPython3.11 `random.Random(seed).sample(range(N),100000)`. The original sampler return order is retained. It reads no truth. Zero support or a missing query cohort stops the stage.
6. `label_targets` is a separate supervision/scoring operation. It joins the frozen target midpoints to the external census350 lineage with tskit's right-continuous interval lookup and writes only one uint8 label per frozen target row in `Ana,CHG,WHG,EHG` order. No labels are returned to observed feature construction.

The roster has separate reference/query Boolean roles. NF and Yam retain both roles. `eligible_donor_indices` excludes both haplotypes of the focal diploid before ranking. `nearest_donors` uses inferred TMRCA with stable haplotype-ID tie breaking. The donor pool is 592 for NF/Yam focal diploids and 594 for PostNeolithic/Present. The current native ARGMix Rust reference loop is not patched by this packet. Its method adapter must use explicit per-focal exclusion before fitting or evaluation. The presence of this helper is not a claim that that downstream requirement is complete.

The nine IID-phase parent IDs and all their seeds are read from the existing 14-parent/77-stream registry. Nothing remints or replaces seeds. Misspecification parents deliberately fail in this IID source, pending the five realized parameter dictionaries. All titles say prospective. `published_recon_*` remains only the existing registry identifier.

## Exact root API invocation

Use the existing `.runtime/env/bin/python -B` from `/workspace-SR003.nfs2/ARG-GNN`. Importing the module has no numerical imports or scientific side effects.

```python
import importlib.util
from pathlib import Path
project = Path('/workspace-SR003.nfs2/ARG-GNN')
packet = project / 'research/reviews/ARGMix_prospective1611_generator_source_20261016_v1'
spec = importlib.util.spec_from_file_location('prospective1611_iid_v1', packet / 'prospective1611_iid_v1.py')
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
config = packet / 'generator_contract.proposed.v1.json'
parent = 'published_recon_train_000'
# Root first admits the parent scope and creates only the dataset-level parent directory.
parent_output = project / 'data/ARGMix_prospective1611_IID_v1' / parent
producer.generate_parent(config, parent, parent_output, guard=guard)
producer.prepare_relate(config, parent, parent_output, parent_output / 'relate', guard=guard)
# Root now reads relate/commands.json and runs its two argv in order, with its recorded cwd.
# guard here is root's real existing resource callback, not a lambda or a proposed stub.
```

After root has genuinely consumed both native completions, its terminal JSON must have the following fields. `commands_ref` is the actual file reference including path, SHA256 and bytes. The Boolean and return codes must record actual waits, not intended outcomes.

```json
{
  "commands_ref": {"path": "actual repository-relative commands.json path", "sha256": "actual SHA256", "bytes": 123},
  "actual_wait_consumed": true,
  "returncodes": [0, 0]
}
```

Then call, still within root's admitted scopes and actual guard callback.

```python
producer.finalize_relate(config, parent, parent_output, parent_output / 'relate',
                        actual_native_terminal_json_path, parent_output / 'inferred', guard=guard)
producer.sample_targets(config, parent, parent_output / 'inferred',
                        parent_output / 'shared_support', guard=guard)
producer.label_targets(config, parent, parent_output, parent_output / 'shared_support',
                       parent_output / 'truth_targets', guard=guard)
```

Generation, input preparation, native execution, inferred bridge and target stages may have separate root admissions. A native stage must be supervised against the finite ceiling in the command manifest and a 86,400-second timeout. Every output directory must be fresh, and no existing parent is overwritten or automatically retried. CPU environment should have `CUDA_VISIBLE_DEVICES=''` and temporary paths strictly inside the admitted parent. This packet creates no launcher, GPU task or persistent service. Metadata-only CLI is `.runtime/env/bin/python -B research/reviews/ARGMix_prospective1611_generator_source_20261016_v1/prospective1611_iid_v1.py --describe`.

## Storage and runtime

No simulated S, T, bytes, memory peak or timing is measured yet. The real panel's 3,720,746 variants are not used as a forecast.

For S retained observed sites and H=3222 haplotypes, the raw packed matrix is exactly `S*ceil(H/8) = 403*S` bytes, and positions are `4*S` bytes. Each VCF row has exactly `2*H=6444` genotype/separator bytes plus a bounded locus prefix. A `.haps` row has the same 6444 allele/separator bytes plus its prefix. Ordered variant TSV rows add tens of bytes each. Per-site text conversion is vectorized with NumPy ASCII layouts, avoiding a Python formatting loop over every haplotype. Output buffering is 64 MiB and callbacks occur at batch boundaries. This does not establish a measured speed gain.

The safety cap is 2,000,000 latent mutation sites and 2,000,000 truth trees. It is a stop condition, not a chosen simulated count. A parent truth table cap is 8 GiB, with a conservative tree-file serialization envelope `2*nbytes+1MiB`. Generation has a 32 GiB total ceiling, Relate input export 16 GiB, native inference 64 GiB and inferred bridge 32 GiB. Shared-support and truth-join stages each have a 32 MiB ceiling. These are unexecuted finite caps rather than predictions or immediately reservable requests. Keeping every maximum for all nine parents would exceed 1 TB. Root must base later admissions and an explicit intermediate retention policy on measured first-parent outputs and actual available capacity. No deletion or reclamation is performed by this source packet.

## Narrow correctness review and remaining work

`static_scientific_review.v1.json` records only AST parse, compile-without-execution and metadata arithmetic. All sampled dates are younger than their population retirement and census350. All admixtures are younger than census350, and founder splits are older. Population labels use names rather than the original AP numeric indices. The map uses cumulative-cM slopes divided by100 for Morgans, with no extrapolation. Mutation/error draw order, once-only error, diploid roles, same-parent support and truth separation are explicit.

No dynamic qualification has occurred. First admitted parent execution must resolve actual library/runtime behavior, native SNP retention, conversion bounds and sample order, then actual resources. If native conversion drops sites or yields uncovered clipped support, the source stops for explicit protocol review. It never silently changes the benchmark.

Still missing are the count/role-adapted released-AP feature/fit route, the current-release ARGMix donor/masked-loss adapter and finite fit config, our variable-panel GNN adapter, all paired fits/scores and the later misspecification dictionaries. Thus this packet completes a concrete generator/bridge source stage. It does not complete the ARGMix benchmark or establish any accuracy gain.
