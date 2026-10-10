# Biological observed-genotype tsinfer+tsdate producer v1

This is an unexecuted, separately versioned observation producer for the four existing biological development parents. The native ARGformer model and author source remain unchanged. It is a transfer of the official historical-sample tsinfer+tsdate workflow to this panel, not a completed reproduction of the published modern simulated task.

Owned entry points are scripts/infer_biological_paths_tsinfer_v1.py and configs/biological_paths_tsinfer_v1.json. Default invocation is source-only. No observed genotype/tree arrays, simulator/oracle trees, ancestry targets, scores or checkpoints were opened in author qualification. The existing Relate manifest was read only as observation metadata to freeze original file hashes and its comparison support.

## Scientific condition and fidelity

Condition: official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1.
New output namespace: data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v1.
The prior proposed native_ARGformer_tsinfer_tsdate_biological_v1 name described an inference family. The more specific condition name discloses the ancient-input task transfer.

The published ARGformer article (DOI 10.1093/bioinformatics/btag438, pinned XML SHA256 53ab26710804f0b9ca6f355095893b365413caf56cf7bb722b39907f6db0619f) specifies tsinfer+tsdate on simulated genotypes without phasing/genotype error. It does not specify package release versions. No tsinfer/tsdate version recipe was found in the pinned ARGformer Python/Markdown/YAML/TOML/text sources at f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596.

The proposed runtime uses the current supported API releases tsinfer 0.5.1 and tsdate 0.2.7, with tskit 1.0.3, numpy 2.4.6 and Python 3.11. This is an explicit version resolution for task transfer, not an assertion that the authors used these versions. SampleData is deprecated but still implemented in pinned tsinfer 0.5.1. It is retained here because the pinned official historical-sample recipe and tsdate.add_sampledata_times explicitly use it.

The official [historical-sample recipe](https://github.com/tskit-dev/tsdate/blob/ee6082b4c17a333d1a54efc4a4be833b2820bca2/docs/historical_samples.md) warns that historical tsinfer topologies combined with variational_gamma dating have not been extensively tested and can have dating/stability issues. Actual CPU qualification and observation-only admission remain pending. The runner preserves any failure, never repairs node identities, and never discards ancient dates to obtain an output.

## Same observations, documented polarization

Inputs remain all four original VCFs and exactly the same 140 diploids / 280 ordered haplotypes: 96 pure references, 64 mixed unlabeled haplotypes and 120 query haplotypes. All original phased query genotypes stay in the observation panel. Query ancestry targets and all truth remain outside inference, dating, vocabulary construction and native model training.

The pinned observation producer src/arggnn/biological_paths_simulation.py (SHA256 f0f4520b8fdfd4074bc78024073dd5b331ecac996911f40af787b6df68f213e8) invokes observed.write_vcf at lines 378-380 with ploidy=2, explicit diploid names, contig 22 and position +1. Its pinned configuration has tskit 1.0.3. The [tskit 1.0.3 writer](https://github.com/tskit-dev/tskit/blob/b62f75f04af81fbf5f0d6fb38953cb97b587737f/python/tskit/vcf.py#L187) assigns REF = variant.alleles[0]. The [default variant encoding](https://github.com/tskit-dev/tskit/blob/b62f75f04af81fbf5f0d6fb38953cb97b587737f/python/tskit/genotypes.py#L59) guarantees allele 0 is the site's ancestral state. The source writer supplies no alternate allele mapping.

Thus REF-as-ancestral is supported specifically by these frozen simulated VCF writer defaults. It is not a general assumption about empirical VCF REF. Runtime parsing also requires the exact original file hashes and the header "##source=tskit 1.0.3". No ancestral FASTA, mutation-age truth or ancestral-state lookup from a tree sequence is used.

Every original VCF site is retained, including multiallelic, singleton and nonsegregating sites. The parser rejects missing/unphased genotypes, invalid nucleotide/allele indices, duplicated coordinates and site-ID mismatches. It never filters invalid records away. Native tsinfer inference versus overlay eligibility is preserved and recorded. A site with a native NaN modern age is excluded from ancestor inference by native rules and retained by the native non-inference-site overlay. No custom age or allele fallback is supplied.

The original site_coordinate_manifest establishes exact VCF_position_1based = original_position_0based + 1 and original site IDs. Diploid header order is flattened as sample_index = 2*diploid_index + haplotype. The separate date file and roster must agree exactly. Native SampleData insertion IDs, actual flagged TS sample order, exact dates, diploid pairing and decoded alleles are checked at the appropriate stages. Observed cohorts and pure reference labels stay in separate rosters, never in inference population/individual/site metadata.

## Official date-aware stages

The [official two-step recipe](https://github.com/tskit-dev/tsdate/blob/ee6082b4c17a333d1a54efc4a4be833b2820bca2/docs/historical_samples.md) is implemented as follows.

1. Construct SampleData from the observed phased VCF, adding each diploid with its actual observed time. No tree-sequence constructor is used for inputs.
2. Select only individuals whose actual date is zero to establish a modern timescale. The source panel includes 32 present-day diploids / 64 haplotypes. Historical individuals keep their dates in the full SampleData and are not relabeled as modern.
3. Run native tsinfer.infer on the actual modern subset, native preprocessing, and native tsdate.date.
4. Obtain native site-age estimates with tsdate.sites_time_from_ts. Apply tsdate.add_sampledata_times to the full panel. That official helper takes the maximum of estimates and the oldest historical derived-allele carrier date. It leaves nonvariable modern site ages as NaN, according to its source semantics.
5. Generate ancestors using dated site times. Insert native historical proxy ancestors using allow_mutation=True, as in the official recipe. Native epsilon=None determines proxy spacing. This creates explicit inferred proxy nodes and allows recurrent mutation conflicts, it does not merge existing ancestral IDs.
6. Match ancestors and then match all samples with force_sample_times=True and overlay_non_inference_sites=True.
7. Perform native preprocessing and redating with variational_gamma, preserving fixed sample dates. Disconnected/unary/dating failures remain failures. Samples must retain exact dates, parent times must be strictly older than child times, and mutation times must be finite and lie on their branches.

The convenience tsinfer.infer API does not accept/forward force_sample_times. Its frequency-age scale cannot be equated with generation dates. The runner does not invent a frequency-to-generation transformation. Its final date-aware stages use the documented native site-age and proxy mechanisms.

Native defaults are frozen explicitly: recombination_rate=None, mismatch_ratio=None, path_compression=True, post_process=True and synchronous num_threads=0. Recombination is not passed to ancestor matching, consistent with the pinned inference source. No seed is invented for deterministic staged APIs which expose no random_seed argument.

Dating is fixed to variational_gamma, mutation_rate=1.25e-8, 25 iterations, 1000 rescaling intervals, five rescaling iterations, match_segregating_sites=False, singletons_phased=True, min_branch_length=1e-8, allow_unary=False and generation units. constr_iterations=None preserves native automatic choice, including its heterochronous constraint handling. set_metadata=True retains native unconstrained estimates for site-age transfer before output sanitation.

One coverage choice differs from unconstrained preprocessing defaults and is disclosed: delete_intervals=[] prevents further tsdate gap/tail deletion, so eligible original scoring support can be preserved. Native tsinfer post_process keeps its erase_flanks=True default, which removes edges outside [first_site,last_site+1) while retaining full sequence length. This rule does not restore those empty flanks. filter_sites=False explicitly retains the native site-preservation default. Native split_disjoint=True and native simplification are retained. Such native upstream operations may reindex/split ancestors and are recorded as inference stages. The final sanitation preserves the dated tree's node IDs and times exactly. No downstream rank/clade/time merge is performed.

## Coverage, genotype and identity admission

Observation-only asset pins are archived in research/reviews/biological_paths_tsinfer_v1_source_audit/observed_asset_pins_from_existing_relate_manifest.json (SHA256 e7781b2a5b6ee9630d12db0edf9401fdba5ac3a9fe9099b69cd444e1c11c492d). They copy the original nine whitelisted observed file hashes and exact comparison support from the existing Relate manifest (SHA256 6872222127c96b3defbfb1b9edb83ba1d5b11b0b60e6888956369bc507cefe28). No prior tree was loaded.

| Parent | Exact original half-open comparison support | Original 100 bp centers retained |
| --- | --- | ---: |
| bio_paths_pilot_train_000 | [117, 1998898) | 19988 |
| bio_paths_pilot_train_001 | [19, 1999991) | 20000 |
| bio_paths_pilot_val_000 | [273, 1999681) | 19994 |
| bio_paths_pilot_val_001 | [116, 1999724) | 19996 |

Native full-length outputs and all original input sites are preserved as stage artifacts. The final observed TS is clipped to this identical comparison support without simplification/relabeling/time repair. Coordinates are shifted by coverage_left, with physical_original = local_tree_position + coverage_left. Original grid centers stay at 50 + 100*grid_index. The original scoring_coordinates.npz edges and centers must match the fixed formula exactly.

Before clipping, all native sites and phased allele strings must agree exactly with all original VCF records. After clipping, all sites inside scored support must agree again. A sidecar joins original site ID, VCF position, native full site ID, clean site ID, original coordinate and local coordinate, including sites outside scored support. No difficult sites or targets are silently removed from inference.

Population/individual/migration/provenance tables and all metadata are removed from the final model TS. Sample flags are retained, technical ancestor/historical flags are removed, and actual nonzero sample times are retained. Node count, node times and actual sample IDs must remain identical to the native dated output. Native unsanitized stages stay archived for provenance.

Connected internal identity is measured from ancestors reached by walking tree.parent from actual samples, excluding isolated non-sample nodes. Connectivity and sharing are evaluated only on marginal-tree intersections with the exact frozen comparison support. Expected empty flanks outside that support are ignored. The native topology must cover the entire specified support, including intervals between scoring centers, and all samples must have one ancestral root throughout it. Gaps or partially uncovered support fail, with no intersection-based shrinkage, tail extension or genealogy repair. At least one connected internal ID reuse across marginal trees in that support is required for representation admission. This is a predeclared minimal diagnostic capability gate, which can be met by shared-root reuse alone. It does not demonstrate substantial/useful sharing, biological identity, or accuracy. Native ID reuse rates and adjacent sharing are recorded. Inadequate sharing fails without invented associations or fallback fitting.

## Proposed isolated CPU runtime and root admission

No package has been installed for this task. Root must first approve a resolved runtime lock containing Python 3.11, numpy 2.4.6, tskit 1.0.3, tsinfer 0.5.1 and tsdate 0.2.7 plus all transitive dependencies and package artifact hashes. PyPI metadata and official release/commit records are archived. The paper did not specify these versions. Do not silently upgrade an existing shared environment.

Use the existing /home/jovyan/miniconda3/bin/conda to create a project-contained environment at .runtime/envs/biological_tsinfer_v1. Resolve/install only after root authorization, using the approved lock and repository-contained package caches. Freeze the interpreter hash, actual installed versions and complete non-pyc environment-file inventory after installation. No inference package import is needed to create that inventory, the runner exposes runtime_inventory(prefix).

A runtime receipt must have kind biological_paths_tsinfer_v1_isolated_runtime, the configured prefix, python_executable_sha256, versions equal to the config pins, and files_sha256 mapping every non-pyc environment file to its hash. Package versions and the complete inventory are checked before and after execution. These installed artifact hashes are separate from upstream source commit evidence.

A storage receipt must have kind biological_paths_tsinfer_v1_storage_admission, project_root, measured_project_bytes, max_added_bytes=20000000000 and limit_bytes=500000000000. The runner also measures current repository storage before admission and each stage, stops at the 450 GB warning threshold, enforces the 500 GB hard limit/reservation, and limits added run storage to 20 GB. All temporary/JIT/cache/output files stay inside the repository. CUDA_VISIBLE_DEVICES must be explicitly empty.

Two separate root-admission receipts are needed, one per mode. Exact schemas with inactive authorizer fields are provided as templates in the audit directory. Each requires kind biological_paths_tsinfer_v1_root_admission, authorizer=root_agent, the exact mode/condition, truth_or_oracle_inputs_authorized=False, and path/SHA256 references for config, implementation, observed_asset_pins, runtime and storage. infer additionally requires a passed same-code/config/runtime cpu_qualification receipt. The authorizer field is an operational workflow opt-in, not cryptographic authentication of a person or agent.

Default source-only command, safe before installing anything:

~~~sh
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" scripts/infer_biological_paths_tsinfer_v1.py
~~~

Later root-only CPU qualification uses the isolated interpreter, --mode qualify-cpu, --expected-config-sha256, --expected-source-sha256, --admission and --expected-admission-sha256. It executes only a fixed hand-authored eight-haplotype genotype table with nonzero dates through the complete native pipeline. Its identity gate uses the fixed original observed-site span [10,61), not the native empty tails of its 2 Mb sequence. All six original sites, phased genotypes and dates are still checked on the full native output. There is no simulator or real observation input. A passed receipt binds the exact code/config/runtime.

Only after that qualification and independent source review may root use --mode infer with a fresh mode-specific admission and all four hashes. The runner then verifies and opens only the nine whitelisted assets in each original observed directory, processes all four parents once, and writes a fresh output directory. Original oracle .trees files are never accepted or loaded. A failed parent/stage preserves its partial manifest and native stage artifacts. There is no alternate algorithm/version/drop-date retry. Any repeat needs a new registered reason and fresh versioned output/admission.

This source-only work has not entered either actual mode. Source/static handmade verification does not qualify native package compatibility, date convergence, runtime costs or output adequacy.

## Separate faithful published modern task plan

If the documented historical task transfer fails or needs material undocumented heuristics, report that limitation for this biological condition. Do not rebrand a heavily changed ancient pipeline as the native paper task. A separately registered ARGformer_published_modern_AmericanAdmixture_tsinfer_tsdate_reference_v1 condition should instead reproduce the original modern task within what published evidence specifies.

That plan uses the paper's AmericanAdmixture 4B18 demographic task: 1600 modern samples (1000 admixed, 200 European, 200 African, 200 East Asian), a 30 Mb chromosome, mutation rate 1.25e-8 and flat recombination rate 1e-8, with error-free phased genotypes. It follows tsinfer+tsdate inference, the author full-path preprocessing/deduplication and the released native ARGformer model. Simulation, demographic catalog version, package versions, unspecified author inference defaults and experimental splits must be separately frozen and independently reviewed. Report any unresolved details as a qualified reproduction limitation. No simulation or new model task is implemented/executed here.

Native ARGformer on this new inference family versus AP/ARGMix/candidate on Relate is an end-to-end inference/method comparison. A representation/model comparison requires separately prepared methods on the identical qualified TS. Existing default-Relate comparisons remain valid with their condition labels. A new inferred ARG requires fresh vocabularies, corpora and checkpoints, never reassignment of the current tree-local-ID checkpoint to new ancestors.

## Source revision after independent flank review

The independent source review blocked script SHA256 74f12e8576cb5a3d64fdca1f9363023ff67c91ccd984067d8a2327668e69a2ef because its full-length identity gate rejected expected native empty flanks. That revision and its original audit remain preserved in research/reviews/biological_paths_tsinfer_v1_source_audit/blocked_revision_v1 and the original review receipts. The narrow revision adds interval-aware identity accounting for both observed inference and handmade CPU qualification. No native inference/dating defaults, original asset/support pins, sample dates, genotypes, model code, or topology are changed. Renewed exact source/receipt pins and independent review are required before actual CPU admission.
