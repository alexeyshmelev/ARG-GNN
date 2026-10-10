# Paired Relate preparation and inference

## Release, reviewed terms and authorization

The official Relate documentation at https://myersgroup.github.io/relate/ lists the Linux x86_64 static **v1.2.4** release, dated 2025-09-13. The authoritative repository is https://github.com/MyersGroup/relate . Its README distinguishes the development branch from published binaries, and its GitHub API currently has no releases/tags. The published archive was downloaded from https://myersgroup.github.io/relate/download/relate_v1.2.4_x86_64_static.tgz after the human reviewed the terms and explicitly replied **“I reviewed, use it” on 2026-10-10**. This authorization covers research use, not external redistribution of licensed code.

The actual archive is retained under `.runtime/tools/relate/`, with 178 inspected entries. Absolute paths, path traversal, links and device entries were rejected before extraction. The installation manifest records the source, release label, binary/script hashes, actual licence PDF, authorization date/scope, help-command return codes and copyright notice. Help commands run successfully. They do not self-report a version, so version provenance comes from the official release archive rather than an invented executable version string.

| Asset | SHA256 |
|---|---|
| Official static v1.2.4 archive | `9d6fe0f25846da9a846099b0d7c8d1f9f529aee0042e92803aef72b82e771438` |
| Relate binary | `71ce82f469dd99749a3eb1847e0704be8785a26be94eae2cb8e1ed80881c7ce8` |
| RelateFileFormats binary | `cb19cb6e277c27704de506f0e8155f8f53d9c3dc8d0f52615a059caa253332ee` |
| Actual reviewed licence PDF | `c55aec267c4f8987f1d67c91cc2fb44a695c3f06244e3bf7be91230e61f2dbe2` |

These are locally computed hashes of official HTTPS assets, not upstream signed checksums. The licence PDF hash exactly matches the reviewed authoritative source at https://raw.githubusercontent.com/MyersGroup/relate/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/20180411_15670_RELATE_Academic_Usage_Licence.pdf . The website's separate licence-link URL returned 404 during inspection. The archive includes the actual PDF and Apple resource-fork files, which are not the licence or executables.

Relate uses a custom Oxford academic-use licence, not MIT/GPL. Exact clause 1.3 reads:

> The User acknowledges and agrees that the University owns all intellectual property rights in the Software. The User shall not have any right, title or interest in or to any results or other output from the Software.

Its opening says not to download or use unless the terms are agreed. The website has a required “I agree with the terms and conditions” checkbox before the download form. Clause 1.1(a) limits use to own internal academic non-commercial research. Clause 1.1(b) restricts third-party service/product use. Clause 1.1(f) requires **Copyright © 2018, University of Oxford** prominently wherever results from the software are used and with descriptions of the software in publications/materials. The human reviewed these terms before authorizing this research use. No vendor was contacted, no checkbox was accepted without that authorization, and licensed binaries/source remain ignored local assets.

Every execution folder gets `NOTICE.txt`, and execution/result provenance includes the copyright notice. Future manuscript, result tables/figures and other materials using Relate output must preserve the notice and citations. Do not commit or redistribute the extracted software. The scholarly software citations are Speidel et al., Nature Genetics 2019, https://www.nature.com/articles/s41588-019-0484-x , and the ancient-sample update, Molecular Biology and Evolution 2021, https://academic.oup.com/mbe/article/38/9/3497/6299394 .

## Preparation contract

`prepare_relate_inputs(parent, observed_ts, directory, sample_order=None, min_mac=1, recombination_rate=None)` reads **cleaned observed-tree mutations** and the external observed sample-role/age manifest. It does not read census labels, internal populations or oracle topology as Relate input. `load_ancient_parent` loads the compact parent targets and `observed.trees.gz`, never the raw census-bearing truth as observations.

Input preparation writes:

| File | Meaning |
|---|---|
| input.vcf | Exact phased diploid genotypes, ancestral REF and derived ALT at eligible observed SNPs |
| input.haps | Independently exported phased haplotypes, 0 ancestral and 1 alternative |
| input.sample | Both IDs equal the original diploid ID, original diploid order |
| sample_ages.txt | One actual sampling age per haplotype, in haps/sample order |
| input.poplabels | Observed pure-source cohort or QUERY role, no query local ancestry |
| constant.map | Explicit uniform-rate pilot genetic map |
| site_manifest.tsv | Original site ID, zero-based physical position and one-based VCF position |
| haplotype_manifest.json | Diploid/haplotype key, original observed leaf ID, observed role/class and age |
| preparation.json | Exact order, filters, coordinate convention, input checksums and genotype-matrix checksum |

Every diploid must have exactly haplotypes 0 and 1, the same sampling age and source role, and a unique sample identity. Every sampled leaf must be accounted for. Query ancestry cannot be supplied as a reference class. Reference source labels and sampling dates are observed information. Queries remain unlabeled in inference inputs.

Only actually segregating biallelic nucleotide sites are exported. The simulated ancestral state must be one of the two observed alleles. Monomorphic, nonbiallelic, nonnucleotide and below-MAC sites are counted explicitly. Missing genotypes and noninteger mutation coordinates cause errors, rather than silent imputation or rounding. The fixed external ancestry grid is unchanged by SNP filtering. Minimum MAC 1 is the small pilot choice, not a claimed ARGMix filtering reproduction.

The synthetic pilot supplies known simulated ancestral alleles **at observed mutation sites only** as a disclosed polarization condition. There is no invented ancestral sequence at unobserved positions and no fabricated ancestor FASTA. Already-polarized `.haps` can run Relate directly, so the real-human ancestral FASTA preparation step is unnecessary here. Real human data or polarization-error conditions require a separate documented policy.

## Exact coordinate conventions

The original msprime coordinate is zero-based. Export uses exactly `p_VCF = p_original + 1`, requiring integer mutation positions. Site position 0 becomes VCF/HAPS position 1. Original position L-1 becomes position L.

For uniform recombination rate r per bp per generation, the map has rows `(1, r*1e8, 0)` and `(L+1, r*1e8, 100*r*L)`. The second column is cM/Mb. Cumulative distance is `100*r*(p_VCF-1)`, so original coordinate zero has distance zero. This preserves genetic distances between SNPs without an origin offset.

The official converter retains the VCF positions directly in tskit. Its source sets sequence length to last SNP position +1 and splits trees at SNP midpoints. See https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp . Therefore converted sites and tree breakpoints must be shifted by **minus one** to return to original coordinates. Do not assume converter output is already zero-based merely because its container is tskit.

`paired_inferred_dataset` checks every inferred site against the explicit exported coordinate list, validates actual sample ages, and remaps leaves by `(sample_id,haplotype)` from `.sample` order. It retains the parent's query-column order even if diploid order changes. It clips both unobserved chromosome tails to the prespecified interval `[first exported SNP, last exported SNP +1)`. It neither extrapolates terminal trees nor silently changes scoring coordinates. Inferred trees are packed in a local covered span and their intervals/positions are restored to **global parent-chromosome normalized coordinates** with declared `window_bounds`.

Output labels are exactly `parent.labels[retained_parent_grid_indexes]`. Position arrays are the exact parent arrays at those indexes. Train/validation/test and replicate identity remain those of the original parent. Report physical coverage and retained-grid coverage. Oracle comparisons must use those same retained coordinates for paired evaluation. Mixed true labels inside an inferred tree are kept.

## Execution API and native commands

`run_paired_relate(parent, preparation, directory, config=RelateConfig(...))` performs one explicit CPU run using the already verified official tools. It checks binary/script/licence hashes and unchanged prepared input files. No download, installation, vendor communication, GPU context or scheduler submission occurs in this function.

The official `RelateParallel.sh` workflow is used by the verified installation. The single-case API defaults to one thread. The authorized v3 panel uses four threads. It executes separate native stages and is followed by `RelateFileFormats --mode ConvertToTreeSequence`. Default mutation rate is 1.25e-8, `-N 30000`, memory allowance .25 GB and explicit seed. **Relate N is effective haploid size**, as stated in its documentation. It is not the msprime diploid population size. `-N 30000` is a fixed inference assumption, not the true varying European demographic model. Population-size re-estimation and `.coal` schedules are separate future conditions.

Relate requires `-o` to be a basename in its current working directory. All native commands run inside a fresh project-owned inference folder with output prefix `inferred`. Absolute paths are used for input files. Existing output folders/files are preserved rather than overwritten. `execution.json` stores argv, working directory, return code, elapsed time and log paths. Any nonzero command status is a failure, and a converted tskit file is not called successful until both native stages return zero and sample/coordinate/tree checks pass.

Example after an ancient parent has been saved:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" -m arggnn.relate_pipeline \
  --parent /workspace-SR003.nfs2/ARG-GNN/data/ancient/train/rep000 \
  --output /workspace-SR003.nfs2/ARG-GNN/data/relate/train/rep000 \
  --run --seed 71
```

Leaving off `--run` prepares files only. The imported `dataset/` has compact tensors, external labels, metadata and an `observed.trees.gz` audit artifact. It does not misname the inferred genealogy as true ancestry. Raw census truth remains in the original parent archive. Copyright and inference provenance remain external metadata, never node input features.

## Sampling ages, phasing and branch-time caveats

The true-genotype pilot has exact phase and no genotype error. VCF and haps preserve haplotype 0 then 1 within each diploid. The sample-age file repeats the actual age of each diploid twice in that same order. Relate conversion stores a leaf per haplotype and does not preserve original simulated internal-node identities. Join via the explicit sample/haplotype manifest, not original tskit IDs or presumed adjacent internal-node IDs.

Inferred inputs contain inferred topology and estimated internal ages only. Internal populations, individual annotations, metadata and extra flags are removed before feature packing. Reference labels come from observed cohort assignments and remain source one-hots. Stable leaf IDs serve correspondence, not embeddings. Replacing the oracle parent's input features does not change imported inference features, which is tested.

Internal branch times are estimated under mutation-rate, fixed-size, sample-age and map assumptions. They are not ground-truth dates and may be biased under the structured European demography. The converter enforces positive branch lengths when estimated parent/child ages conflict. Sample dates themselves must exactly match the supplied manifest. Main experiments must separately examine topology versus date error, mutation-rate/Ne/map misspecification and optional coalescent-rate calibration. No branch-length accuracy claim follows from this smoke test.

A future genotype-error condition must modify observed genotypes with a recorded independent seed while keeping physical ancestry labels external. Phasing-error conditions must save the interval-specific true-to-observed haplotype permutation. Either align haplotype labels using that permutation outside the model or evaluate phase-invariant diploid ancestry dose. Never supply the phase-truth permutation or census truth to the inference/model inputs. These perturbations are not implemented by this Relate exporter yet.

## Actual tiny CPU validation

A single 20 kb stdpopsim ancient-European diagnostic (master seed31, train replicate0) exported 21 eligible SNPs for32 haplotypes. The successful official staged inference used seed71, one thread, mutation1.25e-8 and haploid N30,000. Both native commands returned0. It generated4 inferred trees and252 packed observed nodes. Covered original coordinates were `[169,19327)` bp, physical coverage95.79%. Exactly123 of the original128 grid coordinates were retained, grid coverage96.09375%. Nearest-TMRCA grid accuracy on inferred trees was28.2520324%. The combined inference/conversion/packing run took13.316seconds, not a throughput guarantee. This tiny parent has no ancestry switches and does not establish ordering benefits or empirical ancient-DNA performance.

Evidence is under `.runtime/tools/relate/fixtures/ancient20kb_seed31/`: preparation and haplotype/site manifests, untouched parent archives, successful `inference_parallel/execution.json`, native logs, `diagnostic_parallel.json`, raw converted trees and clean packed inference data. Two earlier engineering failures remain separately recorded: an absolute output prefix rejected by Relate, and a native `--mode All` run that wrote output but returned1 during NFS temporary-directory cleanup. Neither was counted as successful inference. The official staged workflow completed with return0 and was validated afterward. No software source was modified.

Tests cover independent VCF/haps genotype equivalence and actual native VCF conversion, position-zero/endpoint offsets, map-origin arithmetic, exact age/order handling, diploid reordering, tail masks without grid rescaling, hidden oracle-feature exclusion, actual successful fixture label equality and native zero-return codes.

## Resources and next stage

The static release needs no additional package installation for this path. Python uses the existing project runtime for msprime, tskit, numpy and torch. All tools, archives, source assets, decompression scratch, inputs, temporary native files, logs and datasets remain inside `/workspace-SR003.nfs2/ARG-GNN`. Paired inference is CPU work. The primary agent owns GPU scheduling, training, commits and publication. Preserve the reviewed licence scope, copyright and complete experiment records. Keep repository storage below500 GB with a450 GB warning threshold.

Before a main paired benchmark, freeze parent simulation seeds/configuration, variant filtering, inference settings, polarization condition, sample manifest and coverage rule. Generate adequate ancestry tracts on longer independent parents, infer matching trees, and train/evaluate the same model/baseline families separately on true and inferred trees. Report matched-coordinate differences and uncertainty over independent parent replicates. A tsinfer/tsdate backup remains useful as a separately licensed inference family, not a drop-in reproduction of Relate.


## Frozen v3 paired inferred panel

`configs/inferred_common_input_pilot_v3.json` binds the exact v2 source protocol and manifest by SHA256. It keeps the same 16 training and 4 validation parents, models, adapter, eight global windows and training settings. There are no test parents in this pilot. Source metadata, raw tensors and prepared tensors are checked before inference. Each inferred record also binds the source truth archive by hash for target provenance, while genotype export reads only the cleaned observation.

The independent inference master seed is 20261013. Each parent gets a signed-31-bit seed from SHA256 of the UTF-8 string `ARG-GNN/Relate/v1.2.4/{master_seed}/{split}/{replicate}`, interpreting its first eight bytes as an unsigned big-endian integer, taking modulo `2**31-1`, then adding one. The scheme does not consume or replace the ancestry/mutation seeds. The v3 protocol pins mutation rate 1.25e-8, recombination rate 1e-8, effective haploid size 30,000, MAC 1, memory .25 GB, four CPU threads and a 1,200-second native command timeout.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" -m arggnn.prepare_relate_study \
  --config configs/inferred_common_input_pilot_v3.json \
  --output data/ancient_inferred_common_input_pilot_v3
```

The preparation module processes every parent sequentially and fails on any nonzero native status. Existing versioned output is preserved. It writes raw inferred inputs plus the exact reference-adapter cache and integrity fields expected by `train_study.load_cases`. Inferred metadata has an explicit inferred benchmark family, the original source family separately, parent hashes, unchanged parent ancestry/mutation seeds, independent inference settings and exact retained target/position/query alignment.

Before completing the manifest, the actual training loader checks all raw metadata/tensor and prepared-cache bindings. Global fixed windows intersect genuine inferred coverage, without extrapolation or renormalization. Concatenating their positions and labels must recover each paired case exactly once in its original order. `verification.json` records those checks and clipped versus global bounds. The compact tracked `docs/relate_v3_provenance.json` reports the finished panel, with full generated records retained in the dataset manifest.

**Copyright © 2018, University of Oxford** applies to this Relate output and its descriptions.


The completed panel retained 2,559 of 2,560 original parent grid points. Nineteen parents retained all 128. Validation parent 3 retained 127 because source grid index 127 lies outside its observed SNP coverage. Physical coverage averaged 99.81692% and was at least 99.5657%. Both native commands returned zero for every parent. Full preparation took 201.72 seconds, including resource guards, packing, caching and loader/window verification. The sum of recorded native command times was 9.22 seconds. These are distinct timings for this actual run, not throughput guarantees.

The actual loader verified all 20 raw and prepared cases. Prepared inputs have 48 features with the exact original eight-feature prefix and unchanged nonfeature GraphSequence fields. Each case produced eight fixed global tiles, clipped only to its genuine coverage. Their labels and positions concatenate to the original retained case exactly. No raw truth genealogy or test directory is present in the inferred panel. Paired oracle evaluation must exclude validation parent 3's missing source grid index 127, using the recorded retained indexes for every parent.
