# ARGMix deep author-repository audit, 23 October 2026

The deeper search recovers useful ancestor-extraction code, but no unchanged callable reproduction of the April ARGMix benchmark. **The earlier shorthand “ancestor features are absent” is too broad.** They survive in exported low-level Rust/Python APIs. The supported archive-builder/trainer/inference route explicitly disables or rejects them. A 25-reference setting is configurable today, but adding three ancestor tokens to the complete current workflow would require a disclosed source/caller adaptation and new validation. It would still not recover the original data, training configuration or paper weights.

## Scope and fresh public evidence

Only anogena `/workspace-SR003.nfs2/ARG-GNN` was used. Author code was treated as text/data. No author imports, builds, installs, tests, training, inference, GPU operations, Git commands, job changes or signals were performed. Writes are confined to this audit directory.

Fresh GitHub REST requests at 20:20:45–20:20:48 UTC show:

- Public default branch and only advertised branch: `main`, commit `c2481248eb0de5fda659d68f59c51e0e27becd20`.
- No tags, releases or issues in the complete returned lists, and no pagination links.
- Repository metadata says discussions are disabled. The discussions endpoint returns HTTP 410.
- Nine reachable commits, 5–11 August 2026. Earliest commit `b0d0213a3a4bc70e702ab1fdff6cb6c63976b49d` is parentless.

I fetched all nine immutable source snapshots from codeload as data, 2,760,406 compressed bytes total, capped at 4 MiB per response and 6 MiB expanded per snapshot. Every snapshot has the same 85 file paths. Adjacent snapshots contain no added/deleted paths, so no deleted or renamed paper script is hidden within this advertised reachable history. Seven transitions only change README/tutorial/configuration documentation. The sole executable-source transition adds HDF5 shuffle/DEFLATE compression and related validation/output tests. Trainer, attention model, joint model, simulator, archive builder and ancestor extractor are unchanged from the initial release. All 85 public HEAD files are byte-identical to the pinned local author files. The additional local source receipt is not an author file.

The census covers 86 local files, 1,321,307 bytes, 36,561 lines including dependency locks. All files received complete text/hash/link/keyword scans. Python sources additionally received AST function/call inventories, and Rust sources symbol/inline-test inventories. Actual scientific entrypoints and decisive algorithm bodies were traced manually. See `source_census.md`, `source_census.json`, `static_symbol_inventory.json`, `history_snapshots.json`, `history_feature_checks.json` and the eight numbered history diffs. This is static source archaeology, not runtime certification.

One additional explicit source lead matters: `.gitignore:37–38` says “Unit tests live on argmix_dev only” and excludes `tests/`. The named `AI-sandbox/argmix_dev` repository endpoint returned HTTP 404 at 20:24:50 UTC. That can mean nonexistent or inaccessible/private. Root `/scripts/` and model/checkpoint files are also excluded (`.gitignore:7–9,18–29`). I did not infer their contents or claim they do not exist. The public tree contains 73 inline Rust tests and one Python truth-parity oracle, but no top-level unit-test suite. `pixi.toml:13` still advertises that absent `tests/` directory.

Nine core API requests were used. Exact UTC times, HTTP statuses, ETags, source hashes and rate-limit headers are preserved in `http_receipts.json`. Shared quota was 15 remaining at the last request. No authenticated/private access, hidden refs or unreachable commits were inspected.

## Actual callable path

1. `pixi.toml:12` dispatches `python src/argmix_cli.py`. Its `main` maps build-data/build-training/train/test/infer/export to `src/commands/*` (`argmix_cli.py:150–162`).
2. Build-data validates a project, accepts only the forward source, synchronizes a derived runtime configuration and executes `sim/Snakefile`. `argmix_project.py:345–352` explicitly says msprime and other source types are no longer supported. This is not an alternative callable version of the paper coalescent simulator.
3. Build-training creates a backend configuration and invokes the separate Rust `build_training` executable (`commands/data.py:408–430`). That executable loads exact pair/reference/NPZ/map inputs, selects references and query samples from configured population metadata, extracts fast LCA rows, joins first-SNP targets, chunks rows and writes the archive (`build_training.rs:285–362,467–547,638–647`). Its cap counts distinct intervals, with all query haplotypes for retained intervals, rather than being the paper's fixed count of sampled query subgraphs.
4. Train invokes `train_joint_model.py` with train/validation archives (`commands/model.py:330–346`). The trainer validates config, seeds 50, constructs the joint model, loads archives, shuffles training chunks, uses AdamW, and chooses strict minimum mean validation-batch loss with patience (`train_joint_model.py:291–305,391–441,467–588`). Template defaults are 15 references, hidden 128, four layers/four heads, projected feature dimension 16, 1,024-row chunks/512 stride, bidirectional GRU 128, batch 16, maximum 20 epochs, LR 2e-4, weight decay 1e-4, label smoothing 0.1, patience 2, nearest-reference drop-one augmentation only for training. These are current source/template facts, not recovered April hyperparameters.
5. Infer derives config/geometry from a supplied checkpoint, exports the base model to TorchScript, invokes `run_inference.py` and its `ancpath.ShardPoolIterator`, then invokes joint GRU refinement by default (`commands/infer.py:740–758,875–980`; `run_inference.py:924–938`). Joint predictions average softmax probabilities across overlapping chunks (`joint_interval_model.py:1409–1430`). The `--no-joint` branch is real but does not turn current training into a recovered April model.
6. Test invokes `final_test_metrics.py` (`commands/model.py:147–167`). That evaluator runs base or joint inference and joins each predicted interval to all covered truth SNPs; it computes overall/cohort haplotype-SNP accuracy, truth-denominator class accuracy, diploid soft-dose correlation squared, coverage and haplotype jackknife intervals (`final_test_metrics.py:618–651,741–860,1204–1333`; `jackknife.py:9–79`). Default test selection falls back to validation when test replicates are empty and warns about optimism. This is a usable current evaluator, not the April three-IID/five-OOD, 100,000-sampled-subgraph evaluator.

The supported base encoder can also be `weighted` rather than `tmrca_transformer` (`joint_interval_model.py:527–557`). That is an available alternative, with no evidence connecting it to the paper model. Canonical vocabulary retains `<ANC>` as a reserved ID. A vocabulary slot does not prove ancestor use in the supported fit.

## The surviving 25-reference/three-ancestor components

The low-level capability is concrete:

- `src/rust/ancpath/src/sample_iterator.rs:536–585` uses `1 + num_refs + num_ancs`, traverses successive parents, appends ancestors, and pads short paths with the sample. Lines 648–662 emit `<ANC>` tokens. Its Python constructor still accepts `num_ancs` at lines 1261–1283. `lib.rs:114–124` exports this class.
- `fast_lca/engine.rs:932–954` performs the same ancestor-chain construction in the fast engine. `extracted_batch.rs:68–88` accepts the full `1 + refs + ancestors` geometry. `fast_benchmark.rs:15–50` exposes a raw fast-engine wrapper accepting the ancestor argument. `ShardPoolIterator` also retains that low-level argument.
- With supplied suitable inputs and merge disabled, these source components can express 25 references plus three ancestors, 29 extracted nodes. The model itself additionally prepends a learned CLS token, which is distinct from a genealogical node (`tmrca_attn.py:250–279,305–330`). No such run was executed in this audit.

The end-to-end barriers are equally explicit:

| Stage | Decisive source behavior |
|---|---|
| Project/config loader | Rejects `model.num_ancs`, even before normal model creation (`argmix_project.py:483–485`, `config_loader.py:38–42,404–406`). |
| Actual Rust archive executable | Rejects a `num_ancs` key and hardwires zero (`build_training.rs:326–330`). |
| Python compatibility dataset | Rejects nonzero ancestors (`tree_sampler.py:148–152`). |
| Trainer archive geometry | Requires sequence length `num_refs + 1`, excluding ancestor slots (`train_joint_model.py:391–415`). |
| TorchScript export | Rejects ancestor config and derives length `num_refs + 1` (`export_torchscript.py:70–83`). |
| Inference wrapper | Forces zero and rejects a checkpoint with nonzero ancestor count (`run_inference.py:480–482,759–766`). |
| Merging | Merge-on rejects nonzero ancestors; merge-off retains raw component geometry (`subgraph_merge_pipeline.rs:15–50`). |

All nine public snapshots have these same critical restrictions and surviving low-level ancestor chain. There is no older advertised commit with a supported three-ancestor trainer. The generic exported `RustTorchScriptInference` can accept externally supplied token/matrix/model tensors, but it neither supplies paper weights nor recovers the original training path. The `ancpath_incremental` engine is marked deprecated in `pixi.toml:21–23`, has reference-only dimensions (`engine.rs:95–102,646–647`), and is rejected by current inference/config routes. It is not the lost ancestor implementation.

Changing `num_refs` from 15 to 25 is a supported configuration change (`docs/config_reference.md:90`, `guides/hgdp_1kgp.md:261`, `build_training.rs:326`). It would produce 26 extracted nodes in the supported route, not the paper's 29. Reconstructing a 29-node pipeline is feasible engineering using author components, but cannot be described as running an unchanged recovered paper implementation. No claim is made that it would numerically match the April results.

## Data, map, errors and omitted paper assets

The only shipped experiment template is `forward_blank`: AFR/EUR/EAS, train replicate 0, validation 1, no test by default, blank founder-panel header, external founder VCF/map paths, seed 13 and one 30-generation admixture event. It contains no 1,611-person ancient European age/ID roster, no Anatolian/CHG/WHG/EHG demography, and no original empirical map bytes or map hash.

The real simulation algorithm reads phased founder genotypes, interpolates user-supplied cumulative cM into Morgans, draws Poisson tract breakpoints with expected count `generations × map length`, chooses source pools according to proportions and copies founder alleles plus labels (`forward_sim_multievent.py:215–280,340–380`). It uses seed+replicate at line 427. For multievent history it passes each event's supplied absolute `time_gens` value into that mosaic routine (`:548–585`). Only the final ADMIX cohort is emitted into the VCF (`:596–598`), although intermediate label pools are retained. This does not reconstruct the paper's msprime demographic generation with multiple sampled ancient cohorts.

There is a concrete missing-genotype operation: source missing calls are filled with the majority allele (`:273–280,528–543`). That operation is not a 0.001 genotype-error injection. No genotype-error-rate setting or corresponding random-flip operation was found in any current or historical tracked source. The source's `genomic_error_windows` are evaluation error plots, and a 0.001 threshold in `truth_labels.py:298–300` is a missing-SNP diagnostic. Neither implements the paper observation error.

The HGDP guide points to the general HGDP/1KGP resource paper and Relate input-data pages. It suggests GRCh38 ancestor FASTA and a user path called `genetic_map_22.txt` (`guides/hgdp_1kgp.md:62–90`). This is useful acquisition guidance, but it does not identify or certify the April paper's map version/build/domain. No tracked notebooks, trained weights, checkpoint assets, original simulation payloads, extra European recipes or model/data download links were found in the nine advertised source snapshots or public releases. `public_links.json` preserves every non-lock HTTP/FTP link; the root citation is a placeholder and CITATION.cff contains no paper/archive DOI.

## Scientific implication

Fresh archaeology strengthens the evidence that the current public release is a different supported pipeline from the April paper. It also narrows the missing-code claim: author ancestor-extraction building blocks are recoverable and should not be called absent. The public repository does not resolve the original 1,611-person roster, exact map/error mechanism, misspecification draws, complete paper model/optimizer recipe, original sampled evaluator or paper weights. A prospective common-data comparison or a separately validated 25+3 reconstruction remains scientifically possible. Neither source availability nor similar accuracy percentages would establish exact published-table reproduction.

This audit adds no new ancestry accuracy result and makes no superiority claim. It does not inspect inaccessible development repositories, private assets, unreachable commit objects or external unpublished data. HTTP 404 for the explicit development-repository lead is retained as an access limit, not proof of absence.
