# ARGMix paper-revision search v1

No public revision matching the April 2026 paper's 25 references + three ancestors + query and linear TMRCA clamp was found in the inspected resources. This is a bounded source-availability result, not proof that a private, deleted or differently named archive never existed. The existing native bridge remains frozen and its nine CPU tests were not repeated.

## Public evidence

[BioRxiv metadata](https://api.biorxiv.org/details/biorxiv/10.64898/2026.04.05.714076) confirms v1 was posted April 7, 2026, with Cole Shanks, David Bonet, Marçal Comajoan Cara and Alexander G. Ioannidis. [Crossref](https://api.crossref.org/works/10.64898/2026.04.05.714076) agrees and links the author ORCIDs. Full HTML and the API-provided JATS URL returned HTTP 403. The paper's 29-node/linear-clamp specification is retained from the already completed primary-text review in `docs/argformer_control.md`; this search did not freshly inspect the article's code-availability statement.

The supplied author repository [AI-sandbox/argmix](https://github.com/AI-sandbox/argmix) was created August 5. Its public history has nine commits, one branch and no tags or releases. The initial [release commit](https://github.com/AI-sandbox/argmix/commit/b0d0213a3a4bc70e702ab1fdff6cb6c63976b49d), dated August 5, has no parents. Its source archive already rejects ancestor features, defaults to 15 references/log2 distance cap 14 and trains a transformer plus genomic GRU. Its attention, sampler, training module and template are byte-identical to current `c2481248eb0de5fda659d68f59c51e0e27becd20`.

The initial archive SHA256 is `cfdb9024a360fe44d5708bccca8bdf2d181db5503fe1a4651409ca49578651a1` (306,413 bytes, 85 source files; streamed in memory, not extracted or executed). The initial Rust exporter already uses half-float bit encodings. Going back before the later “Compress inference HDF5 outputs” commit does not recover full-precision raw logits.

GitHub's ARGMix-name search returned only this repository. The `shanksc` public repository listing contains no earlier ARGMix repository, and Zenodo's ARGMix query returned zero records. These searches do not cover private/unindexed archives. The initial contributor identity and supplied repository agree with the existing author-source audit; no additional paper-era release link was established.

## Honest benchmark protocol

| Branch | Concrete protocol and missing items |
| --- | --- |
| April v1 reproduction | Pending an author-provided official revision/archive, original training configuration/checkpoint, data/reference-age panel, replicate splits and label/export rules. Require 25 distinct nearest reference haplotype leaves overall, three inferred ancestors and query, plus the paper's linear TMRCA clamp. Current supported training rejects three ancestors. Preserve and clarify the paper/source k_j versus native k_i relative-key discrepancy. |
| Closest available complete author workflow | Freeze current `c2481248...` as **current-release ARGMix transformer+GRU**, executing unchanged author extraction, training and inference with its own explicitly recorded configuration. It is genuine author code but does not reproduce April v1. Keep the full width-128/four-layer template or a matched declared tuning grid; do not substitute the earlier width-32/full-tree/shared-feature adaptation. |
| Paper-configured native component sensitivity | A possible separately reviewed experiment using unchanged native TmrcaAttn and lower-level Rust ancestor extraction. It would require new disclosed glue/training rather than the supported stock CLI and cannot be called exact paper reproduction. It has not been built or run. |

For the current-release benchmark, keep the author population-token/TMRCA inputs, source/reference/query identities, age units and genealogical preprocessing. Use disjoint biological parents/subjects/founders before windows, external outcome labels, matched development splits and tuning budgets. Do not count repeated reference indices as distinct references: the native helper can pad insufficient pools or reduce nearest-drop augmentation. Published 25-nearest requires at least 25 available distinct leaves overall, not 25 diploid individuals or 25 per class.

Preserve stock final probabilities and original component logits separately. Native Rust HDF5 quantizes logits/probabilities, while the GRU averages softmax probabilities across overlapping chunks. Do not rename log averaged probabilities as original logits. Freeze an explicit proper-score/calibration/dose/tract metric contract and biological-parent uncertainty before evaluation. Published-task reproduction and a shared-data comparison remain separate, especially for the current four-founder census benchmark versus AncestralPaths' six path-class/nine-cohort formulation.

## Root-owned build route

Use an exact execution copy under `.runtime/author_exec/`; keep the pristine source and all existing bridges untouched. Build an isolated Linux-64 author runtime with Python 3.12 and the declared Pixi environment. The author lock uses Torch/libtorch 2.12/CUDA 12.9 and Rust tch 0.23 download-libtorch: verify loaded-library, TorchScript and driver compatibility. Current project Python 3.11/Torch 2.6 lacks ancpath, h5py, polars, maturin and workflow packages, with incompatible tskit/pandas versions.

After root's environment review, stock `pixi install` and `pixi run compile` build the ancpath extension and separate `build_training` binary. Keep every environment/cache/Cargo target/temp/log inside this repository. Check stock CLI/export and synthetic Rust extraction on CPU before biological data. Then supply audited development inputs, source/sample-age panel, true ancestry labels, observed coordinate/sample joins and disjoint split manifests. Root owns installations, subsequent training/GPU scheduling and Git.

The matching receipt is `research/protocols/argmix_paper_revision_search_v1.json`. This search used public HTTP reads only and writes only these two new artifacts. It installed nothing, executed no newly fetched source, changed no frozen files and accessed no actual final targets. No scientific readiness or native trained result is claimed.
