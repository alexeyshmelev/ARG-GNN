# Matched public real-data validation plan v1

Status: source and metadata audit, 15 October 2026. No empirical ARG, fit, prediction, PCA or accuracy result was generated in this audit. H018's existing cohort and stability endpoint are reused. New processing, R16 runtime qualification, experiments and compute scheduling remain root-owned. H040 is implemented but untrained and is not changed here.

## What is ready

The practical first study is the frozen H018 ancient cohort: eight reference subjects (16 haplotypes) and twelve query subjects (24 haplotypes), chr22 GRCh37 absolute 0-based [20,000,000,22,000,000). Its dates, source rows, QC eligibility and independent joins already exist. The audit exports the same twenty IDs and a readable metadata roster under [the audit directory](../research/reviews/real_data_PCA_benchmark_audit_20261015_v1/). These exports do not select a new cohort.

| Asset | Public access and concrete product | What it supplies / what remains |
| --- | --- | --- |
| Allentoft ancient genomes | [DOI 10.17894/ucph.d71a6a5a-8107-4fd9-9440-bdafdfe81455](https://doi.org/10.17894/ucph.d71a6a5a-8107-4fd9-9440-bdafdfe81455). Indexed chr22 VCF is 1,654,398,745 bytes, index 28,008 bytes. | 1,664 ancient genomes imputed at 26,198,700 biallelic 1KGP SNPs, GRCh37. 1,492 passing IDs. Published AP is an external comparator, not truth. Named data license is absent from retrieved DataCite rightsList. Full publisher checksums are not exposed. |
| H018 metadata | Existing frozen endpoint, joined supplementary metadata and age contract in experiments/results/public_realdata_metadata_join_v1/. | Reuse the exact R16/Q24 roster and date/QC choices. Do not rerun the earlier inventory/join. Geographic/cultural labels do not prove pure ancestry. |
| High-coverage Ötzi | [ENA PRJEB56570](https://www.ebi.ac.uk/ena/browser/view/PRJEB56570), sample SAMEA114266904, merged run ERR11837081. | Merged mapped BAM 34,206,836,802 bytes plus BAI 9,018,432 bytes; MD5 and exact FTP paths verified in ENA metadata. Four additional library BAMs are the same subject, not independent samples. Wang et al. mapped to hs37d5. This is not ARGMix's processed imputed VCF. |
| Modern HGDP+1KGP | [Koenig et al. primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11216312/), [downloads](https://gnomad.broadinstitute.org/downloads#v3-hgdp-1kg), public metadata and phased BCF bucket recorded in asset_inventory.json. Verified chr22 BCF is 278,689,536 bytes plus a 27,252-byte CSI, with GCS generation and CRC32C metadata. | Harmonized GRCh38, SHAPEIT5 phasing. Paper joint callset has 4,094 subjects / 3,400 inferred unrelated; phasing-source README describes 4,091, so pin the actual released product/metadata intersection. Public release is described as unrestricted. Exact named dataset license was not established here. |
| ARGformer empirical genealogy | [Wohns Zenodo 5512994](https://zenodo.org/records/5512994), version 1.0.0, CC BY 4.0. | GRCh38 chromosome-arm tree sequences inferred with tsinfer 0.2.1 / tsdate 0.1.4. chr22_q is 38,871,562 bytes with published MD5. Includes modern panels, four archaic genomes and an Afanasievo family. It is a separate original-paper replication source, not the common GRCh37 H018 ARG. |
| Observation-error resource | [Zenodo 7993392](https://zenodo.org/records/7993392), CC BY 4.0. | 42 downsampled/imputed ancient genomes, six 0.1x–2x conditions, 3.08–5.54 GB archives. Genotype/imputation validation, not ancestry truth. Existing H018 metadata already audits it. |
| UK Biobank | Controlled access. | Not needed for the first public pilot. The AncestralPaths paper's UKB ancestry/trait figures use ChromoPainter/related components, not automatically AncestralPaths outputs. |

The public components make a useful new experiment feasible. They do not yet reconstruct the exact original merged ARGMix cohort or the exact ARGformer task-specific path corpus.

## Original figures and units

**ARGMix, April v1, DOI 10.64898/2026.04.05.714076.** Fig. 2A is ordinary genotype PCA of Ötzi and present-day Europeans. Fig. 2B is **ancestry-masked genotype PCA**: haplotype alleles assigned other than Anatolian ancestry are set to missing, and covariance is computed on the retained genotype information. It is not PCA of ancestry fractions or neural embeddings. Section 4.5 names snputils, but does not fix its version, selected PCA algorithm, centering/scaling, LD mask, projection policy, missingness threshold or weighting. Current snputils exposes several missing-data algorithms and individual/haplotype/group aggregation options; its current defaults cannot be asserted to reproduce April v1. The plot's Sardinian/Bergamo proximity is a descriptive model-conditional affinity, not proof of direct descent or historical causal continuity.

ARGMix Sections 4.4–4.5 describe 3,720,746 variants / 1,611 subjects after ancient INFO>0.8 filtering, a new GLIMPSE-imputed Ötzi genome, intersection with modern samples, and Beagle 5.4 rephasing. Its panel is 22 Ana, 13 CHG, 42 EHG, 159 Neo including Ötzi, 48 WHG, 13 Yam, 654 post-Neolithic and 660 present-day Europeans. Ana references were chosen by ADMIXTURE K4 on 255,816 MAF>10%, LD-pruned markers, requiring WHG component<0.03. These are inferred proxy labels, not verified pure ancestry. Supplementary Fig. S1 is an ancient-genotype PCA with bold reference symbols, S2 is ADMIXTURE K4, S4 is Anatolian-specific TreeMix with zero migration edges, S5 sums local calls into ancestry fractions, and S6 compares masked f4 statistics. The supplement supplies no extra PCA methods or numeric sample-age table. The precise subject manifest, merged VCF, imputation panel release, build conversion, map, trained paper checkpoint and Fig. 2 calls/settings have not been established in the inspected release. Ancient source data are GRCh37 while the cited harmonized modern resource is GRCh38; the article does not document the coordinate reconciliation in the inspected methods.

Current-release ARGMix c2481248 is a separately labeled transformer+GRU workflow with a GRCh38 chr22 HGDP+1KGP tutorial using 20 unrelated GBR/YRI/JPT founders per class and simulated ADMIX offspring. Its QC PCA uses PLINK2 on merged founders/simulated samples, biallelic A/C/G/T variants, unique IDs, LD pruning (defaults 200, 50, r2=0.2), ten approximate PCs fitted jointly. The source-only plot just subsets those same PCA coordinates. It does not refit a reference PCA or reproduce Fig. 2B. The pruning command passes 200 without a kb suffix even though its config key is prune_window_kb; verify the pinned PLINK2 interpretation rather than advertising 200-kb pruning.

**AncestralPaths, DOI 10.1038/s41586-023-06705-1.** Supplementary Note 3 pp. 27–29 / Fig. S8 compare genotype PCA of simulated chromosomes and MesoNeo/GBR data: both MAF-filtered at 5%, EIGENSOFT smartpca, outlier removal disabled, no projection. Fig. S26 is a genotype PCA used to select diagnostic ancient samples, colored by radiocarbon age. Neither is PCA of inferred ancestry proportions. Fig. S10 is a GNN feature matrix (informative ancestor rows, observed cohort-proportion columns and node age), not an ancestry accuracy heatmap. Figs. S11/S16 are simulated confusion matrices: predicted rows / true columns, normalized by predicted-row sum, so cells show precision. S16 contrasts six paths with BAA versus four paths without BAA. Replicate uncertainty comes from separate simulated tree sequences, not the many correlated sites. Supplementary Fig. S1 is a fineSTRUCTURE/ChromoPainter copying matrix (recipient rows, donor columns), from a different analysis.

The AP diagnostic cohort is 476 diploids including 91 GBR; the selection study's 1,015 ancient genomes and UKB's 408,884 white British subjects are different scopes. Native author code produces a six-path classifier. Store six probabilities, collapse paths 1/5→Ana, 2/6→CHG, 3→WHG, 4→EHG before C4 argmax. The public AP FORMAT header is Number=1518, Type=String, Description="Ancestral Path"; it is not a documented simple per-haplotype integer field. Decode and validate the original encoding in a separate comparator adapter after predictions are sealed. Do not manufacture a probability vector from an unverified string.

**ARGformer, DOI 10.1093/bioinformatics/btag438.** Fig. 2A is simulated genotype PCA (one point per individual). Figs. 2B/C are simulated marginal-path embedding PCA (many points per individual), before/after contrastive tuning. PCs are fitted on AFR/EAS/EUR references and ADMIX is projected. The LAI benchmark uses 10,000 unique admixed paths, a ten-PC clustering rule or nearest-neighbor retrieval. These path-level simulated measurements are not real-data ancestry accuracy.

Figs. 4/5 are empirical top-5 retrieval heatmaps. For query group q and corpus label c:
- vote share v(q,c) = retrieved label-c votes / all top-5 votes for q;
- corpus baseline b(c) = label-c paths / all corpus paths;
- color = log2[v(q,c)/b(c)].

The share is row-normalized; color is log-lift, so a larger raw share can still mean depletion. Preserve v=0 as -infinity and mark b=0 unavailable instead of silently adding an epsilon. The Fig. 4 wording describes query-path rows while the results describe population summaries; export both per-path votes and explicit group aggregates to make this unit unambiguous. Fig. 5B reruns retrieval after Oceanian references are removed. Figs. 4B/S2 are UMAP, not PCA. Reported 3.60% Denisovan or 9.84% Oceanian neighbor fractions are retrieval shares, not admixture percentages. The paper itself cautions about computational archaic phasing artifacts.

Empirical pretraining uses a 90/10 split of unique paths within each task ARG. Supplementary Table S1 gives 688 sample nodes / 184,963-token vocabulary / 7,660,071 train + 851,119 validation paths for archaic analysis, and 665 sample nodes / 119,067 vocabulary / 3,473,898 + 385,989 paths for America–Oceania. These are not independent subjects, chromosomes or simulation replicates. The 95.96% supplemental result is masked-node token accuracy. The reported ridge R2 measures observed path-event count prediction, not local ancestry correctness. Exact empirical IDs/chromosome arms, release checkpoint, PCA plotting code, permutation unit/count and full figure-generation settings are not supplied by the inspected pinned repo.

## Common real-data pipeline

1. Reuse H018 source dates/IDs and freeze method/checkpoint identities on simulations. Qualify R16/Q24 adapters and new age support before processing empirical data. The current R96/Present182 comparison does not admit this cohort. H018 has only 16 reference haplotypes, so the paper-v1 ARGMix 25-reference subgraph is unavailable here. A qualified smaller-panel current-release run is a disclosed adaptation, not a paper-v1 reproduction. Obtain a real GRCh37 chr22 genetic map and ancestral-state sequence with release/hash. Use cal.BP dates and an explicit BP=1950 ↔ inference time convention, retaining fractional generations. Evaluate the already-declared 25/28/29/30-year and date-bound stress cases in separate simulation conditions.

2. Acquire only the fixed chr22 region, ordinary diploid columns and observational fields. Strip AP definitions/values before the first retained observed artifact. Keep provenance for GT/GP/HS/DS and site annotations, with model-accessible fields explicitly allowlisted. H018 admits transversions, strict mask, INFO>=0.8, external RAF-derived MAF>=0.05, valid GP max>=0.8 in all selected subjects and unambiguous ancestral state matching REF/ALT. Require at least 1,000 retained sites. Actual region call/phase QC must pass; the prior 32-record prefix does not qualify this region.

3. Infer one shared observed Relate genealogy for the base GT and one per coherent HS iteration 0–7, with identical sample identities/site set, dates, map and coordinate conversion. Select one HS iteration for each subject across all sites, with the same recorded iteration policy, not new random phase at each SNP. Source HS draws alter genotype and phase together. Preserve observed coverage and mappings; do not derive windows or input ancestry labels from AP. H018 uses original 2-Mb coordinate normalization, 100-bp centers, 250-kb physical tiles and exact covered query-bp, with no extrapolated tails.

4. Apply every fixed method to that same observed genealogy and support. ARGMix gets its native TMRCA/reference representation. AP gets its native 5×9 ordered GNN features and six-path outputs. The new GNN gets its frozen observed-only feature set. ARGformer receives unique full leaf-to-root paths and an ARG-specific vocabulary. A simulation node-ID vocabulary cannot transfer to the real ARG. A new observed-only within-ARG SSL stage and reference-label-only contrastive tuning are transductive adaptation and require a separately frozen budget/protocol. Exclude every query leaf/subject from the retrieval corpus; do not use query path labels. Retain distinct full paths even if truncated inputs or embeddings coincide.

5. Export interval × haplotype × C4 scores with class order, role, coordinate support and adapter identity. Keep AP six-path probabilities too. ARGformer native top-5 vote shares and the existing cosine-similarity downstream C4 adapter are separate readouts; normalized retrieval scores are not calibrated posteriors. Do not pool them under one unnamed probability output. For the common C4 sensitivity endpoint use the predeclared adapter, label its scores accurately, and retain native retrieval plots separately.

6. Evaluate H018's frozen endpoint: natural-log JS between each HS run and base GT, after averaging each query's two haplotype C4 vectors, integrated on identical physical support. Average over eight draws, then median across twelve query subjects; <=0.05 nats passes the fixed observation-stability claim. Also report every subject, method, unscored bp and coverage. This measures observation sensitivity. Published AP concordance, if its encoding is qualified later, is external-method agreement.

## Publication outputs and comparison diagnostics

Produce five labeled panels with downloadable source tables:
- genotype PCA from the fixed QC/LD-pruned dosage matrix, fit on a prespecified reference panel and project queries;
- ancestry-masked genotype PCA using each method's frozen calls, identical sample/SNP panel and explicitly pinned missing-data algorithm;
- C4 ancestry-fraction heatmap, subject rows × source columns, probabilities/scores integrated by exact physical length;
- local painting matrix, haplotype rows × genomic intervals, plus uncertainty/support overlays;
- ARGformer native vote-share and log-lift heatmaps, with corpus class counts alongside.

An optional PCA of the subject × four C4 fractions is a separate ancestry-composition plot. Its rank is at most three because rows sum to one. It cannot reproduce ARGMix Fig. 2B or ARGformer embedding PCA. For an equal-subject structure comparison, separately aggregate embeddings by unique full path within haplotype and then average two haplotype means; disclose that this current author helper is unweighted by genomic span and differs from local-path plots.

Freeze the PCA axes/fitting population, centering/scale, allele orientation, external LD/MAF mask, variant/haplotype/subject aggregation, missingness filter and ancestry calling rule before viewing geographic clusters. For masked PCA report retained-ancestry fraction, per-subject and pairwise SNP overlap, covariance validity, and PC correlation with missingness, coverage, age, site and technical cohort. Never fill masked alleles with reference allele zero. A change in PCA may reflect masking/information loss. Include a mask-only/random-block-mask control matched for retained fraction/tract lengths, leave-one-proxy/site-out sensitivity and a fixed-confidence masking sensitivity chosen on simulation. Use the same missing-data method and sample eligibility across classifiers; disclose exclusions.

Align sign/rotation by reference-only Procrustes when comparing independently fitted PC spaces. Report pairwise-distance/rank correlation and leave-one-chromosome robustness rather than visual resemblance alone. For external AP agreement report length-weighted confusion, per-class agreement, C4 fraction differences, posterior/score JS only where both full vectors exist, and boundary-distance distributions. Do not call the diagonal real-data accuracy. Mask-method correlation is not independent validation.

Subjects, archaeological sites and chromosomes are inferential units. Bootstrap subjects and chromosome/long genomic blocks, not each SNP/path as an independent replicate. The twelve-query/one-region pilot cannot resolve geographic continuity. More balanced, independent modern populations, genome-wide regions and proxy/ARG/phase sensitivity are required for an affinity study. Describe any Bergamo/Sardinian pattern as conditional on the cohort, inferred ARG, ancestry labels and missingness procedure.

## Concrete next commands

These commands reuse existing metadata and have not executed acquisition or scientific processing in this audit. Run from anogena after the root's data/runtime admission for the future extraction.

```sh
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
audit_dir="$ARGGNN_ROOT/research/reviews/real_data_PCA_benchmark_audit_20261015_v1"

# Already exported exact H018 metadata; read these before acquisition.
cat "$audit_dir/H018_subject_ids.txt"
cat "$audit_dir/H018_selected_subject_metadata.tsv"

# Metadata only: exact public Iceman files, sizes and MD5.
cat "$audit_dir/iceman_ENA_read_run.tsv"

# Future region extraction. Preserve external annotations; remove AP immediately.
# Choose a fresh admitted output directory inside this repository.
out="$ARGGNN_ROOT/data/real_data_H018_region_v1"
mkdir -p "$out"
CHR22_REMOTE='https://erda.ku.dk/archives/917f1ac64148c3800ab7baa29402d088/neo/genotypes-public/22.neo_impute_ph.vcf.gz'
bcftools view -I -S "$audit_dir/H018_subject_ids.txt" \
  -r 22:20000001-22000000 -m2 -M2 -v snps -Ou "$CHR22_REMOTE" |
  bcftools annotate -x FORMAT/AP -Oz \
  -o "$out/H018_chr22_20_22Mb.noAP.vcf.gz"
bcftools index -t "$out/H018_chr22_20_22Mb.noAP.vcf.gz"
```

This extracts candidates, not an admitted QC cohort. Before Relate, implement a reviewed H018 validator for external RAF, polarization, all-subject GP, exact phase/HS semantics and sample/time order. Existing simulation preparation commands are not empirical VCF adapters and must not be pointed at this file without that bridge.

For an original ARGMix Fig. 2 replication, first resolve the precise 1,611-ID roster, original merged build/variant list, Ana-reference labels, map and paper-v1 checkpoint/source, plus snputils version and explicit PCA parameters. The public Iceman BAM enables reconstruction but is a 34.2-GB preprocessing branch, not a ready ancestry matrix. For original ARGformer figures, first reconcile the two exact selected-subject/path manifests and chromosome-arm set against Wohns 5512994, then regenerate an ARG-specific vocabulary and pin masking/pooling/contrastive/retrieval settings. Neither branch should delay the small H018 tool-validation pilot.

Evidence and exact URLs: [audit README](../research/reviews/real_data_PCA_benchmark_audit_20261015_v1/README.md), [asset inventory](../research/reviews/real_data_PCA_benchmark_audit_20261015_v1/asset_inventory.json), [figure contracts](../research/reviews/real_data_PCA_benchmark_audit_20261015_v1/figure_contracts.json) and HTTP/source-hash receipts. Existing H018 originals and frozen source pins remain authoritative.
