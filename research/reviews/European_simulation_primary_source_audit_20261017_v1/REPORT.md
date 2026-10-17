# European simulation: primary-source audit

The European condition is supported as a **new study using the released AncestralPaths demography and complete 476-person simulation age roster**. It does not reproduce the AP paper's full simulation experiment or the April ARGMix benchmark. The material differences include BAA at 170 generations in AP prose versus 169 in released code, paper 200 Mb versus example 2 Mb, and ARGMix's 1,611-person panel, empirical chromosome 22 map, added genotype error, admixed references and demographic perturbations.

This is a source-only audit of the European half of the current “Simulation Assumptions and Parameters” TODO. The research ledger was consulted first, particularly H012, H014, H022, H034, H038 and H059. No manuscript, ledger or current record was edited. No simulation, inference, scoring, resource job, Git operation, project code execution, or scientific, truth, checkpoint or genotype payload access occurred. Public age metadata and frozen seed/configuration metadata were checked. The American half of the TODO is outside scope.

## Primary citations and pins

- **AP:** Irving-Pease et al., *The selection landscape and genetic legacy of ancient Eurasians*, Nature 625, 312–320 (2024), [DOI 10.1038/s41586-023-06705-1](https://doi.org/10.1038/s41586-023-06705-1). Supplement locations below give printed page followed by PDF page. Source is the paper-cited [Zenodo 8319452 accepted archive](https://doi.org/10.5281/zenodo.8319452), already established as byte-identical across 17 original files to [commit 77db3cfae100641c35218b87e1451e14f3074666](https://github.com/AliPearson/AncestralPaths/tree/77db3cfae100641c35218b87e1451e14f3074666). That existing archive audit is reused.
- **ARGMix:** Shanks et al., *Graph transformer for ancient ancestry inference*, [bioRxiv v1, 7 April 2026](https://www.biorxiv.org/content/10.64898/2026.04.05.714076v1), [DOI 10.64898/2026.04.05.714076](https://doi.org/10.64898/2026.04.05.714076), [PMC13082007](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082007/). Cached primary paragraphs are in `research/reviews/ARGMix_published_benchmark_audit_20261015_v1/paper_text.txt`: §4.2 at line 41, §4.3 at line 43, and §4.5 at lines 45 and 47. The later [author release c2481248eb0de5fda659d68f59c51e0e27becd20](https://github.com/AI-sandbox/argmix/tree/c2481248eb0de5fda659d68f59c51e0e27becd20) does not certify the April recipe. Existing history-search conclusions are reused; no unchanged search was repeated.
- **Current text:** `paper/current_full_panel_working.tex:258–288`, SHA256 `4c765f123c2762eec92abcf4fd0f553f86d556ff75afdf575458337f3c578e56`. Frozen definition: `configs/expanded_full_panel_v1.json`, SHA256 `662ba8e4613ce85b9aaec9909e873c9cddf9c62672b895e7d9e02f851d518d08`. Study producer `src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py` has SHA256 `ead823158a001dabdbf2224d812e7ef8079b6af4382f6e8e10ca0e0894760e71`; expansion source `src/arggnn/prepare_expanded_full_panel_v1.py` has SHA256 `2ff091677986d5cfec2936c2acf53eeed71c957192f9f048cb47f59d3351913e`. These pin definitions; this audit does not freshly certify deployed runs. All input hashes are in `sources.json`.

## Claim-by-claim result

| Current claim / TODO | Exact support | Finding and limit |
|---|---|---|
| Demographic events, fractions and Ne | AP `simulate.py:38–78`; supplement pp. 25, 27, 30 (PDF 26, 28, 31), Fig. S7 | Supported as released assumptions. AP says population sizes were tuned for agreement with real PCA/Fst. They are not estimates inferred from this study's observations. Fig. S7 numbers were not newly visually inspected; exact numeric support comes from code. |
| 476 diploids / 952 haplotypes; complete ages | AP supplement pp. 27–28 (PDF 28–29); `simulate.py:16–35,82–112`; age file lines 1–476 | Fully recoverable simulation age roster, attached. Modern 91 samples are hardcoded at age zero; ancient dates retain within-cohort source order. Public metadata contain cohort/age rather than named empirical IDs. |
| Zero growth/migration; census 350 | AP `simulate.py:59–80`; frozen config `demography:81–339`; study producer `:334–353` | Zero growth/migration are instantiated study settings; census 350 is explicit author code. Removing census annotations before mutation is a study observation adapter. The model's observed cut at 350 is a separate definition. |
| Hudson, discrete genome, JC69, r = 1e−8, mu = 1.25e−8 | AP `simulate.py:120–124`; supplement p. 28 (PDF 29); study producer `:334–353` | Rates match. Hudson/discrete/JC69 are explicit study pins. Author code does not explicitly set those arguments, seeds or msprime version; do not retroactively attribute these pins to the original run. |
| 2 Mb and 32 TRAIN / 8 VAL | AP `simulate.py:120`; supplement pp. 28, 32 (PDF 29, 33); expansion `:23,77–94` | 2 Mb matches the released 10-replicate example. AP paper describes 200 Mb, three TRAIN and five evaluation sequences. The 32/8 split is this study's design. Eight reserved outer parents are outside the attached seed table; outer payloads were not inspected. |
| Genotype error and map | AP `simulate.py:120–129`, `genetic_map_simulated.txt:1–3`; study producer `:389–401`; ARGMix §4.2 / line 41 | Study writes phased mutations directly, with no injected genotype error; its uniform synthetic map is 1 cM/Mb. Contig label 22 asserts no empirical chromosome/build. ARGMix reports error 0.001 in all samples and a chromosome 22 recombination map. Its exact error mechanism, map identity/build/domain/length remain unresolved. |
| Relate chain | AP `relate.sh:5–9`; supplement p. 32 (PDF 33); study producer `:563–575`; frozen `relate` config | Effective haploid N = 30,000, mu = 1.25e−8 and five size iterations match the released chain. Study explicitly fixes 28 years/generation and final branch re-estimation in Relate 1.2.4; that script defaults to 28 and re-estimates after iterations. ARGMix §4.2 states N/mu, which does not certify identical iterations or the same final tree product. |
| Seeds, software, coordinates | Frozen config `parent_registry:460–1757`, `runtime:1829–1834`, `seed_scheme:1956`; producer `:378–401`; inference helper `:118–162,280–340` | All 40 TRAIN/VAL seed records are attached and independently re-derived. Study pins Python 3.11, msprime 1.3.4, tskit 1.0.3, NumPy 2.4.6 and Relate 1.2.4. Original AP and April ARGMix raw seeds/full version locks remain unresolved. |
| ARGMix panel and eligible ages | ARGMix §2.1 / line 9, §4.1 / line 29, §4.3 / line 43, §4.5 / lines 45, 47 | Distinct 1,611-person panel; BAA excluded; Neo/Yam used as informative admixed references. Exact individual ages/IDs and fully executable eligibility/focal-exclusion rules remain unresolved. Supplement S1 is a PCA (caption lines 8–9), not a complete roster. |
| ARGMix misspecification | ARGMix §4.2 / line 41 | Ne increased/decreased by 1–50%; admixture times older by 1–25%; divergence times changed by ±1–25%. Exact distributions, joint draws, raw seeds and boundary handling are unresolved. This study's unperturbed condition does not implement those experiments. |
| Later ARGMix simulation source | Current `sim/README.md:1–11`; `forward_sim_multievent.py:2,24–27,427`; `build_sample_manifests.py:16,133–137,341–354` | A later forward simulator consumes real reference haplotypes and external metadata/events. It does not recover April's coalescent recipe or freeze the original 1,611-person age table. |

## Event and population-size trace

Times are generations backward from present. Pulse dates are derived from **released cohort maximum age + 1** (`simulate.py:51–54`), rather than hardcoded universal historical estimates. Fractions/splits are explicit at `simulate.py:72–78`.

| Event | Released calculation / constant | Ancestry at event | AP paper comparison |
|---|---:|---|---|
| Present/Bronze | max Bronze 165 + 1 = 166 | Neo 0.5, Yam 0.5 | Supplement p. 25 / PDF 26: 166 |
| BAA | max BAA 168 + 1 = 169 | CHG 0.25, Ana 0.75 | **Paper says 170; retain the source/paper difference.** |
| Yam | max Yam 176 + 1 = 177 | CHG 0.5, EHG 0.5 | Paper: 177 |
| Neo | max Neo 258 + 1 = 259 | WHG 0.25, Ana 0.75 | Paper: 259 |
| WHG/EHG | 600 | NE | Paper: 600 |
| CHG/Ana | 800 | WA | Paper: 800 |
| NE/WA | 1,500 | trunk | Paper says 1,500 generations alongside **15,000 years**. Do not silently reconcile this internal units inconsistency. Study uses code's 1,500 generations. |

Ne from `simulate.py:38–48,60–70`: Present/Bronze 50,000; Yam 20,000; Neo 50,000; EHG 3,000; WHG 2,000; NE 15,000; BAA 20,000; CHG 3,000; Ana 5,000; WA 15,000; trunk 15,000. The declared `N_bronze` variable is unused; actual `present_bronze` uses `N_present = 50000`.

## Complete age roster and panel distinction

`public_author_age_roster.tsv` preserves the exact 476-line public file byte-for-byte (SHA256 `3cadb16abbc0a1913eadb183d324eadc26c035d61e5116cd58675f52a942b7de`). `study_diploid_age_roster.csv` maps every study sample/diploid index to age and source line. The 91 GBR rows are recorded for correspondence only; both simulators hardcode modern ages at zero. Ancient dates are not rounded or replaced by cohort means. The derived 952-haplotype template matches frozen canonical SHA256 `19fc09e3c7186db6ce95fabeea4efadb535e668c505106aba8bcfc6fcc269ceb`.

| Cohort in study order | Study diploids | Age range, generations | Public age-file lines | ARGMix §4.5 count |
|---|---:|---:|---|---|
| present_day | 91 | 0 | GBR 379–469; ignored by simulator | 660 |
| Bronze_Age | 81 | 100–165 | 1–81 | 654 Post-Neolithic; cohort identity unproven |
| BAA | 7 | 136–168 | 470–476 | 0; excluded |
| Neo | 151 | 139–258 | 228–378 | 159; includes Iceman |
| Yam | 13 | 153–176 | 201–213 | 13; age/ID identity unproven |
| WHG | 51 | 210–529 | 150–200 | 48 |
| EHG | 43 | 155–402 | 107–149 | 42 |
| Ana | 25 | 217–360 | 82–106 | 22 |
| CHG | 14 | 184–473 | 214–227 | 13 |
| Total | 476 | — | 476 lines | 1,611 |

The study has 343 query diploids / 686 haplotypes and 133 pure reference diploids / 266 haplotypes. ARGMix's four pure founder counts total 125 individuals; Neo 159 and Yam 13 also act as informative references. Assuming diploidy, 1,611 corresponds to 3,222 haplotypes and a nominal six-cohort reference pool of 594 haplotypes before eligibility/focal exclusion. These are arithmetic totals, **not a verified eligible pool**. Pure reference ages may exceed 350; that census constrains this study's admixed query labels, not reference eligibility. Every query age in the released roster is below 350. AP's separate 1,518-diploid selection panel (supplement p. 28 / PDF 29) is another scope.

## Reproducibility details for integration

`study_TRAIN_VAL_seed_registry.csv` gives ancestry, mutation, initial Relate, size-base, five iteration and final re-estimation seeds for TRAIN 000–031 and VAL 000–007. All 160 purpose-domain hashes and 40 iteration lists were checked without importing project code. Domain: `arggnn|expanded_full_panel_v1|root=101|split={train,val}|parent={index:03d}|purpose={purpose}`. The first eight digest bytes, big-endian, give modulo (2^32−1) + 1 for ancestry/mutations or modulo (2^31−100) + 1 for Relate. Iterations use size-base + 1…5; final re-estimation intentionally reuses size-base + 5.

The synthetic sequence is [0, 2,000,000). Discrete mutation position bp0 becomes VCF POS = bp0 + 1. Map records are 1 → 0 cM and 2,000,001 → 2 cM. Relate inputs retain observed phased biallelic segregating nucleotide variants with minor allele count ≥ 1. Converted positions subtract one; retained support is [first retained VCF POS − 1, last retained VCF POS), equivalently [first SNP bp0, last SNP bp0 + 1). Local coordinates subtract that left offset. Fixed 100 bp scoring centers are 50, 150, …, 1,999,950; uncovered tails are not extrapolated. These are study adapters, not a verified empirical chromosome 22 or April ARGMix coordinate rule.

For integration, cite the AP paper and accepted archive, identify BAA 169 as a released-code value, attach complete age/seed tables and version pins, state no injected genotype error and the synthetic coordinate/map rule, and retain the common-data/released-demography qualification for ARGMix comparisons. Exact numerical or data-protocol reproduction is not established.

`audit.json` records claim statuses; `source_excerpts.md` preserves numbered primary/source passages; `sources.json` records input hashes; `checks.json` records 16 metadata checks; `sha256_manifest.json` seals the files. ROOT owns all subsequent integration.
