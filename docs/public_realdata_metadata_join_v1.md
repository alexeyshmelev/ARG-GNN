# Public real-data metadata join v1

H018 now has a reproducible subject/date/QC join and a frozen, falsifiable observation-sensitivity design. No real-data ARG, model fit, ancestry painting, concordance result or biological accuracy result has been produced. R16 below is a **new, unadmitted condition**, separate from the existing R96 simulation panel. Current adapters and scientific runtimes are not claimed to admit it.

## Source metadata and exact subject joins

The source study is Allentoft et al., *Population genomics of post-glacial western Eurasia*, [DOI10.1038/s41586-023-06865-0](https://doi.org/10.1038/s41586-023-06865-0). Its actual article XML identifies Supplementary Data 1 (new sample details),2–4 (dates/context/reservoir corrections),5–6 (contamination/relatedness) and7 (full ancient-genome roster). The four small XLSX files were obtained from the publisher's static-content.springer.com endpoint and frozen by SHA256. An initial alternate publisher hostname failed DNS; PMC returned HTTP200 download-interstitial HTML, which is retained as failed product evidence and never parsed as XLSX. Every bounded public request, received byte hash, HTTP metadata and rejection is recorded. No ancestry-mixture supplement was requested.

Supplementary Data 7 has 1664 unique subjects. Its Flag 0 IDs equal the 1492 noflag.inds.txt IDs exactly. All 1492 passing IDs join the ordinary chromosome22 VCF columns. Ten Excel IDs are stored as integers; converting those values explicitly to decimal strings resolves their joins without fuzzy aliases. The original header has 1667 ordinary columns and 1664 _rnd columns. Ordinary AltaiNeandertal,Denisova,Vindija33.19 andNEO957 have no full-roster metadata; NEO962 has metadata and an _rnd column but no ordinary column. No NEO957/NEO962 equivalence is inferred. These unmatched/archaic columns are excluded.

The derived join retains source sheet/row and raw field values for identity, date range/mean, dating method, coverage, averageGP, QC flag and geographic/cultural Group label. Supplementary Data 7 ancestry/IBD clustering columnsS–V are excluded. New-sample details, source dating context and contamination fields join separately, including documented starred merged-sample notes. A specimen/library alias does not become another reference leaf. Supplementary Data 6 contains 92 listed relationships: 2 degree 0, 24 parent–offspring,36 siblings and 30 second-degree pairs. No listed degree0/1/2 edge lies among the20 selected subjects. Absence from this limited table does not prove all subjects unrelated.

Of all 1664 subjects, 1644 have a valid positive BP mean inside a positive95% range; nineteen have missing dates and one has an inconsistent range. They remain in the inventory with explicit invalid-date flags and are excluded by the prospective dated-cohort filter. Supplementary Data 7 uses positive calibrated BP. Some new-sample tables store negative values in columns labelled BP; those raw values remain provenance and are never copied into inference as negative sample ages.

## Actual GT phase and HS semantics

A second bounded1,000,000-byte HTTP206 chromosome22 prefix request reproduced the original prefix SHA256, header SHA256 and ETag. Only aggregate/source-QC metadata for the first 32 complete records was retained, at VCF1-based positions16,051,249–16,055,879. Raw VCF rows, raw HS values and AP values were not retained.

All 47,744 passing-subject calls in this probe were phased diploid GT calls; all had valid GP triplets and valid30-bit HS values.40,684 passed maxGP>=0.80 and35,204 passed maxGP>=0.90. This resolves the header wording for these records: the actual calls are phased despite the GT description “Unphased genotypes”. It does not qualify the whole chromosome, proposed region or individual sample call rate. All _rnd calls happened to be missing in this prefix; that is not a genome-wide missingness claim.

The fetched GLIMPSEv1.0.0 phase writer, sampler and ligation source files match their frozen tag-tree Git blob identities; no Git command or build was executed. With actual NMAIN=15, HS contains two allele bits per posterior iteration: allele0=(HS>>(2*m))&1 and allele1=(HS>>(2*m+1))&1. The sampler chooses one iteration per individual and uses it across sites. Ligation aligns corresponding iteration pairs across chunk overlaps. Selecting a new iteration independently at each SNP would destroy phase coherence and is forbidden.

The phase writer emits unphased GT, whereas the sampler writes phased GT and removes HS/GP/DS. The public merged product retains phased GT plus HS/GP, so its later processing cannot be reconstructed from the header alone. The public header says max 16 pairs and the frozen phase source text says max 15; actual NMAIN=15 controls this product. This is source-semantics evidence, not a claim to have reproduced the original executable or full processing. HS draws can alter genotype as well as phase and are not a true phase-error benchmark.

## Separate proxy panel, dates and domain shift

Prospective metadata eligibility requires exact noflag membership, Flag 0, an ordinary VCF column, valid positive dated interval, global coverage>=0.4x and averageGP>=0.98. This gives 1064 metadata-eligible subjects. Within the explicitly frozen geographic/cultural proxy groups, available subject counts are Ana 6, CHG 3, WHG 13, EHG 4, corresponding to maximum 12, 6, 26, 8 haplotypes before region QC/identity qualification. These metadata labels do not prove genetically pure sources. The desired24 haplotypes per class/R96 panel cannot be supplied by these rules.

The conservative NEW panel takes two subjects per proxy class by coverage descending,averageGP descending,then exact ID, with at most one subject per archaeological site. It contains 8 unique reference subjects / 16 haplotypes:

| Proxy | Subjects | Mean cal.BP range |95% CI envelope cal.BP | Working28-year generation range |
| --- | --- | --- | --- | --- |
| Ana | Bar8,Bon002 |8071–10078 |7980–10229 |288.25–359.93 |
| CHG | KK1,SATP |9720–13255 |9550–13380 |347.14–473.39 |
| WHG | Loschbour,Bichon |8050–13665 |7940–13770 |287.50–488.04 |
| EHG | Sidelkino,NEO88 |7858–11258.5 |7756–11336 |280.64–402.09 |

NEO281 is metadata-eligible, with notes merging NEO282/NEO284 into that individual. The one-per-site rule retains higher-coverage KK1 at Kotias Klde and SATP at a different site; it does not assert NEO281 and KK1 are the same person.

The 12 frozen query subjects are NEO935,NEO898,NEO23,NEO739,NEO943,NEO866,NEO945,NEO738,Gjerrild5,NEO28,NEO942,NEO790. They have source Group label Denmark_Neolithic, positive direct dates, mean dates3971–5668 cal.BP and95% CI envelope3851–5747 cal.BP. That cultural label does not prove the simulated Neo ancestry state. Under the explicit working28-year convention they span141.82–202.43 generations; all 20 selected subjects' individual dates and intervals are frozen.

The28-year convention is a predeclared modelling assumption, not established by these metadata tables. BP zero is1950; alignment with model/inference time units is a future root gate. Fractional generations are retained. Frozen25/28/29/30-year stress cases expose sensitivity rather than silently adjusting dates.

Configuration-only inspection of the existing R96 simulation pilot shows pure-reference generation ranges Ana227–310,CHG184–473,WHG216–425,EHG176–389. Actual query ages are the discrete0,135,155,170,200 generations. Some proposed proxy means lie outside the current sampled reference-age support, for example Bichon~488 versus WHG maximum425 under the working28-year convention. This is a visible transfer/domain issue, not a discovered accuracy result. Original simulation source/events/results remain unchanged.

Before any real biological conclusion, root must qualify NEW paired simulation reference-panel conditions R16 versusR96 and original versus metadata-shifted sample-age policies, preserving fixed query correspondence, complete observed trees, source event-time legality and separately held-out exact-support evaluation. Required stress cases include lower/mean/upper95% date bounds,25/28/29/30-year generation conventions, proxy mixture/reference imbalance and ancient observation error/ascertainment. Date bounds are stress cases, not a calibrated dating posterior. No duplicated references, hidden class removal or alteration of the original pulses/census is allowed to manufacture compatibility.

## Frozen falsifiable endpoint

The prospective GRCh37/b37 region is chr22 absolute0-based[20,000,000,22,000,000), chosen without AP. Convert VCF POS by subtracting1, then subtract20,000,000 to obtain the original2Mb regional coordinate. Use retained100bp centers, original2Mb normalization,250kb physical tiles and exact covered bin/query-bp accounting; no tail extrapolation.

Prospective observed site QC requires biallelic A/C/G/T transversions, the MASK_1000G Flag, imputation INFO>=0.8, and external-panel MAF=min(RAF,1-RAF)>=0.05. Target-cohort AF never selects the sites. Every selected subject must have a callable diploid genotype and valid GP with max>=0.8; the same SNP set is used for every run. Polarization requires an unambiguous pan_troglodytes nucleotide matching REF orALT; ambiguous/third/missing states are excluded. REF is not automatically ancestral. Actual GRCh37 recombination map/version/hash is required; no invented constant map. Require at least1000 retained SNPs or fail this fixed admission condition without tuning thresholds.

The primary claim is observation stability of each fixed method/checkpoint: median query posterior-observation Jensen–Shannon sensitivity<=0.05 nats. Compare the phased-GT baseline with each of the fixed coherent HS iterations0–7. For each diploid query, average its two haplotype C4 vectors; integrate natural-log JS divergence over identical represented100bp bins using exact physical weights, divide by exact scored query bp, average across eight draws, then take the median across 12 subjects.0log0 is defined as0; no probability floor is added. Values above0.05 fail the stated stability claim. Report every method separately and all individual sensitivities/support, without selecting a method/checkpoint on real data. Simulation-trained checkpoint identities must be sealed before extraction; disclose any frozen ARGformer within-ARG unsupervised adaptation separately.

This endpoint measures response to posterior genotype/phase uncertainty. It does not establish correct ancestry, true phase, genetic purity, novelty or superiority. AP header definitions and values must be removed before any observed genotype/tree/model artifact. Published AP cannot choose references,QC,features,tuning or labels. Any future concordance uses a separate comparator namespace opened after predictions are sealed and cannot be called biological accuracy.

## Handoff and future gates

Primary evidence is under experiments/results/public_realdata_metadata_join_v1/: completion.json,subject_metadata_join.json,frozen_endpoint.json,age_domain_contract.json,GLIMPSE_phase_HS_semantics.json,GT_HS_GP_bounded_probe.json,independent_reconciliation.json and bounded_source_fetch receipts. Their SHA256 values are sealed in handoff_manifest.json.

The metadata-only script is scripts/public_realdata_metadata_join_v1.py. It uses Python standard-library XLSX/XML extraction, preserves source cells/cached values, requires the dedicated project root and refuses replacement outputs. It invokes no scientific package, fitting, GPU or Git command. Run modes are fetch,inspect,probe,join; existing source/receipt filenames are deliberately fresh-only. A new attempt requires a reviewed new destination/version instead of overwriting these sealed files.

All work used source scripts/project_env.sh on anogena. Accepted small source files plus rejected HTML total 873,379 received / stored bytes; the additional1MB chromosome prefix was received only transiently. No full chromosome or large archive was downloaded; no installation, GPU, simulation truth/score access or ancestry analysis occurred.

Future root gates remain actual target-region acquisition/QC, source genotype/HS provenance across that region, full chromosome publisher checksum availability versus explicitly disclosed limited integrity, map/polarization inputs, new R16 adapter/source/runtime admission, age/panel-size simulation sensitivity, frozen model/adaptation protocol and AP exclusion. A passed stability endpoint cannot remove these gates or supply missing true ancestry labels.
