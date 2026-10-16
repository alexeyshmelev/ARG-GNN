# Danish genotype-only corroboration source

This inactive source freezes the prospectively chosen18-person roster and GRCh37 chr20/21/22[20,30)Mb windows. It is ready for source review, not registered/admitted and not executed. Root owns H055 registration, actual acquisition/scoring, resource callbacks and integration.

The source reuses the existing bounded H018 HTTP Range/BGZF/TBI approach in a chromosome-parameterized standard-library module. The old source is preserved. There is one analysis module, one small root-deferred synthetic fixture, an inactive protocol, the byte-identical prospective roster, and source receipts. No environment install or model dependency is needed.

## Frozen estimator and joins

Earlier: NEO866,NEO945,NEO28,NEO942,NEO790. Later: NEO739,Gjerrild5,NEO735,NEO92,NEO870. All ten retain their exact prospective dates, archaeological sites, coverage and source metadata rows. All eight named proxies remain in the retained dosage table, with unchanged identities and no purity claim.

Primary contrasts:
f4(later,earlier;Sidelkino,Bar8)
f4(later,earlier;KK1,Bar8).

The cached original chr22 header declares GP Number3 Float, GT “Unphased genotypes,” and GLIMPSE_phasev1.0.0. GP uses diploid biallelic order0/0,0/1,1/1. Freeze dosage=(GP01+2GP11)/sumGP when all three probabilities are finite in[0,1], sum>0, and abs(sum-1)<=0.002. This explicit normalization handles printed posterior rounding. There is no GT/DS fallback, phase draw, hard call, or sample maxGP threshold. Full GP triples are not retained; the FP64 expected dosage and validity mask are retained.

Freeze unique-position biallelic ACGT transversions with FILTER=PASS, INFO>=0.8 and MASK_1000G. No target AF, reference-panel MAF, inferred ancestry label or ancestral polarization enters the contrast. Multiple rows at one position are all excluded, rather than choosing a favorable row. Every metadata-eligible SNP is retained even if GP calls are missing. Both contrasts and subject/site deletions use ONE common support requiring all ten queries plus Sidelkino,KK1,Bar8 valid. The other five retained proxies do not impose extra primary missingness exclusions.

Each file must reproduce the exact18 ordinary sample IDs and ordinary-column metadata joins. It must have GP Number3 Float and the expected hs37d5 reference declaration. Exact chromosome labels, original POS1, REF and ALT define the variant key; bp0=POS-1 without liftover, strand matching, aliasing, or coordinate rounding. The source validates conflicting target-contig assembly/length if supplied. The previously observed missing chr22 contig declaration is explicit: b37 reference metadata plus the exact chr22 TBI name is required, with no fabricated contig length.

Source AP ancestry-bearing rows remain bounded opaque RAM input. AP, HS, GT and DS values are never interpreted or retained. Only GP from the selected18 sample fields and the frozen observation-QC metadata are parsed. This is corroboration from the same imputed genotypes, not independent ancestry truth.

## Bounded acquisition and retained artifacts

Root calls run_v1 after review/registration and normal resource admission:

    run_v1(protocol_path, new_absolute_output_directory,
           root_authorization=reviewed_exact_H055_binding,
           before_request=root_existing_network_storage_boundary,
           before_write=root_existing_storage_boundary)

The root_authorization argument must bind registered=True, execution_authorized=True, hypothesis_id=H055, the actual source/protocol SHA256 values and16MiB output cap. This module supplies no accounting/admission implementation. The command-line entry point only describes source. There is no direct acquisition CLI.

Each chromosome uses exact URL/length, strong ETags, If-Match206 ranges and a four-window LRU. Any ignored Range, redirect, changed identity, missing required header field or invalid join fails. The pre-existing chr22 ETag/index SHA are frozen. Chr20/21 strong ETags and index/header hashes will be acquired and recorded by root. Full publisher checksums remain unverified. There is no whole-file HTTP/HTS fallback.

The TBI preflight plans only regional virtual chunks. Compressed transfer is capped at512MiB per chromosome,1.5GiB total. Expanded source is capped at32GiB/chromosome; records at16MiB; duplicate-position groups at8rows; wall time at3600seconds/chromosome. These are unqualified prospective caps, not measured requirements. AP-bearing source VCFs have3331 columns, so selecting18 outputs does not eliminate the cost of reading original regional records.

Durable run output is capped at16,777,216bytes. A64-byte binary header binds the protocol. Each metadata-eligible SNP consumes one160-byte record: chromosome,1Mb block, exact POS1/REF/ALT,18-bit GP-valid mask, metadata/common-support flags, and18FP64 dosages. At100,000rows plus512KiB planned metadata, total is16,524,352bytes. The source hard-stops on a101st-thousand eligible SNP or any actual durable overflow and preserves incomplete output. Actual site counts are unknown. It never thins variants, changes filters/windows/proxies or switches dosage encoding to force fit. If the fixed panel cannot fit, root must review a new capacity/source version before completion.

ranges.bin retains compact byte-range SHA256 records. source_receipt.json retains object ETags/lengths, index/header/regional-stream hashes, joins, virtual chunks and QC counters. blocks.json retains all30fixed1Mb blocks, candidate/common counts, per-proxy f4 sums,18subject-valid counts and10query-by2proxy terms. dosage_records.bin retains per-SNP support, including invalid GP masks, without quantization. A completed receipt appears only after all three chromosomes succeed. Partial files never establish a completed estimate.

The simple reader bounds duplicate input buffering by8x16MiB, plus the current line/decode/range/parser workspace. This is a finite RAM bound, not a claim of a few-megabyte peak. Output-buffer/runtime qualification remains root-deferred.

## Estimand, uncertainty and limits

At each common SNP, p_group is the equal-person mean ALT dosage divided by2; each named proxy contributes its diploid expected ALT fraction. The pooled contrast is an equal-SNP mean over all qualified sites in the three fixed windows, with SNP-count-weighted pooling and separate chromosome means. This differs from a physical-length-weighted model score. A common REF/ALT swap leaves the product sign unchanged; no ancestral allele is needed.

Positive direction is predeclared only. There is no magnitude threshold, significance verdict, ancestry accuracy, Yamnaya fraction, admixture date or population-wide causal interpretation. Five persons/sites per era and three chromosomes are a small finite panel. Genetic unrelatedness and proxy purity are unproven. GP marginal means do not supply a joint genotype posterior or account for imputation/reference bias.

Subject and archaeological-site deletions are the SAME ten units here, since one person was chosen per site. They retain original common SNP support. Chromosome means/sampleSD and leave-one-chromosome estimates are descriptive. Fixed1Mb block sums and delete-block ranges/SD expose genomic sensitivity; no SNP-independent p-values, Gaussian Z scores, or independent-block confidence claim is produced. Proxies remain fixed in the two primary contrasts. Extra retained proxy dosages enable a separately declared future sensitivity, not unregistered primary replacement.

## Validation and root steps

Only AST/metadata/schema/layout checks were performed by this source author. The module was not imported. No genotype row, acquisition, f4 metric, scientific fixture, model/GPU job, guard/admission operation, install or Git operation ran.

Before payload access, root should execute the tiny synthetic-only fixture and review header/TBI/network failure paths under an admitted caller. The fixture covers GP normalization and invalidity, GT/DS/AP invariance, incomplete-call retention, transversion selection, multi-member gzip/TBI and binary record sizes. Actual runtime/source qualification is not claimed. Root then freezes registration/actual callbacks and acquires all three windows. Actual counts, coverage, memory, timing and f4 values remain unknown.
