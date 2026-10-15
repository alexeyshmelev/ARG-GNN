# H018 R16 regional extraction adapter

Ready for root source review/admission. **No empirical acquisition has executed.** Source: src/arggnn/prepare_h018_r16_regional_extraction_v1.py. The API is execute_v1(admission_ref, guard=actual_root_guard). Its only CLI action is --source-report.

It reuses the exact frozen H018 order: Bar8, Bon002, KK1, SATP, Loschbour, Bichon, Sidelkino, NEO88, then the twelve existing query subjects. It preserves eight reference subjects/16 haplotypes, twelve query subjects/24 haplotypes, and GRCh37 chr22 absolute0-based[20,22)Mb. VCF POS remains1-based; regional0-based position is POS−1−20,000,000. Dates remain positive cal.BP1950 with the existing fractional BP/28 working values. Inference time origin and model age support remain unqualified. Proxy labels are not pure-ancestry truth.

## Acquisition and retained products

The client parses the frozen public TBI and performs explicit HTTP Range requests in1,000,000-byte windows. It validates HTTP206, exact Content-Range/Content-Length/full-source size, fixed strong ETag, first1MB received-prefix SHA256 and exact full TBI SHA256. HTTP200/full-file fallback is refused before reading its body. Successful received ranges retain only hashes/provenance. AP-bearing source BGZF bytes stay in bounded RAM and are never written to disk.

The source TBI has one target chunk. The compressed chunk is about71.43MB; the complete unique-window request forecast, including the header and index, is74,028,008bytes. The hard received-byte cap is100,000,000bytes, at most128requests, four cached windows,16MiB per raw record,8GB cumulative decompression and900seconds at request/BGZF boundaries. Requests have30second timeouts. The actual root storage guard is called before every request and retained write.

Before the first retained observed artifact, the adapter selects the20ordinary subject columns, removes all FORMAT/AP definitions/values and source command/history metadata, and allows only FORMAT GT/GP/HS/DS and INFO INFO/RAF/MASK_1000G/pan_troglodytes. It excludes every _rnd column and verifies ordinary-column identities against the frozen metadata. It does not decode AP, use AP for QC, or treat AP as truth.

The observed artifact is an AP-free regional VCF, then an AP-free BGZF VCF and local TBI. Existing bgzip/tabix/bcftools1.24 binaries are SHA-bound, used locally only, and verify sample order and indexed record count. Their remote HTS autoloading is never used. No pysam installation is needed.

The public frozen header lists b37 contigs1–21 but omits22. The derived header explicitly adds ##contig=<ID=22,assembly=b37> using the frozen H018 assembly and source/TBI identity. It does not invent a length or change coordinates. This adaptation is recorded. Frozen metadata and source files are not edited.

The200MB retained-output cap includes the AP-free uncompressed/compressed VCF, local index, site-QC TSV, frozen subject metadata, haplotype roster and compact receipts. Index creation has an additional4MB file-size bound. No raw source chunk or AP-bearing intermediate is retained. Partial AP-free products remain on failure.

## Qualification

Every candidate regional record is retained. The site-QC TSV reports the exact existing H018 criteria, without inventing original-paper filters or silently changing the SNP panel: biallelic A/C/G/T transversions, source strict-mask flag, INFO≥0.8, finite external RAF-derived MAF≥0.05, all20phased callable diploids, valid rounded GP triplets/maxGP≥0.8, valid30bit HS with NMAIN15, and one source pan_troglodytes base matching REF/ALT. Target AF is not used for MAF. GT/GP/HS values are preserved; HS draws are not generated.

The minimum1,000fixed-QC SNP gate is reported separately. If it fails, the H018 cohort/region qualification fails; no threshold or roster is changed. The extraction completion is not an R16 model admission. Genetic map qualification, inference time origin, R16 adapters, observed genealogy inference, model predictions, observation-sensitivity runs and PCA remain future root-owned stages.

## Root admission interface

1. Preserve config.json and source hashes after review.
2. Save runtime.json, which already records the existing HTS runtime; reserve source/config/runtime, all three tool refs, all ten frozen refs, current guard/repair/marker closure and the exact fresh output scope. No admission was created here.
3. Fill admission.draft.json into a new SHA-bound root admission: execution_authorized=true, fresh expiry, storage_ref, runtime_ref, root job ID and exact data output directory. The storage record must be active, guard-enabled, reserve at least200MB and match that directory.
4. Configure the root's current guard and validate its storage. Then import the adapter and call:

    result = acquisition.execute_v1(admission_ref, guard=lambda planned: active_guard.budget_check_v9(
        storage, boundary=True, required_write_bytes=planned))

The root wrapper must bind that callback to the same storage_ref. The adapter independently verifies scope/cap and complete consumer closure. No legacy driver or resource admission is invoked by this source. Capture actual process wait/exit and preserve completion/failure receipts.

## Source-only checks

handmade_check.py and handmade_checks.json cover syntax/pins, frozen ordinary-column joins, exact region endpoints, AP removal, fixed GP/GT/HS/transversion rules,30bit HS decoding, public TBI planning, mocked206accounting/200refusal, BGZF cross-block/virtual-offset reading, and an actual offline bgzip/tabix/bcftools round trip. Genotypes are wholly handmade. The AP-bearing synthetic input stays in RAM; retained synthetic fixtures are noAP only. No public request, empirical genotype read, inference, model, PCA, GPU, installation, Git operation or admission occurred.
