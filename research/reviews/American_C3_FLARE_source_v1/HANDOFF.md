# American C3 FLARE source handoff

**Status:** faithful pinned-native FLARE adapter source is prepared. No project genotype, truth, prediction or result was opened. No JAR was acquired, Java/native algorithm launched, package installed, resource admission created or Git used. Root owns all native qualification and CPU execution.

This supports the registered American EUR/EAS/AFR reconstruction. It does not establish exact numerical ARGformer Table1 reproduction. The publication's exact FLARE binary/options, sample-unit convention and marker-to-recurring-path coordinate rule remain unresolved. No new or guessed paper numbers are provided. Original published precision/recall and their primary source remain in docs/argformer_published_benchmark_audit_v1.md.

## Official pin, licence and requirements

Official repository: [browning-lab/flare](https://github.com/browning-lab/flare). Selected official tag: v0.6.0, commit ec0ad295aeebfacf10d8cbfb6486ab78981de1b1 (2025-10-31). The embedded native version is exactly:

    flare version 0.6.0 [04ad73fd0 31-Oct-2025]

The embedded short hash is the program's version label, not a substitute for the pinned Git commit. Current master was independently verified as d6a6c8fd4479ebdb68529824acd11d2b6bbac4a1. The official releases API returned an empty list. The README links an unversioned faculty-hosted flare.jar, which was not acquired and cannot alone certify tag/binary equivalence.

The runner therefore requires root-qualified unmodified source-build provenance for the pinned tag, an actual project-local JAR SHA, and the exact native version. Root can build the original Java sources under its own admitted scope, preserving the author implementation and recording compiler/JAR argv and JDK. This task compiled no Java. binary_provenance_inactive.json is deliberately NOT a qualification certificate.

Java11 or later is required by the official README. A source build also requires a qualified JDK compiler/jar tool. Current remote PATH presence checks found no java, javac or jar; this is not an exhaustive claim about all installed paths. Root must locate/qualify the actual runtime before execution. The runner verifies executable bytes and the complete java-version output SHA, then the native log version. It never downloads software.

FLARE is Apache-2.0. The official LICENSE-2.0.txt and Java headers were saved and verified. Cite Browning SR, Waples RK, Browning BL (2023), *Fast, accurate local ancestry inference with FLARE*, AJHG110:326–335, DOI10.1016/j.ajhg.2022.12.010. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9943733/). [Pinned official manual](https://github.com/browning-lab/flare/blob/ec0ad295aeebfacf10d8cbfb6486ab78981de1b1/README.md).

## Native algorithm/options retained

All ancestry estimation is done by the official FLARE JAR. There is no replacement classifier, custom emission model, truth-derived prior or injected admixture fraction. The fixed native options are:

    probs=true em=true update-p=false array=false min-maf=0.005 min-mac=50
    gen=10 panel-probs=false nthreads=4 seed=-99999

These preserve documented native inference defaults, apart from requesting posterior output and freezing the CPU thread count. gen10 is the native EM initial value; the simulator's known pulse12 is not supplied as an inferred answer. No model, gt-ancestries, gt-samples or excludemarkers option is supplied. The pinned source retains its original undocumented defaults: states100, IBS step.01/buffer2/haps4/recycle4, EM iterations20/haps100 and native convergence thresholds. Exact source files and hashes are included in the HTTP/source receipts.

The full reconstruction has300 reference diploid samples (600 haplotypes), so native min-mac50 is valid. The existing64-haplotype qualification has only12 reference diploids, so it is not an unchanged-default FLARE canary. The converter rejects this condition. Root should use a separately admitted short chromosome with the full reference roster for a faithful native canary, or separately declare a small-panel parameter deviation. It must not silently lower MAC and call it the published comparator.

## Observation and identity bridge

american_c3_flare_convert_v1.execute(cfg, writer) consumes only four SHA-bound inputs: the current inferred C3 observation manifest, its recorded simulation config, the existing common focal registry, and the manifest's observed genotype/position/allele/roster refs. It requires the upstream inferred sample/allele identity verification and rejects an oracle focal namespace. It never loads a true tree, pulse ancestry, private producer completion or target labels.

Original sample_index is the genotype-column identity. Every adjacent haplotype0/1 pair must share one diploid sample_id, role and reference class. Native VCF sample IDs are deterministic c3i_XXXXXX identifiers, with an explicit original sample ID/index and inferred-node bridge. Genotype left/right phase is exactly haplotype0/1; no rephasing or swap occurs. Reference VCF samples are ordered EUR/EAS/AFR, followed within each class by original identity. Query samples remain in original diploid order. The ref-panel file has the documented two columns: native VCF sample ID and known source label. No query ancestry enters it.

Positions must be strictly increasing discrete bp0 coordinates. VCF POS=bp0+1. Both VCFs carry the identical observed REF/ALT strings and all supported SNP sites, including three/four-allele sites. Native FLARE explicitly supports multiple ALT alleles. No undocumented biallelic filter, allele collapse or minor-allele recoding is applied. Unsupported/non-SNP/duplicate alleles fail rather than being silently dropped.

The converter freezes the expected marker grid using the exact pinned AdmixReader rule: reference allele counts, second-largest count at a multiallelic site, threshold=max(min-mac50,ceil(float32(min-maf.005)*reference haplotypes)). It does not pre-remove these records from VCFs; the JAR performs its native filter. The sealer requires every and only expected native marker, so missing output cannot silently remove difficult calls.

## Real simulated map, common support and outputs

The two-anchor PLINK cM map derives from the SHA-bound simulation config's recorded r=1e-8, matching parent, split, sample counts and sequence length. It has columns chromosome, markerID, cM, one-based physical position:

    1 map_start 0 1
    1 map_end 30 30000001

The second line represents the 30Mb domain's right boundary. Thus genetic position at a VCF marker is100*r*(POS-1), with linear interpolation. The entire30Mb interval is30cM. No real-human chromosome map is renamed, stretched or fabricated.

The original common focal registry is unchanged: same raw full-path ID, inferred sample node, original genotype-column sample index, bp0 coordinate, parent and EUR/EAS/AFR order. All its records are sealed before private truth. The prescribed reconstruction readout is nearest retained native marker, with midpoint ties assigned to the right marker. First/last marker values extend to the observation-domain edges; the number of edge-carried focals is reported. There is no purity filter, missing-call deletion or truth-based support change. This is a transparent reconstruction convention, not a recovered Table1 coordinate rule.

The native .anc.vcf.gz, .model, .global.anc.gz and .log are retained byte-for-byte with SHA refs, plus argv/stdout/JVM provenance. FLARE has no native continuous-tract table in this release. The optional derived_native_marker_tracts.tsv merges native AN hard calls over the same midpoint cells, explicitly labelled as derived, preserving each original haplotype index and full domain.

Pinned EstimatedAncestry uses TWO decimal places for ANP1/ANP2. AdmixRecBuilder computes AN1/AN2 from unrounded probabilities. Serialized ANP can sum to.99/1.01, and rounded argmax can lose the native call. Consequently:
- The full original VCF and raw focal ANP decimal strings/numeric values are preserved without floor or renormalization.
- Original AN1/AN2 supply classification calls. Common predictions.npz transports those calls as one-hot C3, as the existing native-retrieval comparator does. These one-hots are call transport, not estimated confidence.
- Raw focal posterior sidecars retain original ancestry order, native indices, canonical C3 values and the selected native marker.
- Common P/R/F1 comes from the original american_c3_evaluate_v1 confusion-count scorer, not an argmax of rounded ANP or a new classifier.

## Exact root execution API

Four new modules are owned here:

- american_c3_flare_common_v1: pinned constants, phased pairing, observed MAC convention and bounded gzip I/O.
- american_c3_flare_convert_v1: execute(cfg, writer) creates VCFs, panel file, actual uniform map, bridge and expected marker grid. completed.json status is completed_C3_FLARE_observation_conversion.
- american_c3_flare_run_v1: execute(cfg, writer) accepts conversion_ref, qualified jar_ref/binary_provenance_ref and qualified Java path/hash/version-output hash. It runs only official native argv. Status is completed_C3_native_FLARE.
- american_c3_flare_evaluate_v1: mode=seal_native accepts native_run_ref plus the identical focal_registry_ref and no truth. Status is sealed_C3_native_FLARE_calls_and_raw_posteriors. mode=score accepts that completion, its prediction_ref, common focal_registry_ref and separately bound external_truth_ref. It delegates to the existing scorer, which verifies complete predictions before opening truth.

Inactive config drafts are american_c3_flare_{convert,run,evaluate}_inactive.json. Set actual refs only in root-activated copies. For score, use a separate admitted scope/config with mode=score. The binary provenance template remains invalid until root verifies source build/JAR and fills actual refs. The runner requires the exact binary_provenance_ref JSON manifest in the existing storage reservation's consumer_bindings. That SHA-bound manifest carries the exact cfg.jar_ref; the runner separately streams and verifies the actual project-local JAR SHA before constructing the Java command. The JAR itself is not a consumer binding because the immutable storage API admits only .py/.sh/.json bindings. Root must retain and charge the JAR and build artifacts in their own admitted artifact scopes.

All modules use the existing american_c3_common_v1.cli authority and Writer: exact root admission/SHA, execution_enabled true, GPU_authorized false, qualified Python runtime/CUDA_VISIBLE_DEVICES empty, exact ordered source closure, current capacity-v2 storage adapter/marker authority and a fresh output directory. No new resource guard is introduced. Root must bind actual producer/helper/config/runtime and JSON binary-provenance refs under the current authority. The qualified JAR remains separately SHA-verified and retained in a charged artifact scope.

Metadata-only source descriptions:

    source scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.american_c3_flare_convert_v1
    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.american_c3_flare_run_v1
    CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m arggnn.american_c3_flare_evaluate_v1

Future execution uses each module's --execute --admission <project-path> --expected-admission-sha256 <digest>. No actual admission, binary or native launch is supplied by this task. The native runner uses no shell, fixes English numeric locale and a project-local JVM tmpdir, bounds its own child products/stdout/tmp and timeout, checks the existing storage guard during execution, and preserves partial failure without fallback.

## Bounded resource forecast

resource_forecast.json contains exact count formulas using the inactive full1600-haplotype source config, not observations or measured output. At the declared1,000,000-site ceiling and up to that many native-retained markers:

| Stage | Count-bound bytes | Inactive scope proposal |
|---|---:|---:|
| VCF/map/bridge conversion |3,790,500,784|8,000,000,000|
| Native ancestry/model/global/log/stdout + JVM tmp |33,709,032,408|42,000,000,000|
| Focal/raw-posterior seal +10M derived-tract cap |981,901,312|2,000,000,000|

These allow gzip/BGZIP expansion and assume no compression saving. Native marker filtering can reduce output but is not budgeted as a guaranteed saving. Existing observation files/JAR/source build artifacts require their own retained scopes; these stage bounds are not a bulk35-parent reservation.

The native proposal is4 FLARE threads, JVM ActiveProcessorCount4, Xmx16GiB, up to1GiB project-local scratch and a4h timeout. Xmx is not a total-RSS bound; root must qualify native/JVM overhead and enforce its actual host-memory/CPU scope. No production runtime is measured or promised. Failure preserves partial files and never reduces markers/samples or changes options.

The adapters mmap observed genotypes, retain O(S) coordinate/index arrays, stream VCF rows and keep only two native marker rows plus O(Nfocals) posterior arrays. Python containers, mapped pages and serialization overhead still need root RSS qualification. Source/API checks compile/import only; no numerical genotype fixture or native canary was run.

## Scientific reporting boundary

For actual runs, report original native version/JAR/source/Java/options, genotype and phased-identity provenance, observed marker retention by allele count, common focal projection and edge counts, actual estimated model/global ancestry, and one3x3 confusion matrix per independent ARG. Derive class P/R/F1 from each run's counts, then report between-ARG mean/sample SD and paired differences under the existing C3 protocol. Keep American C3 independent from European C4 results.

The method remains the original FLARE biological model of reference-haplotype copying with native HMM/EM estimation. This source handoff creates no accuracy claim, real-human biological discovery or numerical equivalence to the article's unresolved benchmark pipeline.

## Verification artifacts

Primary HTTP receipts pin the official tag/head, README, licence, parameter parsing, marker filtering, genotype/ancestry writer, posterior formatting, map reader and primary-paper metadata. source_API_check.json records successful imports/compilation with inactive defaults and no scientific arrays or Java/JAR invocation. source_closure.json and completion.json identify the exact final adapter sources. No model/author/current-job source was modified.

## Metadata binding correction

The unadmitted runner was corrected after the immutable v4 scope API was found to restrict consumer bindings to .py/.sh/.json. Native options, author code, reference/query conversion, projection, outputs and scoring are unchanged. Only the JSON build-provenance manifest is required as a consumer binding. It must pin the same exact JAR ref as the activated config, and bound() independently verifies the actual JAR bytes. The inactive provenance template still requires genuine root build/runtime qualification. The source check exercises metadata-only acceptance/rejection with fabricated manifests and no JAR access.
