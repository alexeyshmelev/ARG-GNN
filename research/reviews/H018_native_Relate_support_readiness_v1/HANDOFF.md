# H018 Relate support readiness: source evidence only

All 2,074 frozen sites must remain, including the 277 actual-40 monomorphic sites reported by root after corrected preparation. The canonical erratum records original 217 and extension 2,074 counts, with incorrect 434/4,148 reports preserved. Basis: data/H018_R16_observed_bridge_v2/completed.json, SHA256 b25d3f587dc5669b85410c7fca9bafd80f2f22ef7532e1c374538b7f2e339672, supplied by root. This audit did not read that output or any genotype payload. Independent actual readback was running at the last root update.

## Monomorphic preservation is a testable hypothesis

Pinned upstream [anc_builder.cpp lines 1051–1064](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/src/anc_builder.cpp#L1051-L1064) explicitly maps all-derived rows to the root and keeps all-ancestral rows with zero mutation branches. [The converter adds every unique-position site](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L289-L336), then [exports single-branch mutations](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L386-L405). This can represent both constant vectors without deleting rows. The inspected ingestion loop also does not filter zero/N carrier counts.

These source paths do not qualify the separately pinned installed release binary. Its build identity against the inspected commit remains unproven. In particular, multiple-branch mutations, flips, all-constant inputs and boundary rows need exact tests. Keep the current native blocker until root separately qualifies the exact binaries and full path.

Defensible options are unchanged all-site native qualification, a separately registered minimal lossless source correction for a concrete defect, or an unsupported-native result preserving prepared data. A reduced native run followed by restoring rows, filtering constants, intersecting sites or inventing flanking trees cannot establish this full-mask baseline.

## Fractional tip dates

[Finalize writes ages using %f](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/pipeline/Finalize.cpp#L199-L208), giving six fractional decimal digits. [The direct binary converter inserts the parsed doubles as sample times](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L338-L347). Proposed policy for root freeze: retain original calBP1950/28 .17g input bytes, expect the parsed six-decimal export, and bound source-to-native error by 0.5e-6 generations plus eight source-double ULP, with rtol=0. Each sample node must also equal its actual parsed .anc age within at most one double ULP. Qualify C locale and the actual serialization path. Avoid text-table/DumpStream precision and post-inference age replacement.

This tip bound does not qualify internal dates. Internal reconstruction uses floats, and the converter has an [explicit chronology adjustment loop](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L359-L374). Compare before/after times, record every adjustment and reject unreviewed repairs.

## Exact proof and physical support

Root's separately admitted tiny qualification should include both constant patterns at first/interior/last rows, phased heterozygotes, ancestral-ALT orientation, fractional ages, nonuniform map and duplicate-coordinate rejection. For actual data require identical 2,074 ordered native/site rows and decode all 82,960 biological nucleotide calls for the exact 40-haplotype roster. Check every one of the 277 constant vectors. Reject missing, displaced, reordered or rescued sites and all unproved allele/flip differences.

Raw converter sites retain local bp1. The exact physical join is local bp0 = absolute POS bp1 -20,000,001 = haps POS bp1 -1. The [raw sequence endpoint is last SNP +1](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L273-L304), with [first edge0 and midpoint internal boundaries](https://github.com/MyersGroup/relate/blob/b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74/include/file_formats/ConvertToTreeSequence.cpp#L376-L434). Record exact edge coverage for every retained site and distinguish requested raw [1,10,000,001) from actual support, including artificial left one bp and any unsupported terminal flank. Map endpoints do not prove full 10 Mb coverage.

Do not reuse the generic 280-haplotype/2 Mb runner: it filters monomorphic rows, uses a uniform map and clips support. A future adapter needs a separate frozen coordinate/coverage policy preserving every site and native edge support. Keep original normalization caveats.

readiness.json supplies source pins, line citations and the detailed proof checklist. Fourteen static source checks pass. No wrapper, native command, payload read, admission, guard, GPU or model execution occurred.
