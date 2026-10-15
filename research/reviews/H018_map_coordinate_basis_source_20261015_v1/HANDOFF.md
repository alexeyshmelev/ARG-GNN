# H018 map coordinate basis: source evidence and next root helper

The pinned GLIMPSE map is consumed in **absolute one-based base positions**, the same numeric basis as VCF text POS. Use `position_basis="absolute_bp1"`. Its actual three columns are **pos, chr, cM**; the middle column is chromosome22, not recombination rate. Use `source_format="pos_chr_cM"`, `exact_header_tokens=["pos","chr","cM"]` and `cumulative_distance_units="cM"`.

The evidence chain is closed at GLIMPSE commit103ca071ea9e4e648192916142e5ed353ac94e27:

- `genotype_reader.cpp:119` takes `line_main->pos + 1`: HTSlib's internal BCF position is zero-based, while VCF text POS is one-based. Line123 passes that position to the variant constructor.
- `variant.cpp:29` assigns `this->bp = bp;` without another offset.
- Retained `variant_map.cpp:61` indexes the unchanged variant.bp.
- Retained `gmap_reader.cpp:46` reads source map column0 as bp unchanged, and line47 reads column2 as cM.
- Retained `variant_map.cpp:67` matches source pos_bp directly against variant.bp. Line89 interpolates using these same values, without a coordinate conversion.

`coordinate_evidence.json` gives exact source references, hashes and line anchors. The two newly needed public source files were absent from the existing source inventory. Only the explicitly authorized genotype_reader.cpp (6,703 bytes) and additional variant.cpp (1,907 bytes) were fetched, once each in memory, with exact URL, byte count and Git blob verification. Their complete copyright-bearing source and separate fetch receipts remain here. Total new public bytes: **8,610**. No bulk or scientific payload was fetched.

The map already acquired by root has SHA256 `75b0026033139f7c2515eda80c330d93d1eb728374a31e0a713589c0d413b24f`, Git blob `eec8ceff2fdff0d70badcb09c18e5c6cc5b04854`, 322,186 compressed bytes and 994,107 expanded bytes. Source metadata inspection found45,329 three-column rows, all chromosome22, strictly increasing positions and finite nondecreasing cM. Positions span16,051,347–51,229,805; cumulative cM spans0–74.109562. The source release path is `maps/genetic_maps.b37/chr22.b37.gmap.gz`, matching H018's frozen GRCh37/b37 assembly contract without liftover.

Both helper boundaries are bracketed by actual source rows:

| Absolute bp1 boundary | Left source bp1/cM | Right source bp1/cM |
| --- | --- | --- |
| 20,000,001 | 19,999,760 /11.064291 | 20,004,704 /11.068168 |
| 30,000,001 | 29,997,574 /34.193560 | 30,002,440 /34.193565 |

There are12,278 source knots strictly inside those boundaries. The existing helper should retain all of them and add two boundary interpolation rows, giving **12,280 output rows**. Decimal arithmetic on the source's printed cM values gives a boundary-to-boundary distance of **23.12908250576842884448860530169713366774 cM**. This is source-map arithmetic, not a genealogy result. Float output may differ in trailing rounding. Divide cM by100 for Morgans.

`map_semantics.inactive.json` is now fully populated with these facts and exact semantic_evidence_refs. It remains deliberately inactive. Root should review it and seal a separate copy with `status="root_reviewed_source_semantics"`, then separately admit the existing `H018_genealogy_next_source_v1/qualification_source.py:write_root_metadata_stage_v1` under the current actual guard. `root_helper_call.inactive.json` supplies the exact map ref, helper ref and fresh output-stage basename; guard/storage/active-semantics references remain root-owned placeholders.

The proposed helper output path is `research/reviews/H018_genealogy_next_source_v1/root_metadata_stage_basis_v1`. It needs a fresh matching output reservation: the acquisition reservation only owns its acquired_v1 directory. The helper's existing compact bundle cap is4,000,000 bytes. No old source/guard needs alteration. No helper was invoked by this task.

The conversion is:

- map source bp1 and VCF text POS already share the same basis: do not add1 to either source value;
- absolute bp0 = VCF POS minus1;
- regional bp0 = VCF POS minus1 minus20,000,000;
- local haps/map bp1 = absolute bp1 minus20,000,000;
- absolute map boundaries20,000,001 and30,000,001 become local map boundaries1 and10,000,001.

The helper keeps each interior public knot, interpolates only those two boundaries and subtracts a constant genetic origin. Output rates are derived as delta(cM)/delta(bp)*1e6 in cM/Mb. Source rate-rounding tolerance is null because the source has a chromosome column. GLIMPSE's separately defined uniform-map overload and out-of-range extrapolation branches are not selected; no uniform map substitution or extrapolation is needed here. Native Relate conversion/actual SNP joins still require separate root execution and verification.

This establishes the pinned source's coordinate interface, units and regional map coverage. It does not independently establish historical map calibration/population origin, exact empirical SNP/haplotype joins, ancestry, inferred genealogy coverage or runtime success. H018's original434-site failure and extended4,148-site acquisition remain unchanged.

Fourteen bounded source/metadata checks passed. No actual VCF, tree, model, checkpoint or target was read. No guard, admission, process/GPU check, metadata/map helper, native inference, installation or Git operation was performed. Root retains every execution/accounting decision.
