# ARGMix original metadata recovery

The original public metadata now recover the identities and recorded dates of **116 reference individuals**, grouped as **13 CHG, 42 EHG, 48 WHG and 13 Yamnaya**. These counts exactly match ARGMix Section 4.5 after applying the stated quality and prior-painting requirements. The complete original 1,611-person roster remains unresolved. No arbitrary sample exclusions, farmer ancestry assignments or synthetic dates were introduced.

## What was recovered

The original [AncestralPaths supplementary workbook](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06705-1/MediaObjects/41586_2023_6705_MOESM4_ESM.xlsx) contains the full ancient metadata in Table S1 and the actual reference identities in Table S1.2.1. Its SHA256 is `f83c22d013aec5830788f8f3dab69daa48b16e0795095513e4a0a74a4d112e3b`. The 64 MB original is retained in ignored `sources/raw/`, not intended for Git.

Table S1 identifies 1,492 samples passing the published Flag=0 criterion. These identities agree with the independent public `postglacial_noflag.inds.txt` and the Allentoft supplementary metadata. Nine numeric IDs require only a documented Excel `.0` serialization normalization. The flag values `0` and `0.0` are equivalent. No biological filtering was added.

Intersecting that verified QC list with the public 1,015 previously painted ancient samples gives **992 individuals**. The published reference groups then divide as follows.

| Group | Recovered QC and painted candidates | ARGMix count | Status |
|---|---:|---:|---|
| CHG | 13 | 13 | Exact public reference identities recovered |
| EHG | 42 | 42 | Exact public reference identities recovered |
| WHG | 48 | 48 | Exact public reference identities recovered |
| Yamnaya | 13 | 13 | Exact public reference identities recovered |
| Combined farmers, before Iceman | 183 | 180 | Three exclusions still unidentified |
| Other ancient samples | 693 | 654 | Thirty-nine exclusions still unidentified |

The original Yamnaya reference list contains 15 individuals. RISE509 and RISE511 are absent from the public painted set, so the specified intersection naturally leaves the published 13. Matching these four reference groups is meaningful source recovery. It does not authenticate the authors' unreleased final roster file.

The 183 farmer candidates comprise 5 FarmerAnatolian, 40 FarmerEarly, 102 FarmerMiddle and 36 FarmerLate individuals. ARGMix additionally used its own K=4 ADMIXTURE run on 255,816 filtered variants and a WHG proportion below 0.03 to distinguish 22 Anatolian references from Neolithic farmers. Neither the three remaining exclusions nor that individual-level ADMIXTURE assignment were recovered. The Iceman is explicitly an additional NF individual, giving the paper's total NF count of 159 after its final selection.

## Date discrepancies that must remain visible

Both the AP Table S1 and the Allentoft full metadata agree on every recovered upper bound, lower bound and mean date. However, the public painted-sample TSV differs from Table S1 in **166 mean ages**, including 160 individuals passing QC. Moreover, **107 painted references have different mean dates between Table S1 and Table S1.2.1 inside the same published workbook**. All 107 reference-table values agree with the repository TSV.

| Individual | Full metadata, Table S1 | Reference table / painted TSV |
|---|---:|---:|
| Ukraine_N1 | 10643 | 7250 |
| Ukraine_HG1 | 7250 | 10643 in painted TSV, not an AP reference |
| syltholm | 5759.5 | 7709.5 |
| NEO669 | 9800 | 7950 |
| NEO310 | 6550 | 7150 |

These are source disagreements, not random-seed differences or floating-point noise. Many other differences are small calibration revisions. The tables retain the reported dates without choosing one silently. The MesoNeo methods describe calibrated and reservoir-corrected radiocarbon intervals and their midpoints. The recovered raw values are not already simulation generations. The ARGMix date-source choice, generation interval and point-date policy were not established, so `selected_simulation_age` and `age_generations` remain empty.

For the Iceman, the original paper gives 3350–3120 calibrated BCE and public sample SAMEA114266904, study PRJEB56570. The archive identifies ERR11837081 as the combined BAM candidate. Its ARGMix sample alias, exact selected BAM and single simulation date remain unconfirmed. The calibrated interval is preserved without inventing a point age.

## Present-day Europeans

The original HGDP/1KGP metadata contain 788 EUR candidates with unique sample IDs. Published release flags give 666 and intersection with the released chromosome-22 header gives 663. Excluding related samples and intersecting with that header gives 668. None establishes the reported 660.

The public original 1KGP Phase 3 panel contains 503 Europeans. Only 501 occur in the current high-coverage metadata and 492 in its chromosome-22 header. The current HGDP metadata provide 155 European candidates, 153 in that header. Thus a hypothesized 503+157 split does not recover the panel from these released resources. No duplicate IDs explain the difference. These policies are recorded as diagnostics, not treated as the authors' selection.

## Reusable outputs and how to reproduce

All derived tables are under `derived/`.

- `reconstructed_CHG_EHG_WHG_Yam_reference_116.tsv` contains public donor identities, source groups, all available date versions and spreadsheet row references.
- `ancient_QC_painted_992_candidates.tsv` retains the verified ancient candidate intersection and explicit unresolved membership flags.
- `farmer_183_candidates_unsplit.tsv` and `nonreference_ancient_693_candidates.tsv` expose exactly where the 42 remaining ancient exclusions are unknown.
- `modern_EUR_788_candidates.tsv` retains original population labels, quality flags, header membership and original Phase 3 membership.
- `ap_reference_list_318.tsv`, `all_1664_ancient_source_metadata.tsv` and `qc_1492_sample_ids.tsv` preserve the source lists needed to inspect the joins.
- `recovery_summary.json`, `modern_selection_diagnostics.json` and `source_manifest.json` contain counts, every date discrepancy and input checksums.

From the server repository root, run the reviewed plaintext script:

```bash
python3 -B research/argmix_roster_recovery_20261027/public_metadata/recover_public_roster.py
```

The script parses cached workbook values without running formulas or external workbook links. It reads only the relevant metadata sheets, validates uniqueness and the independent QC lists, preserves source differences, and writes only this recovery folder. It does not emit a purported complete simulation roster, launch experiments, alter the active benchmark or modify author code.

The next necessary information is the final sample manifest or equivalent documented filters resolving the 42 ancient and modern selections, the K=4 per-person assignments, and the actual age convention used for ARGMix simulations. Public metadata already replace invented donor identities and provide authentic dated candidates. They do not yet justify claiming an exact reconstruction of the full ARGMix panel.
