# Real-data and PCA benchmark audit, 15 October 2026

Completed source/metadata audit. No genotype, tree, ancestry, model/checkpoint payload or scientific analysis was acquired or run. No GPU, installs, admissions, scheduler/guard changes, H040 changes, commits or outside-repository writes were performed.

The practical next study reuses H018: **8 reference subjects / 16 haplotypes, 12 queries / 24 haplotypes, chr22 GRCh37 [20,22) Mb**. Its exact ID and readable metadata exports are ready in this directory. R16 empirical adapters, target-region QC, map/polarization, age support and root admission remain outstanding.

- [Validation plan](../../../docs/real_data_validation_plan_v1.md): matched inputs, readouts, stability endpoint, publication panels and concrete extraction command.
- [Asset inventory](asset_inventory.json): exact URLs, access/license distinctions, public IDs, sizes, phasing/build and missing original-paper assets.
- [Figure contracts](figure_contracts.json): 19 figure/quantity records with axes, normalizations and replicate units.
- [Frozen IDs](H018_subject_ids.txt), [metadata roster](H018_selected_subject_metadata.tsv), [export provenance](H018_metadata_export_receipt.json).
- [Source/evidence hashes](evidence_file_hashes.json), [artifact hashes](artifact_hashes.json), [verification receipt](completion.json).

The central comparison distinctions are:

| Figure | What is measured |
| --- | --- |
| ARGMix Fig. 2B | Genotype PCA after masking non-Anatolian haplotype alleles. Not PCA of ancestry fractions. |
| ARGMix S5 | Fractions from summed local ancestry calls. |
| AP S8/S26 | Genotype PCA; S26 selected diagnostic subjects. |
| AP S11/S16 | Simulated precision matrices: predicted rows / true columns, each predicted row normalized. |
| AP S1 | Separate fineSTRUCTURE/ChromoPainter copying matrix. |
| ARGformer Fig. 2A vs 2B/C | Individual genotype PCA versus local path-embedding PCA on simulations. |
| ARGformer Fig. 4/5 | Top-5 retrieval shares; color is log2 lift relative to corpus composition. Not admixture percentages. |

Verified public assets:

- Allentoft chr22 VCF: **1,654,398,745 bytes**, plus **28,008-byte TBI**. AP strings must be stripped from observed artifacts.
- Modern HGDP+1KGP chr22 SHAPEIT5 **BCF: 278,689,536 bytes**, plus **27,252-byte CSI**. GCS generation/CRC32C metadata is frozen. The ARGMix guide names a VCF while the public release supplies BCF.
- Ötzi merged BAM: ENA **ERR11837081 / SAMEA114266904**, **34,206,836,802 bytes**, plus **9,018,432-byte BAI**. Exact URLs and published MD5 are in the inventory and ENA TSV. The four library BAMs are the same individual.
- Wohns chr22_q dated genealogy: **38,871,562 bytes**, MD5 **f2737a8063735710944b0eab983df7bd**, CC BY 4.0.

The original ARGMix merged 1,611-subject VCF/reference-date panel/checkpoint/PCA configuration is not established in the inspected current repo. Ancient inputs are GRCh37, the modern harmonized resource is GRCh38, and original-paper coordinate reconciliation is not documented in inspected methods. Current snputils has multiple missing-data algorithms; its latest defaults do not reproduce an unpinned paper call.

ARGformer original empirical task IDs/chromosome-arm sets/checkpoints/figure-generation settings are not established by its current source. Its 90/10 split is over unique paths inside one ARG; its supplemental 95.96% score is masked-token accuracy, not real ancestry accuracy. A real ARG needs its own node vocabulary and a declared transductive adaptation protocol.

Source acquisition reused H018 metadata and author pins. New bounded reads were public full-text XML, repository text/tree metadata, ENA TSV manifests and GCS JSON object metadata. HTTP receipts retain URLs and hashes. No bulk data request was made. ARGMix primary XML and extracted supplement text were supplied by the parallel paper audit and are cited by their existing paths/hashes.
