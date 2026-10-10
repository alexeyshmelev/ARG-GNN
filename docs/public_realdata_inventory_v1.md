# Public ancient genotype inventory v1

H018 is a data inventory. No real-data ancestry hypothesis or accuracy result has been tested.

The AncestralPaths article cites Allentoft et al., *Population genomics of post-glacial western Eurasia*, DOI10.1038/s41586-023-06865-0, whose data availability links the public full analysis genotype archive at [DOI10.17894/ucph.d71a6a5a-8107-4fd9-9440-bdafdfe81455](https://doi.org/10.17894/ucph.d71a6a5a-8107-4fd9-9440-bdafdfe81455). Actual article XML, DataCite metadata, archive landing page, published file catalogue, README, passing-subject IDs, chromosome22 index and bounded header receipt were fetched on2026-10-10. Their bytes and hashes are recorded. The landing page explicitly exposes a public JSON file catalogue.

## Available product and QC evidence

The archive README describes1664 ancient diploid genomes imputed with GLIMPSE at26,198,700 biallelic1000Genomes SNPs on GRCh37. Chromosome22 is a1,654,398,745-byte indexed BGZF VCF. The actual header contains3331 sample columns, including pseudo-haploid _rnd columns. The1,492 IDs in noflag.inds.txt mark the paper's passing ancient subjects. Header count alone does not establish the exact diploid/pseudo-haploid/extra-column composition. That join requires a new explicit subject inventory.

The VCF is unfiltered. Its README records INFO imputation quality, MASK_1000G, CAP_TARGET_1240K, and FORMAT genotype dosage and posterior fields. Freeze sample/date/relatedness and SNP/genotype QC before selecting any real-data study cohort. Do not use all raw subjects/sites automatically. The actual GT header description says unphased despite the _ph filename. Determine actual phased calls and HS sampled-haplotype semantics before ARG inference. Never invent phase.

The same public VCF contains FORMAT AP, the published AncestralPaths paintings for the authors' selected individuals and SNPs. These are **external comparator outputs, not known true ancestry**. They must be removed from every observed genotype/tree/model-input artifact and must never determine reference identity, validation eligibility, features or model tuning. Real-data concordance can be compared with these published outputs after predictions are sealed. It cannot establish sitewise ancestry accuracy.

The archive does not expose full artifact checksums in its catalogue, instead saying please request explicitly. No contact was sent. Received metadata/index/header bytes have localSHA256 receipts, with HTTP ETag/Last-Modified/Content-Range. Those are not claims of a verified full chromosome checksum or publisher-certified content hash. The1Mb bounded chromosome-prefix request returned HTTP206 with exact length/range. Only the complete header was retained, and the full chromosome has not been downloaded.

## Separate error-validation product

The cited imputation benchmark, DOI10.1038/s41467-023-39202-0, exposes [Zenodo7993392](https://zenodo.org/records/7993392), an openCC-BY4.0 collection of42 downsampled/imputed ancient genomes. Its coverage archives range from3.08GB to5.54GB and provide six0.1x–2.0x conditions. The checksum-verifiedREADME says VCFs are split by chromosome. No coverage archive has been downloaded. This is a promising observation-error/robustness resource, not evidence of local ancestry truth.

## Next condition

Join the paper's supplementary biological metadata to the passing IDs and actual diploid genotype columns. Count unique usable pure-source proxies and admixed/query subjects after QC without consulting AP. Freeze a public real-data cohort, GRCh37 region/map/ancestral-state handling and a falsifiable concordance or robustness endpoint. Apply simulation-trained methods, disclose ARGformer's within-ARG adaptation and preserve published AP only in an external comparison namespace. Use a new lower-reference condition if the desired96 pure haplotypes are unavailable, with no duplicated reference leaves.

Evidence: experiments/results/public_realdata_metadata_inventory_v1.json, public_realdata_product_catalogue_v1.json, public_postglacial_file_inventory_v1.json, public_postglacial_chr22_header_probe_v1.json. Original small source files remain under .runtime/author_sources/ancestralpaths_public_realdata_inventory_v1/.
