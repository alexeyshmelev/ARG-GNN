# Independent prospective1611 generator review

Verdict: source-only pass with declared prospective scientific deviations. No simulation, native command, package import, numerical fixture, array read/hash, process probe, admission or Git operation was performed.

All four packet source/config/document/receipt pins independently match. The complete source parses and compiles without execution. Top-level imports are standard library only. The existing registry contains77 distinct streams and exactly nine implemented IID-phase parents, each100000planned examples. The source explicitly rejects five deferred misspecification parents.

## Biological and coordinate checks

SampleSet(1,ploidy=2) creates one diploid individual, not one haplotype. This was independently verified against the installed msprime SampleSet documentation and _insert_sample_sets source, which inserts n*ploidy nodes and one paired individual for n=1. The generator verifies3222 ordered sample nodes, their actual ages/populations and every diploid pair.

The exact roster arithmetic is1611diploids,3222haplotypes,594donor haplotypes,2972query haplotypes and250pure-founder references. NF/Yam retain dual roles. Focal-diploid identity exclusion removes both haplotypes before nearest-TMRCA ranking. Stable haplotype IDs break ties. Thus NF/Yam focal donors number592, while PostNeolithic/Present focal donors number594.

All declared samples precede the retirement of their sampling population. Present/PostNeolithic ages0/135 precede136, NF180/185 precede186 and Yam160 precedes161. Census350 follows all admixtures and all sample ages, and precedes founder divergences600/800/1500. Classes are joined by explicit population names, preventing alphabetical population-ID order from changing labels. The separate truth join climbs to the first census350 ancestor at the frozen target coordinate.

Empirical rates use delta cumulative cM divided by100*delta bp, which supplies Morgans per bp to msprime. The declared interval is35178458bp with74.109562cM and45329knots, with no map extrapolation. Relate export uses cM/Mb rates and shifts both map and haplotype coordinates by+1. The conversion subtracts that same1, clips edges to the observed SNP interval and records the local-to-rebased offset. This matches the project's existing official-Relate coordinate contract in src/arggnn/relate_pipeline.py. The map contents themselves were not reparsed during this source-only review.

One independent PCG64 Bernoulli0.001 draw is made per haplotype at each simulated mutation site, before the sole post-error segregating filter. The retained error realization feeds VCF and packed states, then native haps. No second error draw or truth-derived feature is present. Census trees/populations remain in a separate truth directory. Observed roster exports sampling ages, roles and synthetic cohort identities, while sampling_population and true trees are omitted. Native-converted mutation tables are explicitly forbidden as genotype features.

## Native bridge and support

prepare_relate creates native format inputs and exact pinned1.2.4argv, without subprocess launch. Root supplies genuine successful waits. finalize_relate checks the actual command-manifest identity, sample count/ages, exact exported SNP positions and complete single-root clipped trees. It preserves inferred node IDs and returns explicit sample-index/stable-ID-to-node joins.

Sample order has a precise limit: count and age checks cannot distinguish a permutation among same-age samples. The bridge relies on the native Relate/ConvertToTreeSequence index-preserving .haps/.sample column contract. It must not be described as independent identity recovery from the output. Root must execute the exact pinned native commands and adapters must preserve those explicit joins, rather than reconstructing cohort or query ranges from node age. A native success or sample-order observation has not occurred in this review.

Sampling uses only inferred observations. Zero-based native tree modulo10/100 selects TRAIN/VAL support, IID uses all positive-span trees, and edge trees remain eligible. One uniform without-replacement sample of100000tree/query pairs is frozen before truth. No cohort- or class-conditioned redraw is allowed. Missing support/cohort stops. Midpoints are mapped to full rebased coordinates and joined with right-continuous tskit lookup. Reported future sampled-target accuracy will be a tree/query-target estimand, distinct from base-pair span accuracy on the older100bp benchmark.

## Scientific deviations and root prerequisites

This is a prospective common-data benchmark inspired by published counts, not an exact ARGMix/AP paper reproduction. Synthetic integer sample dates, the185generation older-NF proxy, deleting the unused BAA branch, explicit error model, empirical chr22 interval, modulo support rule, midpoint targets and current-release algorithms are all new/disclosed choices. The older-NF proxy is not an identified Iceman. Relate uses N30000,mu1.25e-8 and --mode All without the older development pipeline's extra demographic/branch refinements. No comparison to published numerical table values is licensed by this source pass.

A first root-admitted full parent must establish actual native API/output compatibility, exact retained SNPs, coordinate coverage, command waits and resources. Capacity/retention caps are unmeasured ceilings, not forecasts. Preserve failed outputs and stop without redraw, shortening or silent SNP/support changes. Root must bind the exact source/config and supervise existing guards, expiry, RSS and full native output accounting. Native commands, methods, all shared-row adapters, full fits, IID/OOD scores and paper claims remain unfinished. In particular the current Rust ARGMix focal-diploid exclusion and masked shared-target loss adapters are not implemented by this generator.
