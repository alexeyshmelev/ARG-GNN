# Observed variant evidence audit v1

This source/manifest-only audit covers the same four development parents. It establishes available inputs and prospective controls, without computing allele statistics, fitting models or claiming gains.

## Available observations and joins

Each parent has data/biological_paths_execution_pilot_v1/{parent}/observed/genotypes.vcf, plus sample_manifest.json and site_coordinate_manifest.json. Producer code writes 140 phased diploids (280 haplotypes): 120 queries, 96 pure references and 64 mixed references. Only pure references carry observed C4 source labels (Ana/CHG/WHG/EHG, 24 haplotypes each). Observed ages are available. Query/mixed local ancestry remains unavailable to features.

Original site counts are 10,213/10,163/10,553/10,144 for train000/train001/val000/val001. tsinfer metadata records all sites and exact phased genotypes/ages retained. Relate accepts phased, complete biallelic nucleotide GTs and drops 17/16/20/21 multiallelic sites, with no further converter attrition. Original VCF hashes agree across both producers.

Join VCF header sample IDs and phase index to (sample_id, haplotype) and sample_index=2*diploid_index+haplotype, then to each inferred roster's actual node ID. Never join internal ancestor IDs across inference methods. Sites join by original site ID and zero-based physical position: VCF_POS=position+1; clipped graph coordinates add coverage-left. This synthetic contig22 is not GRCh37.

Keep existing coverage [117,1998898), [19,1999991), [273,1999681), [116,1999724), and 19,988/20,000/19,994/19,996 retained centers 50+100*i. Score the existing clipped bins only, without variant-density or confidence exclusions.

## Native consumption

ARGMix consumes nearest-reference cohort tokens and transformed pairwise TMRCA matrices, then final-layer features and cM span in its GRU. ARGformer consumes ancestor-node-ID paths, with pure-reference contrastive labels. AncestralPaths consumes ancestor-shell observed-cohort fractions and ancestor ages. These audited native paths have no direct genotype/allele feature channel. Mutation counts in AncestralPaths setup are inventory values, not classifier inputs.

Both inferred tree producers retain site/mutation tables, including inferred mutation placements. tsinfer additionally verifies native mutation times against branch bounds. This is source/receipt evidence, not a payload inspection or a claim that placements are correct.

## Prospective donor allele evidence

A genotype channel can use class-conditional categorical allele counts from pureR96 only. At site s, estimate p(c,s,a)=(k(c,s,a)+alpha)/(n(c,s)+K(s)*alpha), where k counts called donor haplotypes carrying nucleotide a, n counts called donor haplotypes and K counts declared site alleles. A fixed positive symmetric pseudocount, for example alpha=0.5, avoids hard zeros. This needs no ancestral polarization. VCF REF is a producer-supplied simulated ancestral convention, not empirical ancestral-state validation.

Aggregate query haplotype log probabilities in prespecified physical windows around the unchanged scoring centers (a 10kb window is a reasonable first hypothesis), clipping windows at observed coverage. Include callable counts and empty-window masks. Current parser/receipts require complete phase, but future missing calls must reduce n and be masked. Keep multiallelic encoding explicit.

Linked SNPs and paired haplotypes are dependent: weighted log scores are composite evidence, not calibrated independent-site likelihoods. Freeze LD weighting/window choices using TRAIN only. Leave both haplotypes of a focal donor diploid out when evaluating donor self-prediction. Pure donors span 176–473 generations, queries 0–200: donor frequencies are temporally sampled proxies, not contemporaneous source frequencies. Date strata or weights require declared assumptions and identical availability in controls. Never assign source classes to mixed references or queries.

## Fair graph controls and uncertainty

Use one fixed inferred graph and its exact original SNP list for genotype-only, graph-only and fusion arms, with identical observed rosters, pure labels, dates, windows, scored support and evaluation. Graph-only omits direct allele channels but still encodes SNP-derived inference. Compare fusion against genotype-only with the identical genotype channel; topology removal or a prespecified sham must retain allele inputs and capacity/training allowances. Native Relate-versus-tsinfer results cannot isolate graph architecture because their variant filtering and training differ. A matched cross-producer experiment would require separately declared inference from identical variants. No inferred ancestry label becomes an input.

A whole-tree GNN could attach mutation support, recurrent/back-mutation burden and allele-pattern compatibility to edges, accounting for branch duration, span and observed-site density. Assign placements through the edge above mutation.node at its site; root mutations have no such edge. Zero mutations do not prove an erroneous edge. Placements reuse the same genotypes, mapping may be ambiguous, and dates/phase may be wrong. These are reliability covariates, not independent evidence or posterior confidence. Mutation-support ablation and equal-information genotype controls are required before attributing benefit to graph uncertainty.
