# H028 observed-variant residual discovery

H028 asks whether original observed SNP evidence improves ancestry accuracy beyond the frozen, selected H025 genomic smoother, and whether the same SNP evidence plus observed graph reference context adds value. This is an input-information discovery experiment. It makes no new-GNN, calibrated-likelihood or superiority claim.

## Fixed prospective features

Read the original observed VCF, source sample roster and original site coordinates for the same two TRAIN and two development VAL parents. Join VCF diploid IDs and phase indices to (sample_id, haplotype), then to each graph/emitter roster's actual inferred node IDs. Retain every original nucleotide site, including multiallelic sites. No imputation, phase repair, inferred source labels or query/mixed-reference local ancestry enters preparation.

The donor panel is pureR96 only, split by observed sample age at 350 generations: 84 younger donors with C4 counts [24,16,22,22], 12 older donors with counts [0,8,2,2]. At each site, p(c,a)=(called donor count for allele a+0.5)/(called donor count+K*0.5), with K equal to declared site alleles. Only these observed proxy labels are accepted. Older Ana has no donors: its evidence stays exactly zero, and it is excluded from that cohort's feature normalization. This does not set the query's final Ana ancestry probability to zero.

Aggregate query haplotype log evidence within [center-5000,center+5000), using original parent physical coordinates and clipping only to the existing observed coverage. Windows cross artificial250kb tile boundaries. Every original 1kb physical block has total observed-site weight1, giving a fixed density-weighted composite score. This is a heuristic for correlated evidence, not exact LD correction. A missing query call contributes no evidence. If a globally present donor class has no called donor at a site, omit that site for every class in the same cohort. An empty cohort/query window stays zero.

Within each younger/older cohort, use stable softmax over the available class composite scores with equal class priors. These are feature fields, not calibrated ancestry posteriors. Center each cohort field across actual query haplotypes of the same exact observed age at each original scoring center. Uncalled rows stay zero. Counts and empty-window diagnostics are recorded without dropping scored bins.

The graph field reuses H026's fixed native-reference context: inverse-class-balanced exp(-native metric-tree distance/scale), normalized separately in younger/older cohorts, with fixed scales350/1400 generations. H026's additional census/sham fields are unused. Both channels have the same eight cohort/C4 coordinates and the same centering. Frozen inferred trees retain the same original observations and all 280 samples.

## Matched fit and score

Arms are genotype_only, graph_only and genotype_graph_fusion. Fusion is the fixed arithmetic mean of the two commensurate fields. Every arm fits the same zero-initialized, bias-free 8-to4 head with 32 parameters. There is no baseline-logit head input, temperature, intercept, learned modality gate or independent calibration path. The selected H025 genomic-smoothing epoch 10 model is frozen and recomputed exactly on the original250kb tiles. Its logits are the output skip only.

New observation-only preparation is protected by a reader that rejects truth/oracle payloads. It completes and writes observation_features_ready.json before TRAIN query target arrays open. TRAIN120 exact C4 span mass is used only in the original loss. VAL targets open only through the unchanged scorer after predictions are sealed.

One FP32 seed 101, 10 epochs, 16 TRAIN tiles per epoch and 160 optimizer updates per arm. Use the inherited AdamW recipe and earliest minimum equal-parent development VAL Present64 raw-probability NLL checkpoint. Keep the full original 120-query, physical-bin support and all inherited per-class, boundary, short-tract, dosage and calibration metrics. An NLL change does not establish an overall ancestry-accuracy gain.

Fixed fields stay in RAM. Retain compact metrics plus frozen-baseline and selected/last predictions and checkpoints only. The output cap is 1,000,000,000 bytes. Superseded own predictions/checkpoints are pruned through the inherited exclusive writer.

## Evidence and limits

The author checked one hand-derived allele/callability case and actual source/metadata joins for all four parents. No original VCF/tree payload, weights, query targets, GPU, live fit or broad test suite was opened/run during authoring. Sources remained unpinned during construction; receipt.json records completed source hashes for the root's subsequent active-input freeze.

Temporal donor drift, sparse older classes, SNP dependence, in-sample TRAIN emitter/teacher predictions and repeated development VAL exposure remain confounds. The graph also encodes all 280 observed haplotypes, including unlabeled mixed 64, whereas donor frequencies compress the panel to pureR96 counts. Fixed mean fusion tests one limited representation. Results can support this discovery contrast, not an isolated topology mechanism or generalization claim.
