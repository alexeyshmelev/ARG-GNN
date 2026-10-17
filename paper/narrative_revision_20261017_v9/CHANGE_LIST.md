# Narrative candidate, 17 October 2026

Source-only candidate for ROOT integration. Canonical manuscript, captions, figures, bibliography and runtime were not edited.

- Rebuilt the abstract around admixture biology, the representation gap, TreeLoom, measured development results and tract interpretation. LAI is expanded at first use.
- Reframed the introduction around changing query/reference relationships and persistent sample identity. Related work explains the conceptual relationship to existing genealogy methods and standard neural operations.
- Defined GNN, MLP, SNP and layer normalization, and named the sample memory operation as a bidirectional exponential moving average. Kept every equation body unchanged. The auxiliary structural residual formulas now appear in the supplement, leaving their biological meaning and data flow in the main methods.
- Made the two applications of the shared tree block, separate sampled readout, structural flags, memory residual and final BiGRU easier to follow. TreeLoom-NoMem explicitly retains both tree traversals, structural residuals and the final GRU.
- Consolidated reused validation, one seed, unfinished test, pipeline input/training differences and reproduction scope in evaluation design. Retained the exact AncestralPaths deficit and release/classifier mismatch, ARGMix 15+0 versus 25+3 difference, visible scientific TODOs and full supplementary conditions.
- Preserved the main comparison of TreeLoom with completed comparators and its single matched memory ablation. Kept all tables and their values verbatim. The lower boundary F1 of TreeLoom remains explicit in abstract, results and discussion. Changed causal improvement wording to observed arm differences.
- Tightened discussion around genealogy representation, the specific comparison scope, tract recovery and empirical validation. Retained the earlier Danish condition and counterevidence in the supplement.
- Removed needless compound adjectives and one duplicated FLARE limitation. No semicolons were introduced. The abstract is 152 words by whitespace count, and the main paper is about 8% shorter by the same measure.

No models, installation, jobs, Git operations or compilation were run. ROOT owns integration and build verification.
