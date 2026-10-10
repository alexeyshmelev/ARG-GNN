# ARG-GNN

Research into graph neural networks for sets and ordered interval-bearing graph sequences, with local ancestry inference from ancestral recombination graphs as the primary application.

Status: research design and reproducible infrastructure. No empirical performance or novelty claims are established yet.

The project runs on anogena in /workspace-SR003.nfs2/ARG-GNN. Read AGENTS.md before contributing.

## Research requirements

- Compare with graph-independent, pooled set, ordered graph sequence and ARG-specific methods.
- Evaluate on disjoint simulation replicates and demographic shifts.
- Measure predictive accuracy, calibration, boundary behavior, runtime and memory.
- Test whether predictions remain stable when an unchanged genomic interval is split into identical adjacent trees.
- Record complete provenance and use independent paper reviewers.
- Produce an AAAI-template manuscript and portable Overleaf sources.

## Environment

Source scripts/project_env.sh before installing packages or running Python.
The project conda environment is .runtime/env. Runtime state, datasets and checkpoints are excluded from Git.
The tracked code and experiment summaries are pushed regularly.
