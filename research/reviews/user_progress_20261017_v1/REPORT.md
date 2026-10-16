# Current progress

Native ARGformer runs on physical GPU 0. A stable checkpoint observation verifies **64,500 AdamW updates**, including 42,500 beyond the current resume and one completed epoch. Its full 227,990 update fit and ancestry evaluation remain incomplete.

European modern span accuracy remains **97.0778 ± 0.5674%** for TreeLoom, **94.4461 ± 1.1491%** for current ARGMix and **93.7830 ± 0.8385%** for derived dense AncestralPaths. These are eight reused validation simulations and one optimizer seed. Matched NoMem reaches **97.0000 ± 0.4978%**. Boundary F1 is **18.3807 ± 1.3141%** for TreeLoom versus **19.0488 ± 1.4275%** for NoMem. No independent test or overall top ranking is established.

American VAL000–002 predictions continue. VAL003 execution failed before prediction completion because an adapter misread nested checkpoint metadata. The repaired prediction preparation completed and its exact one-shot execution is launched, with no worker completion yet. VAL004 native path preparation is running with the complete original data. Native FLARE retains **98.0100%** accuracy and **97.7151%** macro F1 on one parent and 10,000 focal records. There is no multi-parent SD or American GNN or ARGformer score yet.

Current American predictions score the simpler backbone. The full TreeLoom transfer now instantiates with **32,132 parameters**, both structural residuals and sample memory. H060 registers600additional full-parent updates per arm with the same accepted backbone weights. Source and CPU runtime reviews passed. Actual admission77023 waits on the existing scheduler mutex while native path preparation waits on NFS. No scientific run, new update or score is claimed.

H059 freezes the European independent comparison and eight reserved simulations before generation. Interpreter, package and native binary identities match the original configuration. Generation and all 40 primary plus eight secondary products remain future work.

The ten-page paper rewrite and Figure 1 v5 are already published and visually checked. Experiment values are unchanged. Complete independent benchmarks, useful architectural novelty and stronger biological validation remain unfinished.
