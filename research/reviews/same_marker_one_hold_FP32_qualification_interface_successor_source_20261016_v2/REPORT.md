# H058 one-hold observed-roster interface correction

The genuine ROOTv3 attempt failed before the observed archive read at the exact sample/date/phase join. Its arithmetic checks are empty and optimizer updates are zero. Preserve the original FAIL, actual direct wait, source and admission history.

Authenticated compact TRAIN000 metadata has952 source rows,952 derived rows and952 mapping rows. All952 agree on sample ID, sample index, haplotype, age, role, reference class and source observed node. The inference receipt's mapping is a list of explicit two-field dictionaries. The original loop binds each dictionary to native_id and compares a derived integer to that dictionary. All952 such comparisons are false. The intended integer field agrees on every row, and all mapping sample indices agree with both rosters in declared order. The native receipt and expanded wrapper contain exactly the same mapping.

qualify_one_observed_hold_v2.py is a fresh successor. It requires exactly952 typed, ordered mapping dictionaries with exactly sample_index and inferred_tree_node_id fields, then joins both roster sample indices and the derived scalar node to those fields. All original sample/date/phase/class checks remain. It rejects bool aliases for mapping indices/nodes and the joined roster index/node fields. The original source is untouched.

The exact text/AST delta replaces only the old join loop with mapping assignment, fail-closed mapping-schema assertion and corrected join loop. The entire module AST is otherwise identical. Source parsing and compilation did not import or run the module. Core/null/plan hashes, first-hold selection, FP32 seed101, all tolerances/oracles/masks/null/gradient logic, optimizer0 and the original600s/8GiB/8threads ROOT callback contract are unchanged.

The current fullpilot's eight source files do not repeat this loop. Its existing full-panel roster join indexes explicit mapping dictionary fields correctly. This source inspection establishes no fullpilot runtime or feasibility result.

No arrays, native trees, VCF, scoring/truth payloads, VAL/TEST data or checkpoint payloads were opened. No scientific/project imports, numerical execution, process/GPU probes, job launches, accounting or Git calls occurred. This source correction grants no admission or automatic replay. ROOT owns independent delta review, experiment-ledger integration and actual fresh admission.
