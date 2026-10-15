# Full native GST TRAIN coverage

The unchanged author parser retains **10,439 of 10,500 TRAIN reactions**, with 61 original None drops and no fatal parse exception. These contain **37,410 reactant graphs, 364,213 atoms and 34,648 positive atoms**. The largest reaction has 100 atoms and 20 molecule graphs. No malformed bond token was omitted in the retained reactions. Original order, features, edges and edit labels are preserved. VALID and TEST were not read.

At the native 20 epochs and batch 16, this denominator implies **13,060 optimizer updates per fit**. The unchanged scheduler uses T_max=13,040. These are calculated training budgets. No model was trained in this count stage.

[Counts and native-array/order hashes](counts.json) · [Actual execution receipt](actual_exec_root_v1.json) · [Source registration](registration.json)

This resolves full TRAIN coverage beyond the completed first-1,000 pilot. The next baseline suite will use the existing author CPU runtime, four native modes and five seeds, with full VALID and TEST withheld. It is a disclosed full-data coverage extension. Our molecular adapter and final paired TEST comparison remain unfinished.
