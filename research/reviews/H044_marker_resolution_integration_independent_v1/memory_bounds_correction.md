# H044 resource review correction: exported state has96 channels

Root identified a bound error missed in my earlier independent core review. The scientific forward is consistent; the frozen logical_bounds_v1 helper and my quoted state/gather byte estimates used64 channels incorrectly. Preserve those receipts as historical. The corrected resource formulas below supersede those estimates. No constructor, forward, corpus, checkpoint, target, runtime/GPU/accounting inspection or scientific execution by this reviewer.

Source proof: MutationRoutedWholeTreeGNNV1 lines276–279 produces context width32, GRU(input_size35, hidden_size32, bidirectional=True) history width64, and classifier Linear(96,32). H044 concatenates that context and complete history, so the exported state is96. Its marker residual intentionally reads the first32 channels. No model parameter, activation shape, checkpoint layout, numerical forward or scientific schedule is changed by correcting the resource formulas.

Verified helper: src/arggnn/marker_resolution_memory_bounds_v2.py SHA f821c69037adf97a05bf6537c427ba41cebcff6f4256d24aa848d2315669e7ce. It wraps the frozen helper and replaces the two affected named bounds, and exposes a constructor-interface assertion without data/forward execution.

- Exported state:4*T*H*96 bytes, versus old4*T*H*64; exactly1.5 times the old amount. At H952 this is365,568*T bytes.
- Saved readout state+feature inputs:4*P*Q*(96+4*6). At P20,000/Q686 this is6,585,600,000bytes. Old4,829,440,000 was low by1,756,160,000bytes.
- At2048 centers/B64, query-first selected-state block is50,331,648bytes. The old hold-first all952 block was748,683,264bytes. My prior33,554,432/499,122,176 figures also require1.5 times correction.
- Prefix, evidence cache, grid logits/probabilities, targets, parameters and96-product probability byte formulas are unaffected. None establishes total allocator/autograd peak or training fit.

The previously issued prep-only review at722e57a9 must be rebound after this helper is included in the successor prep and source manifest. Actual admission remains root-owned. Frozen core27ecd306 and old source/receipt closures remain unchanged.
