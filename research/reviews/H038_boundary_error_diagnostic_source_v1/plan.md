# H038 Present182 boundary diagnostic

Registered phase: H038_boundary_error_diagnostic_20261015_v1. Root authorized a single CPU execution after source/pin and handmade checks.

The only predictions are the existing selected H038 aligned pass30 and zero-concordance pass25 products on the original eight internal VAL simulations. The code verifies all16 seals and all8 observed geometry maps before target access. It calls the unchanged full_span_score on Present182 and requires exact equality to the original boundary counts/F1 and accuracy. No smoothing, relabeling, reselection, retraining, new predictions or outer data are permitted. Results apply to H038. H039 primary-only results acquire no boundary claim.

Outputs are compact per-parent summaries, across-simulation mean±sampleSD(ddof1), pooled histograms and matrices, and a short biological interpretation. No per-query tract list, candidate vector or dense array is saved. Total output stays below10MB.

Fixed measures:
- True and predicted tract lengths on exact scored support, plus uncensored true/internal predicted distributions.
- Original class-agnostic one-to-one1kb boundary precision/recall/F1, absolute match distances and unmatched-event distances.
- Internal A→B→A islands ≤1kb and ≤10kb. Strict false requires zero exact true support for B. Majority-wrong requires less than50% B support. Boundary attribution is deduplicated per query and threshold. This is attribution, not a correction.
- Four-class true/predicted/matched/missed/extra transition counts and a16x16 matched-transition matrix to expose wrong-class positional matches.
- Class-agnostic geometric oracle recall ceiling using existing scored-bin edges at which readout_hold changes. It includes100bp snapping and omits native holds with no retained center. It is a positional upper bound, not an attainable model estimate. Zero within-hold argmax changes is verified first.
- Missed true changes with no candidate within1kb versus candidate collisions, separately from model misses despite a nearby hold edge.

Interpretation will use the observed decomposition: short false islands support studying biologically constrained tract continuity in a future TRAIN-only model; low ceiling supports richer within-tree readout; high ceiling with low recall points toward ancestry-transition recognition rather than grid geometry. Class-agnostic F1 may count a nearby wrong transition as a match. Tolerance and scoring remain fixed throughout. No modeling choice is selected or fitted in this diagnostic.

Handmade checks cover hold-edge construction, false-island overlap/deduplication, wrong-transition positional matches, geometric recall upper bound, single-candidate collisions, zero-boundary undefined F1 and inclusive length cutoffs. They contain no VAL data.
