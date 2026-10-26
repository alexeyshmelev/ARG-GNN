# Figure communication review, revision 5

## Why revision 4 was rejected

The owner could not trace the depicted computation. Earlier readers reconstructed the conceptual architecture, but that test did not establish that individual tensor operands, row identities, repeated operations and assemblies were visible. Their broad conceptual success is preserved as historical evidence. It is not acceptance of the new requirements.

## Current acceptance test

A fresh agent receives only the two frozen images and a neutral reading task. It receives no manuscript, source code, prior explanations or review history. It must trace a named leaf across the illustrated intervals and a named ancestor through the encoder, identify aggregation axes and resulting tensors, distinguish both uses of genomic history, and flag any unsupported inference.

The root compares that explanation with the implementation. Material errors or ambiguity require revision and a new reader with no history. Images and their hashes are frozen for each reading. A reader pass measures figure communication, not model quality, novelty, benchmark validity or owner approval.

## Design checklist

- Exactly two main figures.
- Named input and output tensors at every key transformation.
- Separate physical inputs for addition, multiplication, concatenation and subtraction.
- All nine node rows remain visible until explicit leaf selection.
- Leaf features are scattered only into leaf rows.
- Tree order and persistent leaf identity determine sequence row assembly.
- Interval length and center distances are measured on the chromosome drawing.
- The complete encoder expansion has a matching ENC parent marker.
- Both rounds reuse parameters and retain all nodes.
- Memory follows leaf identity even when group membership changes.
- Direction averaging and averaging current group members are distinct operations.
- Memory and structural corrections are placed at their specified node rows.
- Every color denotes a named semantic role.
- Tensor labels and operations remain legible without overlaps at paper scale.

## Status

The third independent image-only reader reconstructed every tested key path correctly. The accepted image hashes are sealed in reader_acceptance.json. Compilation and publication checks are recorded in integration.json.

## Draft issues caught before the independent reading

- The initial prediction drawing had overflowing card headers and footers. It was reauthored with bare tensors and bounded operations.
- Gate and correction operands could appear to be serial transformations. Inputs are being separated at their operators.
- Some schematic node matrices hid ancestor rows, and some vectors were drawn with multiple rows. Row counts are checked against the displayed shapes.
- The group assignment table did not show the context source for sequence assembly. Gather inputs and the three interval context sources must be explicit.
- A worked memory recurrence for one leaf lacked the assembly step for all leaves. The recurrence axis and returned matrices must be visible.
- Interval coordinates alone did not show which distances become the three geometry features. The measured spans are marked.

These are design-review findings, not results from a fresh image-only reader.

## Independent round 1: not accepted

The fresh reader correctly traced most of q, r1 and b, including all three ways of using other intervals. It nevertheless had to infer several connections and repetition scopes. Leaf initialization, the base residual arrow, compound U/O ports and interval-specific flag definitions were not sufficiently explicit. The reader also miscopied the five-digit learned-scale coefficient, motivating digit grouping. The independent source audit caught two incorrect toy site labels. These findings are repaired before a second fresh reading.

[Full independent account](review_round1/reader_report.md) · [Root comparison](review_round1/result.json) · [Independent source conformance](source_conformance.md).

## Independent round 2: correct key flow, final labels requested

The new reader correctly traced q from observations to probabilities, r1 through groups d/b/b, and b through both rounds and both corrections. It identified aggregation axes, shared parameters, and all three uses of genomic history. The report nevertheless noted that the GRU layer count and final downward map width were not explicit. These are made visible before a final new reader. Theta is also marked as one scalar parameter. The immutable round 2 images retain the reviewed version.

[Full independent account](review_round2/reader_report.md) · [Root comparison](review_round2/result.json).

The remaining report notes about exact statistical estimators, empty-data handling, endpoint rules and GRU cell internals describe the boundary between an architecture figure and a complete implementation specification. They do not justify guessing those details. The paper and code remain the sources for those policies.

## Independent round 3: accepted for key-flow reconstruction

A new reader received only the final two PNGs. It correctly reconstructed q from observations to probabilities, r1 through the complete bidirectional memory sequence, and b through both tree rounds and both correction paths. Its equations match the source contract, including normalized saved child messages, recipient exclusion, the original-base correction, group centering, current-interval decay and the final concatenations. It identified one GRU layer per direction, the downward map width and the scalar learned decay parameter. The root comparison found no incorrect key computation in these traces.

[Full independent account](review_round3/reader_report.md) · [Root comparison and exact image pins](review_round3/result.json).

The reader separately identified preprocessing estimators and exceptional-case conventions that cannot be recovered from the images alone. Those are preserved in its report rather than concealed or guessed. The accepted scope is the complete key architecture flow and the pictured worked example, not a numerical reimplementation of every preprocessing policy.

[Figure 1 source closure](source_conformance_round2_architecture.md) · [Figure 2 source closure](source_conformance_round2_encoder.md). Both include the final label clarifications and artifact hashes.
