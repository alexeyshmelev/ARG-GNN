# Full-parent binding successor

The sealed packedv2 parent stores both per-hold query rows and a concatenated parent scatter vector. The v2 binder checked original observation tensors and each hold's immutable tensors but did not initially prove that the concatenated vector matched the observed hold/query order, or retain its mutation stamp.

This successor initially compares that vector with the original observed `bank.sample_index[:, bank.query_column]` order using bounded CPU tiles. Each expected tile and compatible actual tile has at most1048576 elements, so their combined transient numeric allocation is at most16MiB plus the original query-column index conversion. No full expected parent join or persistent numeric tensor is added. Every original hold, site and query remains present.

The binding retains the concatenated vector's pointer/version/shape/dtype/device stamp. Its existing validation now checks that stamp alongside all original current bank, episode and per-hold stamps. Integration repeats the complete version validation after residual construction and immediately before scatter-row transfer and index_add. Setup also checks original CPU/device observation and episode versions before and after its comparisons, preventing a changed source from being silently stamped after comparison.

The scientific packed CSR core remains imported from its exact sealedv2 path. Original evidence math, layers, FP32 parameters, marker blocks, sparse layout, checkpoint behavior and one-hold qualification source are unchanged. Static AST comparison confirms the integration body matchesv2 after only successor names/imports and removal of the one added pre-scatter guard call. All existing binder helper ASTs remain identical.

Future root-owned qualification should verify acceptance of a genuine complete observed parent and rejection of initially permuted or incorrect concatenated scatter rows, in-place mutation after binding, replacement of that tensor, and original bank/episode/hold mutation before the scatter boundary. Use small chunk sizes as well as the default to exercise both query and hold tiling. Existing one-hold numerical qualification remains a separate unchanged prerequisite. No such qualification was executed here.

The root caller must authenticate the new source files and sealed dependencies through the existing source reader and resource admission. This source handoff grants no loading, qualification, training, truth access, process or storage authority.
