# Immutable correction of ARGMix producer/context attribution

The sealed legacy adapter remains unchanged and is classified as a development adaptation. It is not an author-faithful implementation of the documented current CLI producer. My earlier log-distance ranking attribution applied to legacy SampleIterator, not the current CLI fast engine.

Pinned author commit is c2481248eb0de5fda659d68f59c51e0e27becd20. README.md52–60 documents build-training before train. src/commands/data.py408–430 invokes the Rust build_training binary. src/rust/ancpath/src/bin/build_training.rs530–547 constructs FastSubgraphEngine with max_count0. src/argmix_project.py121–122 leaves both train and validation subgraph caps unlimited. The producer collects rows before build_chunks at638 and groups/orders them per query before1024/512 windows at724–777. Thinning that collected context to100kshared pairs is a substantive change.

The fast engine quantizes node times at src/rust/ancpath/src/fast_lca/engine.rs285–299. At128–129 it stores relative_distance_uncapped, defined at206–210 as absolute difference between quantized focal time and quantized TMRCA. Selection at148–150 delegates to src/rust/ancpath_incremental/src/ref_selection.rs, which sorts distance then original reference-vector index at31–45. Log2 matrix encoding is separate and must not be confused with reference ranking.

The producer reads references into a Rust HashMap at build_training.rs900–915 and constructs its reference vector from that map at466–470. Its tie order is therefore an actual runtime vector, not a promised stable input-file order. A new paired prospective route should capture that vector once as observed metadata before labels and reuse it across eligibility flags.

Original training drops rows for which TruthResolver returns no label. first_snp requires an observed SNP in the interval, with floor/ceil boundary conventions at npz_truth.rs320–327. No purity threshold is the relevant declared common-target setting. We cannot fill every neighboring context row with extra truth labels while claiming the same500klabel budget. The successor must retain complete observed native context and mark only exact shared targets as supervised/scored, with this loss-interface adaptation disclosed. Independent native1024/512 windows containing a target must retain every observed row in that canonical window. No arbitrary smaller radius, row cap or query thinning is acceptable.

The April25reference+3ancestor paper is a separate unrecovered implementation. No current-release context choice establishes its numerical reproduction. Existing20epoch maximum is prospective, while current advanced defaults use3epochs.

Exact source hashes and excerpts are preserved in source_evidence.v1.json. This correction involved only source/metadata reads and writes in this new folder. No arrays, builds, numerical imports, fits, probes, guards, admissions or Git operations occurred.
