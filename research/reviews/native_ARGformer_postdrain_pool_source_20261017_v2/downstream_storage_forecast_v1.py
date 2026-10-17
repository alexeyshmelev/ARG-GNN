"""Source-derived tensor-byte forecast only; no models/checkpoints/payloads."""
WIDTH=768;SEQUENCE_LENGTH=128;FLOAT32_BYTES=4;INT64_BYTES=8
VIEWS=("contrastive/train","contrastive/val","query_paths/train","query_paths/val")
def forecast_v1(counts):
    rows={key:counts["datasets"][key]["rows"]for key in VIEWS}
    if not all(type(n)is int and n>0 for n in rows.values()):raise ValueError("Exact full original view cardinalities")
    # Author always saves head_norm vectors, input IDs and attention masks, and
    # conditionally scalar labels/extant IDs. Each of the four full saves occurs
    # once. Combined train+val tensors remain RAM-only in the inherited I/O path.
    per_row=WIDTH*FLOAT32_BYTES+2*SEQUENCE_LENGTH*INT64_BYTES+2*INT64_BYTES
    products={key:dict(rows=n,FP32_embedding_bytes=n*WIDTH*FLOAT32_BYTES,
        saved_tensor_payload_upper_bytes=n*per_row,
        explicit_small_ZIP_pickle_metadata_allowance_bytes=16*2**20,
        prospective_serialized_bound_bytes=n*per_row+16*2**20)for key,n in rows.items()}
    tensors=sum(r["saved_tensor_payload_upper_bytes"]for r in products.values())
    serialized=sum(r["prospective_serialized_bound_bytes"]for r in products.values())
    return dict(schema="original_four_full_FP32_author_embedding_storage_forecast_v1",parent_id=counts["parent_id"],
        learned_head_width=WIDTH,original_max_sequence_length=SEQUENCE_LENGTH,products=products,
        FP32_vectors_only_bytes=sum(r["FP32_embedding_bytes"]for r in products.values()),
        all_saved_tensor_payload_upper_bytes=tensors,all_four_prospective_serialized_bound_bytes=serialized,
        retained_embedding_scope_cap_bytes=56*2**30,retained_contrastive_scope_cap_bytes=8*2**30,
        retained_retrieval_scope_cap_bytes=2**30,retained_whole_downstream_scope_cap_bytes=65*2**30,
        contrastive_transient_regular_checkpoints=2,checkpoint_serialized_file_failstop_bytes=3*2**30,
        checkpoint_events_preserved="20ba",disclosed_keep1_retention_only=True,
        original_four_full_author_saves_preserved=True,combined_embedding_disk_copy_added=False,
        storage_admitted=False,measured_serialized_fit=False,measured_RAM_or_GPU_fit=False,
        default_prior_downstream_GPU_proposal_bytes=74*2**30,
        nominal_shared_native_allocator_bytes=32*2**30,nominal_shared_native_allocator_fit_unproved=True,
        CPU_retrieval_substitution=False)
