def assemble_admitted_parent(core, frozen, torch, loaded_cpu_tiles, tile_records, parent_record, geometry, namespace):
    # Root must validate original observed/native-tree producer bindings first.
    assert geometry in ("age", "plain")
    assert len(loaded_cpu_tiles) == len(tile_records) == 8
    assert [r["tile_id"] for r in tile_records] == list(range(8))
    leaf_ids = loaded_cpu_tiles[0].bank.sample_ids
    contexts = []
    for prepared, record in zip(loaded_cpu_tiles, tile_records):
        bank, cache = prepared.bank, prepared.bank.query_cache
        assert bank.leaf_x.device.type == "cpu" and bank.leaf_x.dtype == torch.float32
        assert bank.leaf_x.shape[1:] == (280, 47) and cache.normalized_response.shape[1:] == (280, 280)
        assert prepared.layout.anchors.numel() == 32 and prepared.num_scales == 3
        assert torch.equal(bank.sample_ids, leaf_ids)
        assert torch.equal(bank.sample_ids[cache.query_indices], cache.query_ids)
        assert torch.equal(bank.sample_ids[cache.reference_indices], cache.reference_ids)
        assert cache.query_ids.numel() == 184 and cache.reference_ids.numel() == 96
        original_ids = tuple(record["original_tree_ids"])
        assert len(original_ids) == bank.leaf_x.shape[0]
        fingerprints = tuple(frozen.fingerprint_sha({
            "parent_id": parent_record["parent_id"], "native_namespace": namespace,
            "original_tree_id": tree_id, "leaf_ids": leaf_ids,
            "query_ids": cache.query_ids, "reference_ids": cache.reference_ids,
            "feature_columns": bank.feature_columns, "leaf_x": bank.leaf_x[g],
            "plain_normalized_response": cache.normalized_response[g],
        }) for g, tree_id in enumerate(original_ids))
        contexts.append(core.TileContextV3(
            prepared.layout, prepared.operators[geometry], cache.query_ids, cache.reference_ids,
            original_ids, geometry=geometry, feature_fingerprints=fingerprints,
            parent_id=parent_record["parent_id"], split=parent_record["split"], namespace=namespace))
    parent = core.stitch_context_v3(tuple(contexts), fixture=False)
    assert parent.layout.anchors.numel() == 256
    assert torch.equal(parent.layout.anchors, torch.cat([x.layout.anchors for x in loaded_cpu_tiles]))
    assert torch.equal(parent.layout.weights, torch.cat([x.layout.weights for x in loaded_cpu_tiles]))
    leaf = torch.cat([x.bank.leaf_x for x in loaded_cpu_tiles])
    response = torch.cat([x.bank.query_cache.normalized_response for x in loaded_cpu_tiles])
    assert not leaf.requires_grad and not response.requires_grad
    return parent, leaf, response, leaf_ids
