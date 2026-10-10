"""Handmade topology/component checks, never official task/biological evidence."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tskit

from arggnn import official_argformer_bridge as bridge


def handmade():
    t=tskit.TableCollection(1000.)
    for _ in range(3):t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0)
    a=t.nodes.add_row(time=10);b=t.nodes.add_row(time=20)
    for left,right,ancestor in ((0.,500.,a),(500.,1000.,b)):
        for leaf in range(3):t.edges.add_row(left,right,ancestor,leaf)
    t.sort();return t.tree_sequence()


def test_pinned_source_and_no_supplied_assets():
    info=bridge.validate_source()
    assert info['commit']==bridge.COMMIT and len(info['files'])==60
    assert not any(row['path'].endswith(('.pt','.pkl','.trees','.h5','.safetensors')) for row in info['files'])
    assert not info['full_model_import_performed']


def test_observed_full_path_dedup_and_half_open_coordinate_identity():
    c=bridge.extract_observed_paths(handmade(),namespace='synthetic_A',sample_ids=[2,0])
    assert c.paths==((2,3),(2,4),(0,3),(0,4))
    actual=c.readout_path_ids([100.,500.,900.],[0,2])
    np.testing.assert_array_equal(actual,[[2,0],[3,1],[3,1]])
    with pytest.raises(ValueError):c.readout_path_ids([1000.],[0])
    with pytest.raises(ValueError):c.readout_path_ids([100.,100.],[0])


def test_real_author_tokenizer_skip_extant_preserves_full_path_identity():
    c=bridge.extract_observed_paths(handmade(),namespace='synthetic_A',sample_ids=[0,1])
    v=bridge.build_vocabulary(c);encoded=bridge.tokenize_official(c,v,max_seq_len=8)
    assert encoded['input_ids'].shape==(4,8)
    assert encoded['extant_node_id_raw'].tolist()==[0,0,1,1]
    assert encoded['input_ids'][0].tolist()==[2,v.official_values()['node_id_tokens']['node_3'],3,0,0,0,0,0]
    assert torch.equal(encoded['input_ids'][0],encoded['input_ids'][2])
    assert len(encoded['full_paths_before_truncation'])==4  # Never dedup by identical model inputs.
    assert encoded['attention_mask'][0].tolist()==[1,1,1,0,0,0,0,0]


def test_namespace_collision_never_means_cross_arg_transfer():
    a=bridge.extract_observed_paths(handmade(),namespace='ARG_A',sample_ids=[0,1])
    b=replace(a,namespace='ARG_B');va=bridge.build_vocabulary(a);vb=bridge.build_vocabulary(b)
    assert va.official_values()==vb.official_values()
    assert va.fingerprint()!=vb.fingerprint()
    with pytest.raises(ValueError,match='namespace'):bridge.tokenize_official(b,va)


def test_unseen_nodes_reject_without_pad_or_unknown_substitution():
    c=bridge.extract_observed_paths(handmade(),namespace='ARG_A',sample_ids=[0,1])
    v=bridge.ArgVocabulary(c.namespace,(0,1,3))
    with pytest.raises(KeyError,match='node_4'):bridge.tokenize_official(c,v)


def test_author_truncation_is_recorded_without_losing_full_path_occurrences():
    c=bridge.PathCorpus('ARG_A',((1,4,8,9),(2,4,8,9)),((0,1,0.,100.),(1,2,0.,100.)),(1,2),(0.,100.))
    v=bridge.build_vocabulary(c);x=bridge.tokenize_official(c,v,max_seq_len=3,skip_extant_tokens=True)
    assert x['input_ids'].shape==(2,3) and x['full_paths_before_truncation']==c.paths
    assert x['input_ids'][0].tolist()==[2,v.official_values()['node_id_tokens']['node_4'],3]


def test_actual_author_cpu_attention_component_and_released_sliding_rejection(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');torch.set_num_threads(1)
    receipt=bridge.author_component_cpu_probe()
    assert receipt['author_component_forward_backward']
    assert 'Sliding window' in receipt['released_sliding_window_stock_fallback_rejection']
    assert not receipt['full_official_model_forward'] and not torch.cuda.is_initialized()


def test_downstream_readout_keeps_original_meaning_and_has_no_query_target_parameter():
    q=torch.eye(4)[:2];r=torch.eye(4)
    kw=dict(query_ids=[100,101],reference_ids=[0,1,2,3],query_namespace='ARG_A',reference_namespace='ARG_A',
        temperature=2.,class_order=['ANA','CHG','WHG','EHG'])
    result=bridge.downstream_reference_similarity_logits(q,r,[0,1,2,3],**kw)
    torch.testing.assert_close(result['raw_readout_logits'],torch.tensor([[.5,0,0,0],[0,.5,0,0]],dtype=torch.float64))
    assert result['official_local_ancestry_classifier'] is False and not result['query_ancestry_targets_used']
    kw['reference_namespace']='ARG_B'
    with pytest.raises(ValueError,match='namespace'):bridge.downstream_reference_similarity_logits(q,r,[0,1,2,3],**kw)


@pytest.mark.parametrize('bad',([0,1,2,2],[0.,1.,2.,3.]))
def test_missing_class_or_noninteger_observed_reference_labels_fail(bad):
    with pytest.raises(ValueError):bridge.downstream_reference_similarity_logits(torch.eye(4)[:1],torch.eye(4),bad,
        query_ids=[10],reference_ids=[0,1,2,3],query_namespace='A',reference_namespace='A',temperature=1.,class_order=['ANA','CHG','WHG','EHG'])


def test_real_author_contrastive_loss_with_explicit_toy_encoder():
    result=bridge.author_contrastive_cpu_probe()
    assert result['actual_author_contrastive_loss_executed'] and result['loss']>0
    assert not result['full_official_model_forward'] and not torch.cuda.is_initialized()


def test_downstream_extreme_temperature_rejects_nonfinite_scores():
    with pytest.raises(ValueError,match='range'):
        bridge.downstream_reference_similarity_logits(torch.eye(4)[:1],torch.eye(4),[0,1,2,3],
            query_ids=[10],reference_ids=[0,1,2,3],query_namespace='A',reference_namespace='A',
            temperature=1e-320,class_order=['ANA','CHG','WHG','EHG'])


def test_real_author_pooling_and_streaming_retrieval_on_handmade_embeddings():
    result=bridge.author_embedding_retrieval_cpu_probe()
    assert result['actual_author_pooling_and_streaming_retrieval']
    assert result['retrieval_indices']==[[0],[1]]
    assert result['first_node_with_skipped_leaf_means_first_ancestor']
    assert not result['full_official_model_forward'] and not torch.cuda.is_initialized()
