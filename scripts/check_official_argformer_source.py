#!/usr/bin/env python
"""Read pinned author metadata and optionally exercise reviewed CPU components.

No author CLI, dataset loader, checkpoint, training loop or GPU is launched.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import tskit

from arggnn.official_argformer_bridge import (
    validate_source,dependency_inventory,extract_observed_paths,build_vocabulary,
    tokenize_official,author_component_cpu_probe,author_contrastive_cpu_probe,author_embedding_retrieval_cpu_probe,downstream_reference_similarity_logits,
)
from arggnn.resources import ROOT


def synthetic_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU checks require CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    tables=tskit.TableCollection(1000.)
    for _ in range(6):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0)
    ancestor=tables.nodes.add_row(time=1000.)
    for leaf in range(6):tables.edges.add_row(0.,1000.,ancestor,leaf)
    tables.sort();ts=tables.tree_sequence()
    corpus=extract_observed_paths(ts,namespace='synthetic_ARG_A_not_biological_evidence',sample_ids=list(range(6)))
    vocab=build_vocabulary(corpus);tokens=tokenize_official(corpus,vocab,max_seq_len=128,skip_extant_tokens=True)
    ids=corpus.readout_path_ids([125.,500.,875.],[5,4])
    assert ids.shape==(3,2) and tokens['input_ids'].shape==(6,128)
    probe=author_component_cpu_probe()
    readout=downstream_reference_similarity_logits(torch.eye(4)[:2],torch.eye(4),[0,1,2,3],
        query_ids=[4,5],reference_ids=[0,1,2,3],query_namespace=corpus.namespace,
        reference_namespace=corpus.namespace,temperature=1.,class_order=['ANA','CHG','WHG','EHG'])
    assert readout['raw_readout_logits'].shape==(2,4) and not torch.cuda.is_initialized()
    return {'status':'passed_synthetic_components_only','paths':len(corpus.paths),'vocab_size':len(vocab.node_ids)+4,
        'sequence_length':128,'skip_extant_tokens':True,'vocabulary_sha256':vocab.fingerprint(),
        'author_tokenization_methods_executed':True,'readout_path_id_shape':list(ids.shape),
        'attention_component':probe,'contrastive_component':author_contrastive_cpu_probe(),'embedding_retrieval_component':author_embedding_retrieval_cpu_probe(),'downstream_adapter_explicitly_not_author_classifier':True,
        'full_official_model_import_or_forward':False,'training':False,'gpu':False,
        'actual_data_or_targets':False,'scientific_reproduction':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-cpu',action='store_true')
    parser.add_argument('--output',help='Fresh in-project receipt path')
    args=parser.parse_args(argv)
    source=validate_source();inventory=dependency_inventory()
    result={'schema_version':1,'kind':'official_argformer_source_cpu_receipt','source':source,
        'dependencies':inventory,'missing_dependencies':[k for k,v in inventory.items() if v is None],
        'original_author_sources_modified':False,'upstream_scripts_executed':False,
        'released_mlm_full_forward_available':False,'published_task_reproduction':False}
    if args.synthetic_cpu:result['synthetic_cpu']=synthetic_cpu()
    validate_source()
    body=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if args.output:
        lexical=Path(args.output)
        if '..' in lexical.parts:raise ValueError('Traversal forbidden')
        output=(lexical if lexical.is_absolute() else ROOT/lexical).resolve()
        if not output.is_relative_to(ROOT) or not output.parent.is_dir():raise ValueError('Existing in-project output parent required')
        with output.open('xb') as handle:handle.write(body)
    print(json.dumps({'source_tree_sha256':source['source_tree_sha256'],'missing_dependencies':result['missing_dependencies'],
        'synthetic_cpu':result.get('synthetic_cpu',{}).get('status'),'published_task_reproduction':False,'output':args.output}))


if __name__=='__main__':main()
