"""Original GST USPTO parser/models/training, with TEST withheld.

This invocation bypasses the author's run(), which opens TEST. Its unchanged
train_one is called with full VALID in both evaluation slots; the final score
is selected-checkpoint VALID AP. This is a disclosed CPython3.14.3/Torch2.10+cpu
runtime port and invocation change, without a numerical paper-reproduction claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
from importlib import metadata
import json
import locale
import math
import os
from pathlib import Path
import sys

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
AUTHOR_COMMIT='6469d56aa131463325058b897b1e435c6d2edcd2'
AUTHOR_SOURCE_SHA256={
    'scripts/eval_rxn_center.py':'57fd803410d57c9b05d0719cdb5bd318d9d90b2cdebbc86a1260c37a3c3354b3',
    'src/graph_set_transformer/models/gst.py':'bc08c15d55247b32bda98e4b03fcaee988ceb8f654423240a33ca632ce11ef72',
}
MODEL_MODES=('gsc-ca','gsc-bc','gcn-ds','gcn-st')
DATA_DIRECTORY='data/gst_uspto_native_payload_v2_r001/data'

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):result.update(block)
    return result.hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def project_path(value):
    path=(ROOT/value).resolve();path.relative_to(ROOT);return path

def native_arguments(data_dir,output):
    return argparse.Namespace(data_dir=str(data_dir),out_dir=str(output),device='cpu',
        quick=False,hidden=128,layers=1,num_heads=4,dropout=.1,epochs=20,
        batch_size=16,lr=.001,weight_decay=.0001,seeds=1,seed_offset=0,
        max_train=1000,max_val=None,max_test=None,equalize_params='smaller',
        models=list(MODEL_MODES))

def protocol_report():
    return {'schema':'native_GST_USPTO_VALID_only_v1','status':'inactive_until_explicit_train',
        'author_commit':AUTHOR_COMMIT,'author_source_sha256':AUTHOR_SOURCE_SHA256,
        'args':vars(native_arguments(DATA_DIRECTORY,'ROOT_EXCLUSIVE_NEW_OUTPUT')),
        'invocation_change':'Bypass original run(); call unchanged train_one(model, TRAIN, VALID, VALID, args, seed0, cpu). Final evaluation slot is full VALID, so the terminal score is selected-checkpoint VALID AP.',
        'TEST_parsed_or_scored':False,
        'runtime_port':'CPython3.14.3/Torch2.10.0+cpu; saved author run used Python3.14.4/Torch2.10.0+cu128 with CPU execution.',
        'numerical_paper_reproduction_claim':False}

def import_native(author_root):
    for name,expected in AUTHOR_SOURCE_SHA256.items():
        require(sha(author_root/name)==expected,'Pinned original GST source changed: '+name)
    # Ordinary full-module import. No AST slicing, rewriting or function wrappers.
    sys.path[:0]=[str(author_root/'scripts'),str(author_root/'src')]
    native=importlib.import_module('eval_rxn_center')
    require(Path(native.__file__).resolve()==(author_root/'scripts/eval_rxn_center.py').resolve(),
            'Imported runner must be the pinned full author module')
    gst=importlib.import_module('graph_set_transformer.models.gst')
    require(Path(gst.__file__).resolve()==(author_root/'src/graph_set_transformer/models/gst.py').resolve(),
            'Imported GST model must be the unchanged pinned author model')
    return native

def execute(author_root,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit empty CUDA visibility required')
    require(sys.version_info[:3]==(3,14,3),'Registered CPU runtime uses CPython3.14.3')
    author_root=project_path(author_root);output=project_path(output)
    require(not output.exists(),'Exclusive new result directory required')
    native=import_native(author_root)
    require(native.torch.__version__=='2.10.0+cpu','Registered CPU runtime uses Torch2.10.0+cpu')
    require(not native.torch.cuda.is_available(),'CPU-only native GST invocation required')
    require(tuple(native.MODEL_KEYS)==MODEL_MODES,'Original four model modes/order changed')
    data_dir=project_path(DATA_DIRECTORY);args=native_arguments(data_dir,output)
    output.mkdir(parents=True)
    record=protocol_report()
    record.update(status='initializing',created_utc=datetime.now(timezone.utc).isoformat(),
        caller_ref={'path':str(Path(__file__).resolve().relative_to(ROOT)),'sha256':sha(__file__)},
        author_root=str(author_root),args=vars(args),models={},
        runtime={'python':sys.version.split()[0],'packages':{name:metadata.version(name)
            for name in ('torch','torch-geometric','numpy','scipy','scikit-learn','rdkit')},
            'default_text_encoding':locale.getpreferredencoding(False),
            'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']},
        entry_torch_CPU_RNG_sha256=hashlib.sha256(native.torch.get_rng_state().numpy().tobytes()).hexdigest(),
        initialization_scope='Original equalize_models and model construction precede train_one reseeding. Entry RNG is recorded by hash, without adding an initialization seed or claiming exact author initialization.',
        target_endpoint='Full original-native-retained VALID pooled per-atom average precision',
        native_functions_unchanged=['load_split','_parse_line','_atom_features','_mol_to_arrays',
            'collate','atom_feature_dim','equalize_models','build_model','train_one','evaluate'],
        selection_scope='Original strict-better VALID AP checkpoint selection and original final-state fallback; no TEST access')
    receipt=output/'receipt.json'
    native.write_json(receipt,record)
    try:
        record['split_refs']={split:{'path':str((data_dir/filename).relative_to(ROOT)),
            'sha256':sha(data_dir/filename),'bytes':(data_dir/filename).stat().st_size}
            for split,filename in (('train','train.txt'),('valid','valid.txt'))}
        # Preserve original TRAIN then VALID parsing and native retained-row limits.
        train_set=native.load_split(data_dir/'train.txt',args.max_train)
        val_set=native.load_split(data_dir/'valid.txt',args.max_val)
        record['split_audit']={'train':native.audit_split('train',train_set),
                              'valid':native.audit_split('valid',val_set)}
        in_dim=native.atom_feature_dim()
        hiddens,params=native.equalize_models(in_dim,args.hidden,args.layers,
                                           args.num_heads,args.equalize_params)
        record.update(status='fitting_native_TRAIN_VALID',in_dim=in_dim,
            model_hidden_dims=hiddens,model_param_counts=params)
        native.write_json(receipt,record)
        for mode in MODEL_MODES:
            seed=0
            # Preserve author run() construction order, followed by native reseeding.
            model=native.build_model(mode,in_dim,hiddens[mode],args.layers,
                                     args.num_heads,dropout=args.dropout).to('cpu')
            record['models'][mode]={'status':'fitting','seed':seed,'hidden':hiddens[mode],
                                   'parameters':native.count_params(model)}
            native.write_json(receipt,record)
            # train_one unconditionally evaluates its final argument after restoring
            # the VALID-selected weights. The same full VALID object fills that slot.
            valid_ap,elapsed,history=native.train_one(
                model,train_set,val_set,val_set,args,seed,'cpu')
            valid_epochs=[(score,index+1) for index,score in enumerate(history['val_ap'])
                          if not math.isnan(score)]
            selected_epoch=None
            if valid_epochs:
                best=max(score for score,_ in valid_epochs)
                selected_epoch=next(index for score,index in valid_epochs if score==best)
            checkpoint=output/f'{mode}.seed0.VAL_selected.pt'
            native.torch.save({'model_state':model.state_dict(),'model_key':mode,'seed':seed,
                'selected_epoch':selected_epoch,'selected_VALID_AP':valid_ap,
                'selection_scope':record['selection_scope'],'args':vars(args),
                'author_commit':AUTHOR_COMMIT,'author_source_sha256':AUTHOR_SOURCE_SHA256,
                'split_refs':record['split_refs'],'runtime':record['runtime'],
                'TEST_parsed_or_scored':False,'numerical_paper_reproduction_claim':False},checkpoint)
            record['models'][mode].update(status='completed_native_TRAIN_VALID',
                selected_VALID_AP=valid_ap,selected_epoch=selected_epoch,
                native_final_state_fallback_used=not bool(valid_epochs),
                native_train_one_seconds=elapsed,history=history,
                checkpoint={'path':str(checkpoint.relative_to(ROOT)),'sha256':sha(checkpoint)},
                TEST_parsed_or_scored=False)
            native.write_json(receipt,record)
            print(json.dumps({'model':mode,'seed':seed,'selected_VALID_AP':valid_ap,
                              'selected_epoch':selected_epoch,'TEST_parsed_or_scored':False}),flush=True)
            del model
        record.update(status='completed_four_native_modes_seed0_VALID_only',
                      completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error))
        raise
    finally:native.write_json(receipt,record)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--author-root')
    parser.add_argument('--output')
    args=parser.parse_args()
    if not args.train:print(json.dumps(protocol_report(),sort_keys=True,indent=2));return
    require(args.author_root and args.output,'Explicit pinned author tree and exclusive output required')
    execute(args.author_root,args.output)

if __name__=='__main__':main()
