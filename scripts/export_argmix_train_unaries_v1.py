#!/usr/bin/env python3
"""CPU frozen native ARGMix TRAIN inference; in-sample stacking inputs, no refit."""
from __future__ import annotations
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,traceback

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
PREFIX=ROOT/'.runtime/official_argmix_abi210_v1/cu129/env'
NATIVE=ROOT/'.runtime/official_argmix_abi210_v1/cu129/source/src'
FIT={'path':'experiments/results/official_argmix_biological_metadata_repair_v3/fit/stock_fit.json','sha256':'d148a4b9c95d7d590e7346b3af1d10f9a211bd86aa89ff64a3cb3c6b6254b02b'}
CHECKPOINT={'path':'experiments/results/official_argmix_biological_metadata_repair_v3/fit/checkpoints/2026-10-10_20-53-45_joint_tmrca_gru_h128_nh4_nl4_c4_chunk1024/best_model.pt','sha256':'778fdf2a736f5357cfd35ddcedef20a8f3130c8dd2cb2cca70594dab691aa19f'}
FRAMES={'path':'experiments/results/official_argmix_biological_pilot_v2/observed/observed_frames.json','sha256':'b8de4b503bfebcf7eb71b6861879826f24731deb37aabe68b5283f6f07f07531'}
BACKEND={'path':'experiments/results/official_argmix_biological_metadata_repair_v3/fit/training_backend.yaml','sha256':'0807ea933eebdb42aedb827e28000cefba1cd5f4b4c6680b1d5005b2022a1aad'}
EXPORT={'path':'experiments/results/official_argmix_biological_metadata_repair_v3/inference/source_base_logits_and_final.pt','sha256':'a83377e955a1a56edd88edd75fc6f47c591400516474ea8435afb117e3c6c13c'}
VOCAB={'path':'experiments/results/official_argmix_biological_metadata_repair_v3/inference/source_base_logits_and_final_vocab.json','sha256':'437311b9281a1f4787fb69d6db5bb85629e2557e87c995fa5ef6c48462138203'}
SOURCE_PINS={
 'src/arggnn/official_argmix_biological.py':'4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f',
 '.runtime/official_argmix_abi210_v1/cu129/source/src/run_inference.py':'41ecbc14605d73779a97b9d814f870a86351e0fbcda6ad123e77f3579bbaa464',
 '.runtime/official_argmix_abi210_v1/cu129/source/src/run_joint_inference.py':'ab4eefcf2de42b4439ed66f39c289a4a98c14916a9d159f52c0f0d01a4fe692b',
 'configs/official_argmix_biological_pilot_v1.json':'25829cdb811da918db4cb7fad5ee607a44610bd1f04bc40dbe2f5564a6d58eb2'}
PARENTS=('bio_paths_pilot_train_000','bio_paths_pilot_train_001')
C4=['ANA','CHG','WHG','EHG']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def ref(path):return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path)}
def bound(record):
    path=ROOT/record['path']
    if path.resolve()!=path or not path.is_file() or sha(path)!=record['sha256']:raise ValueError('changed frozen artifact: '+str(path))
    return path
def json_value(record):return json.loads(bound(record).read_bytes())
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def confined(value):
    path=Path(value);path=path if path.is_absolute() else ROOT/path
    if ROOT not in path.parents or path.resolve()!=path or '..' in path.parts:raise ValueError('canonical project output required')
    return path
def native_environment(output):
    env=os.environ.copy()
    for key in ('PYTHONHOME','LIBTORCH_BYPASS_VERSION_CHECK','LIBTORCH','LIBTORCH_INCLUDE','LIBTORCH_LIB','LIBTORCH_STATIC','LIBTORCH_CXX11_ABI','TORCH_CUDA_VERSION'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
      PYTHONPATH=str(NATIVE),CONDA_PREFIX=str(PREFIX),LIBTORCH_USE_PYTORCH='1',HDF5_DIR=str(PREFIX),
      LD_LIBRARY_PATH=str(PREFIX/'lib')+':'+str(PREFIX/'lib/python3.12/site-packages/torch/lib'),
      WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PATH']=str(PREFIX/'bin')+os.pathsep+env.get('PATH','')
    short=ROOT/'.runtime/tmp/amtu'/hashlib.sha256(str(output).encode()).hexdigest()[:12]
    short.mkdir(parents=True,exist_ok=True)
    for key in ('TMPDIR','TMP','TEMP'):env[key]=str(short)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    for key in ('XDG_CACHE_HOME','TORCH_HOME','HF_HOME','MPLCONFIGDIR','WANDB_DIR','NUMBA_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR'):
        directory=output/'cache'/key.lower();directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
    return env

def execute(output):
    import h5py,numpy as np,yaml
    for name,digest in SOURCE_PINS.items():bound({'path':name,'sha256':digest})
    fit=json_value(FIT);frames=json_value(FRAMES)
    if fit['checkpoint']!=CHECKPOINT or fit['status']!='passed':raise ValueError('exact completed frozen ARGMix fit required')
    checkpoint=bound(CHECKPOINT);export=bound(EXPORT);vocab=bound(VOCAB)
    backend=yaml.safe_load(bound(BACKEND).read_text())
    if output.exists():raise ValueError('fresh TRAIN export output required')
    output.mkdir(parents=True)
    env=native_environment(output)
    manifest={'kind':'argmix_frozen_train_unaries_v1','status':'running','emitter_condition':'native_ARGMix_Relate_frozen_emitter',
      'training_predictions':'in_sample_not_out_of_fold','in_sample':True,'emitter_refit':False,
      'notice':'The emitter was fitted on these two TRAIN parents. These are in-sample stacking inputs, not cross-fitted predictions. VAL was also used for the native checkpoint selection.',
      'device':'cpu','GPU_used':False,'class_order':C4,'fitted_checkpoint_ref':CHECKPOINT,'native_fit_ref':FIT,
      'observed_frames_ref':FRAMES,'exported_base_model_ref':EXPORT,'source_pins':SOURCE_PINS,
      'exporter_ref':ref(Path(__file__)),'parents':{}}
    def run(name,argv,directory):
        command=[str(PREFIX/'bin/python'),'-B',str(NATIVE/name),*map(str,argv)]
        log=directory/(name+'.log')
        with log.open('xb') as stream:
            done=subprocess.run(command,cwd=NATIVE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
        save(directory/(name+'.command.json'),{'argv':command,'returncode':done.returncode,'source_ref':ref(NATIVE/name),'device':'cpu','log_ref':ref(log)})
        if done.returncode:raise RuntimeError('native frozen inference failed; see '+str(log))
    try:
        for parent in PARENTS:
            frame=frames['parents'][parent]
            if frame['split']!='train':raise ValueError('TRAIN parents only')
            directory=output/parent;directory.mkdir()
            pairs=directory/'inference_pairs.yaml'
            # The exact stock inference iterator sees only observed genealogy and declared reference roles.
            pair=[{'rel':str(bound(frame['tree'])),'references':str(bound(frame['references']))}]
            pairs.write_text(yaml.safe_dump(pair,sort_keys=False))
            config=copy.deepcopy(backend)
            config['processing']['exclude_edge_trees']=False
            config['inference']['device']='cpu'
            # Inference ignores native supervision; remove training pair pointers entirely.
            config['paths']={'val_pairs':str(pairs),'references':str(bound(frame['references'])),'save_dir':str(directory)}
            config_path=directory/'full_support_inference.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
            base=directory/'source_base.h5'
            run('run_inference.py',['--config',config_path,'--val_pairs',pairs,'--references',bound(frame['references']),
              '--torchscript_model',export,'--vocab_file',vocab,'--checkpoint',checkpoint,
              '--output_dir',directory,'--output_name','source_base','--probs_h5',base,
              '--h5_feature_source','final_layer','--h5_include_probs','true','--h5_final_layer_dim','16',
              '--device','cpu','--num_data_workers','1','--max_count','0','--max_predict_haplotypes','0','--drop_top_m_nearest','0'],directory)
            refined=directory/'stock_refinement';refined.mkdir()
            run('run_joint_inference.py',['--config',config_path,'--checkpoint',checkpoint,'--input_h5',base,
              '--output_dir',refined,'--output_prefix','source_refined','--device','cpu'],directory)
            h5path=refined/'source_refined.h5'
            with h5py.File(h5path,'r') as h:
                probability=np.asarray(h['refined_probs'],dtype=np.float32)
                node=np.asarray(h['node_id'],dtype=np.int32)
                left=np.asarray(h['tree_start'],dtype=np.float64);right=np.asarray(h['tree_end'],dtype=np.float64)
            if probability.shape!=(len(node),4) or not np.isfinite(probability).all() or np.any(probability<=0) or not np.allclose(probability.sum(-1),1,atol=1e-5,rtol=0) or np.any(right<=left):
                raise ValueError('finite strictly positive native refined C4 probabilities and valid intervals required; no floor')
            coordinate=bound(frame['coordinates'])
            with np.load(coordinate,allow_pickle=False) as archive:coords={key:archive[key].copy() for key in archive.files}
            grid=np.full((len(coords['grid_indexes']),120,4),np.nan,dtype=np.float32)
            supported=np.zeros(grid.shape[:2],dtype=bool)
            columns={int(value):index for index,value in enumerate(coords['query_nodes'])}
            for index,identity in enumerate(node):
                if int(identity) not in columns:raise ValueError('non-query native row')
                column=columns[int(identity)]
                mask=(coords['local_positions_bp']>=left[index])&(coords['local_positions_bp']<right[index])
                if np.any(supported[mask,column]):raise ValueError('overlapping native output interval')
                grid[mask,column]=probability[index];supported[mask,column]=True
            if not supported.all():raise ValueError('missing original TRAIN grid support; no denominator reduction')
            gridpath=directory/'sealed_C4_grid_probabilities.npz'
            with gridpath.open('xb') as stream:np.savez_compressed(stream,probabilities=grid,logits=np.log(grid).astype(np.float32),supported=supported,**coords)
            roster=bound(frame['roster'])
            manifest['parents'][parent]={'split':'train','grid_prediction':ref(gridpath),
              'native_refined_HDF5':ref(h5path),'observed_roster':ref(roster),'coordinate_ref':ref(coordinate),
              'shape':list(grid.shape),'query_count':120,'all_supported':True,'class_order':C4,
              'local_offset_bp':frame['local_offset_bp'],'coverage_original_bp0':frame['coverage_original_bp0'],
              'original_grid_formula_bp0':'50 + 100*grid_index',
              'unary_semantics':'finite float32 log(native overlapping-chunk averaged C4 probability), not original raw logits',
              'native_row_semantics':'refined_probs at half-open [tree_start,tree_end) in emitter local bp, node_id observed query join',
              'probability_floor':None,'in_sample':True}
            save(directory/'prediction_seal.json',manifest['parents'][parent])
            print(json.dumps({'parent':parent,'stage':'frozen_TRAIN_unaries_exported','shape':list(grid.shape),'device':'cpu','in_sample':True}),flush=True)
        bound(CHECKPOINT)
        for name,digest in SOURCE_PINS.items():bound({'path':name,'sha256':digest})
        manifest.update(status='complete',exit_code=0,model_modified=False,query_truth_read=False,TEST_read=False)
    except BaseException as error:
        manifest.update(status='failed_partial_preserved',exit_code=1,error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:save(output/'train_unaries_manifest.json',manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    parser.add_argument('--native',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'inactive_source_only','device':'cpu','parents':PARENTS,'fitted_checkpoint_ref':CHECKPOINT,
          'source_pins':SOURCE_PINS,'emitter_refit':False,'training_predictions':'in_sample_not_out_of_fold',
          'manifest_kind':'argmix_frozen_train_unaries_v1','output_manifest':'train_unaries_manifest.json'},indent=2));return
    if not args.output:raise ValueError('new root output directory required')
    output=confined(args.output)
    if not args.native:
        if output.exists():raise ValueError('fresh output required')
        # Route native import caches before exec. Export output itself is created only by the native process.
        staging=output.parent/(output.name+'_runtime')
        if staging.exists():raise ValueError('fresh runtime cache directory required')
        staging.mkdir(parents=True)
        env=native_environment(staging)
        os.execve(PREFIX/'bin/python',[str(PREFIX/'bin/python'),'-B',str(Path(__file__).resolve()),'--native',*sys.argv[1:]],env)
    if Path(sys.prefix).resolve()!=PREFIX.resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('exact native CPU interpreter and hidden CUDA required')
    execute(output)

if __name__=='__main__':main()
