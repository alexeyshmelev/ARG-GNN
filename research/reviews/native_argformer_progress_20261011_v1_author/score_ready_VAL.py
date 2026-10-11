#!/usr/bin/env python3
"""Score already sealed native VAL C4 retrievals with the unchanged established scorer."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).resolve().parent
DEST=ROOT/'experiments/results/native_argformer_tsinfer_fit_v7_summary_v2'
VAL=(('bio_paths_pilot_val_000','val000'),('bio_paths_pilot_val_001','val001'))
PINS={
 'scripts/score_official_argformer_retrieval_biological_v1.py':'2f4fc18efa5e83e7c1c3c72ce01cd3783fa03c83dc62225a52b99733c0d56f37',
 'scripts/score_official_argformer_biological_v1.py':'82e031a9b22bf469ed044000c87a345537c1255ee398705137fa47f6d72b68e4',
 'scripts/train_ancestralpaths_biological_v1.py':'826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056',
 'configs/official_argformer_biological_readout_v1.json':'0758e9f2cdcf738d75ddcafdeb03a87d3be86f92836075f30cd2e55937efc96d'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 receipts=[];seals=[];missing=[]
 for parent,label in VAL:
  path=ROOT/'experiments/results/native_argformer_tsinfer_fit_v7'/parent/'retrieve'/(label+'_retrieve_r003')/'receipt.json'
  if not path.exists():missing.append(parent);continue
  x=json.loads(path.read_bytes())
  if x['status']!='passed' or x['action']!='retrieve' or x['parent_id']!=parent:raise ValueError('passed exact VAL retrieval required')
  seal=x['prediction_seal']
  if sha(ROOT/seal['path'])!=seal['sha256']:raise ValueError('sealed probability bytes changed')
  receipts.append(ref(path));seals.append(seal)
 if missing:
  print(json.dumps({'status':'pending_both_VAL_seals','missing':missing,'truth_read':False,'ancestry_accuracy_available':False}));return
 if not args.execute:
  print(json.dumps({'status':'both_VAL_seals_ready_for_CPU_scoring','receipts':receipts,'truth_read':False}));return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit empty CUDA visibility required')
 if DEST.exists():raise ValueError('summary destination already exists; do not overwrite')
 for name,value in PINS.items():
  if sha(ROOT/name)!=value:raise ValueError('established scoring source/config changed')
 sys.path.insert(0,str(ROOT/'src'))
 import numpy as np
 legacy=load('_existing_native_VAL_C4_truth',ROOT/'scripts/score_official_argformer_biological_v1.py')
 scorer=load('_existing_native_VAL_retrieval_metrics',ROOT/'scripts/score_official_argformer_retrieval_biological_v1.py')
 common,helper=legacy.source_policy(PINS['configs/official_argformer_biological_readout_v1.json'])
 parents={}
 for parent,label in VAL:
  directory=ROOT/'data/official_argformer_tsinfer_parent_v3'/parent/'prepared'
  prepared=json.loads((directory/'prepared_manifest.json').read_bytes())
  roster=json.loads((directory/'observed_roster.json').read_bytes())
  contract=prepared['coordinate_contract'];indexes=np.asarray(contract['retained_fixed100bp_indexes'],dtype=np.int64)
  coverage=np.asarray(contract['coverage_original_bp0'],dtype=np.float64)
  queries=[r for r in roster if r['role']=='query']
  cohorts=np.asarray([r['observed_cohort'] for r in queries])
  if len(queries)!=120 or int((cohorts=='present_day').sum())!=64:raise ValueError('Q120/Present64 roster differs')
  edges=100*np.r_[indexes,indexes[-1]+1].astype(np.float64)
  edges[0]=max(edges[0],coverage[0]);edges[-1]=min(edges[-1],coverage[1])
  parents[parent]={'parent_id':parent,'split':'val','directory':ROOT/'data/biological_paths_execution_pilot_v1'/parent,
   'positions':50+100*indexes,'edges':edges,'retained_indexes':indexes,'coverage_original':coverage,
   'samples':roster,'query_slots':np.asarray([r['sample_index'] for r in queries],dtype=np.int64),
   'raw_query_nodes':np.asarray([r['source_observed_node_id'] for r in queries],dtype=np.int64),'cohorts':cohorts}
 DEST.mkdir()
 aggregate=DEST/'both_VAL_predictions_sealed_before_truth.json'
 save(aggregate,{'status':'both_VAL_predictions_sealed_before_truth','predictions':seals,'all_k':[1,5],'truth_scoring_started':False})
 rows=[]
 keys=('accuracy','macro_f1','balanced_accuracy','nll','infinite_nll_query_bp','probability_floor_used','brier',
       'calibration_ece','denominator_query_bp','query_count','represented_bin_support_bp','represented_span_bp',
       'original_parent_query_bp','score_scope','confusion_truth_rows_prediction_columns_query_bp',
       'boundaries','per_class','dosage_mse_per_diploid_bp_class','dosage_mae_per_diploid_bp_class')
 for seal in seals:
  value=scorer.score_parent(parents[seal['parent_id']],seal,common,helper,legacy,ref(aggregate))
  result={'parent_id':seal['parent_id'],'prediction_seal':{'path':seal['path'],'sha256':seal['sha256']},'results_by_k':{}}
  for k,item in value['results_by_k'].items():
   result['results_by_k'][k]={'condition':item['condition'],'exact_support_metrics':{}}
   for name,m in item['exact_support_metrics'].items():
    compact={key:m[key] for key in keys};compact['nll_is_infinite']=m['nll_is_infinite']
    result['results_by_k'][k]['exact_support_metrics'][name]=scorer.jsonable(compact)
  save(DEST/(seal['parent_id']+'_compact_ancestry_metrics.json'),result);rows.append(result)
 means={}
 for k in ('1','5'):
  means[k]={}
  for group in ('primary_present_day_64','secondary_ancient_56','all_queries_120'):
   values=[r['results_by_k'][k]['exact_support_metrics'][group] for r in rows]
   means[k][group]={name:sum(v[name] for v in values)/len(values) for name in ('accuracy','macro_f1','balanced_accuracy','brier','calibration_ece','dosage_mse_per_diploid_bp_class','dosage_mae_per_diploid_bp_class')}
   means[k][group]['nll_is_infinite']=any(v['nll_is_infinite'] for v in values)
   means[k][group]['nll']=None if means[k][group]['nll_is_infinite'] else sum(v['nll'] for v in values)/len(values)
 comparison_path=ROOT/'experiments/results/official_argmix_biological_metadata_repair_v3_summary/summary.json'
 comparison=json.loads(comparison_path.read_bytes())
 argmix_accuracy=comparison['equal_parent_means']['primary_present_day_64']['accuracy']
 summary={'kind':'native_ARGformer_tsinfer_actual_ancestry_summary_v2','status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
  'retrieval_receipts':receipts,'aggregate_prediction_seal_ref':ref(aggregate),'score_source_pins':PINS,
  'rows':rows,'equal_parent_means':means,'primary_k':1,'prespecified_sensitivity_k':5,'C4_order':['ANA','CHG','WHG','EHG'],'MLM_accuracy_is_not_ancestry_accuracy':True,
  'probability_floor':False,'calibration_fit':False,'training_or_model_inference_in_summary':False,'GPU_used':False,
  'transductive_fits':True,'query_ancestry_used_in_native_fit':False,
  'disclosures':['Native encoder was fitted separately within each observed ARG, including VAL observations: five MLM epochs over all280 samples and one pure-reference contrastive epoch. No query ancestry was used for native fitting.',
   'European heterochronous C4 on current tsinfer/tsdate trees is an adapted biological task. This is not original-paper task or numeric reproduction.',
   'Primary default top1 and prespecifiedtop5 use exhaustive same-ARG pure-reference full-path corpus and raw hard votes. Exact hard-vote zeros preserve infiniteNLL; no probability floor or fitted calibration.',
   'Common Present64 support allows descriptive scores against supervised Relate ARGMix. Genealogy, training task and optimization differ; two development parents do not establish superiority or independent confirmation.'],
  'descriptive_ARGMix_comparison':{'summary_ref':ref(comparison_path),'Relate_ARGMix_present64_accuracy':argmix_accuracy,
   'native_tsinfer_top1_present64_accuracy':means['1']['primary_present_day_64']['accuracy'],
   'native_minus_ARGMix_accuracy_percentage_points':100*(means['1']['primary_present_day_64']['accuracy']-argmix_accuracy),
   'matched_topology_or_training':False,'superiority_established':False}}
 save(DEST/'summary.json',summary)
 print(json.dumps({'status':'complete','summary_ref':ref(DEST/'summary.json'),'actual_Present64_ancestry_accuracy_top1':means['1']['primary_present_day_64']['accuracy'],
 'actual_Present64_ancestry_accuracy_top5':means['5']['primary_present_day_64']['accuracy'],'MLM_accuracy_reported_as_ancestry':False},indent=2))
if __name__=='__main__':main()
