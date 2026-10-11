"""H023: unchanged completed H020 prediction diagnostic, old exposed VAL only."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,os,types
import numpy as np
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=Path(__file__).resolve().parent
PARENTS=('bio_paths_pilot_val_000','bio_paths_pilot_val_001')
SEEDS=(101,202,303);METHODS=('candidate_age_d2','candidate_plain_d2')
BOUNDARIES=np.arange(250000.,1750001.,250000.)
BANDS=('[0,10000)','[10000,25000)','[25000,50000)','[50000,125000]','>125000_terminal_accounting')
VERIFIED={};META={};ASTS={}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as s:
  for block in iter(lambda:s.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def path(value):
 p=Path(value);p=p if p.is_absolute() else ROOT/p
 if ROOT not in p.parents or p.resolve()!=p or any(x.is_symlink() for x in [p,*p.parents] if x!=ROOT):
  raise ValueError('Canonical immutable project input required')
 return p
def reference(p):p=path(p);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def check_ref(r):
 p=path(r['path'])
 if sha(p)!=r['sha256']:raise ValueError('Input hash drift: '+str(p))
 VERIFIED[str(p.relative_to(ROOT))]=r['sha256'];return p
def metadata(p,expected=None):
 p=path(p);r=reference(p)
 if expected is not None and r['sha256']!=expected:raise ValueError('Metadata hash drift')
 META[r['path']]=r['sha256'];return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as s:json.dump(v,s,indent=2,allow_nan=False);s.write('\n')
def capsule(source,names,ns):
 source=path(source);tree=ast.parse(source.read_text());nodes=[]
 for name in names:
  n=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
  ASTS[str(source.relative_to(ROOT))+':'+name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
  nodes.append(copy.deepcopy(n))
 module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
 exec(compile(module,str(source),'exec'),ns)
 return ns

# Registration and ledger are consulted before reading any prediction/truth arrays.
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
decisions=metadata('research/experiment_decisions.json')
h023=next(x for x in decisions['decisions'] if x['id']=='H023')
if 'terminal' not in json.dumps(h023).lower():raise ValueError('Await root registered terminal accounting correction before outcome inspection')
ledger=path('research/experiment_ledger.md');ledger_text=ledger.read_text()
assert '## H020:' in ledger_text
META[str(ledger.relative_to(ROOT))]=sha(ledger)
write(OUT/'registration_snapshot.json',{'H023':h023,'decisions_read':reference('research/experiment_decisions.json'),'ledger_read':reference(ledger),
 'ground_truth_authority':'Root explicitly authorized original external scorer for two already-exposed VAL parents only; no TEST/new parents.',
 'four_prespecified_bands_unchanged':True,'terminal_accounting_correction_before_outcomes':True})
cfg=metadata('configs/biological_candidate_pilot_v1.json','4151e7e156db3f188512bcc738b43597407d7707bc77c420509165c2e9bfc2f1')
candidate_source=check_ref({'path':'src/arggnn/biological_candidate.py','sha256':'6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3'})
scorer_source=check_ref({'path':'scripts/train_ancestralpaths_biological_v1.py','sha256':'826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056'})
ns={'ROOT':ROOT,'Path':Path,'hashlib':hashlib,'json':json,'np':np}
truth_ns=capsule(candidate_source,('sha','path','checked','ExternalTruthCapability'),dict(ns))
helpers=capsule(scorer_source,('intersections','boundary_match','tracts','length_summary','full_span_score'),{'np':np})
arg_summary=metadata('experiments/results/official_argmix_biological_metadata_repair_v3_summary/summary.json')
assert arg_summary['source_seed']==50 and arg_summary['native_selected_epoch']==20
assert arg_summary['current_native_author_code_model_objective_postprocessing_preserved'] is True
arg_receipt=metadata(arg_summary['native_inference_receipt']['path'],arg_summary['native_inference_receipt']['sha256'])
seals={x['parent_id']:x for x in arg_receipt['prediction_seals']}
assert set(seals)==set(PARENTS) and arg_receipt['exit_code']==0
published={x['parent_id']:x['groups']['primary_present_day_64'] for x in arg_summary['rows']}
cand_summaries={};cand_receipts={}
for seed in SEEDS:
 cand_summaries[seed]=metadata(f'experiments/results/biological_candidate_pilot_v1_seed{seed}_summary/summary.json')
 cand_receipts[seed]=metadata(cand_summaries[seed]['native_receipt']['path'],cand_summaries[seed]['native_receipt']['sha256'])
 assert cand_receipts[seed]['status']=='completed_three_conditions_one_seed' and cand_receipts[seed]['TEST_parents']==0
 for name in METHODS:assert cand_receipts[seed]['models'][name]['status']=='completed'
frozen_h020=metadata('experiments/results/readme_biological_summary_v4/summary.json')

# Exact physical regions: no distance cap and no artificial 0/2Mb boundaries.
cuts=sorted(set([0.,2_000_000.,*BOUNDARIES,*((BOUNDARIES[:-1]+BOUNDARIES[1:])/2),*(b+sign*r for b in BOUNDARIES for sign in (-1,1) for r in (10000.,25000.,50000.,125000.))]))
cuts=np.asarray([x for x in cuts if 0<=x<=2_000_000.])
regions=[[] for _ in BANDS]
for left,right in zip(cuts[:-1],cuts[1:]):
 d=float(np.abs(BOUNDARIES-(left+right)/2).min())
 band=0 if d<10000 else 1 if d<25000 else 2 if d<50000 else 3 if d<=125000 else 4
 regions[band].append((left,right))
assert [sum(r-l for l,r in x) for x in regions]==[140000.,210000.,350000.,1050000.,250000.]
# Endpoint/intersection handmade checks are independent of any outcomes.
toy_edges=np.array([0.,190000.,260000.,2000000.]);toy_truth=np.array([[0.,230000.],[230000.,2000000.]])
tb,tr,tw=helpers['intersections'](toy_edges,toy_truth)
assert tw.sum()==2000000.
def region_weights(left,right):
 return np.stack([sum(np.maximum(0.,np.minimum(right,r)-np.maximum(left,l)) for l,r in pieces) for pieces in regions],axis=1)
toy=region_weights(np.array([0.]),np.array([2000000.]))
assert np.array_equal(toy[0],np.array([140000.,210000.,350000.,1050000.,250000.]))

def scores(probabilities,parent,truth):
 bins,rows,spans=helpers['intersections'](parent['edges'],truth.intervals)
 left=np.maximum(parent['edges'][bins],truth.intervals[rows,0])
 right=np.minimum(parent['edges'][bins+1],truth.intervals[rows,1])
 weights=region_weights(left,right)
 assert np.array_equal(weights.sum(1),spans)
 columns=np.flatnonzero(parent['cohorts']=='present_day')
 assert len(columns)==64 and np.array_equal(columns,np.arange(64))
 den=weights.sum(0)*64;good=np.zeros(5);nll=np.zeros(5);inf=np.zeros(5);brier=np.zeros(5)
 confusion=np.zeros((5,4,4))
 for start in range(0,len(bins),1024):
  stop=start+1024;p=probabilities[bins[start:stop]][:,columns].astype(np.float64);y=truth.c4[rows[start:stop]][:,columns]
  pred=p.argmax(-1);selected=np.take_along_axis(p,y[...,None],-1)[...,0]
  positive=selected>0;logs=-np.log(np.where(positive,selected,1.))
  b=np.sum(p*p,-1)+1.-2.*selected
  for k in range(5):
   w=weights[start:stop,k,None];good[k]+=np.sum(w*(pred==y));inf[k]+=np.sum(w*~positive)
   nll[k]+=np.sum(w*logs);brier[k]+=np.sum(w*b)
   np.add.at(confusion[k],(y.ravel(),pred.ravel()),np.broadcast_to(w,y.shape).ravel())
 bands=[{'band':band,'denominator_query_bp':float(den[i]),'correct_query_bp':float(good[i]),'error_query_bp':float(den[i]-good[i]),
  'accuracy':float(good[i]/den[i]),'nll':None if inf[i] else float(nll[i]/den[i]),'NLL_is_infinite':bool(inf[i]),
  'infinite_nll_query_bp':float(inf[i]),'brier':float(brier[i]/den[i]),
  'confusion_truth_rows_prediction_columns_query_bp':confusion[i].tolist()} for i,band in enumerate(BANDS)]
 overall={'denominator_query_bp':float(den.sum()),'correct_query_bp':float(good.sum()),'accuracy':float(good.sum()/den.sum()),
  'nll':None if inf.sum() else float(nll.sum()/den.sum()),'NLL_is_infinite':bool(inf.sum()),
  'infinite_nll_query_bp':float(inf.sum()),'brier':float(brier.sum()/den.sum())}
 # Unchanged external scorer independently reconstructs the published overall endpoint.
 old=helpers['full_span_score'](probabilities,parent,truth,columns,cfg['metrics'])
 for field in ('accuracy','nll','brier','denominator_query_bp','infinite_nll_query_bp'):
  assert (old[field] is None and overall[field] is None) or abs(old[field]-overall[field])<1e-12,(field,old[field],overall[field])
 assert np.array_equal(confusion.sum(0),old['confusion_truth_rows_prediction_columns_query_bp'])
 return bands,overall,{'accuracy':old['accuracy'],'nll':old['nll'],'brier':old['brier'],'denominator_query_bp':old['denominator_query_bp'],'infinite_nll_query_bp':old['infinite_nll_query_bp']}

results=[];supports={};truths={};inputs=[]
for parent_id in PARENTS:
 seal=seals[parent_id];frame=seal['frame'];assert frame['split']=='val' and frame['parent_id']==parent_id
 roster=metadata(frame['roster']['path'],frame['roster']['sha256']);queries=[r for r in roster if r['role']=='query']
 assert len(roster)==280 and len(queries)==120
 qslots=np.asarray([r['sample_index'] for r in queries]);qnodes=np.asarray([r['inferred_tree_node_id'] for r in queries])
 raw_nodes=np.asarray([r['source_observed_node_id'] for r in queries]);cohorts=np.asarray([r['observed_cohort'] for r in queries])
 assert sum(cohorts=='present_day')==64 and np.array_equal(qslots[:64],np.arange(64))
 arg_pred=seal['grid_prediction'];check_ref(arg_pred)
 with np.load(path(arg_pred['path']),allow_pickle=False) as z:
  probs=z['probabilities'].copy();supported=z['supported'].copy()
  positions=z['positions_bp'].copy();edges=z['bin_edges_bp'].copy();indexes=z['grid_indexes'].copy();nodes=z['query_nodes'].copy()
 assert supported.all() and seal['full_support_complete'] and seal['missing_grid_query_cells']==0
 assert np.array_equal(nodes,qnodes) and probs.shape==(len(positions),120,4)
 assert positions.dtype==np.float64 and edges.dtype==np.float64
 assert np.array_equal(positions,50.+100.*indexes)
 assert np.array_equal(edges,np.clip(np.r_[positions-50.,positions[-1]+50.],*frame['coverage_original_bp0']))
 assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(-1),1.,atol=1e-6,rtol=1e-5)
 parent={'parent_id':parent_id,'split':'val','raw_query_nodes':raw_nodes,'edges':edges,'positions':positions,
  'query_slots':qslots,'samples':roster,'cohorts':cohorts,'coverage_original':np.asarray(frame['coverage_original_bp0']),'retained_indexes':indexes.tolist()}
 # Genuine truth decoded ONLY by unchanged original scorer capability after sealed prediction SHA validation.
 truth=truth_ns['ExternalTruthCapability'](parent,cfg,'VAL_after_sealed_prediction',arg_pred)
 truths[parent_id]=truth;truth_record={'path':str(path(truth.provenance['path']).relative_to(ROOT)),'sha256':truth.provenance['sha256']}
 check_ref(truth_record);metadata(cfg['external_truth_manifest']['path'],cfg['external_truth_manifest']['sha256'])
 bands,total,replay=scores(probs,parent,truth)
 expected=published[parent_id]
 assert abs(total['accuracy']-expected['accuracy'])<1e-12 and total['denominator_query_bp']==expected['full_retained_support_denominator_query_bp']
 assert abs(total['nll']-expected['NLL'])<1e-12 and abs(total['brier']-expected['Brier'])<1e-12
 results.append({'method':'current_release_ARGMix','seed':50,'parent_id':parent_id,'selected_epoch':arg_summary['native_selected_epoch'],
  'prediction':arg_pred,'truth':truth_record,'bands':bands,'overall':total,'unchanged_external_scorer_overall_replay':replay,'published_overall':expected})
 inputs.append({'method':'current_release_ARGMix','seed':50,'parent_id':parent_id,'prediction':arg_pred,'truth':truth_record,
  'selected_epoch':20,'declared_native_checkpoint':seal['checkpoint'],'roster':frame['roster'],'original_coordinates':frame['coordinates'],
  'native_final_estimand':seal['native_final_estimand']})
 supports[parent_id]={'coverage_original_bp0':frame['coverage_original_bp0'],'represented_bin_support_bp':[float(edges[0]),float(edges[-1])],
  'query_sample_indexes':qslots.tolist(),'query_node_ids':qnodes.tolist(),'query_raw_node_ids':raw_nodes.tolist(),'present_day_query_indexes':list(range(64)),
  'positions_array_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),'edges_array_sha256':hashlib.sha256(edges.tobytes()).hexdigest(),
  'retained_index_array_sha256':hashlib.sha256(indexes.tobytes()).hexdigest(),'band_denominators_query_bp':[x['denominator_query_bp'] for x in bands]}
 for seed in SEEDS:
  receipt=cand_receipts[seed];summary=cand_summaries[seed]
  for method in METHODS:
   model=receipt['models'][method];metric_ref=next(r for r in model['metrics'] if r['parent_id']==parent_id)
   metric=metadata(metric_ref['path'],metric_ref['sha256']);pred=metric['sealed_prediction'];check_ref(pred)
   assert metric['selected_epoch']==model['best_epoch']==summary['conditions'][method]['selected_epoch']
   assert metric['model']==method and metric['seed']==seed and metric['split']=='val'
   assert path(metric['external_truth_provenance']['path'])==path(truth_record['path']) and metric['external_truth_provenance']['sha256']==truth_record['sha256']
   with np.load(path(pred['path']),allow_pickle=False) as z:
    candidate=z['C4_probabilities'].copy()
    assert np.array_equal(z['positions_bp'],positions) and np.array_equal(z['bin_edges_bp'],edges)
    assert np.array_equal(z['retained_fixed100bp_indexes'],indexes) and np.array_equal(z['query_node_ids'],qnodes)
    assert np.array_equal(z['query_sample_indexes'],qslots) and int(z['normalization_length_bp'])==2000000
    assert np.array_equal(z['coverage_original_bp0'],parent['coverage_original'])
   assert candidate.shape==probs.shape and np.isfinite(candidate).all() and (candidate>=0).all()
   assert np.allclose(candidate.sum(-1),1.,atol=1e-6,rtol=1e-5)
   bands,total,replay=scores(candidate,parent,truth)
   expected=metric['C4_exact_support_metrics']['primary_present_day_64']
   for field in ('accuracy','nll','brier','denominator_query_bp','infinite_nll_query_bp'):
    assert (total[field] is None and expected[field] is None) or abs(total[field]-expected[field])<1e-12
   results.append({'method':method,'seed':seed,'parent_id':parent_id,'selected_epoch':model['best_epoch'],
    'prediction':pred,'truth':truth_record,'bands':bands,'overall':total,'unchanged_external_scorer_overall_replay':replay,
    'published_overall':{k:expected[k] for k in ('accuracy','nll','brier','denominator_query_bp','infinite_nll_query_bp')}})
   inputs.append({'method':method,'seed':seed,'parent_id':parent_id,'selected_epoch':model['best_epoch'],
    'prediction':pred,'truth':truth_record,'metrics':metric_ref,'declared_selected_checkpoint':model['checkpoint'],
    'fit_receipt':cand_summaries[seed]['native_receipt']})
   print(json.dumps({'completed':method,'seed':seed,'parent':parent_id,'overall_accuracy':total['accuracy']}),flush=True)

def mean_rows(rows):
 return {'accuracy':float(np.mean([x['accuracy'] for x in rows])),
  'nll':None if any(x['NLL_is_infinite'] for x in rows) else float(np.mean([x['nll'] for x in rows])),
  'NLL_is_infinite':any(x['NLL_is_infinite'] for x in rows),'brier':float(np.mean([x['brier'] for x in rows]))}
fit_means=[];method_means={};parent_means={}
for method in (*METHODS,'current_release_ARGMix'):
 seeds=SEEDS if method in METHODS else (50,)
 for seed in seeds:
  rr=[x for x in results if x['method']==method and x['seed']==seed]
  fit_means.append({'method':method,'seed':seed,'overall':mean_rows([x['overall'] for x in rr]),
   'bands':[{'band':BANDS[k],**mean_rows([x['bands'][k] for x in rr])} for k in range(5)]})
 fit=[x for x in fit_means if x['method']==method]
 method_means[method]={'optimizer_seeds':list(seeds),'aggregation':'equal biological-parent metric mean per separate fit, then mean separate fits; no probability ensemble',
  'overall':mean_rows([x['overall'] for x in fit]),'bands':[{'band':BANDS[k],**mean_rows([x['bands'][k] for x in fit])} for k in range(5)]}
 parent_means[method]={parent:{'overall':mean_rows([x['overall'] for x in results if x['method']==method and x['parent_id']==parent]),
  'bands':[{'band':BANDS[k],**mean_rows([x['bands'][k] for x in results if x['method']==method and x['parent_id']==parent])} for k in range(5)]} for parent in PARENTS}
for method in METHODS:
 published_mean=frozen_h020['candidate_seed_group_means'][method]['primary_present_day_64']['accuracy']
 assert abs(method_means[method]['overall']['accuracy']-published_mean)<1e-12
assert abs(method_means['current_release_ARGMix']['overall']['accuracy']-arg_summary['equal_parent_means']['primary_present_day_64']['accuracy'])<1e-12
contrasts={}
for method in METHODS:
 contrasts[method]={'equal_parent_separate_fit_mean_difference_percentage_points':
  [(method_means[method]['bands'][k]['accuracy']-method_means['current_release_ARGMix']['bands'][k]['accuracy'])*100 for k in range(5)],
  'overall_accuracy_difference_percentage_points':(method_means[method]['overall']['accuracy']-method_means['current_release_ARGMix']['overall']['accuracy'])*100,
  'per_parent_mean_difference_percentage_points':{p:[(parent_means[method][p]['bands'][k]['accuracy']-parent_means['current_release_ARGMix'][p]['bands'][k]['accuracy'])*100 for k in range(5)] for p in PARENTS},
  'near0_10k_vs_far50_125k_difference_of_accuracy_gaps_percentage_points':
   ((method_means[method]['bands'][0]['accuracy']-method_means['current_release_ARGMix']['bands'][0]['accuracy'])-
    (method_means[method]['bands'][3]['accuracy']-method_means['current_release_ARGMix']['bands'][3]['accuracy']))*100}
report={'kind':'biological_tile_boundary_diagnostic_v1','hypothesis_id':'H023','source_hypothesis':'H020','status':'completed_exploratory_DEVELOPMENT_diagnostic',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parents':list(PARENTS),'independent_biological_parents':2,'TEST_parents':0,
 'artificial_interior_boundaries_bp':BOUNDARIES.tolist(),'distance_bands':list(BANDS),'terminal_stratum':'accounting only, separately displayed; no capped distances or artificial domain-edge boundaries',
 'distance_assignment':'pointwise exact physical-bp intersection with bands; constant original prediction bins and unmodified full truth intervals',
 'primary_queries':64,'separate_fit_results':results,'equal_parent_means_per_separate_fit':fit_means,'equal_parent_then_separate_fit_means':method_means,
 'per_parent_then_separate_fit_means':parent_means,'candidate_ARGMix_accuracy_contrasts':contrasts,'support_and_identity':supports,
 'prediction_truth_selection_inputs':inputs,'verified_input_file_sha256':VERIFIED,'metadata_read_sha256':META,'unchanged_original_scorer_AST_sha256':ASTS,
 'analysis_source':reference(__file__),'original_fit_source_pin_maps':{'candidates':{str(seed):cand_receipts[seed]['source_sha256'] for seed in SEEDS},'ARGMix_inference_owned_code':arg_receipt['owned_code_sha256'],'ARGMix_author_commit':arg_summary['author_commit']},'original_scorer_and_band_sums_recover_H020_accuracy_NLL_Brier':True,
 'no_probability_floor':True,'posthoc_renormalization':False,'predictions_postprocessing_targets_models_and_selection_unchanged':True,
 'GPU_models_fits_or_installations_used':False,'live_budget_or_ledger_modified':False,'significance_causal_or_superiority_claim':False,
 'limits':'Two repeatedly exposed development VAL parents; optimizer seeds are separate fits, not independent biological replicates.'}
write(OUT/'summary.json',report)
print(json.dumps({'completed':True,'method_means':method_means,'accuracy_contrasts':contrasts},indent=2))
