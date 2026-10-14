"""Registered H038 Present182 boundary diagnostic. Import performs no data reads."""
from pathlib import Path
import os,sys,json,hashlib,io,importlib.util,time,datetime,traceback
import numpy as np
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H038_boundary_error_diagnostic_source_v1'
IDS=tuple('ap_expanded_val_%03d'%i for i in range(8))
ARMS=('aligned','zero_concordance')
PHASE='H038_boundary_error_diagnostic_20261015_v1'
CAP=10_000_000
LABELS=['Ana','CHG','WHG','EHG']
BINS=np.array([0,100,500,1000,5000,10000,50000,100000,500000,2000000.],dtype=float)
def req(x,msg):
 if not x:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def jval(x):
 if isinstance(x,(np.ndarray,np.generic)):return jval(x.tolist())
 if isinstance(x,dict):return {str(k):jval(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jval(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes()))
def save(name,x):
 b=(json.dumps(jval(x),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 req(Path(name).name==name and sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(b)<CAP,'Output cap/name')
 with (HERE/name).open('xb') as f:f.write(b)
 return dict(path=str((HERE/name).relative_to(ROOT)),sha256=sha(b))
def emit(stage,**kw):print(json.dumps(jval(dict(stage=stage,utc=utc(),**kw))),flush=True)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Reader:
 def __init__(self):self.truth_allowed=False;self.reads={}
 def read(self,r,**kw):
  req(isinstance(r,dict) and 'path'in r and 'sha256'in r,'Bound read')
  p=ROOT/r['path'];req(not Path(r['path']).is_absolute() and '..'not in Path(r['path']).parts and p.resolve()==p and p.is_file(),'Canonical project file')
  req('ap_expanded_outer_' not in str(p),'No outer data')
  if 'truth'in p.parts or 'oracle'in p.parts:req(self.truth_allowed and any(pid in p.parts for pid in IDS),'Sealed8VAL-only truth')
  if p.suffix=='.npz':req(any(pid in str(p) for pid in IDS),'Only8VAL arrays')
  b=p.read_bytes();req(sha(b)==r['sha256'],'SHA drift '+r['path']);self.reads[r['path']]=r['sha256'];return b
 def decode_json(self,r):return json.loads(self.read(r))
def histogram(values):
 a=np.asarray(values,dtype=float)
 return dict(count=len(a),edges_bp=BINS,counts=np.histogram(a,BINS)[0],
  mean_bp=float(a.mean()) if len(a) else None,
  quantiles_bp={str(q):float(np.quantile(a,q)) if len(a) else None for q in [0,.1,.25,.5,.75,.9,.99,1]},
  fractions_at_or_below={str(int(c)):float(np.mean(a<=c)) if len(a) else None for c in [1000,10000]},
  total_bp=float(a.sum()))
def geometry_candidates(edges,holds):
 req(len(edges)==len(holds)+1 and holds.ndim==1 and np.all(np.diff(holds)>=0),'Monotone native-hold map')
 return np.asarray(edges)[1:-1][holds[1:]!=holds[:-1]]
def transitions(tracts):
 return [(float(tracts[i]['right_bp']),int(tracts[i]['class_0based']),int(tracts[i+1]['class_0based'])) for i in range(len(tracts)-1)]
def nearest_distance(x,candidates):
 if not len(candidates):return None
 i=np.searchsorted(candidates,x)
 return min(abs(x-candidates[j]) for j in [i-1,i] if 0<=j<len(candidates))
def correct_span(tract,truth):
 left,right=tract['left_bp'],tract['right_bp'];total=0.
 for t in truth:
  if t['right_bp']<=left:continue
  if t['left_bp']>=right:break
  if t['class_0based']==tract['class_0based']:total+=max(0,min(right,t['right_bp'])-max(left,t['left_bp']))
 return float(total)
def diagnose(group,candidates,helper):
 # group is the unchanged full_span_score result, including transient tract lists.
 true_lengths=[];true_uncensored=[];pred_lengths=[];pred_internal=[];miss_dist=[];extra_dist=[];match_errors=[]
 matrices={k:np.zeros((4,4),dtype=np.int64) for k in ['true','predicted','matched_true','missed_true','extra_predicted']}
 cross=np.zeros((16,16),dtype=np.int64)
 counts=dict(true=0,predicted=0,matched=0,missed=0,extra=0,oracle_matched=0,
  true_no_candidate_within_1kb=0,missed_no_candidate_within_1kb=0,
  true_candidate_competition_gap=0,positional_matches_same_transition=0,
  extra_pred_near_true_within_1kb=0)
 short={str(c):dict(predicted_tract_count=0,predicted_tract_bp=0.,error_query_bp=0.,
  internal_ABA_island_count=0,strict_zero_true_support_islands=0,
  majority_wrong_islands=0,strict_false_island_bp=0.,strict_false_islands_both_boundaries_extra=0,
  unique_boundaries_in_strict_false_islands=0,unique_extra_boundaries_in_strict_false_islands=0) for c in [1000,10000]}
 for qr in group['per_query']:
  true,pred=qr['true_tracts'],qr['predicted_tracts']
  tt,pt=transitions(true),transitions(pred);tb=[x[0] for x in tt];pb=[x[0] for x in pt]
  match=qr['boundaries'];oracle=helper.boundary_match(tb,candidates.tolist(),1000.)
  req(oracle['matched']>=match['matched'],'Raw hold decoder exceeds proposed geometric ceiling')
  tmap={x[0]:x[1:] for x in tt};pmap={x[0]:x[1:] for x in pt}
  matched_t={x['true_bp'] for x in match['matches']};matched_p={x['predicted_bp'] for x in match['matches']}
  for name,rows in [('true',tt),('predicted',pt)]:
   for _,a,b in rows:matrices[name][a,b]+=1
  for item in match['matches']:
   a,b=tmap[item['true_bp']];c,d=pmap[item['predicted_bp']]
   matrices['matched_true'][a,b]+=1;cross[a*4+b,c*4+d]+=1
   counts['positional_matches_same_transition']+=int((a,b)==(c,d));match_errors.append(item['absolute_error_bp'])
  far_true=0
  for x,a,b in tt:
   distance=nearest_distance(x,candidates);far=distance is None or distance>1000
   far_true+=int(far)
   if x not in matched_t:
    matrices['missed_true'][a,b]+=1;counts['missed_no_candidate_within_1kb']+=int(far)
    if distance is not None:miss_dist.append(distance)
  counts['true_no_candidate_within_1kb']+=far_true
  counts['true_candidate_competition_gap']+=len(tb)-oracle['matched']-far_true
  for x,a,b in pt:
   if x not in matched_p:
    matrices['extra_predicted'][a,b]+=1;dist=nearest_distance(x,np.asarray(tb))
    if dist is not None:
     extra_dist.append(dist);counts['extra_pred_near_true_within_1kb']+=int(dist<=1000)
  for k,key in [('true','true_boundaries'),('predicted','predicted_boundaries'),('matched','matched'),('missed','missed'),('extra','extra')]:
   counts[k]+=match[key]
  counts['oracle_matched']+=oracle['matched']
  true_lengths.extend(t['right_bp']-t['left_bp'] for t in true)
  true_uncensored.extend(t['right_bp']-t['left_bp'] for t in true if not t['support_left_censored'] and not t['support_right_censored'])
  strict_edges={str(c):set() for c in [1000,10000]}
  for i,t in enumerate(pred):
   width=t['right_bp']-t['left_bp'];pred_lengths.append(width)
   internal=0<i<len(pred)-1
   if internal:pred_internal.append(width)
   correct=correct_span(t,true)
   for cutoff in [1000,10000]:
    if width>cutoff:continue
    s=short[str(cutoff)];s['predicted_tract_count']+=1;s['predicted_tract_bp']+=width;s['error_query_bp']+=width-correct
    island=internal and pred[i-1]['class_0based']==pred[i+1]['class_0based']!=t['class_0based']
    if island:
     s['internal_ABA_island_count']+=1
     s['majority_wrong_islands']+=int(correct/width<.5)
     if correct==0:
      s['strict_zero_true_support_islands']+=1;s['strict_false_island_bp']+=width
      strict_edges[str(cutoff)].update([t['left_bp'],t['right_bp']])
      s['strict_false_islands_both_boundaries_extra']+=int(t['left_bp']not in matched_p and t['right_bp']not in matched_p)
  for cutoff,points in strict_edges.items():
   short[cutoff]['unique_boundaries_in_strict_false_islands']+=len(points)
   short[cutoff]['unique_extra_boundaries_in_strict_false_islands']+=len(points-matched_p)
 req(all(counts[k]==group['boundaries'][v] for k,v in [('true','true_boundaries'),('predicted','predicted_boundaries'),('matched','matched'),('missed','missed'),('extra','extra')]),'Original matcher counts differ')
 error_bp=float(group['denominator_query_bp']-np.trace(group['confusion_truth_rows_prediction_columns_query_bp']))
 for s in short.values():
  s['fraction_of_total_error_query_bp']=s['error_query_bp']/error_bp if error_bp else None
  s['fraction_of_predicted_tract_count']=s['predicted_tract_count']/len(pred_lengths) if pred_lengths else None
  s['fraction_of_predicted_boundaries_in_strict_false_islands']=s['unique_boundaries_in_strict_false_islands']/counts['predicted'] if counts['predicted'] else None
  s['fraction_of_extra_boundaries_in_strict_false_islands']=s['unique_extra_boundaries_in_strict_false_islands']/counts['extra'] if counts['extra'] else None
 recall_upper=counts['oracle_matched']/counts['true'] if counts['true'] else None
 return dict(original_boundary_metrics=group['boundaries'],accuracy=group['accuracy'],error_query_bp=error_bp,
  count_decomposition=counts,short_predicted_tracts_and_islands=short,
  distributions=dict(true_support_clipped=histogram(true_lengths),true_uncensored_only=histogram(true_uncensored),
   predicted_support_clipped=histogram(pred_lengths),predicted_internal_only=histogram(pred_internal),
   missed_true_distance_to_nearest_hold_edge=histogram(miss_dist),
   extra_pred_distance_to_nearest_true_change=histogram(extra_dist),
   positional_match_absolute_error=histogram(match_errors)),
  class_transition_matrices=matrices,matched_true_vs_predicted_transition_16x16=cross,
  positional_matched_same_transition_fraction=counts['positional_matches_same_transition']/counts['matched'] if counts['matched'] else None,
  representable_boundary_recall_upper_bound=dict(value=recall_upper,matched=counts['oracle_matched'],
   true_boundaries=counts['true'],candidate_edges_per_query=len(candidates),tolerance_bp=1000,
   definition='Maximum-cardinality class-agnostic positional oracle match to scored bin edges where native readout_hold changes',
   attainability='Geometric diagnostic ceiling only. No predicted labels, fitted oracle or score correction.',
   includes_existing_grid_snapping=True,unrepresented_native_holds_omitted=True))
def stat(v):
 if any(x is None for x in v):return dict(mean=None,sample_sd=None,n=8,ddof=1,undefined_parents=sum(x is None for x in v))
 a=np.asarray(v,dtype=float);return dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),n=8,ddof=1)
def aggregate(cells):
 metrics={k:stat([c['original_boundary_metrics'][k] for c in cells]) for k in ['precision','recall','f1']}
 metrics['accuracy']=stat([c['accuracy'] for c in cells])
 metrics['representable_boundary_recall_upper_bound']=stat([c['representable_boundary_recall_upper_bound']['value'] for c in cells])
 counts={k:sum(c['count_decomposition'][k] for c in cells) for k in cells[0]['count_decomposition']}
 dist={}
 for key in cells[0]['distributions']:
  ds=[c['distributions'][key] for c in cells];n=sum(d['count'] for d in ds)
  dist[key]=dict(count=n,edges_bp=BINS,pooled_counts=np.sum([d['counts'] for d in ds],axis=0),
   pooled_count_weighted_mean_bp=sum(d['total_bp'] for d in ds)/n if n else None,
   parent_median_bp_mean_SD=stat([d['quantiles_bp']['0.5'] for d in ds]))
 return dict(mean_SD_across8simulations=metrics,pooled_count_decomposition=counts,
  pooled_transition_matrices={k:np.sum([c['class_transition_matrices'][k] for c in cells],axis=0) for k in cells[0]['class_transition_matrices']},
  pooled_matched_transition_16x16=np.sum([c['matched_true_vs_predicted_transition_16x16'] for c in cells],axis=0),
  pooled_distributions=dist,short_tract_parent_summaries={cut:{
   k:stat([c['short_predicted_tracts_and_islands'][cut][k] for c in cells])
   for k in cells[0]['short_predicted_tracts_and_islands'][cut]} for cut in ['1000','10000']},
  positional_matched_same_transition_fraction_mean_SD=stat([c['positional_matched_same_transition_fraction'] for c in cells]))
def run():
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 req(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']),'One BLAS/OMP thread')
 sys.path.insert(0,str(ROOT/'src'));start=time.monotonic();inp=json.loads((HERE/'inputs.json').read_text());reader=Reader()
 handmade=json.loads((HERE/'handmade_source_checks.json').read_text());req(handmade['source_ref']==ref(HERE/'boundary_diagnostic.py') and handmade['input_ref']==ref(HERE/'inputs.json') and handmade['passed'] is True and handmade['VAL_arrays_or_truth_read'] is False,'Pinned handmade checks before data access')
 decisions=json.loads((ROOT/'research/experiment_decisions.json').read_text());found=[]
 def walk(x):
  if isinstance(x,dict):
   if x.get('phase',x.get('id'))==PHASE:found.append(x)
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(decisions);req(len(found)==1 and found[0]==inp['registered_amendment'],'Exact prospective root registration')
 for p,h in inp['source_sha256'].items():reader.read(dict(path=p,sha256=h))
 meta={k:reader.decode_json(v) for k,v in inp['metadata_refs'].items()}
 helpers=module('scripts/train_ancestralpaths_biological_v1.py','boundary_original_metric')
 ap=module('research/reviews/expanded_native_AP_score_root_assistance_v1/root_score_helper.py','boundary_scoring_only')
 from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1
 from arggnn.mutation_routed_fast_primary_v1 import verify_seal_v1
 from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
 aligned=meta['aligned_input'];cfg=meta['H038_config'];gen=meta['generation_config']
 req(cfg['metrics']['boundary_tolerance_bp']==1000 and gen['parent_registry']==meta['registration']['parent_registry'],'Original geometry/scorer policy')
 index=read_full_cache_index_v1(inp['metadata_refs']['observation_cache'],reader,registry=gen['parent_registry'],
  entry_refs=aligned['observation_entry_refs'],source_refs=aligned['observation_cache_source_refs'])
 parents={};holds={};candidates={};seals={};checks=[]
 for pid in IDS:
  parent=ap.read_h038_scoring_only_v1(index,pid,reader);parents[pid]=parent
  rec=index['parents'][pid];keys=rec['model_spec']
  req(keys['readout_hold']=='model.readout_hold' and keys['spans']=='model.spans','Original observed geometry keys')
  with np.load(io.BytesIO(reader.read(rec['array_ref'])),allow_pickle=False) as z:
   h=z[keys['readout_hold']].copy();sp=z[keys['spans']].copy()
  req(h.shape==parent['positions'].shape and h.dtype.kind in 'iu' and np.all((h>=0)&(h<len(sp))) and np.all(sp>0),'Native hold map')
  holds[pid]=h;candidates[pid]=geometry_candidates(parent['edges'],h)
 for arm in ARMS:
  complete=meta[arm+'_completed'];selected=complete['selected']
  req(complete['arm']==arm and complete['seed']==101 and complete['outer_parents_accessed']==0
   and selected['pass_index']==inp['selected_passes'][arm],'Existing frozen H038 selection only')
  seals[arm]={c['parent_id']:c['prediction_seal'] for c in selected['cells']}
  req(set(seals[arm])==set(IDS),'All8seals')
  for pid in IDS:
   p=verify_seal_v1(seals[arm][pid],parents[pid],reader);h=holds[pid];delta=0.;hard=0
   for i in range(1,len(h),1024):
    ids=np.arange(i,min(i+1024,len(h)));ids=ids[h[ids]==h[ids-1]]
    if len(ids):
     delta=max(delta,float(np.max(np.abs(p[ids]-p[ids-1]))))
     hard+=int(np.count_nonzero(p[ids].argmax(-1)!=p[ids-1].argmax(-1)))
   req(hard==0,'Within-hold hard changes invalidate geometric bound')
   checks.append(dict(arm=arm,parent_id=pid,within_hold_hard_changes=hard,max_within_hold_probability_difference=delta))
   del p
   emit('raw_probability_hold_geometry_verified',arm=arm,parent_id=pid)
 req(not any('/truth/'in p for p in reader.reads),'No truth before all raw seals/geometry')
 save('all8_prediction_geometry_preflight.json',dict(checks=checks,all_seals_and_geometry_before_truth=True,
  read_refs=reader.reads,source_ref=ref(HERE/'boundary_diagnostic.py'),input_ref=ref(HERE/'inputs.json')))
 reader.truth_allowed=True;all_seals=[r for arm in ARMS for r in seals[arm].values()];cells={arm:[] for arm in ARMS}
 for pid in IDS:
  parent=parents[pid];binding=meta['target_bindings']['parents'][pid];ap._check_parent(parent,binding,reader)
  target=FullPanelExternalTargetV1(parent,binding,'VAL_after_sealed_predictions',reader,sealed_records=all_seals)
  columns=np.flatnonzero(parent['cohorts']=='present_day');req(len(columns)==182,'Present182')
  for arm in ARMS:
   p=verify_seal_v1(seals[arm][pid],parent,reader)
   g=helpers.full_span_score(p,parent,target,columns,cfg['metrics'])
   original=next(x for x in meta[arm+'_diagnostics'].get('cells',meta[arm+'_diagnostics'].get('per_parent',[])) if x['parent_id']==pid)['groups']['primary_present_day_182']
   req(g['boundaries']==original['boundaries'] and abs(g['accuracy']-original['accuracy'])<1e-12,'Exactly preserved original score')
   result=diagnose(g,candidates[pid],helpers);result.update(scored_native_hold_span_distribution=histogram(np.diff(np.r_[parent['edges'][0],candidates[pid],parent['edges'][-1]])),parent_id=pid,arm=arm,prediction_ref=seals[arm][pid],
    support_bp=[float(parent['edges'][0]),float(parent['edges'][-1])],target_ref=binding['truth_npz'],
    within_hold_verification=next(x for x in checks if x['arm']==arm and x['parent_id']==pid))
   save(arm+'_'+pid+'.json',result);cells[arm].append(result)
   emit('Present182_boundary_diagnosed',arm=arm,parent_id=pid,F1=g['boundaries']['f1'],
    representable_recall_upper_bound=result['representable_boundary_recall_upper_bound']['value'])
   del p,g
  del target
 report=dict(schema='H038_boundary_error_diagnostic_summary_v1',models={k:aggregate(v) for k,v in cells.items()},
  class_order=LABELS,SD_definition='ddof1 across8VALsimulations; one frozen fit per arm',
  boundary_definition='Original C4 label-change positions, class-agnostic greedy 1to1 tolerance1000bp; coverage-clipped support; no domain endpoints',
  selection_unchanged=True,postprocessing=False,new_predictions=False,outer_read=False,GPU_used=False,
  H039_boundary_metrics_inferred=False,scorer_ref=dict(path='scripts/train_ancestralpaths_biological_v1.py',sha256=inp['source_sha256']['scripts/train_ancestralpaths_biological_v1.py']),
  source_ref=ref(HERE/'boundary_diagnostic.py'),input_ref=ref(HERE/'inputs.json'),elapsed_seconds=time.monotonic()-start,
  caveats=['Oracle recall ceiling assumes arbitrary label changes at every retained grid hold edge, ignores transition class; not an attainable score or new predictor.',
   'Islands are predefined internal A→B→A tracts≤1kb/10kb; strict false means exact zero true support for B. Majority-wrong means below50% B support.',
   'Strict false-island boundary attribution is deduplicated within each query at each predefined length threshold; raw position precision/recall/F1 remain unchanged.',
   'Pooled histograms/counts differ from equal-parent mean±SD. Endpoint-censored truth tracts are also summarized separately.',
   'Existing1kb matches can join wrong biological class transitions; matched transition matrix exposes this without redefining F1.'])
 report_ref=save('summary.json',report)
 save('completed.json',dict(schema='H038_boundary_diagnostic_CPU_completed_v1',summary_ref=report_ref,
  elapsed_seconds=time.monotonic()-start,actual_diagnostic_completed=True,new_predictions=False,outer_arrays_read=False,
  original_boundary_and_accuracy_cells_exactly_reproduced=16,GPU_used=False,source_ref=ref(HERE/'boundary_diagnostic.py'),
  input_ref=ref(HERE/'inputs.json'),completed_utc=utc()))
 emit('completed',summary_ref=report_ref,elapsed_seconds=time.monotonic()-start)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  emit('failed',error=str(e),traceback=traceback.format_exc());raise
