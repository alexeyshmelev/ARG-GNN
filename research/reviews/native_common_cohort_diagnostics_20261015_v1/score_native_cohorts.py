"""CPU-only H038 amendment: diagnostics of original sealed native VAL products."""
from pathlib import Path
import os,sys,hashlib,json,io,importlib.util,time,datetime,csv,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/native_common_cohort_diagnostics_20261015_v1'
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
IDS=tuple('ap_expanded_val_%03d'%i for i in range(8))
CAP=10_000_000
CLASS=['Ana','CHG','WHG','EHG']
def req(ok,msg):
    if not ok: raise ValueError(msg)
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def jval(x):
    if isinstance(x,np.ndarray): return jval(x.tolist())
    if isinstance(x,np.generic): return jval(x.item())
    if isinstance(x,dict): return {str(k):jval(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jval(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def save(name,x):
    payload=(json.dumps(jval(x),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())
    req(Path(name).name==name and used+len(payload)<=CAP,'Output name/cap')
    with (HERE/name).open('xb') as f:f.write(payload)
    return {'path':str((HERE/name).relative_to(ROOT)),'sha256':digest(payload)}
def emit(stage,**x): print(json.dumps(jval(dict(stage=stage,utc=utc(),**x))),flush=True)
class Reader:
    def __init__(self):
        self.targets_allowed=False
        self.reads={}
    def read(self,r,**kwargs):
        req(isinstance(r,dict) and 'path'in r and 'sha256'in r,'Bound record required')
        lexical=Path(r['path']);req(not lexical.is_absolute() and '..'not in lexical.parts,'Canonical relative path')
        p=ROOT/lexical;req(p.is_file() and p.resolve()==p,'Canonical regular file')
        req('ap_expanded_outer_' not in str(p),'No outer reads')
        istruth='truth' in lexical.parts or 'oracle' in lexical.parts
        if istruth:req(self.targets_allowed and any(pid in lexical.parts for pid in IDS),'All seals before VAL truth; no TRAIN')
        if p.suffix=='.npz':req(any(pid in str(p) for pid in IDS),'Only eight VAL arrays')
        b=p.read_bytes();req(digest(b)==r['sha256'],'SHA differs: '+str(lexical))
        self.reads[str(lexical)]=r['sha256']
        return b
    def decode_json(self,r):return json.loads(self.read(r))
def mod(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stat(values):
    missing=sum(v is None for v in values)
    vals=np.array([v for v in values if v is not None],dtype=float)
    if missing:
        return dict(mean=None,sample_sd=None,undefined_parent_count=missing,
                    defined_parent_count=len(vals),n=8,ddof=1)
    return dict(mean=float(vals.mean()),sample_sd=float(vals.std(ddof=1)),n=8,ddof=1)
def summary(cells):
    rows=[]
    for name in cells[0]['groups']:
        groups=[c['groups'][name] for c in cells]
        row=dict(cohort=name,queries=groups[0]['query_count'],
                 independent_validation_simulations=8,optimizer_seeds=1)
        for k in ['accuracy','balanced_accuracy','macro_f1','nll','brier','calibration_ece',
                  'dosage_mse_per_diploid_bp_class','dosage_mae_per_diploid_bp_class']:
            row[k]=stat([g[k] for g in groups])
        for k in ['precision','recall','f1']:
            row['boundary_'+k]=stat([g['boundaries'][k] for g in groups])
        C=np.sum([g['confusion_truth_rows_prediction_columns_query_bp'] for g in groups],axis=0)
        norm=np.divide(C,C.sum(1)[:,None],out=np.full_like(C,np.nan),where=C.sum(1)[:,None]>0)
        row['pooled_confusion_query_bp']=C
        row['pooled_confusion_truth_row_normalized']=norm
        row['per_class_mean_SD']=[
            dict(class_0based=c,class_label=CLASS[c],
                 **{k:stat([g['per_class'][c][k] for g in groups])
                    for k in ['precision','recall','f1','span_iou']})
            for c in range(4)]
        reliability=[]
        for b in range(15):
            bins=[g['calibration_reliability_bins'][b] for g in groups]
            w=sum(x['query_bp'] for x in bins)
            reliability.append(dict(bin_index=b,query_bp=w,
              mean_confidence=sum(x['query_bp']*(x['mean_confidence'] or 0) for x in bins)/w if w else None,
              accuracy=sum(x['query_bp']*(x['accuracy'] or 0) for x in bins)/w if w else None))
        row['pooled_calibration_bins']=reliability
        row['pooled_boundary_counts']={k:sum(g['boundaries'][k] for g in groups)
            for k in ['true_boundaries','predicted_boundaries','matched','missed','extra']}
        rows.append(row)
    return dict(rows=rows)
def run():
    started=time.monotonic();inp=json.loads((HERE/'inputs.json').read_text());reader=Reader()
    req(os.environ['CUDA_VISIBLE_DEVICES']=='','CPU only')
    # Live experiment ledger may change, so bind the registered amendment object.
    decisions=json.loads((ROOT/'research/experiment_decisions.json').read_text())
    matches=[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('phase')=='unanswered_common_cohort_diagnostics_20261015':matches.append(x)
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(decisions)
    req(len(matches)==1 and matches[0]==inp['amendment'] and matches[0]['CPU_only']
        and not matches[0]['new_model_predictions'] and not matches[0]['outer_truth_authorized'],
        'Prospectively registered CPU amendment required')
    for p,h in inp['source_sha256'].items():reader.read(dict(path=p,sha256=h))
    meta={k:reader.decode_json(v) for k,v in inp['metadata_refs'].items()}
    # Author source inventory, pure read only. No fit/predict/guard/resource call.
    from arggnn.official_argmix_biological import source_inventory
    source_inventory(require_copy=True)
    ap=mod(inp['AP_helper'],'cohort_shared_AP')
    am=mod(inp['ARGMix_helper'],'cohort_shared_ARGMix')
    helpers=mod('scripts/train_ancestralpaths_biological_v1.py','cohort_original_span')
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1
    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelExternalTargetV1
    from arggnn.mutation_routed_fast_primary_v1 import mass_cache_v1,score_primary_v1
    cfg=meta['H038_config'];aligned=meta['aligned_input'];gen=meta['generation_config']
    req(gen['parent_registry']==meta['registration']['parent_registry'],'Original registry')
    req([r['parent_id'] for r in gen['parent_registry'] if r['split']=='val']==list(IDS),'EightVAL')
    index=read_full_cache_index_v1(inp['metadata_refs']['observation_cache'],reader,
        registry=gen['parent_registry'],entry_refs=aligned['observation_entry_refs'],
        source_refs=aligned['observation_cache_source_refs'])
    parents={pid:ap.read_h038_scoring_only_v1(index,pid,reader) for pid in IDS}
    bindings={pid:meta['target_bindings']['parents'][pid] for pid in IDS}
    ac,af,aconfig=meta['AP_completed'],meta['AP_fit'],meta['AP_config']
    mc,mf=meta['ARGMix_completed'],meta['ARGMix_fit']
    req(ac['schema']=='expanded_native_ap_driver_completed_v2' and ac['TRAIN_count']==32 and ac['VAL_count']==8
        and ac['epochs']==5 and ac['checkpoint_selection']=='final_epoch5'
        and not ac['VAL_selection'] and not ac['VAL_targets_read'] and not ac['outer_included'],'APcompletion')
    req(af['source_ref']==aconfig['source_ref'] and af['config_digest']==ap._digest(aconfig)
        and af['checkpoint_selection']=='final_epoch5' and af['epochs']==5 and not af['VAL_selection']
        and af['training_rows']==ac['realized_training_rows'],'APfit')
    reader.read(aconfig['source_ref'])
    for r in aconfig.get('helper_refs',{}).values():reader.read(r)
    for p,h in mf['owned_code_sha256'].items():reader.read(dict(path=p,sha256=h))
    req(mc['schema']=='expanded_native_argmix_GPU_fit_root_v1_completed'
        and mc['status']=='completed_native_fit_all8_VAL' and mc['TRAIN_parents']==32
        and mc['VAL_parents']==8 and mc['outer_parents']==0 and mc['source_seed']==50
        and not mc['original_paper_reproduction'] and not mc['truth_opened_for_external_score']
        and mf['kind']=='expanded_native_argmix_stock_fit_v1','ARGMixfit')
    ms={s['parent_id']:s for s in mc['VAL_seals']};ams={}
    req(set(ms)==set(IDS)==set(ac['VAL_prediction_seals']),'All 8 seals')
    # Phase 1 verifies every AP C4/native6 and ARGMix raw grid before any target.
    for pid in IDS:
        keys=ap._check_parent(parents[pid],bindings[pid],reader)
        s=reader.decode_json(ac['VAL_prediction_seals'][pid])
        req(s['schema']=='expanded_native_ap_prediction_seal_v1' and s['parent_id']==pid
            and s['split']=='val' and s['source_ref']==aconfig['source_ref']
            and s['config_digest']==af['config_digest'] and s['fit_ref']==ac['fit_ref']
            and s['checkpoint_ref']==af['checkpoint_ref'] and s['query_keys']==keys
            and s['query_haplotypes']==686 and s['BAA_queries']==14
            and s['C4_class_order']==CLASS and s['native_path_axis_1based']==list(range(1,7))
            and s['C4_collapse_before_argmax']=='p1+p5,p2+p6,p3,p4'
            and s['native_six_argmax_unmasked'] and not s['targets_read'] and not s['VAL_selection']
            and not s['outer_included'] and not s['calibration_fitted'],'APmanifest')
        p,ids=ap._common(reader,s,parents[pid]);f=ap._focal(reader,s,parents[pid],ids)
        del p,ids,f;ams[pid]=s
        s=ms[pid]
        req(s['grid_prediction']['path']==f'experiments/results/expanded_native_argmix_full32TRAIN8VAL_fit_v1/VAL/{pid}/sealed_C4_grid_probabilities.npz'
            and s['checkpoint']==mf['checkpoint'] and s['frame']['config_sha256']==mf['config_sha256']
            and s['full_support_complete'] and s['missing_grid_query_cells']==0
            and s['native_final_estimand']==am.ESTIMAND and not s['truth_opened_for_external_score']
            and not s['inference_only_edge_option'] and 0<=s['native_replay_max_probability_error']<=1e-6,
            'ARGMixseal')
        p=am.grid_v1(s,parents[pid],reader);del p
        emit('both_baselines_raw_seals_verified',parent_id=pid)
    before=dict(reader.reads)
    req(not any('/truth/' in p for p in before),'No truth before seals')
    save('all8_raw_seals_preflight.json',dict(all8_verified_before_truth=True,
      native_AP_C4_and_native6=True,native_ARGMix=True,
      source_and_metadata_refs_verified=before,amendment=matches[0],
      author_ARGMix_inventory_verified=True,GPU_used=False))
    reader.targets_allowed=True
    cells={'native_AP':[],'native_ARGMix':[]}
    records=[inp['metadata_refs']['ARGMix_completed']]+list(ac['VAL_prediction_seals'].values())
    for pid in IDS:
        parent=parents[pid]
        target=FullPanelExternalTargetV1(parent,bindings[pid],'VAL_after_sealed_predictions',reader,
            sealed_records=records)
        cohorts=parent['cohorts']
        strata={'primary_present_day_182':np.flatnonzero(cohorts=='present_day'),
            'secondary_ancient_504':np.flatnonzero(cohorts!='present_day'),'all_queries_686':np.arange(686)}
        strata.update({'age_stratum_'+c:np.flatnonzero(cohorts==c) for c in ['Bronze_Age','BAA','Neo','Yam']})
        for model in cells:
            p=ap._common(reader,ams[pid],parent)[0] if model=='native_AP' else am.grid_v1(ms[pid],parent,reader)
            groups={}
            for name,col in strata.items():
                g=helpers.full_span_score(p,parent,target,col,cfg['metrics'])
                per=g.pop('per_query');g.pop('retained_original_grid_indexes')
                short=[t['correct_overlap_fraction'] for r in per for t in r['short_true_tracts']]
                g['short_tract_summary']=dict(count=len(short),mean_correct_overlap_fraction=float(np.mean(short)) if short else None,
                    cutoff_bp=cfg['metrics']['short_tract_cutoff_bp'])
                groups[name]=g
            # Verify the prior FP32 primary result with the unchanged H038 scorer.
            cache=mass_cache_v1(parent,target.target_mass(parent,helpers),
                target_provenance=target.provenance,expected_present_queries=182)
            primary=score_primary_v1(p,cache,backend='numpy')
            cell=jval(dict(parent_id=pid,split='val',groups=groups,
                original_FP32_primary=primary,target_ref=bindings[pid]['truth_npz']))
            cells[model].append(cell)
            save(model+'_'+pid+'.json',cell)
            emit('parent_diagnostics_scored',model=model,parent_id=pid,
                accuracy=groups['primary_present_day_182']['accuracy'],
                boundary_f1=groups['primary_present_day_182']['boundaries']['f1'])
            del p,cache
        del target
    report=dict(schema='native_common_cohort_diagnostics_v1',generated_utc=utc(),
        models={k:summary(v) for k,v in cells.items()},class_order=CLASS,
        scope='eight existing internal VAL simulations, original frozen model selection and raw probabilities',
        metric_arithmetic='unchanged full_span_score FP64 physical integration; original_FP32_primary separately',
        SD_definition='sample SD, ddof1, across8independentVALsimulations. One existing fit per method; not optimizerseedSD.',
        cohort_aliases_to_paper_not_assumed=True,calibration_fitted=False,probability_floor=False,
        posthoc_renormalization=False,new_predictions=False,retraining=False,outer_arrays_read=False,
        GPU_used=False,elapsed_seconds=time.monotonic()-started,
        input_ref=dict(path=str((HERE/'inputs.json').relative_to(ROOT)),sha256=digest((HERE/'inputs.json').read_bytes())),
        new_source_ref=dict(path=str((HERE/'score_native_cohorts.py').relative_to(ROOT)),sha256=digest((HERE/'score_native_cohorts.py').read_bytes())),
        all8_seals_verified_before_truth=True)
    ref=save('native_cohort_metrics_mean_SD.json',report)
    save('per_parent_metrics.json',cells)
    out=io.StringIO();fields=['model','cohort','queries','accuracy_percent_mean','accuracy_percent_sampleSD',
          'boundary_f1_percent_mean','boundary_f1_percent_sampleSD','nll_mean','brier_mean','ECE_mean']
    w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
    for model,rep in report['models'].items():
        for r in rep['rows']:
            w.writerow(dict(model=model,cohort=r['cohort'],queries=r['queries'],
                accuracy_percent_mean=100*r['accuracy']['mean'],
                accuracy_percent_sampleSD=100*r['accuracy']['sample_sd'],
                boundary_f1_percent_mean=None if r['boundary_f1']['mean'] is None else 100*r['boundary_f1']['mean'],
                boundary_f1_percent_sampleSD=None if r['boundary_f1']['sample_sd'] is None else 100*r['boundary_f1']['sample_sd'],
                nll_mean=r['nll']['mean'],brier_mean=r['brier']['mean'],ECE_mean=r['calibration_ece']['mean']))
    payload=out.getvalue().encode()
    req(sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())+len(payload)<CAP,'CSV cap')
    (HERE/'native_cohort_metrics.csv').write_bytes(payload)
    save('completed.json',dict(schema='native_common_cohort_CPU_completed_v1',summary_ref=ref,
      actual_computation_completed=True,all8_raw_seals_before_truth=True,GPUs_used=0,
      retraining=False,new_predictions=False,outer_arrays_read=False,
      baseline_models=2,VAL_parents=8,cohorts=7,elapsed_seconds=time.monotonic()-started,
      source_sha256=inp['source_sha256'],output_bytes=sum(p.stat().st_size for p in HERE.iterdir() if p.is_file()),
      completed_utc=utc()))
    emit('all_native_cohort_diagnostics_completed',elapsed_seconds=time.monotonic()-started,summary_ref=ref)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        emit('failed',error=str(e),traceback=traceback.format_exc())
        raise
