"""H052 fixed native trade baselines under a disclosed strict year scheduler."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,importlib.util,io,json,os,resource,sys,time,traceback,urllib.request,zipfile
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/TGB_trade_causal_baselines_root_20261016_v1'
PROTOCOL=HERE/'protocol.root_frozen_v1.json'
RENEWAL=dict(path='research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json',sha256='66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7')

def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 p=ROOT/r['path'];assert p.resolve()==p and ref(p)==r;return json.loads(p.read_bytes())
def guard():
 from arggnn import gpu45_c3cache_h050_pool_v1 as p
 p.install_marker_renewal(RENEWAL);return p.current_guard()
def load_author(r):
 p=ROOT/r['path'];assert ref(p)==r
 sp=importlib.util.spec_from_file_location('_H052_original_TGB_author_heuristics',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def predict_from_edges(author,edges):
 import numpy as np
 models=[author.PersistantForecaster(255),author.MovingAverage(255,window=7)]
 years=list(range(1987,2017));out=np.zeros((2,30,255,255),dtype=np.float64)
 # The public255country schema is fixed, but only complete past-year observed
 # edge vectors can enter each native class state. No label object is passed.
 for year in range(1986,2017):
  if year>=1987:
   for a,model in enumerate(models):
    out[a,year-1987]=np.stack([model.query_dict(n) for n in range(255)])
  matrix,present=edges[year]
  for model in models:
   for n in np.flatnonzero(present):model.update_dict(int(n),matrix[n].copy())
 return years,out

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--storage');pa.add_argument('--sha256');a=pa.parse_args()
 assert a.storage and a.sha256
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
 resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,32*2**20))
 started=time.monotonic();g=guard();storage=read(dict(path=a.storage,sha256=a.sha256))
 g.budget_check_v9(storage,first=True,boundary=True,required_write_bytes=32*2**20)
 protocol=json.loads(PROTOCOL.read_text());assert ref(PROTOCOL) in storage['reservation']['consumer_bindings']
 assert ref(Path(__file__).resolve()) in storage['reservation']['consumer_bindings']
 import numpy as np
 import sklearn
 from sklearn.metrics import ndcg_score
 assert sklearn.__version__==protocol['metric']['version']
 assert protocol['split']['TEST']==[2014,2015,2016]
 url='https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-trade.zip'
 with urllib.request.urlopen(url,timeout=45) as response:blob=response.read(13*2**20+1)
 assert len(blob)==protocol['archive_bytes'] and hashlib.sha256(blob).hexdigest()==protocol['archive_sha256']
 z=zipfile.ZipFile(io.BytesIO(blob));assert len(z.infolist())==2 and sum(x.file_size for x in z.infolist())==45725083
 names={Path(x.filename).name:x.filename for x in z.infolist()};assert set(names)=={'tgbn-trade_edgelist.csv','tgbn-trade_node_labels.csv'}
 node_ids={};edges={};year_rows={}
 with z.open(names['tgbn-trade_edgelist.csv']) as raw:
  reader=csv.reader(io.TextIOWrapper(raw,encoding='utf-8',newline=''));assert next(reader)==['year','nation','trading nation','weight']
  for row in reader:
   year=int(row[0]);u,v=row[1:3];w=float(row[3]);assert np.isfinite(w) and w>0
   for name in (u,v):
    if name not in node_ids:node_ids[name]=len(node_ids)
   u,v=node_ids[u],node_ids[v];assert max(u,v)<255
   if year not in edges:edges[year]=(np.zeros((255,255),dtype=np.float64),np.zeros(255,dtype=bool))
   matrix,present=edges[year];assert matrix[u,v]==0;matrix[u,v]=w;present[u]=True;year_rows[year]=year_rows.get(year,0)+1
 assert len(node_ids)==255 and sum(year_rows.values())==468245 and sorted(edges)==list(range(1986,2017))
 mapping_sha=hashlib.sha256(json.dumps(node_ids,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 audit=read(protocol['payload_audit_ref']);assert mapping_sha==audit['country_mapping_sha256']
 author=load_author(protocol['author_heuristics_ref']);years,predictions=predict_from_edges(author,edges)
 assert predictions.shape==(2,30,255,255) and np.isfinite(predictions).all()
 output=ROOT/storage['output_directory'];assert not output.exists();output.mkdir()
 pred_path=output/'predictions.npy'
 # Full-precision native heuristic predictions, sealed for ALL source countries
 # in ALL forecast years before opening any target-label member.
 with pred_path.open('xb') as h:np.save(h,predictions,allow_pickle=False);h.flush();os.fsync(h.fileno())
 seal=dict(schema='H052_all_native_forecasts_sealed_v1',created_utc=datetime.now(timezone.utc).isoformat(),prediction_ref=ref(pred_path),protocol_ref=ref(PROTOCOL),shape=list(predictions.shape),dtype=str(predictions.dtype),country_mapping_sha256=mapping_sha,label_file_opened_before_seal=False,all_255_source_predictions=True,forecast_years=years,source_ref=ref(Path(__file__).resolve()))
 with (output/'prediction_seal.json').open('x') as h:json.dump(seal,h,indent=2);h.write('\n')
 print('ALL_FORECASTS_SEALED '+json.dumps(seal),flush=True)
 labels={}
 with z.open(names['tgbn-trade_node_labels.csv']) as raw:
  reader=csv.reader(io.TextIOWrapper(raw,encoding='utf-8',newline=''));assert next(reader)==['year','nation','trading nation','weight']
  for row in reader:
   year=int(row[0]);u,v=node_ids[row[1]],node_ids[row[2]];w=float(row[3])
   if year not in labels:labels[year]=(np.zeros((255,255),dtype=np.float64),np.zeros(255,dtype=bool))
   matrix,present=labels[year];assert matrix[u,v]==0;matrix[u,v]=w;present[u]=True
 assert sorted(labels)==years
 results={}
 for arm,name in enumerate(['native_persistence','native_EMA_window7']):
  rows=[]
  for year in years:
   truth,present=labels[year];score=float(ndcg_score(truth[present],predictions[arm,year-1987,present],k=10,ignore_ties=False))
   rows.append(dict(target_year=year,queried_countries=int(present.sum()),ndcg_at10=score,latest_allowed_edge_year=year-1))
  splits={}
  for split,periods in protocol['split'].items():
   if split in ['q70','q85']:continue
   scores=np.asarray([r['ndcg_at10'] for r in rows if r['target_year'] in periods]);assert len(scores)==len(periods)
   splits[split]=dict(mean_ndcg_at10=float(scores.mean()),sample_SD_over_years=float(scores.std(ddof=1)),year_count=int(scores.size),years=periods,queried_country_years=sum(r['queried_countries'] for r in rows if r['target_year'] in periods))
  results[name]=dict(splits=splits,by_year=rows)
 g.budget_check_v9(storage,boundary=True,required_write_bytes=64*1024)
 result=dict(schema='H052_native_causal_trade_baseline_results_v1',completed_utc=datetime.now(timezone.utc).isoformat(),protocol_ref=ref(PROTOCOL),prediction_seal_ref=ref(output/'prediction_seal.json'),results=results,original_paper_reproduction=False,method_adaptations=protocol['initialization'],no_fitting=True,no_hyperparameter_selection=True,GPU_used=False,native_class_numeric_precision='Unchanged NumPyFP64 author baseline arithmetic, no neural/FP64 training',scikit_learn_version=sklearn.__version__,seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 with (output/'results.json').open('x') as h:json.dump(result,h,indent=2);h.write('\n')
 g.budget_check_v9(storage,boundary=True,required_write_bytes=0)
 print('H052_RESULTS '+json.dumps(result),flush=True)
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except SystemExit:raise
 except BaseException:traceback.print_exc(file=sys.stdout);sys.exit(1)
