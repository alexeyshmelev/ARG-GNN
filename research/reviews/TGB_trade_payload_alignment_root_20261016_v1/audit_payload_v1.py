"""Root RAM-only official TGB trade CSV alignment audit. No fitting or scoring."""
from pathlib import Path
from datetime import datetime,timezone
import urllib.request,io,zipfile,csv,hashlib,json,sys,traceback,collections,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/TGB_trade_payload_alignment_root_20261016_v1'
URL='https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-trade.zip'
SHA='697e9ca432c03aa6d4a998a75a1e49310bc78682dbe8cec2d8452b5a6604c9fe'
def main():
 import numpy as np
 started=time.monotonic()
 with urllib.request.urlopen(URL,timeout=45) as r:
  blob=r.read(13*2**20+1);headers=dict(r.headers);status=r.status
 assert len(blob)==12780237 and hashlib.sha256(blob).hexdigest()==SHA
 z=zipfile.ZipFile(io.BytesIO(blob));assert len(z.infolist())==2
 assert sum(x.file_size for x in z.infolist())==45725083
 names={Path(x.filename).name:x.filename for x in z.infolist()}
 assert set(names)=={'tgbn-trade_edgelist.csv','tgbn-trade_node_labels.csv'}
 node_ids={};tables={};counts={};metadata={}
 for kind,filename in [('edges','tgbn-trade_edgelist.csv'),('labels','tgbn-trade_node_labels.csv')]:
  matrices={};seen={};year_rows=collections.Counter();duplicates=0;first=[];nonfinite=0;negative=0;zero=0;ordered=True;prev=None;rawdigest=hashlib.sha256()
  # Exact official loader first-appearance node vocabulary, then label lookup.
  with z.open(names[filename]) as raw:
   source=io.TextIOWrapper(raw,encoding='utf-8',newline='');reader=csv.reader(source);header=next(reader)
   assert header==['year','nation','trading nation','weight'],header
   for row in reader:
    assert len(row)==4
    year=int(row[0]);u,v=row[1:3];weight=float(row[3])
    if kind=='edges':
     for country in (u,v):
      if country not in node_ids:node_ids[country]=len(node_ids)
    assert u in node_ids and v in node_ids
    uid,vid=node_ids[u],node_ids[v];assert max(uid,vid)<255
    if year not in matrices:matrices[year]=np.zeros((255,255),dtype=np.float32);seen[year]=np.zeros((255,255),dtype=np.uint8)
    duplicates+=int(seen[year][uid,vid]>0);seen[year][uid,vid]=1
    # Author loaders overwrite repeated label keys; audit duplicates explicitly.
    matrices[year][uid,vid]=weight;year_rows[year]+=1
    nonfinite+=int(not np.isfinite(weight));negative+=int(weight<0);zero+=int(weight==0)
    if prev is not None and year<prev:ordered=False
    prev=year
    if len(first)<3:first.append(row)
  assert len(node_ids)==255
  all_sums=np.concatenate([m.sum(axis=1,dtype=np.float64)[s.any(axis=1)] for y,m in sorted(matrices.items()) for s in [seen[y]]])
  metadata[kind]=dict(rows=sum(year_rows.values()),year_histogram=dict(sorted(year_rows.items())),distinct_years=len(year_rows),year_min=min(year_rows),year_max=max(year_rows),duplicate_year_source_destination_keys=duplicates,negative_weights=negative,nonfinite_weights=nonfinite,zero_weight_rows=zero,nondecreasing_years=ordered,first_three_rows=first,source_row_sum_min=float(all_sums.min()),source_row_sum_max=float(all_sums.max()),source_row_sum_max_abs_error_to_one=float(np.abs(all_sums-1).max()),queried_sources_by_year={y:int(s.any(axis=1).sum()) for y,s in sorted(seen.items())})
  tables[kind]=matrices;counts[kind]=seen
 mapping=json.dumps(node_ids,separators=(',',':'),ensure_ascii=False).encode()
 shifts={}
 for shift in (-1,0,1):
  by_year=[];cells=0;exact=0;maxerr=0.;l1=0.;compared_sources=0
  for year,labels in sorted(tables['labels'].items()):
   edge=tables['edges'].get(year+shift)
   if edge is None:continue
   sources=counts['labels'][year].any(axis=1);a=labels[sources];b=edge[sources];delta=np.abs(a-b)
   n=int(a.size);nexact=int((a==b).sum());mx=float(delta.max()) if n else 0
   cells+=n;exact+=nexact;maxerr=max(maxerr,mx);l1+=float(delta.sum(dtype=np.float64));compared_sources+=int(sources.sum())
   by_year.append(dict(label_year=year,edge_year=year+shift,sources=int(sources.sum()),max_abs_difference=mx,exact_cells=nexact,total_cells=n))
  shifts[str(shift)]=dict(compared_years=len(by_year),compared_source_years=compared_sources,exact_cells=exact,total_cells=cells,max_abs_difference=maxerr,mean_absolute_difference=l1/cells if cells else None,by_year=by_year)
 timestamps=np.concatenate([np.full(n,y,dtype=np.int32) for y,n in metadata['edges']['year_histogram'].items()])
 q70,q85=np.quantile(timestamps,[.7,.85]);ys=sorted(tables['labels'])
 out=dict(schema='TGB_trade_actual_payload_alignment_audit_v1',observed_utc=datetime.now(timezone.utc).isoformat(),root_authorizer='root_agent',archive=dict(url=URL,sha256=SHA,compressed_bytes=len(blob),uncompressed_bytes=sum(x.file_size for x in z.infolist()),HTTP_status=status,members=[dict(name=x.filename,size=x.file_size,crc32=x.CRC) for x in z.infolist()]),data=metadata,country_count=len(node_ids),country_mapping_sha256=hashlib.sha256(mapping).hexdigest(),country_mapping_order='Exact official edge-loader first appearance, source then destination',edge_label_shift_comparisons=shifts,official_quantile_cutoffs=dict(q70=float(q70),q85=float(q85)),target_label_year_splits=dict(TRAIN=[y for y in ys if y<=q70],VAL=[y for y in ys if q70<y<=q85],TEST=[y for y in ys if y>q85]),dataset_payload_persisted=False,model_constructed=False,fitting_performed=False,predictive_metric_computed=False,precision='FP32 matrix storage with FP64 audit reductions only, no model training',seconds=time.monotonic()-started)
 raw=(json.dumps(out,indent=2,sort_keys=True)+'\n').encode();assert len(raw)<64*1024
 dest=HERE/'actual_payload_alignment_v1.json'
 with dest.open('xb') as h:h.write(raw)
 print(json.dumps(dict(receipt=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),data=metadata,shift_summary={k:{kk:vv for kk,vv in v.items() if kk!='by_year'} for k,v in shifts.items()},splits=out['target_label_year_splits'],seconds=out['seconds'])),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc(file=sys.stdout);sys.exit(1)
