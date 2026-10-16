"""H056 bounded H36 extraction source. Inert; root injects all operational IO.
Minimal successor of H018 prospective_worker_v3 Range/BGZF and QC; no controller.
"""
from collections import OrderedDict,Counter
from urllib.request import Request
import datetime,gzip,hashlib,json,math,re,struct,time,zlib
from research.reviews.H056_H36_implementation_20261016_v1.contract_v1 import (H,R,Q,PEOPLE,BEGIN,END,CLASSES,SUBJECTS,PROTOCOL_REF,require,sha,encoded,validate_chromosome_v1,validate_counts_v1,validate_roster_identity_v1)
BEGIN,END=20_000_000,30_000_000
ALLOWED_FORMAT=("GT","GP","HS","DS")
ALLOWED_INFO=("INFO","RAF","MASK_1000G","pan_troglodytes")
NET_CAP=500_000_000;OUTPUT_CAP=200_000_000;WINDOW=1_000_000
MAX_HTTP_RANGE_BYTES=8*WINDOW;MAX_COMPRESSED_CACHE_BYTES=32*WINDOW
MAX_LINE=16*1024*1024;MAX_DECOMPRESSED=40_000_000_000;MAX_REQUESTS=640
ACQUISITION_SECONDS=1800
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def validate_asset_v1(asset):
 require(isinstance(asset,dict) and set(asset)=={"chromosome","url","bytes","index_url","index_bytes","window_bp0","vcf_ETag","index_ETag","index_sha256","prefix_sha256"},"Exact frozen asset plus root-qualified object identities")
 c=validate_chromosome_v1(asset["chromosome"])
 require(asset["chromosome"] in ("20","21","22"),"Fixed H056 chromosome")
 expected={"20":(2529400869,45986),"21":(1626647184,27903),"22":(1654398745,28008)}[asset["chromosome"]]
 url="https://erda.ku.dk/archives/917f1ac64148c3800ab7baa29402d088/neo/genotypes-public/"+asset["chromosome"]+".neo_impute_ph.vcf.gz"
 require((asset["bytes"],asset["index_bytes"])==expected and asset["url"]==url and asset["index_url"]==url+".tbi" and asset["window_bp0"]==[BEGIN,END],"Exact protocol URL/length/window")
 for key in ("vcf_ETag","index_ETag"):
  v=asset[key];require(isinstance(v,str) and len(v)>=2 and v.startswith('"') and v.endswith('"') and not v.startswith("W/"),"Root-qualified strong object ETag")
 for key in ("index_sha256","prefix_sha256"):
  require(isinstance(asset[key],str) and re.fullmatch("[0-9a-f]{64}",asset[key]),"Root-qualified exact index/prefix digest")
 if asset["chromosome"]=="22":
  require(asset["vcf_ETag"]=='"629c1f19-605efa5ff6274"' and asset["index_sha256"]=="7c35737a4f03c40b1ce149c837f87c92495c88b1948c79da5246be72ed085b40" and asset["prefix_sha256"]=="4564d66dc800ef5573ab043decfc89596868391a933d0aaadd9a9ac16c21ee2d","Preserve preexisting chr22 source identity")
 return asset

def frozen_header(lines,subjects,chromosome):
 validate_chromosome_v1(chromosome);require(tuple(subjects)==SUBJECTS,"Exact frozen H056 subject order")
 meta=[];sample_names=None;nmain=None;source_b37=False;source_contig=False
 reference="##reference=ftp://ftp.1000genomes.ebi.ac.uk//vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz"
 for _,line in lines:
  text=line.decode()
  if text.startswith("##NMAIN="):
   require(nmain is None,"Unique NMAIN declaration");nmain=int(text.split("=",1)[1])
  if text==reference:source_b37=True;meta.append(text)
  if text.startswith("##contig=") and "assembly=b37" in text:source_b37=True
  if text.startswith("##fileformat=") or text.startswith("##FILTER="):meta.append(text)
  if text.startswith("##contig=<ID="+chromosome+",") or text=="##contig=<ID="+chromosome+">":
   require("assembly=" not in text or "assembly=b37" in text,"No contradictory target assembly")
   meta.append(text);source_contig=True
  match=re.match(r"##(FORMAT|INFO)=<ID=([^,>]+)",text)
  if match and match.group(2) in (ALLOWED_FORMAT if match.group(1)=="FORMAT" else ALLOWED_INFO):meta.append(text)
  if text.startswith("#CHROM\t"):
   fields=text.split("\t");sample_names=fields[9:];break
 require(sample_names is not None and len(sample_names)==3331 and len(set(sample_names))==3331 and nmain==15,"Frozen3331-column/NMAIN15 source")
 require(all(s in sample_names and not s.endswith("_rnd") for s in subjects),"Exact ordinary IDs")
 ordinary=[s for s in sample_names if not s.endswith("_rnd")]
 require([ordinary.index(s) for s in subjects]==[336,354,341,342,361,340,292,1364,1595,1632,1331,1630,1567,1555,0,1552,1366,1596],"Frozen ordinary subject/index joins")
 require(source_b37,"Original source b37 assembly evidence")
 if not source_contig:meta.append("##contig=<ID="+chromosome+",assembly=b37>")
 defs={m.group(1) for x in meta if (m:=re.match(r"##FORMAT=<ID=([^,>]+)",x))}
 require({"GT","GP","HS"}<=defs,"Original GT/GP/HS definitions")
 meta+=["##H056_region_bp0="+chromosome+":"+str(BEGIN)+"-"+str(END),
  "##H056_fields_policy=AP_removed;GT_GP_HS_DS_only;original_H018_QC_all18",
  "##H056_coordinate=VCF_POS_bp1_preserved;regional_bp0=POS-1-20000000","##NMAIN=15"]
 raw=("\n".join(meta)+"\n"+"\t".join(["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]+list(subjects))+"\n").encode()
 return raw,[sample_names.index(s) for s in subjects],dict(chromosome=chromosome,source_sample_columns=3331,selected_subjects=list(subjects),NMAIN=15,source_target_contig_present=source_contig,derived_target_header_without_length=not source_contig,original_GT_header_description="Source calls GT unphased; exact selected-record phase is checked by QC")

def numeric(value):
 # Scalar numeric VCF text only: malformed bare INFO/RAF flags are not1.0.
 if not isinstance(value,str) or value in ("",".") or value.strip()!=value:return None
 try:
  f=float(value);return f if math.isfinite(f) else None
 except (ValueError,TypeError):return None

class RangeSource:
 def __init__(self,asset,*,opener,before_request):
  validate_asset_v1(asset);require(callable(opener) and callable(before_request),"Root admitted HTTP opener and resource boundary")
  self.asset=dict(asset)
  self.before_request=before_request;self.started=time.monotonic()
  self.opener=opener;self.received=0;self.requests=[];self.cache=OrderedDict();self.decompressed=0
  self.planned_windows=None;self.timings=Counter();self.stats=Counter()
 def bind_plan_v1(self,keys):
  require(self.planned_windows is None and type(keys)is list and keys==sorted(set(keys))
   and keys and keys[0]==0 and all(type(k)is int and 0<=k<(self.asset["bytes"]+WINDOW-1)//WINDOW for k in keys),
   'Exact bounded compressed-window plan')
  require(sum(min(WINDOW,self.asset["bytes"]-k*WINDOW) for k in keys)+self.asset["index_bytes"]<NET_CAP,
   'Planned network bytes remain within original cap')
  self.planned_windows=frozenset(keys)
 def fetch(self,url,start,end,total,etag=None):
  require(time.monotonic()-self.started<=ACQUISITION_SECONDS,'Regional acquisition runtime cap')
  check_started=time.monotonic()
  if self.before_request is not None:self.before_request()
  self.timings['before_request_callback_seconds']+=time.monotonic()-check_started
  remaining=ACQUISITION_SECONDS-(time.monotonic()-self.started)
  require(remaining>0,'Regional acquisition runtime cap after boundary check')
  require(url in [self.asset["url"],self.asset["index_url"]] and 0<=start<=end<total and end-start+1<=MAX_HTTP_RANGE_BYTES,'Exact bounded source Range')
  n=end-start+1
  require(self.received+n+1<=NET_CAP and len(self.requests)<MAX_REQUESTS,'Network byte/request cap')
  receipt=dict(url=url,start_byte=start,end_byte=end,total_source_bytes=total,requested_utc=utc(),bytes=0)
  self.requests.append(receipt)
  headers={'User-Agent':'ARG-GNN-H018-R16-20-30Mb-extension/1','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
  if etag:headers['If-Match']=etag
  http_started=time.monotonic();opened=False
  try:
   with self.opener(Request(url,headers=headers),timeout=min(30,remaining)) as r:
    opened=True;open_seconds=time.monotonic()-http_started
    self.timings['http_open_seconds']+=open_seconds;receipt['http_open_seconds']=open_seconds
    receipt.update(status=r.status,ETag=r.headers.get('ETag'),Content_Range=r.headers.get('Content-Range'))
    require(r.status==206,'Refuse source that ignores Range (no full-file fallback)')
    require(r.url==url and r.headers.get('Content-Range')==f'bytes {start}-{end}/{total}','Source URL/range/total differs')
    require(r.headers.get('Content-Length')==str(n),'Exact bounded Content-Length')
    require(r.headers.get('Content-Encoding') in [None,'identity'],'No transfer content encoding')
    if etag:require(r.headers.get('ETag')==etag,'Source ETag changed')
    read_started=time.monotonic()
    try:b=r.read(n+1)
    finally:
     body_seconds=time.monotonic()-read_started
     self.timings['http_body_read_seconds']+=body_seconds;receipt['http_body_read_seconds']=body_seconds
    self.received+=len(b);receipt.update(bytes=len(b),sha256=sha(b),Last_Modified=r.headers.get('Last-Modified'))
    require(len(b)==n and self.received<=NET_CAP,'Range body length/cap differs')
   return b
  finally:
   seconds=time.monotonic()-http_started
   receipt.update(http_total_seconds=seconds,finished_utc=utc())
   self.timings['http_total_seconds']+=seconds
   if not opened:self.timings['http_open_failed_seconds']+=seconds
 def bytes(self,start,n):
  require(0<=start<=start+n<=self.asset["bytes"] and n<=65536,'Bounded BGZF read')
  parts=[];pos=start
  while pos<start+n:
   key=pos//WINDOW
   require(self.planned_windows is not None and key in self.planned_windows,'BGZF read outside frozen TBI window plan')
   if key not in self.cache:
    self.stats['compressed_cache_misses']+=1;keys=[key]
    # Prefix identity remains exactly the frozen original 1 MB window.
    while key!=0 and len(keys)<MAX_HTTP_RANGE_BYTES//WINDOW:
     following=keys[-1]+1
     if following not in self.planned_windows or following in self.cache:break
     keys.append(following)
    a=key*WINDOW;b=min((keys[-1]+1)*WINDOW,self.asset["bytes"])-1
    chunk=self.fetch(self.asset["url"],a,b,self.asset["bytes"],self.asset["vcf_ETag"])
    for cached_key in keys:
     offset=(cached_key-key)*WINDOW;part=chunk[offset:offset+WINDOW]
     if cached_key==0:require(sha(part)==self.asset["prefix_sha256"],'Frozen received first1MB identity differs')
     while len(self.cache)>=MAX_COMPRESSED_CACHE_BYTES//WINDOW:self.cache.popitem(last=False)
     self.cache[cached_key]=part
    self.stats['prefetched_windows']+=len(keys)
    self.stats['compressed_cache_peak_windows']=max(self.stats['compressed_cache_peak_windows'],len(self.cache))
   else:self.stats['compressed_cache_hits']+=1
   self.cache.move_to_end(key);part=self.cache[key];offset=pos-key*WINDOW
   take=min(start+n-pos,len(part)-offset);require(take>0,'Empty byte range')
   parts.append(part[offset:offset+take]);pos+=take
  return b''.join(parts)
 def block(self,pos):
  require(time.monotonic()-self.started<=ACQUISITION_SECONDS,'Regional decode runtime cap')
  h=self.bytes(pos,12);require(h[:4]==b'\x1f\x8b\x08\x04','BGZF magic/flags')
  xlen=struct.unpack_from('<H',h,10)[0];require(6<=xlen<=1024,'BGZF extra length')
  extra=self.bytes(pos+12,xlen);off=0;size=None
  while off+4<=len(extra):
   tag=extra[off:off+2];n=struct.unpack_from('<H',extra,off+2)[0];require(off+4+n<=len(extra),'BGZF subfield')
   if tag==b'BC':require(n==2 and size is None,'Unique BGZF BC');size=struct.unpack_from('<H',extra,off+4)[0]+1
   off+=4+n
  require(size is not None and 26<=size<=65536,'BGZF block size')
  raw=self.bytes(pos,size);decode_started=time.monotonic()
  try:data=zlib.decompress(raw,31)
  finally:self.timings['zlib_decompress_seconds']+=time.monotonic()-decode_started
  require(len(data)<=65536,'BGZF expanded block')
  self.stats['BGZF_blocks_decoded']+=1
  self.decompressed+=len(data);require(self.decompressed<=MAX_DECOMPRESSED,'Expanded-source cap')
  return data,pos+size
 def diagnostics_v1(self):
  return dict(wall_elapsed_seconds=time.monotonic()-self.started,received_bytes=self.received,
   request_attempts=len(self.requests),decompressed_bytes=self.decompressed,
   max_http_range_bytes=MAX_HTTP_RANGE_BYTES,max_compressed_cache_bytes=MAX_COMPRESSED_CACHE_BYTES,
   live_compressed_cache_windows=len(self.cache),planned_windows=None if self.planned_windows is None else len(self.planned_windows),
   timings_seconds=dict(self.timings),counts=dict(self.stats),raw_source_bytes_retained_on_disk=False)
 def lines(self,begin_vo,end_vo=None):
  pos=begin_vo>>16;first=begin_vo&65535;pending=bytearray();line_vo=begin_vo
  while True:
   data,nextpos=self.block(pos)
   if not data:require(not pending,'Truncated VCF line');return
   require(first<=len(data),'BGZF virtual offset')
   cursor=first
   while cursor<len(data):
    if not pending:
     line_vo=(pos<<16)|cursor
     if end_vo is not None and line_vo>=end_vo:return
    stop=data.find(b'\n',cursor)
    if stop<0:
     pending.extend(data[cursor:]);require(len(pending)<=MAX_LINE,'VCF record byte cap');break
    pending.extend(data[cursor:stop]);require(len(pending)<=MAX_LINE,'VCF record byte cap')
    yield line_vo,bytes(pending);pending.clear();cursor=stop+1
   pos=nextpos;first=0

def tbi_chunks(raw,asset):
 validate_asset_v1(asset);chrom=asset['chromosome'];begin,end=BEGIN,END
 require(sha(raw)==asset["index_sha256"],'Exact frozen public TBI hash')
 data=gzip.decompress(raw);require(len(data)<2_000_000 and data[:4]==b'TBI\x01','TBI schema/cap')
 n,fmt,seq,beg_col,end_col,meta,skip,ln=struct.unpack_from('<8i',data,4);off=36
 require((fmt,seq,beg_col,end_col,meta,skip)==(2,1,2,0,35,0),'VCF tabix coordinate schema')
 names=data[off:off+ln].rstrip(b'\0').decode().split('\0');off+=ln;found=None
 for i in range(n):
  nb,=struct.unpack_from('<i',data,off);off+=4;bins={}
  require(0<=nb<100000,'TBI bin count')
  for _ in range(nb):
   b,nc=struct.unpack_from('<Ii',data,off);off+=8;require(0<=nc<100000,'TBI chunk count')
   chunks=[]
   for _ in range(nc):
    chunks.append(struct.unpack_from('<QQ',data,off));off+=16
   bins[b]=chunks
  nl,=struct.unpack_from('<i',data,off);off+=4;require(0<=nl<1000000,'TBI linear count')
  linear=struct.unpack_from('<'+'Q'*nl,data,off);off+=8*nl
  if names[i]==chrom:found=(bins,linear)
 require(found is not None and names==[chrom],'Exact declared public chromosome TBI')
 bins,linear=found;wanted={0}
 for first,shift in [(1,26),(9,23),(73,20),(585,17),(4681,14)]:
  wanted.update(range(first+(begin>>shift),first+((end-1)>>shift)+1))
 minimum=linear[begin>>14] if begin>>14<len(linear) else 0
 chunks=sorted((max(a,minimum),b) for k in wanted for a,b in bins.get(k,[]) if b>minimum and a<b)
 merged=[]
 for a,b in chunks:
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((a,b))
 require(merged,'No region tabix chunks')
 windows={0}
 for a,b in merged:
  windows.update(range((a>>16)//WINDOW,min(asset["bytes"]-1,(b>>16)+65535)//WINDOW+1))
 estimate=sum(min(WINDOW,asset["bytes"]-key*WINDOW) for key in windows)+asset["index_bytes"]
 require(estimate<NET_CAP,'Region compressed plan exceeds byte bound')
 return merged,dict(index_raw_bytes=len(data),estimated_network_upper_bytes=estimate,
  planned_windows=sorted(windows),virtual_chunks=[[int(a),int(b)] for a,b in merged],index_sha256=asset["index_sha256"])

def clean_record(raw,indexes,chromosome):
 validate_chromosome_v1(chromosome);require(len(indexes)==PEOPLE and len(set(indexes))==PEOPLE,"Exact H056 ordinary columns")
 f=raw.decode().split('\t');require(len(f)==3340,'Source VCF column count')
 chrom,pos,vid,r,a,qual,flt,info_text,fmt=f[:9];pos=int(pos)
 if chrom!=chromosome or not BEGIN<=pos-1<END:return None
 tags=fmt.split(':');require(len(tags)==len(set(tags)),'Unique source FORMAT names')
 info={}
 for item in info_text.split(';'):
  if not item or item=='.':continue
  k,sep,v=item.partition('=');require(k not in info,'Unique source INFO names');info[k]=v if sep else True
 values=[];subject_qc=[]
 for index in indexes:
  parts=f[9+index].split(':');require(len(parts)<=len(tags),'VCF FORMAT cardinality')
  parts+=['.']*(len(tags)-len(parts));g=dict(zip(tags,parts))
  values.append(':'.join(g.get(t,'.') for t in ALLOWED_FORMAT))
  gt=g.get('GT','.');phased=gt in ['0|0','0|1','1|0','1|1']
  gp=[numeric(x) for x in g.get('GP','.').split(',')]
  valid_gp=len(gp)==3 and all(x is not None and 0<=x<=1 for x in gp) and abs(sum(gp)-1)<=.002
  max_gp=max(gp) if valid_gp else None
  try:h=int(g.get('HS','.'));valid_hs=0<=h<2**30
  except ValueError:valid_hs=False
  subject_qc.append(dict(phased_diploid=phased,GP_valid=valid_gp,GP_max=max_gp,HS_valid30=valid_hs))
 keepinfo=';'.join(k if info[k] is True else k+'='+info[k] for k in ALLOWED_INFO if k in info) or '.'
 # No AP value, source history or excluded sample is ever emitted.
 output='\t'.join([chrom,str(pos),vid,r,a,qual,flt,keepinfo,':'.join(ALLOWED_FORMAT)]+values)+'\n'
 require('AP' not in output.split('\t')[8].split(':'),'AP must not be retained')
 bases=set('ACGT');biallelic=len(r)==len(a)==1 and r in bases and a in bases and r!=a
 transversion=biallelic and (r,a)not in [('A','G'),('G','A'),('C','T'),('T','C')]
 score=numeric(info.get('INFO'));raf=numeric(info.get('RAF'));maf=min(raf,1-raf) if raf is not None and 0<=raf<=1 else None
 ancestral=info.get('pan_troglodytes');polarized=isinstance(ancestral,str) and ancestral in bases and ancestral in [r,a]
 qc=dict(biallelic_ACGT=biallelic,transversion=transversion,strict_mask=info.get('MASK_1000G') is True,
  INFO_ge_0_8=score is not None and score>=.8,external_RAF_MAF_ge_0_05=maf is not None and maf>=.05,
  all18_phased_diploid=all(x['phased_diploid'] for x in subject_qc),
  all18_GP_valid=all(x['GP_valid'] for x in subject_qc),
  all18_GP_max_ge_0_8=all(x['GP_max'] is not None and x['GP_max']>=.8 for x in subject_qc),
  all18_HS_valid30=all(x['HS_valid30'] for x in subject_qc),ancestral_in_REF_ALT=polarized)
 passed=all(qc.values())
 row=dict(chrom=chrom,POS_bp1=pos,absolute_bp0=pos-1,region_bp0=pos-1-BEGIN,REF=r,ALT=a,
  FILTER=flt,INFO=score,RAF_external=raf,MAF_external=maf,pan_troglodytes=ancestral if isinstance(ancestral,str) else None,
  **qc,H018_fixed_QC_pass=passed)
 return output.encode(),row,subject_qc

class Buffered:
 def __init__(self,out,name):self.out=out;self.name=name;self.data=bytearray()
 def write(self,b):
  self.data.extend(b)
  if len(self.data)>=1_000_000:self.flush()
 def flush(self):
  if self.data:self.out.append(self.name,bytes(self.data));self.data.clear()

def execute_v1(ctx):
 """Root-injected acquisition core; no files, native tools, guard or admission implementation.
 Root supplies admitted opener, before_request(), writer.append(name,bytes) and
 output_cap_bytes. It owns compression/indexing, independent count/readback
 seals, genuine waits and actual completion metadata. This return is not a
 native/model admission or a durable completion.
 """
 require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True and ctx.get("protocol_ref")==PROTOCOL_REF,"Actual root H056 acquisition context")
 validate_roster_identity_v1(ctx["roster"]);asset=validate_asset_v1(ctx["asset"]);chrom=asset["chromosome"]
 require(callable(ctx.get("opener")) and callable(ctx.get("before_request")) and callable(getattr(ctx.get("writer"),"append",None)),"Root-owned network/resource/write interfaces")
 cap=ctx["output_cap_bytes"];require(type(cap)is int and 0<cap<=OUTPUT_CAP,"Root count-derived output cap within inherited extraction ceiling")
 used=0
 class Sink:
  def append(self,name,data):
   nonlocal used
   require(name in {"H056.chr"+chrom+".noAP.vcf","regional_site_QC.tsv"} and isinstance(data,bytes) and used+len(data)<=cap,"AP-free bounded root append")
   ctx["writer"].append(name,data);used+=len(data)
 sink=Sink();net=RangeSource(asset,opener=ctx["opener"],before_request=ctx["before_request"])
 index=net.fetch(asset["index_url"],0,asset["index_bytes"]-1,asset["index_bytes"],asset["index_ETag"])
 chunks,plan=tbi_chunks(index,asset);net.bind_plan_v1(plan["planned_windows"])
 header,indexes,hreport=frozen_header(net.lines(0),SUBJECTS,chrom)
 vcf=Buffered(sink,"H056.chr"+chrom+".noAP.vcf");vcf.write(header)
 qc=Buffered(sink,"regional_site_QC.tsv");qcfields=None
 counters=Counter();per_subject=[Counter() for _ in SUBJECTS];seen_vo=-1;last_pos=-1
 constants=0;qualified_ids=[];qualified_vectors=hashlib.sha256()
 for a,b in chunks:
  for vo,line in net.lines(a,b):
   if vo<=seen_vo:continue
   seen_vo=vo
   if line.startswith(b"#"):continue
   result=clean_record(line,indexes,chrom)
   if result is None:continue
   data,row,subjectqc=result;pos=row["POS_bp1"]
   require(pos>last_pos,"Strict unique candidate positions; ambiguous duplicate positions fail")
   last_pos=pos;vcf.write(data);counters["candidate_sites"]+=1
   # Exactly one pass count: exclude it from the generic Boolean loop.
   passed=row["H018_fixed_QC_pass"]
   counters["sites"]+=int(passed)
   for key,v in row.items():
    if type(v)is bool and key!="H018_fixed_QC_pass":counters[key+"_sites"]+=int(v)
   for dst,vals in zip(per_subject,subjectqc):
    for key in ("phased_diploid","GP_valid","HS_valid30"):dst[key]+=int(vals[key])
    dst["GP_max_ge_0_8"]+=int(vals["GP_max"] is not None and vals["GP_max"]>=.8)
   if passed:
    f=data.decode().rstrip("\n").split("\t");tags=f[8].split(":");gt=tags.index("GT")
    codes=[int(x) for sample in f[9:] for x in sample.split(":")[gt][::2]]
    require(len(codes)==H,"All qualified H36 phase calls")
    constants+=int(len(set(codes))==1);qualified_ids.append([chrom,pos,f[3],f[4]])
    nt="".join((f[3],f[4])[v] for v in codes);qualified_vectors.update((str(pos)+":"+nt+"\n").encode())
   if qcfields is None:qcfields=list(row);qc.write(("\t".join(qcfields)+"\n").encode())
   qc.write(("\t".join("" if row[k]is None else str(int(row[k])) if type(row[k])is bool else str(row[k]) for k in qcfields)+"\n").encode())
 require(counters["candidate_sites"]>0,"No fixed-region candidates")
 vcf.flush();qc.flush()
 counts=dict(candidate_sites=counters["candidate_sites"],sites=counters["sites"],constants=constants)
 # Underqualification remains recorded, with all partial outputs retained by root.
 qualified=counts["sites"]>=1000
 if qualified:validate_counts_v1(counts)
 return dict(schema="H056_H36_extraction_core_result_v1",protocol_ref=PROTOCOL_REF,chromosome=chrom,counts=counts,
  status="acquired_QC_qualified_root_completion_and_readback_required" if qualified else "inconclusive_under1000_qualified_sites",
  minimum_retained_SNPs=1000,AP_used_or_retained=False,HS_draws_materialized=False,GT_modified=False,
  FILTER_recorded_outside_QC_conjunction=True,qualified_identity_sha256=sha(json.dumps(qualified_ids,separators=(",",":")).encode()),
  original_nucleotide_vectors_sha256=qualified_vectors.hexdigest(),serialized_bytes=used,header=hreport,network_plan=plan,
  network=net.diagnostics_v1(),request_receipts=net.requests,counters=dict(counters),per_subject_QC_counts=[dict(x) for x in per_subject],
  independent_actual_count_and_serialized_readback_seals_required=True,compression_indexing_and_genuine_completion_root_owned=True)
