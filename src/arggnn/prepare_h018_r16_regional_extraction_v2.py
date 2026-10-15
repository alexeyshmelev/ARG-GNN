"""Bounded H018 R16/Q24 Allentoft regional extraction. Inert on import.

Root calls execute_v1(admission_ref, guard=actual_storage_guard). No network or
genotype acquisition without that root admission. AP-bearing source bytes stay
in bounded RAM; the first observed artifact is selected-subject, AP-free VCF.
"""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict,Counter
from urllib.request import Request,urlopen
import argparse,csv,datetime,gzip,hashlib,io,json,math,os,re,resource,struct,subprocess,sys,time,zlib
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='src/arggnn/prepare_h018_r16_regional_extraction_v2.py'
BASE='https://erda.ku.dk/archives/917f1ac64148c3800ab7baa29402d088/neo/genotypes-public/'
VCF_URL=BASE+'22.neo_impute_ph.vcf.gz';INDEX_URL=VCF_URL+'.tbi'
SOURCE_BYTES=1654398745;INDEX_BYTES=28008
VCF_ETAG='"629c1f19-605efa5ff6274"'
PREFIX_SHA='4564d66dc800ef5573ab043decfc89596868391a933d0aaadd9a9ac16c21ee2d'
INDEX_SHA='7c35737a4f03c40b1ce149c837f87c92495c88b1948c79da5246be72ed085b40'
BEGIN,END=20_000_000,22_000_000
ALLOWED_FORMAT=('GT','GP','HS','DS')
ALLOWED_INFO=('INFO','RAF','MASK_1000G','pan_troglodytes')
NET_CAP=100_000_000;OUTPUT_CAP=200_000_000;WINDOW=1_000_000
MAX_LINE=16*1024*1024;MAX_DECOMPRESSED=8_000_000_000;MAX_REQUESTS=128
def require(ok,msg):
 if not ok:raise ValueError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def ref(path):
 p=Path(path);return dict(path=str(p.relative_to(ROOT)),sha256=sha(p.read_bytes()))
def bound(record):
 p=ROOT/record['path']
 require(not Path(record['path']).is_absolute() and '..'not in Path(record['path']).parts and p.resolve()==p and p.is_file(),'Canonical project file')
 b=p.read_bytes();require(sha(b)==record['sha256'],'Bound file SHA differs: '+record['path']);return b
def read_json(record):return json.loads(bound(record))
def decode_hs(hs,m,nmain=15):
 require(type(hs)is int and 0<=hs<2**(2*nmain) and 0<=m<nmain,'Invalid HS/m')
 return (hs>>(2*m))&1,(hs>>(2*m+1))&1
class RangeSource:
 def __init__(self,opener=urlopen,before_request=None):
  self.before_request=before_request;self.started=time.monotonic()
  self.opener=opener;self.received=0;self.requests=[];self.cache=OrderedDict();self.decompressed=0
 def fetch(self,url,start,end,total,etag=None):
  require(time.monotonic()-self.started<=900,'Regional acquisition runtime cap')
  if self.before_request is not None:self.before_request()
  require(url in [VCF_URL,INDEX_URL] and 0<=start<=end<total and end-start+1<=WINDOW,'Exact bounded source Range')
  n=end-start+1
  require(self.received+n+1<=NET_CAP and len(self.requests)<MAX_REQUESTS,'Network byte/request cap')
  receipt=dict(url=url,start_byte=start,end_byte=end,total_source_bytes=total,requested_utc=utc(),bytes=0)
  self.requests.append(receipt)
  headers={'User-Agent':'ARG-GNN-H018-R16-region/1','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
  if etag:headers['If-Match']=etag
  with self.opener(Request(url,headers=headers),timeout=30) as r:
   receipt.update(status=r.status,ETag=r.headers.get('ETag'),Content_Range=r.headers.get('Content-Range'))
   require(r.status==206,'Refuse source that ignores Range (no full-file fallback)')
   require(r.url==url and r.headers.get('Content-Range')==f'bytes {start}-{end}/{total}','Source URL/range/total differs')
   require(r.headers.get('Content-Length')==str(n),'Exact bounded Content-Length')
   require(r.headers.get('Content-Encoding') in [None,'identity'],'No transfer content encoding')
   if etag:require(r.headers.get('ETag')==etag,'Source ETag changed')
   b=r.read(n+1);self.received+=len(b);receipt.update(bytes=len(b),sha256=sha(b),Last_Modified=r.headers.get('Last-Modified'))
   require(len(b)==n and self.received<=NET_CAP,'Range body length/cap differs')
  return b
 def bytes(self,start,n):
  require(0<=start<=start+n<=SOURCE_BYTES and n<=65536,'Bounded BGZF read')
  parts=[];pos=start
  while pos<start+n:
   key=pos//WINDOW
   if key not in self.cache:
    a=key*WINDOW;b=min(a+WINDOW,SOURCE_BYTES)-1
    chunk=self.fetch(VCF_URL,a,b,SOURCE_BYTES,VCF_ETAG)
    if a==0:require(sha(chunk)==PREFIX_SHA,'Frozen received first1MB identity differs')
    self.cache[key]=chunk
    while len(self.cache)>4:self.cache.popitem(last=False)
   self.cache.move_to_end(key);part=self.cache[key];offset=pos-key*WINDOW
   take=min(start+n-pos,len(part)-offset);require(take>0,'Empty byte range')
   parts.append(part[offset:offset+take]);pos+=take
  return b''.join(parts)
 def block(self,pos):
  require(time.monotonic()-self.started<=900,'Regional decode runtime cap')
  h=self.bytes(pos,12);require(h[:4]==b'\x1f\x8b\x08\x04','BGZF magic/flags')
  xlen=struct.unpack_from('<H',h,10)[0];require(6<=xlen<=1024,'BGZF extra length')
  extra=self.bytes(pos+12,xlen);off=0;size=None
  while off+4<=len(extra):
   tag=extra[off:off+2];n=struct.unpack_from('<H',extra,off+2)[0];require(off+4+n<=len(extra),'BGZF subfield')
   if tag==b'BC':require(n==2 and size is None,'Unique BGZF BC');size=struct.unpack_from('<H',extra,off+4)[0]+1
   off+=4+n
  require(size is not None and 26<=size<=65536,'BGZF block size')
  raw=self.bytes(pos,size);data=zlib.decompress(raw,31);require(len(data)<=65536,'BGZF expanded block')
  self.decompressed+=len(data);require(self.decompressed<=MAX_DECOMPRESSED,'Expanded-source cap')
  return data,pos+size
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
def tbi_chunks(raw,chrom='22',begin=BEGIN,end=END):
 require(sha(raw)==INDEX_SHA,'Exact frozen public TBI hash')
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
 require(found is not None and names==['22'],'Exact public chromosome22 TBI')
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
  windows.update(range((a>>16)//WINDOW,min(SOURCE_BYTES-1,(b>>16)+65535)//WINDOW+1))
 estimate=sum(min(WINDOW,SOURCE_BYTES-key*WINDOW) for key in windows)+INDEX_BYTES
 require(estimate<NET_CAP,'Region compressed plan exceeds byte bound')
 return merged,dict(index_raw_bytes=len(data),estimated_network_upper_bytes=estimate,
  virtual_chunks=[[int(a),int(b)] for a,b in merged],index_sha256=INDEX_SHA)
def frozen_header(lines,subjects):
 meta=[];sample_names=None;nmain=None;source_b37=False;source_chr22=False
 for _,line in lines:
  text=line.decode()
  if text.startswith('##NMAIN='):nmain=int(text.split('=',1)[1])
  if text.startswith('##contig=') and 'assembly=b37' in text:source_b37=True
  if text.startswith('##fileformat=') or text.startswith('##FILTER='):meta.append(text)
  if text.startswith('##contig=<ID=22,') or text=='##contig=<ID=22>':meta.append(text);source_chr22=True
  match=re.match(r'##(FORMAT|INFO)=<ID=([^,>]+)',text)
  if match and match.group(2) in (ALLOWED_FORMAT if match.group(1)=='FORMAT' else ALLOWED_INFO):meta.append(text)
  if text.startswith('#CHROM\t'):
   fields=text.split('\t');sample_names=fields[9:];break
 require(sample_names is not None and len(sample_names)==3331 and nmain==15,'Frozen source sample header/NMAIN15')
 require(len(set(sample_names))==len(sample_names) and all(s in sample_names and not s.endswith('_rnd') for s in subjects),'Exact ordinary subject IDs')
 require(source_b37,'Frozen public source b37 assembly metadata')
 if not source_chr22:meta.append('##contig=<ID=22,assembly=b37>')
 defs={m.group(1) for x in meta if (m:=re.match(r'##FORMAT=<ID=([^,>]+)',x))}
 require(set(['GT','GP','HS']).issubset(defs),'Required original GT/GP/HS definitions')
 meta+=['##H018_region_bp0=22:20000000-22000000','##H018_fields_policy=FORMAT_AP_removed_before_retained_artifact;GT_GP_HS_DS_only',
  '##H018_coordinate=VCF_POS_1based_preserved;region_bp0=POS-1-20000000',
  '##NMAIN=15']
 header='\n'.join(meta)+'\n'+'\t'.join(['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','FORMAT']+subjects)+'\n'
 return header.encode(),[sample_names.index(s) for s in subjects],dict(source_sample_columns=len(sample_names),selected_subjects=subjects,
  excluded_pseudohaploid_columns=sum(s.endswith('_rnd') for s in sample_names),
  selected_ordinary_indexes0=[[s for s in sample_names if not s.endswith('_rnd')].index(x) for x in subjects],NMAIN=nmain,
  source_chr22_contig_definition_present=source_chr22,
  derived_chr22_header_added_without_length=not source_chr22,
  chromosome_definition_basis='Frozen H018 GRCh37/b37 protocol plus source b37 contigs and exact public TBI name22; no coordinate conversion or fabricated length.',
  original_GT_header_description='Source header calls GT unphased; actual selected-record phase is independently counted.')
def numeric(value):
 try:
  f=float(value);return f if math.isfinite(f) else None
 except (ValueError,TypeError):return None
def clean_record(raw,indexes):
 f=raw.decode().split('\t');require(len(f)==3340,'Source VCF column count')
 chrom,pos,vid,r,a,qual,flt,info_text,fmt=f[:9];pos=int(pos)
 if chrom!='22' or not BEGIN<=pos-1<END:return None
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
 ancestral=info.get('pan_troglodytes');polarized=ancestral in bases and ancestral in [r,a]
 qc=dict(biallelic_ACGT=biallelic,transversion=transversion,strict_mask=info.get('MASK_1000G') is True,
  INFO_ge_0_8=score is not None and score>=.8,external_RAF_MAF_ge_0_05=maf is not None and maf>=.05,
  all20_phased_diploid=all(x['phased_diploid'] for x in subject_qc),
  all20_GP_valid=all(x['GP_valid'] for x in subject_qc),
  all20_GP_max_ge_0_8=all(x['GP_max'] is not None and x['GP_max']>=.8 for x in subject_qc),
  all20_HS_valid30=all(x['HS_valid30'] for x in subject_qc),ancestral_in_REF_ALT=polarized)
 passed=all(qc.values())
 row=dict(chrom=chrom,POS_bp1=pos,absolute_bp0=pos-1,region_bp0=pos-1-BEGIN,REF=r,ALT=a,
  FILTER=flt,INFO=score,RAF_external=raf,MAF_external=maf,pan_troglodytes=ancestral if isinstance(ancestral,str) else None,
  **qc,H018_fixed_QC_pass=passed)
 return output.encode(),row,subject_qc
class Output:
 def __init__(self,path,guard):
  require(callable(guard),'Root actual storage guard callback required')
  self.path=path;self.guard=guard;self.used=0;self.files={}
  guard(8192);path.mkdir(parents=False,exist_ok=False)
 def append(self,name,data):
  require(Path(name).name==name and self.used+len(data)+8192<=OUTPUT_CAP,'Regional output byte cap')
  self.guard(len(data)+8192)
  if name not in self.files:self.files[name]=(self.path/name).open('xb')
  self.files[name].write(data);self.files[name].flush();self.used+=len(data)
 def json(self,name,value):self.append(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
 def close(self):
  for f in self.files.values():f.flush();os.fsync(f.fileno());f.close()
  self.files={}
 def adopt(self,name,maximum):
  p=self.path/name;require(p.resolve()==p and p.is_file() and p.stat().st_size<=maximum,'Bounded native output')
  self.used+=p.stat().st_size;require(self.used+8192<OUTPUT_CAP and sum(x.stat().st_size for x in self.path.iterdir() if x.is_file())+8192<OUTPUT_CAP,'Output total cap')
class Buffered:
 def __init__(self,out,name):self.out=out;self.name=name;self.data=bytearray()
 def write(self,b):
  self.data.extend(b)
  if len(self.data)>=1_000_000:self.flush()
 def flush(self):
  if self.data:self.out.append(self.name,bytes(self.data));self.data.clear()
def tool_command(path,args,limit=16384):
 r=subprocess.run([str(path),*args],capture_output=True,timeout=60,
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
 require(r.returncode==0 and len(r.stdout)+len(r.stderr)<=limit,'Native HTS qualification command failed/bound')
 return r.stdout.decode()
def runtime_report_v1(cfg):
 tools={};versions={}
 for name,r in cfg['tools'].items():
  bound(r);p=ROOT/r['path'];tools[name]=r;versions[name]=tool_command(p,['--version'])
 return dict(python=sys.version,python_executable=sys.executable,prefix=sys.prefix,
  tools=tools,versions=versions,GPU_used=False,remote_HTS_autofetch_used=False,install_performed=False)
def execute_v1(admission_ref,*,guard):
 """Only root's admitted caller supplies its current actual storage guard."""
 require(callable(guard),'No unguarded CLI execution')
 ad=read_json(admission_ref);cfg=read_json(ad['config_ref'])
 require(ad['schema']=='H018_R16_regional_extraction_root_admission_v1' and ad['root_authorizer']=='root_agent'
  and ad['execution_authorized'] is True and ad['GPU_authorized'] is False
  and ad['source_ref']==ref(ROOT/SOURCE) and ad['output_cap_bytes']==OUTPUT_CAP
  and ad['network_byte_cap']==NET_CAP,'Exact root H018 acquisition admission')
 expiry=datetime.datetime.fromisoformat(ad['expires_at_utc'].replace('Z','+00:00'))
 require(expiry.tzinfo is not None and expiry>datetime.datetime.now(datetime.timezone.utc),'Fresh root admission')
 for r in cfg['frozen_refs'].values():bound(r)
 endpoint=read_json(cfg['frozen_refs']['endpoint'])
 subjects=(ROOT/cfg['frozen_refs']['subject_ids']['path']).read_text().splitlines()
 expected=[s for group in endpoint['reference_subjects'].values() for s in group]+endpoint['query_subjects']
 require(subjects==expected and len(subjects)==len(set(subjects))==20 and all(not x.endswith('_rnd') for x in subjects),'Frozen H018 cohort/order')
 require(endpoint['region']['interval_bp0']==[BEGIN,END] and endpoint['reference_haplotypes']==16
  and endpoint['query_haplotypes']==24,'Frozen H018 R16/Q24 region')
 outdir=ROOT/ad['output_directory'];require(not Path(ad['output_directory']).is_absolute()
  and '..'not in Path(ad['output_directory']).parts and outdir.parent.resolve()==outdir.parent
  and outdir.is_relative_to(ROOT/'data'),'Fresh canonical data output scope')
 storage=read_json(ad['storage_ref'])
 require(storage.get('active') is True and storage.get('consumer_uses_v9_guard') is True
  and storage['output_directory']==ad['output_directory'] and storage['reserved_new_bytes']>=OUTPUT_CAP,
  'Root actual guarded reservation must match acquisition output scope/cap')
 # Resource journal permits source/runtime metadata only. The exact JSON
 # manifest binds raw tools/TBI/TSV/text transitively, preserving all SHA checks.
 closure=[ad['source_ref'],ad['config_ref'],ad['runtime_ref'],ad['consumer_manifest_ref']]
 require(all(r in storage['reservation']['consumer_bindings'] for r in closure),'Source/config/runtime/closure manifest must be reserved')
 manifest=read_json(ad['consumer_manifest_ref'])
 require(manifest['schema']=='H018_R16_exact_transitive_consumer_closure_v1'
  and manifest['source_ref']==ad['source_ref'] and manifest['config_ref']==ad['config_ref']
  and manifest['runtime_ref']==ad['runtime_ref']
  and manifest['tool_refs']==cfg['tools'] and manifest['frozen_refs']==cfg['frozen_refs'],
  'Exact tool/data transitive closure required')
 for r in [*manifest['tool_refs'].values(),*manifest['frozen_refs'].values()]:bound(r)
 runtime=runtime_report_v1(cfg);require(runtime==read_json(ad['runtime_ref']),'Pinned source-only HTS runtime changed')
 out=Output(outdir,guard);net=RangeSource(before_request=lambda:guard(0));start=time.monotonic()
 try:
  out.json('started.json',dict(admission_ref=admission_ref,source_ref=ref(ROOT/SOURCE),config_ref=ad['config_ref'],
   started_utc=utc(),scope='H018 bounded observed acquisition/qualification only',runtime=runtime))
  local_index=bound(cfg['frozen_refs']['public_TBI'])
  remote_index=net.fetch(INDEX_URL,0,INDEX_BYTES-1,INDEX_BYTES)
  require(sha(remote_index)==sha(local_index)==INDEX_SHA,'Public source/index identity differs')
  chunks,plan=tbi_chunks(remote_index)
  header,indexes,hreport=frozen_header(net.lines(0),subjects)
  # Sanitize in RAM. Only this noAP20-subject header reaches an observed artifact.
  vcf=Buffered(out,'H018_R16_Q24.chr22_20_22Mb.noAP.vcf');vcf.write(header)
  counters=Counter();per_subject=[Counter() for _ in subjects];qc=None;seen_vo=-1;last_pos=-1;duplicate_positions=0
  qcfields=None
  for a,b in chunks:
   for vo,line in net.lines(a,b):
    if vo<=seen_vo:continue
    seen_vo=vo
    if line.startswith(b'#'):continue
    result=clean_record(line,indexes)
    if result is None:continue
    data,row,subjectqc=result;pos=row['POS_bp1'];require(pos>=last_pos,'Source regional position order')
    duplicate_positions+=int(pos==last_pos);last_pos=pos;vcf.write(data);counters['candidate_sites']+=1
    counters['H018_fixed_QC_pass_sites']+=int(row['H018_fixed_QC_pass'])
    for key,v in row.items():
     if type(v)is bool:counters[key+'_sites']+=int(v)
    for dst,vals in zip(per_subject,subjectqc):
     for key in ['phased_diploid','GP_valid','HS_valid30']:dst[key]+=int(vals[key])
     dst['GP_max_ge_0_8']+=int(vals['GP_max'] is not None and vals['GP_max']>=.8)
    if qc is None:
     qc=Buffered(out,'regional_site_QC.tsv');qcfields=list(row)
     qc.write(('\t'.join(qcfields)+'\n').encode())
    qc.write(('\t'.join('' if row[k]is None else str(int(row[k])) if type(row[k])is bool else str(row[k]) for k in qcfields)+'\n').encode())
  require(counters['candidate_sites']>0,'No source regional records')
  vcf.flush();qc.flush();out.close()
  # Compress the AP-free file only, through a capped stdout sink.
  bgzip=ROOT/cfg['tools']['bgzip']['path'];tabix=ROOT/cfg['tools']['tabix']['path'];bcf=ROOT/cfg['tools']['bcftools']['path']
  name='H018_R16_Q24.chr22_20_22Mb.noAP.vcf.gz'
  child=subprocess.Popen([str(bgzip),'-@','1','-c',str(outdir/'H018_R16_Q24.chr22_20_22Mb.noAP.vcf')],
   stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TMPDIR':str(outdir)})
  try:
   while True:
    b=child.stdout.read(1_000_000)
    if not b:break
    out.append(name,b)
   stderr=child.stderr.read(16385);code=child.wait(timeout=20)
   require(code==0 and len(stderr)<=16384,'bgzip actual wait/bound failed')
  except BaseException:
   if child.poll()is None:child.kill();child.wait()
   raise
  out.close();max_index=4_000_000;out.guard(max_index+8192)
  def index_limits():resource.setrlimit(resource.RLIMIT_FSIZE,(max_index,max_index))
  native=subprocess.run([str(tabix),'-p','vcf',str(outdir/name)],capture_output=True,timeout=60,
   preexec_fn=index_limits,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TMPDIR':str(outdir)})
  require(native.returncode==0 and len(native.stdout)+len(native.stderr)<=16384,'tabix actual wait/bound')
  out.adopt(name+'.tbi',max_index)
  actual_subjects=tool_command(bcf,['query','-l',str(outdir/name)]).splitlines()
  require(actual_subjects==subjects,'Native HTS sample-order qualification')
  native_count=int(tool_command(bcf,['index','-n',str(outdir/name)]).strip())
  require(native_count==counters['candidate_sites'],'Native tabix indexed record count')
  actual_header=tool_command(bcf,['view','-h',str(outdir/name)],limit=65536)
  require('##FORMAT=<ID=AP,'not in actual_header and '_rnd'not in actual_header.split('#CHROM')[-1],'Native noAP/ordinary header qualification')
  rows=list(csv.DictReader(io.StringIO(bound(cfg['frozen_refs']['subject_metadata']).decode()),delimiter='\t'))
  require([r['subject_id'] for r in rows]==subjects and [int(r['ordinary_VCF_column_index0']) for r in rows]==hreport['selected_ordinary_indexes0'],'Exact frozen metadata/header ordinary-column join')
  roster=[]
  for row in rows:
   age=float(row['age_BP_mean']);lo=float(row['age_BP_lower95']);hi=float(row['age_BP_upper95'])
   require(0<lo<=age<=hi and abs(float(row['working_age_generations_BP_div_28'])-age/28)<1e-9,'Frozen age convention')
   for hap in [0,1]:roster.append(dict(subject_id=row['subject_id'],haplotype=hap,role=row['role'],
    proxy_class=row['proxy_class'] or None,age_mean_cal_BP1950=age,age95CI_cal_BP1950=[lo,hi],
    working_age_generations_BP_div_28=age/28,inference_time_origin_qualified=False,
    pure_ancestry_verified=False,source_VCF_column0=indexes[len(roster)//2]))
  out.append('sample_order.txt',('\n'.join(subjects)+'\n').encode())
  out.append('subject_metadata.tsv',bound(cfg['frozen_refs']['subject_metadata']))
  out.json('haplotype_roster.json',roster)
  out.json('source_download_receipt.json',dict(requests=net.requests,received_bytes=net.received,
   range_cap_bytes=NET_CAP,plan=plan,source_URL=VCF_URL,source_ETag=VCF_ETAG,
   source_full_bytes=SOURCE_BYTES,source_full_publisher_checksum_verified=False,
   public_index_sha256=INDEX_SHA,raw_AP_bearing_bytes_retained=False,first_observed_artifact_AP_free=True))
  qualification=dict(candidate_sites=counters['candidate_sites'],fixed_H018_QC_pass_sites=counters['H018_fixed_QC_pass_sites'],
   minimum_required=1000,region_observation_QC_passed=counters['H018_fixed_QC_pass_sites']>=1000,
   fixed_filters=dict(endpoint['observation_QC']),site_counts=dict(counters),duplicate_source_positions=duplicate_positions,
   per_subject=[dict(subject_id=s,**v) for s,v in zip(subjects,per_subject)],source_header=hreport,
   source_INFO_and_external_RAF_preserved=True,target_AF_used_for_MAF=False,AP_used=False,
   GT_modified=False,HS_sampled=False,posterior_semantics='HS changes genotype and phase; fixed iteration must be coherent across sites in future sensitivity stage',
   genetic_map_qualified=False,R16_model_runtime_qualified=False,inference_or_PCA_performed=False,
   ancestral_state='Only exact single-base source INFO/pan_troglodytes matching REF/ALT passes the frozen criterion; no REF-as-ancestral substitution.')
  out.json('qualification.json',qualification)
  out.json('completed.json',dict(schema='H018_R16_regional_extraction_completed_v1',status='extracted_and_characterized_not_model_admitted',
   source_ref=ref(ROOT/SOURCE),admission_ref=admission_ref,config_ref=ad['config_ref'],runtime=runtime,
   observed_VCF_ref=ref(outdir/name),observed_index_ref=ref(outdir/(name+'.tbi')),
   site_QC_ref=ref(outdir/'regional_site_QC.tsv'),qualification_ref=ref(outdir/'qualification.json'),
   regional_observation_QC_passed=qualification['region_observation_QC_passed'],network_bytes=net.received,
   output_bytes_before_completion=out.used,elapsed_seconds=time.monotonic()-start,completed_utc=utc(),
   actual_bgzip_wait_exit0=True,actual_tabix_wait_exit0=True,native_index_count=native_count,
   GPU_used=False,original_H018_roster_or_protocol_changed=False,ancestry_truth_available=False,
   empirical_genealogy_inference=False,model_prediction=False,real_PCA=False))
  out.close();return ref(outdir/'completed.json')
 except BaseException as e:
  out.close()
  try:
   out.json('failed.json',dict(status='failed_partial_preserved',error_type=type(e).__name__,error=str(e),
    network_bytes=net.received,HTTP_requests=net.requests,raw_AP_bearing_bytes_retained=False,scientific_completion_claimed=False,elapsed_seconds=time.monotonic()-start))
   out.close()
  except BaseException:pass
  raise
def source_report_v1():
 return dict(status='source_only_no_acquisition',source_ref=ref(ROOT/SOURCE),
  fixed_region_bp0=[BEGIN,END],subjects=20,ref_haplotypes=16,query_haplotypes=24,
  network_cap_bytes=NET_CAP,output_cap_bytes=OUTPUT_CAP,root_guard_callback_required=True,
  direct_remote_HTS_autofetch=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-report',action='store_true')
 args=parser.parse_args()
 if not args.source_report:parser.error('Root must call execute_v1 with its actual storage guard; no unguarded CLI execution.')
 print(json.dumps(source_report_v1(),indent=2))
