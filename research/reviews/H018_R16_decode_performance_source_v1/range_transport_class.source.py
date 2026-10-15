class RangeSource:
 def __init__(self,opener=urlopen,before_request=None):
  self.before_request=before_request;self.started=time.monotonic()
  self.opener=opener;self.received=0;self.requests=[];self.cache=OrderedDict();self.decompressed=0
  self.planned_windows=None;self.timings=Counter();self.stats=Counter()
 def bind_plan_v1(self,keys):
  require(self.planned_windows is None and type(keys)is list and keys==sorted(set(keys))
   and keys and keys[0]==0 and all(type(k)is int and 0<=k<(SOURCE_BYTES+WINDOW-1)//WINDOW for k in keys),
   'Exact bounded compressed-window plan')
  require(sum(min(WINDOW,SOURCE_BYTES-k*WINDOW) for k in keys)+INDEX_BYTES<NET_CAP,
   'Planned network bytes remain within original cap')
  self.planned_windows=frozenset(keys)
 def fetch(self,url,start,end,total,etag=None):
  require(time.monotonic()-self.started<=ACQUISITION_SECONDS,'Regional acquisition runtime cap')
  check_started=time.monotonic()
  if self.before_request is not None:self.before_request()
  self.timings['before_request_callback_seconds']+=time.monotonic()-check_started
  remaining=ACQUISITION_SECONDS-(time.monotonic()-self.started)
  require(remaining>0,'Regional acquisition runtime cap after boundary check')
  require(url in [VCF_URL,INDEX_URL] and 0<=start<=end<total and end-start+1<=MAX_HTTP_RANGE_BYTES,'Exact bounded source Range')
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
  require(0<=start<=start+n<=SOURCE_BYTES and n<=65536,'Bounded BGZF read')
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
    a=key*WINDOW;b=min((keys[-1]+1)*WINDOW,SOURCE_BYTES)-1
    chunk=self.fetch(VCF_URL,a,b,SOURCE_BYTES,VCF_ETAG)
    for cached_key in keys:
     offset=(cached_key-key)*WINDOW;part=chunk[offset:offset+WINDOW]
     if cached_key==0:require(sha(part)==PREFIX_SHA,'Frozen received first1MB identity differs')
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
