"""Inactive H055 source: bounded original-VCF GP dosage/f4, no model/phase use.
Root calls run_v1 only after source review, registration and normal admissions.
All network/payload/writes are inert on import. No direct acquisition CLI.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from collections import OrderedDict, Counter
import csv, datetime, hashlib, io, json, math, re, struct, time, zlib

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BEGIN,END=20_000_000,30_000_000
BLOCK=1_000_000
CAP=16*1024*1024
MAX_ROWS=100_000
RECORD=struct.Struct("<BBIBBIH18d2x")  # 160 bytes, documented in protocol.
RANGE_RECEIPT=struct.Struct("<BQ I32s")  # chromosome, start, bytes, SHA256.
WINDOW=1_048_576
MISSING=float("nan")
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bound(ref):
    p=ROOT/ref["path"]
    need(not Path(ref["path"]).is_absolute() and ".." not in Path(ref["path"]).parts and p.resolve()==p and p.is_file(),"Canonical bound source")
    raw=p.read_bytes();need(sha(raw)==ref["sha256"],"Bound source hash changed")
    return raw
def finite(x):
    try: y=float(x);return y if math.isfinite(y) else None
    except (ValueError,TypeError):return None
def dosage_gp(raw):
    # VCF diploid biallelic GP order 0/0,0/1,1/1. No GT/DS fallback.
    p=[finite(x) for x in raw.split(b",")]
    if len(p)!=3 or any(x is None or not 0<=x<=1 for x in p):return None
    s=sum(p)
    if s<=0 or abs(s-1)>0.002:return None
    return (p[1]+2*p[2])/s
def unpack_gzip(raw,limit):
    # TBI may contain multiple BGZF/gzip members; consume every member.
    pending=raw;parts=[];used=0
    while pending:
        d=zlib.decompressobj(31);part=d.decompress(pending,limit-used+1)
        used+=len(part)
        need(used<=limit and d.eof and not d.unconsumed_tail,"Bounded complete gzip members")
        parts.append(part);pending=d.unused_data
    return b"".join(parts)

class Writer:
    def __init__(self,path,before_write):
        self.path=Path(path);self.before=before_write;self.total=0;self.files={}
        need(callable(before_write) and self.path.is_absolute() and self.path.parent.resolve()==self.path.parent and self.path.is_relative_to(ROOT),"Root owned canonical output")
        need(not self.path.exists(),"Fresh output required")
        self.before(4096);self.path.mkdir()
    def put(self,name,raw):
        need(Path(name).name==name and self.total+len(raw)<=CAP-16384,"16MiB durable cap; no thinning")
        self.before(len(raw))
        if name not in self.files:self.files[name]=(self.path/name).open("xb")
        self.files[name].write(raw);self.files[name].flush();self.total+=len(raw)
    def js(self,name,obj):
        self.put(name,(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    def close(self):
        for f in self.files.values():f.flush();__import__("os").fsync(f.fileno());f.close()
        self.files={}

class Region:
    def __init__(self,asset,before_request,writer):
        self.a=asset;self.before=before_request;self.out=writer
        self.received=0;self.requests=0;self.expanded=0;self.cache=OrderedDict();self.start=time.monotonic()
        self.identities={}
    def head(self,url,total):
        self.before(0);self.requests+=1
        with urlopen(Request(url,method="HEAD",headers={"Accept-Encoding":"identity","User-Agent":"ARG-GNN-H055/1"}),timeout=30) as r:
            need(r.status==200 and r.url==url and int(r.headers.get("Content-Length","-1"))==total,"Object metadata identity/size")
            tag=r.headers.get("ETag")
            need(isinstance(tag,str) and tag.startswith('"') and tag.endswith('"') and not tag.startswith("W/"),"Strong ETag required")
            expected=self.a.get("expected_vcf_ETag") if url==self.a["url"] else None
            if expected is not None:need(tag==expected,"Previously authenticated source ETag changed")
            self.identities[url]=dict(ETag=tag,bytes=total,Last_Modified=r.headers.get("Last-Modified"))
    def fetch(self,url,start,n,total):
        need(time.monotonic()-self.start<=3600 and 0<=start<start+n<=total and n<=WINDOW,"Bounded regional request/time")
        need(self.received+n+1<=512*1024*1024 and self.requests<1800,"512MiB chromosome transfer/request cap")
        self.before(n+1);self.requests+=1;tag=self.identities[url]["ETag"]
        h={"Range":f"bytes={start}-{start+n-1}","If-Match":tag,"Accept-Encoding":"identity","User-Agent":"ARG-GNN-H055/1"}
        with urlopen(Request(url,headers=h),timeout=30) as r:
            need(r.status==206 and r.url==url and r.headers.get("Content-Range")==f"bytes {start}-{start+n-1}/{total}","No full-file/redirect fallback")
            need(r.headers.get("Content-Length")==str(n) and r.headers.get("ETag")==tag and r.headers.get("Content-Encoding") in (None,"identity"),"Exact Range length/version")
            raw=r.read(n+1);self.received+=len(raw);need(len(raw)==n,"Range body length")
        self.out.put("ranges.bin",RANGE_RECEIPT.pack(int(self.a["chromosome"]),start,n,hashlib.sha256(raw).digest()))
        return raw
    def read(self,start,n):
        total=self.a["bytes"];need(0<=start<=start+n<=total and n<=65536,"BGZF byte read")
        out=[];pos=start
        while pos<start+n:
            key=pos//WINDOW
            if key not in self.cache:
                a=key*WINDOW;self.cache[key]=self.fetch(self.a["url"],a,min(WINDOW,total-a),total)
                while len(self.cache)>4:self.cache.popitem(last=False)
            self.cache.move_to_end(key);part=self.cache[key];off=pos-key*WINDOW
            take=min(start+n-pos,len(part)-off);need(take>0,"Nonempty BGZF range")
            out.append(part[off:off+take]);pos+=take
        return b"".join(out)
    def block(self,pos):
        h=self.read(pos,12);need(h[:4]==b"\x1f\x8b\x08\x04","BGZF magic")
        xlen=struct.unpack_from("<H",h,10)[0];need(6<=xlen<=1024,"BGZF extra size")
        extra=self.read(pos+12,xlen);off=0;size=None
        while off+4<=len(extra):
            key,n=struct.unpack_from("<2sH",extra,off);off+=4
            need(off+n<=len(extra),"BGZF extra field")
            if key==b"BC":
                need(n==2 and size is None,"Unique BC");size=struct.unpack_from("<H",extra,off)[0]+1
            off+=n
        need(size is not None and 26<=size<=65536,"BGZF block size")
        data=unpack_gzip(self.read(pos,size),65536);self.expanded+=len(data)
        need(self.expanded<=32*1024**3,"32GiB chromosome decode cap")
        return data,pos+size
    def lines(self,begin_vo,end_vo=None):
        pos,first=begin_vo>>16,begin_vo&65535;pending=bytearray()
        while True:
            data,nxt=self.block(pos)
            if not data:need(not pending,"No truncated final line");return
            need(first<=len(data),"BGZF virtual offset");cursor=first
            while cursor<len(data):
                if not pending:
                    vo=(pos<<16)|cursor
                    if end_vo is not None and vo>=end_vo:return
                stop=data.find(b"\n",cursor)
                if stop<0:
                    pending.extend(data[cursor:]);need(len(pending)<=16*1024*1024,"16MiB source record cap");break
                pending.extend(data[cursor:stop]);need(len(pending)<=16*1024*1024,"16MiB source record cap")
                yield vo,bytes(pending)
                pending.clear();cursor=stop+1
            pos,first=nxt,0

def chunks_tbi(raw,chrom):
    data=unpack_gzip(raw,8*1024*1024)
    need(data[:4]==b"TBI\x01","TBI format")
    n,fmt,seq,beg,end,meta,skip,ln=struct.unpack_from("<8i",data,4);off=36
    need(n==1 and (fmt,seq,beg,end,meta,skip)==(2,1,2,0,35,0) and 0<ln<100,"Exact VCF index schema")
    names=data[off:off+ln].rstrip(b"\0").decode().split("\0");off+=ln
    need(names==[chrom],"Exact chromosome index; no aliases")
    nb,=struct.unpack_from("<i",data,off);off+=4;need(0<=nb<=100000,"TBI bins");bins={}
    for _ in range(nb):
        key,nc=struct.unpack_from("<Ii",data,off);off+=8;need(0<=nc<=100000,"TBI chunks")
        need(key not in bins,"Unique TBI bin");bins[key]=[]
        for _ in range(nc):bins[key].append(struct.unpack_from("<QQ",data,off));off+=16
    nl,=struct.unpack_from("<i",data,off);off+=4;need(0<=nl<=1000000 and off+8*nl<=len(data),"TBI linear index")
    linear=struct.unpack_from("<"+"Q"*nl,data,off);wanted={0}
    for first,shift in [(1,26),(9,23),(73,20),(585,17),(4681,14)]:
        wanted.update(range(first+(BEGIN>>shift),first+((END-1)>>shift)+1))
    minimum=linear[BEGIN>>14] if BEGIN>>14<nl else 0
    selected=sorted((max(a,minimum),b) for key in wanted for a,b in bins.get(key,[]) if b>minimum and a<b)
    merged=[]
    for a,b in selected:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    need(merged,"No regional chunks")
    return merged

def header_v1(lines,subjects,cfg):
    raw=[];names=None
    for _,line in lines:
        need(line.startswith(b"#"),"Header ends before CHROM")
        raw.append(line+b"\n")
        need(sum(map(len,raw))<=262144,"Header size cap")
        if line.startswith(b"#CHROM\t"):names=line.decode().split("\t")[9:];break
    t=b"".join(raw).decode()
    need(names is not None and len(names)==len(set(names))==3331,"Frozen ordinary/public header catalogue")
    need([line for line in t.splitlines() if line.startswith("##FORMAT=<ID=GP,")]==[cfg["expected_GP_definition"]],"Unique expected GP definition")
    need([line for line in t.splitlines() if line.startswith("##reference=")]==[cfg["expected_reference_line"]] and "assembly=b37" in t,"Unique GRCh37 source metadata")
    need(all(s in names and not s.endswith("_rnd") for s in subjects),"Exact ordinary subjects, no alias or pseudohaploid substitution")
    for line in t.splitlines():
        if line.startswith("##contig=<ID="+cfg["chromosome"]+","):
            m=re.search(r"(?:,|<)assembly=([^,>]+)",line)
            need(m is None or m.group(1)=="b37","Conflicting target assembly")
            m=re.search(r"(?:,|<)length=([0-9]+)",line)
            lengths={"20":63025520,"21":48129895,"22":51304566}
            need(m is None or int(m.group(1))==lengths[cfg["chromosome"]],"Conflicting GRCh37 contig length")
    indexes=[names.index(s) for s in subjects];ordinary=[s for s in names if not s.endswith("_rnd")]
    need([ordinary.index(s) for s in subjects]==cfg["expected_ordinary_indexes0"],"Exact roster/header join")
    return indexes,dict(header_sha256=sha(t.encode()),sample_columns=len(names),selected_subjects=subjects,source_indexes0=indexes,
        ordinary_indexes0=cfg["expected_ordinary_indexes0"],GP_definition=cfg["expected_GP_definition"],
        reference_line=cfg["expected_reference_line"],target_contig_definition_present=bool(re.search(r"##contig=<ID="+cfg["chromosome"]+r"[,>]",t)),
        ancestry_AP_parsed_or_retained=False)

def site_v1(raw,indexes,chrom):
    f=raw.split(b"\t");need(len(f)==3340,"Exact original VCF width")
    c,pos,vid,ref,alt,qual,flt,info,fmt=f[:9];pos=int(pos)
    if c.decode()!=chrom or not BEGIN<=pos-1<END:return None,"outside_window"
    if not(len(ref)==len(alt)==1 and ref in b"ACGT" and alt in b"ACGT" and ref!=alt):return None,"not_biallelic_SNV"
    if (ref,alt) in [(b"A",b"G"),(b"G",b"A"),(b"C",b"T"),(b"T",b"C")]:return None,"transition"
    if flt!=b"PASS":return None,"FILTER_not_PASS"
    # Only two observation-QC fields are interpreted. No target AF/AP/ancestral label.
    inf={}
    for field in info.split(b";"):
        key,sep,value=field.partition(b"=")
        if key in (b"INFO",b"MASK_1000G"):
            need(key not in inf,"Unique selected INFO tags");inf[key]=value if sep else True
    quality=finite(inf.get(b"INFO"))
    if quality is None or quality<0.8:return None,"imputation_INFO_lt_0_8"
    if inf.get(b"MASK_1000G") is not True:return None,"not_strict_mask"
    tags=fmt.split(b":");need(len(tags)==len(set(tags)),"Unique FORMAT tags")
    gpi=tags.index(b"GP") if b"GP" in tags else None
    doses=[];mask=0
    for j,index in enumerate(indexes):
        # Extract only GP; do not decode phase, HS, AP, GT or DS values.
        parts=f[9+index].split(b":",gpi+1) if gpi is not None else []
        d=dosage_gp(parts[gpi]) if gpi is not None and len(parts)>gpi else None
        doses.append(d if d is not None else MISSING)
        if d is not None:mask|=1<<j
    return (pos,ref[0],alt[0],mask,doses),"metadata_eligible"

def sd(values):
    return (sum((x-sum(values)/len(values))**2 for x in values)/(len(values)-1))**0.5 if len(values)>1 else None
def descriptive(values):
    v=[x for x in values if x is not None]
    return dict(n=len(v),min=min(v) if v else None,max=max(v) if v else None,sample_SD=sd(v))
def run_v1(protocol_path,output_dir,*,root_authorization,before_request,before_write):
    """Root callback boundary only. No guard/admission implementation is provided."""
    protocol_path=Path(protocol_path);praw=protocol_path.read_bytes();cfg=json.loads(praw)
    auth=root_authorization
    need(callable(before_request) and callable(before_write),"Root supplies existing resource boundaries")
    need(auth.get("hypothesis_id")=="H055" and auth.get("execution_authorized") is True and auth.get("registered") is True
        and auth.get("protocol_sha256")==sha(praw) and auth.get("source_sha256")==sha(Path(__file__).read_bytes())
        and auth.get("output_cap_bytes")==CAP,"Root-reviewed registered exact source/protocol")
    need(cfg["schema"]=="Danish_genotype_f4_protocol_v1" and RECORD.size==160,"Frozen protocol/schema")
    roster_raw=bound(cfg["roster_ref"]);rows=list(csv.DictReader(io.StringIO(roster_raw.decode()),delimiter="\t"))
    subjects=[r["subject_id"] for r in rows];need(subjects==cfg["subjects"] and len(set(subjects))==18,"Exact prospective18-person roster")
    for ref in cfg["source_refs"]:bound(ref)
    earlier=[subjects.index(s) for s in cfg["earlier"]];later=[subjects.index(s) for s in cfg["later"]]
    proxies=[subjects.index(s) for s in ["Sidelkino","KK1"]];bar=subjects.index("Bar8")
    required=sum(1<<j for j in earlier+later+proxies+[bar])
    need(len(earlier)==len(later)==5 and len(set(earlier+later+proxies+[bar]))==13,"Frozen nonoverlapping primary subjects")
    need(len(cfg["assets"])==3 and [a["chromosome"] for a in cfg["assets"]]==["20","21","22"],"Exact three windows")
    for ids,lo,hi in [(cfg["earlier"],4800,6000),(cfg["later"],3500,4500)]:
        for s in ids:
            row=rows[subjects.index(s)]
            need(row["proposed_role"]=="query" and row["country"]=="Denmark" and row["dating_source"]=="direct","Frozen Danish metadata role")
            need(lo<=float(row["age_calBP1950_lower95"])<=float(row["age_calBP1950_mean"])<=float(row["age_calBP1950_upper95"])<=hi,"Frozen whole date-bound era")
            need(0.5<=float(row["coverage_x"])<=2,"Predeclared coverage band")
    query=earlier+later
    need(len({rows[j]["site"] for j in query})==10,"Ten different archaeological sites")
    out=Writer(output_dir,before_write);net_report=[];counts=Counter();nrows=0
    blocks=[dict(chromosome=str(c),block_bp0=[BEGIN+b*BLOCK,BEGIN+(b+1)*BLOCK],candidate_SNPs=0,
        primary_SNPs=0,sums=[0.,0.],valid_GP_counts=[0]*18,query_proxy_terms=[[0.]*10 for _ in range(2)]) for c in [20,21,22] for b in range(10)]
    try:
        out.put("candidate_roster.metadata.v1.tsv",roster_raw)
        out.js("started.json",dict(protocol_sha256=sha(praw),source_sha256=sha(Path(__file__).read_bytes()),started_utc=utc(),registered_hypothesis="H055"))
        out.put("dosage_records.bin",b"H055GP01"+hashlib.sha256(praw).digest()+b"\0"*24)
        for ai,asset in enumerate(cfg["assets"]):
            chrom=asset["chromosome"];need(chrom==str(20+ai),"Frozen chromosome order")
            net=Region(asset,before_request,out)
            net.head(asset["url"],asset["bytes"]);net.head(asset["index_url"],asset["index_bytes"])
            index=net.fetch(asset["index_url"],0,asset["index_bytes"],asset["index_bytes"])
            if asset.get("expected_index_sha256") is not None:need(sha(index)==asset["expected_index_sha256"],"Authenticated public index changed")
            chunks=chunks_tbi(index,chrom)
            windows={0}
            for a,b in chunks:windows.update(range((a>>16)//WINDOW,min(asset["bytes"]-1,(b>>16)+65535)//WINDOW+1))
            estimate=sum(min(WINDOW,asset["bytes"]-k*WINDOW) for k in windows)+asset["index_bytes"]
            need(estimate<=512*1024*1024,"Index plan exceeds transfer cap; root must review new version")
            hc={**cfg,"chromosome":chrom};indexes,hreport=header_v1(net.lines(0),subjects,hc)
            stream_hash=hashlib.sha256();last_vo=-1;last_pos=-1;pending=[]
            def emit(group):
                nonlocal nrows
                if not group:return
                if len(group)>1:
                    counts["duplicate_position_groups_excluded"]+=1;counts["duplicate_position_rows_excluded"]+=len(group);return
                raw=group[0];result,reason=site_v1(raw,indexes,chrom);counts[reason]+=1
                if result is None:return
                need(nrows<MAX_ROWS,"16MiB plan unsupported: >100000 eligible SNPs; no thinning")
                pos,r,a,mask,doses=result;b=(pos-1-BEGIN)//BLOCK;block=blocks[ai*10+b]
                block["candidate_SNPs"]+=1
                for j in range(18):block["valid_GP_counts"][j]+=int(bool(mask&(1<<j)))
                support=(mask&required)==required
                flags=1|(2 if support else 0)
                out.put("dosage_records.bin",RECORD.pack(int(chrom),b,pos,r,a,mask,flags,*doses));nrows+=1
                if not support:return
                block["primary_SNPs"]+=1
                change=sum(doses[j] for j in later)/10-sum(doses[j] for j in earlier)/10
                for k,p in enumerate(proxies):
                    diff=(doses[p]-doses[bar])/2
                    block["sums"][k]+=change*diff
                    for q,j in enumerate(query):block["query_proxy_terms"][k][q]+=doses[j]/2*diff
            for a,b in chunks:
                for vo,line in net.lines(a,b):
                    if vo<=last_vo:continue
                    last_vo=vo
                    if line.startswith(b"#"):continue
                    first=line.split(b"\t",2);need(len(first)==3,"VCF fixed fields")
                    c=first[0].decode();pos=int(first[1])
                    if c!=chrom or not BEGIN<=pos-1<END:continue
                    need(pos>=last_pos,"Source variant order")
                    counts["regional_source_rows"]+=1;stream_hash.update(line+b"\n")
                    if pending and pos!=last_pos:emit(pending);pending=[]
                    pending.append(line);need(len(pending)<=8,"Duplicate position group cap")
                    last_pos=pos
            emit(pending)
            net_report.append(dict(chromosome=chrom,objects=net.identities,index_sha256=sha(index),
                merged_virtual_chunks=chunks,planned_unique_range_bytes=estimate,received_bytes=net.received,
                requests=net.requests,expanded_bytes=net.expanded,regional_rows_sha256=stream_hash.hexdigest(),header=hreport))
        n=sum(b["primary_SNPs"] for b in blocks);need(n>0,"No primary support; no scientific estimate")
        chroms=[]
        for ai,c in enumerate([20,21,22]):
            ns=sum(b["primary_SNPs"] for b in blocks[ai*10:ai*10+10])
            need(ns>0,"Empty chromosome primary support is inconclusive, never replaced")
            sums=[sum(b["sums"][k] for b in blocks[ai*10:ai*10+10]) for k in range(2)]
            chroms.append(dict(chromosome=str(c),SNPs=ns,sums=sums,means=[s/ns for s in sums]))
        totals=[sum(b["sums"][k] for b in blocks) for k in range(2)]
        terms=[[sum(b["query_proxy_terms"][k][q] for b in blocks) for q in range(10)] for k in range(2)]
        leave_subject=[]
        for q,j in enumerate(query):
            vals=[]
            for k in range(2):
                es,ls=sum(terms[k][:5]),sum(terms[k][5:])
                vals.append(((ls/5-(es-terms[k][q])/4) if q<5 else ((ls-terms[k][q])/4-es/5))/n)
            leave_subject.append(dict(subject=subjects[j],site=rows[j]["site"],values=vals))
        leave_chrom=[dict(chromosome=x["chromosome"],values=[(totals[k]-x["sums"][k])/(n-x["SNPs"]) if n>x["SNPs"] else None for k in range(2)]) for x in chroms]
        leave_block=[dict(chromosome=b["chromosome"],block_bp0=b["block_bp0"],values=[(totals[k]-b["sums"][k])/(n-b["primary_SNPs"]) if n>b["primary_SNPs"] else None for k in range(2)]) for b in blocks if b["primary_SNPs"]]
        summary=dict(schema="H055_descriptive_GP_f4_v1",primary_SNPs=n,metadata_eligible_retained_SNPs=nrows,contrasts=cfg["contrasts"],
            pooled_equal_SNP_means=[x/n for x in totals],chromosomes=chroms,
            positive_direction_observed=[all(x["means"][k]>0 for x in chroms) and totals[k]>0 for k in range(2)],
            leave_subject_or_site_out=leave_subject,subject_site_units_identical=True,leave_chromosome_out=leave_chrom,leave_1Mb_block_out=leave_block,
            descriptive_chromosome_SD=[sd([x["means"][k] for x in chroms]) for k in range(2)],
            descriptive_subject_delete_SD=[sd([x["values"][k] for x in leave_subject]) for k in range(2)],
            descriptive_block_delete_range=[descriptive([x["values"][k] for x in leave_block]) for k in range(2)],
            no_SNP_independent_pvalues=True,no_ancestry_accuracy_or_mixture_fraction=True,shared_imputed_data_not_independent_truth=True)
        out.js("blocks.json",blocks);out.js("source_receipt.json",dict(chromosomes=net_report,counts=dict(counts),GP_only=True,AP_HS_GT_DS_used=False,publisher_checksum_verified=False))
        out.js("summary.json",summary)
        out.js("completed.json",dict(status="completed_registered_genotype_only_corroboration",completed_utc=utc(),protocol_sha256=sha(praw),
            retained_rows=nrows,record_bytes=RECORD.size,durable_bytes_before_completion=out.total,cap_bytes=CAP))
        out.close()
        return summary
    except BaseException:
        out.close()
        try:out.js("failed.json",dict(status="failed_partial_preserved_no_completed_estimate",error_type=__import__("sys").exc_info()[0].__name__));out.close()
        except BaseException:pass
        raise

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--describe",action="store_true");args=parser.parse_args()
    if not args.describe:parser.error("No acquisition CLI. Root must call reviewed run_v1 after H055 registration/admission.")
    print(json.dumps(dict(status="inactive_source_only",durable_cap_bytes=CAP,rows_cap=MAX_ROWS,record_bytes=RECORD.size,GP_only=True)))
