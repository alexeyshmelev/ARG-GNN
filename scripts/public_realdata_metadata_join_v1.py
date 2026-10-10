#!/usr/bin/env python
"""Bounded public H018 metadata acquisition/join only; no ancestry/genotype modelling."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,re,zipfile
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from xml.etree import ElementTree as ET
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=ROOT/".runtime/author_sources/public_realdata_metadata_join_v1"
RESULTS=ROOT/"experiments/results/public_realdata_metadata_join_v1"
SPRINGER="https://static-content.springer-cdn.com/esm/art%3A10.1038%2Fs41586-023-06865-0/MediaObjects/"
URLS={f"MOESM{i}.xlsx":SPRINGER+f"41586_2023_6865_MOESM{i}_ESM.xlsx" for i in (3,4,5,6)}
URLS["GLIMPSE_v1_tree.json"]="https://api.github.com/repos/odelaneau/GLIMPSE/git/trees/v1.0.0?recursive=1"
MAX_BYTES=10_000_000
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
 with Path(p).open("x") as f:f.write(json.dumps(data,indent=2,allow_nan=False)+"\n")
def fetch(attempt):
 BASE.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
 receipt={"kind":"public_realdata_metadata_join_v1_bounded_source_fetch","hypothesis_id":"H018","started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":"sample/date/contamination/relatedness tables and GLIMPSE source metadata only","requests":[],"full_chromosome_downloaded":False,"large_archive_downloaded":False,"ancestry_analysis_executed":False}
 for name,url in URLS.items():
  row={"name":name,"url":url,"maximum_bytes":MAX_BYTES,"requested_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
  try:
   if (BASE/name).exists():raise FileExistsError("Refuse to overwrite source bytes")
   with urlopen(Request(url,headers={"User-Agent":"ARG-GNN-H018-metadata/1","Accept":"*/*"}),timeout=20) as response:
    raw=response.read(MAX_BYTES+1)
    row.update(HTTP_status=response.status,final_url=response.url,headers={k:response.headers.get(k) for k in ("Content-Type","Content-Length","ETag","Last-Modified")})
    if len(raw)>MAX_BYTES:raise ValueError("Metadata response exceeded hard byte bound")
    (BASE/name).write_bytes(raw)
    row.update(status="received",path=str((BASE/name).relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  except Exception as e:row.update(status="failed",error_type=type(e).__name__,error=str(e))
  receipt["requests"].append(row)
  print(json.dumps(row),flush=True)
 receipt["finished_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
 save(RESULTS/("bounded_source_fetch_"+attempt+".json"),receipt)
def workbook_rows(p):
 ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
 with zipfile.ZipFile(p) as z:
  if sum(i.file_size for i in z.infolist())>150_000_000:raise ValueError("Expanded workbook byte bound")
  shared=[]
  if "xl/sharedStrings.xml" in z.namelist():
   shared=["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si",ns)]
  rels={x.attrib["Id"]:x.attrib["Target"] for x in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
  result={}
  for sheet in ET.fromstring(z.read("xl/workbook.xml")).findall("m:sheets/m:sheet",ns):
   target=rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
   member=target.lstrip("/") if target.startswith("/") else "xl/"+target
   rows=[]
   for row in ET.fromstring(z.read(member)).findall("m:sheetData/m:row",ns):
    cells={}
    for c in row.findall("m:c",ns):
     t=c.attrib.get("t");v=c.find("m:v",ns)
     value=v.text if v is not None else None
     if t=="s" and value is not None:value=shared[int(value)]
     elif t=="inlineStr":value="".join(c.find("m:is",ns).itertext())
     elif t not in ("str","e") and value is not None:
      try:value=float(value);value=int(value) if value.is_integer() else value
      except ValueError:pass
     if value is not None:cells[c.attrib["r"]]=value
    rows.append(cells)
   result[sheet.attrib["name"]]=rows
  return result
def inspect():
 for p in BASE.glob("springer*.xlsx"):
  print(p.name)
  for name,rows in workbook_rows(p).items():print(json.dumps({"sheet":name,"rows":len(rows),"first":rows[:4]},ensure_ascii=False))

def probe():
 import collections,zlib
 old=ROOT/".runtime/author_sources/ancestralpaths_public_realdata_inventory_v1"
 url="https://erda.ku.dk/archives/917f1ac64148c3800ab7baa29402d088/neo/genotypes-public/22.neo_impute_ph.vcf.gz"
 row={"kind":"public_realdata_metadata_join_v1_GT_HS_bounded_probe","hypothesis_id":"H018","url":url,"requested_range":"bytes=0-999999","started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"raw_VCF_or_AP_values_retained":False,"maximum_compressed_bytes":1000000,"maximum_decompressed_bytes":50000000,"record_limit":32}
 RESULTS.mkdir(parents=True,exist_ok=True)
 try:
  with urlopen(Request(url,headers={"User-Agent":"ARG-GNN-H018-metadata/1","Range":"bytes=0-999999"}),timeout=20) as response:
   row.update(HTTP_status=response.status,headers={k:response.headers.get(k) for k in ("Content-Length","Content-Range","ETag","Last-Modified")})
   if response.status!=206 or response.headers.get("Content-Range")!="bytes 0-999999/1654398745":raise ValueError("Exact bounded source206/length required")
   raw=response.read(1000001)
  if len(raw)!=1000000:raise ValueError("Bounded response byte count differs")
  row.update(received_bytes=len(raw),sha256_received_compressed_prefix=hashlib.sha256(raw).hexdigest())
  blocks=[];cursor=0;size=0
  while cursor+18<=len(raw):
   extra_len=int.from_bytes(raw[cursor+10:cursor+12],"little")
   extra=raw[cursor+12:cursor+12+extra_len];offset=0;blocksize=None
   while offset+4<=len(extra):
    key=extra[offset:offset+2];n=int.from_bytes(extra[offset+2:offset+4],"little")
    if key==b"BC" and n==2:blocksize=int.from_bytes(extra[offset+4:offset+6],"little")+1
    offset+=4+n
   if not blocksize or cursor+blocksize>len(raw):break
   data=zlib.decompress(raw[cursor:cursor+blocksize],wbits=31);size+=len(data)
   if size>50000000:raise ValueError("Decompressed metadata probe bound")
   blocks.append(data);cursor+=blocksize
  text=b"".join(blocks).decode();lines=text.splitlines()
  if not text.endswith("\n"):lines=lines[:-1]
  header="\n".join(line for line in lines if line.startswith("#"))+"\n"
  if header!=(old/"postglacial_chr22_VCF_header.txt").read_text():raise ValueError("Frozen complete header changed")
  samples=next(line for line in lines if line.startswith("#CHROM")).split("\t")[9:]
  passing=set((old/"postglacial_noflag.inds.txt").read_text().split())
  groups={"passing_ordinary":collections.Counter(),"other_ordinary":collections.Counter(),"pseudo_rnd":collections.Counter()}
  stats={s:collections.Counter() for s in samples if s in passing}
  positions=[];formats=set();hs_range=[]
  for line in (l for l in lines if not l.startswith("#")):
   fields=line.split("\t")
   if len(fields)!=9+len(samples):raise ValueError("Complete source variant/sample correspondence differs")
   fmt=fields[8].split(":");formats.add(fields[8]);positions.append(int(fields[1]))
   lookup={k:fmt.index(k) for k in ("GT","GP","HS") if k in fmt}
   for sample,value in zip(samples,fields[9:]):
    # FORMAT/AP remains uninterpreted and is never copied to an output.
    values=value.split(":",len(fmt)-1)
    group="pseudo_rnd" if sample.endswith("_rnd") else ("passing_ordinary" if sample in passing else "other_ordinary")
    counters=[groups[group]]+([stats[sample]] if sample in stats else [])
    gt=values[lookup["GT"]] if "GT" in lookup else "."
    kind="GT_missing" if "." in gt else ("GT_phased_diploid" if "|" in gt else ("GT_unphased_diploid" if "/" in gt else "GT_haploid"))
    for counter in counters:counter["calls"]+=1;counter[kind]+=1
    if "HS" in lookup and values[lookup["HS"]]!=".":
     hs=int(values[lookup["HS"]])
     if group=="passing_ordinary":hs_range.append(hs)
     for counter in counters:counter["HS_nonmissing"]+=1;counter["HS_valid30bits"]+=int(0<=hs<2**30)
    if "GP" in lookup and values[lookup["GP"]]!=".":
     gp=list(map(float,values[lookup["GP"]].split(",")))
     valid=len(gp)==3 and min(gp)>=0 and max(gp)<=1 and abs(sum(gp)-1)<=.002
     for counter in counters:
      counter["GP_nonmissing"]+=1;counter["GP_valid_triplet_rounded_tolerance"]+=int(valid)
      counter["GP_max_ge_0_80"]+=int(valid and max(gp)>=.8);counter["GP_max_ge_0_90"]+=int(valid and max(gp)>=.9)
   if len(positions)>=32:break
  if not positions:raise ValueError("No complete data rows within bounded probe")
  row.update(status="bounded_GT_HS_GP_metadata_observed",complete_records_used=len(positions),first_position_bp1=positions[0],last_position_bp1=positions[-1],FORMAT_layouts=sorted(formats),groups={k:dict(v) for k,v in groups.items()},passing_subject_call_QC={k:dict(v) for k,v in stats.items()},passing_HS_min=min(hs_range) if hs_range else None,passing_HS_max=max(hs_range) if hs_range else None,header_sha256=hashlib.sha256(header.encode()).hexdigest(),scope="first32 complete source records only; not chromosome/region/sample QC admission")
 except Exception as error:row.update(status="failed_preserved",error_type=type(error).__name__,error=str(error))
 row["finished_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
 save(RESULTS/"GT_HS_GP_bounded_probe.json",row)
 print(json.dumps({k:v for k,v in row.items() if k!="passing_subject_call_QC"}))

FROZEN_METADATA={
 "springer_MOESM3.xlsx":"2d77f3d228da0f1649f080f54fb26c8fe48866112527628764295fbd54756e52",
 "springer_MOESM4.xlsx":"cfeb1b729e300a59775cece61b1fb759cd47a829b21599286dafdb058c23039e",
 "springer_MOESM5.xlsx":"db8150d74679681acdf592245f5d8d70fe2c971b0774cbfb2a1831a135b37ba1",
 "springer_MOESM6.xlsx":"dbc4091808be4b648985f0e77a7215088acb2ac4aa48479b512d7a4199b6a828",
 "GLIMPSE_v1_tree.json":"b27b3729108b32e5be99d43f26dfa7e6486a3ebf237bb58b92346771d3b4e5f7",
 "GLIMPSE_phase_genotype_writer.cpp":"9d4a35a9c5224d52f766fb52a59a7ffe228a8c5f21770b36e72b11bf3b12d89b",
 "GLIMPSE_sample_algorithm.cpp":"40d5ba7df123722212363aad7db406a7750050a458464b41d0ac2b7952d9c1bf",
 "GLIMPSE_ligate_algorithm.cpp":"4fc5481d6b8ce84d6593313d0b8f02eaeda3031b3f90a3a527b4f30e048ed881"}
PROXY_GROUPS={
 "Ana_proxy":("Anatolia_Neolithic_Aceramic","Anatolia_Neolithic"),
 "CHG_proxy":("Georgia_Mesolithic",),
 "WHG_proxy":("Luxembourg_Mesolithic","Switzerland_Mesolithic","France_Mesolithic",
               "Iberia_Mesolithic","Italy_Mesolithic","Britain_Mesolithic"),
 "EHG_proxy":("Russia_Mesolithic",)}
def column(row):
 return {re.sub(r"\d","",k):v for k,v in row.items()}
def sample_id(value):
 # Excel contains ten integer subject IDs; normalize them explicitly to decimal
 # strings. The only stripped annotations are documented trailing star markers.
 if type(value) is int:return str(value)
 if not isinstance(value,str):raise ValueError("Unsupported subject identifier type")
 return value.rstrip("*").strip()
def table(num,sheet_index,header_row):
 p=BASE/f"springer_MOESM{num}.xlsx"
 if sha(p)!=FROZEN_METADATA[p.name]:raise ValueError("Frozen metadata drift")
 sheets=workbook_rows(p);name=list(sheets)[sheet_index];rows=sheets[name]
 headers=column(rows[header_row-1])
 return name,headers,[(int(next(iter(row)).lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),column(row))
                       for row in rows[header_row:] if row and "A" in column(row)]
def join():
 import collections,math
 for name,expected in FROZEN_METADATA.items():
  if sha(BASE/name)!=expected:raise ValueError("Bound source bytes changed: "+name)
 old=ROOT/".runtime/author_sources/ancestralpaths_public_realdata_inventory_v1"
 hp=old/"postglacial_chr22_VCF_header.txt";np=old/"postglacial_noflag.inds.txt"
 if sha(hp)!="936547916f31f512897b609858823c4ce3f76fecc1b749b77b1d8c64e04c4027" or sha(np)!="2191e8ac9bbed677ff0ac82c98d70d0358317952d172ff4ebd3958943f3c21d9":raise ValueError("Original subject/header source drift")
 samples=next(x for x in hp.read_text().splitlines() if x.startswith("#CHROM")).split("\t")[9:]
 passing=set(np.read_text().split());ordinary={s for s in samples if not s.endswith("_rnd")}
 name,headers,rows=table(6,0,11)
 ids=[sample_id(r["A"]) for _,r in rows]
 if len(ids)!=1664 or len(set(ids))!=1664 or len(samples)!=3331 or len(set(samples))!=3331:raise ValueError("Source subject cardinality/uniqueness differs")
 if {sample_id(r["A"]) for _,r in rows if r.get("R")==0}!=passing:raise ValueError("Flag0 and exact passing ID file disagree")
 supplemental={}
 for key,num,index,header in (("sample_details",3,0,15),("dating_context",4,0,8),
                              ("contamination",5,0,11)):
  s,h,rs=table(num,index,header);mapping=collections.defaultdict(list)
  for rownum,r in rs:
   mapping[sample_id(r["A"])].append({"sheet":s,"row":rownum,
     "fields":{h.get(c,c):v for c,v in r.items()},"documented_star_markers":str(r["A"]).count("*")})
  supplemental[key]=mapping
 related_name,related_headers,relrows=table(5,1,16)
 relations=[];adj=collections.defaultdict(set)
 for n,r in relrows:
  if "B" not in r:continue
  a,b=sample_id(r["A"]),sample_id(r["B"]);degree=r.get("AC")
  relations.append({"sample1":a,"sample2":b,"degree":degree,"KING_robust":r.get("J"),"source_sheet":related_name,"source_row":n})
  if degree==0 or degree==2 or str(degree).startswith("1"):
   adj[a].add(b);adj[b].add(a)
 joined=[]
 for rownum,r in rows:
  sid=sample_id(r["A"]);upper,lower,mean=(r.get(c) for c in ("I","J","K"))
  dated=all(isinstance(v,(int,float)) and math.isfinite(v) for v in (upper,lower,mean))
  dated=dated and upper>=mean>=lower>0
  eligible=(sid in passing and sid in ordinary and r.get("R")==0 and dated
            and isinstance(r.get("P"),(int,float)) and r["P"]>=.4
            and isinstance(r.get("Q"),(int,float)) and r["Q"]>=.98)
  row={"subject_id":sid,"source_ID_value":r["A"],"source_ID_type":type(r["A"]).__name__,
       "source_sheet":name,"source_row":rownum,
       "metadata":{headers[c]:v for c,v in r.items() if c in headers and len(c)==1 and "A"<=c<="R"},
       "ordinary_column_index0":samples.index(sid) if sid in ordinary else None,
       "pseudo_rnd_column_index0":samples.index(sid+"_rnd") if sid+"_rnd" in samples else None,
       "passing_ID_file":sid in passing,"valid_positive_BP_date_range":bool(dated),
       "metadata_QC_eligible":bool(eligible),"coverage_x":r.get("P"),"average_GP":r.get("Q"),
       "group_label":r.get("E"),"age_BP_95CI":[upper,lower],"age_BP_mean":mean,
       "dating_source":r.get("L"),"site":r.get("B"),"country":r.get("C"),
       "listed_close_relation_subjects":sorted(adj[sid]),
       "absence_of_listed_relationship_not_proof_of_unrelatedness":True}
  for key,mapping in supplemental.items():row[key]=mapping.get(sid,[])
  joined.append(row)
 avail={};selected=[];selection={}
 for label,groups in PROXY_GROUPS.items():
  candidates=[r for r in joined if r["metadata_QC_eligible"] and r["group_label"] in groups]
  candidates.sort(key=lambda r:(-r["coverage_x"],-r["average_GP"],r["subject_id"]))
  avail[label]={"metadata_QC_eligible_subjects":len(candidates),"candidate_IDs":[r["subject_id"] for r in candidates],"maximum_diploid_haplotypes_before_identity_or_region_QC":2*len(candidates)}
  sites=set();accepted=[]
  for r in candidates:
   sid=r["subject_id"];site=(r["country"],r["site"])
   if site in sites or adj[sid].intersection(selected):continue
   accepted.append(sid);selected.append(sid);sites.add(site)
   if len(accepted)==2:break
  if len(accepted)!=2:raise ValueError("Predeclared two-subject proxy panel unavailable: "+label)
  selection[label]=accepted
 queries=[r for r in joined if r["metadata_QC_eligible"] and r["group_label"]=="Denmark_Neolithic"]
 queries.sort(key=lambda r:(-r["coverage_x"],-r["average_GP"],r["subject_id"]))
 query_ids=[]
 for r in queries:
  sid=r["subject_id"]
  if sid in selected or adj[sid].intersection(selected+query_ids):continue
  query_ids.append(sid)
  if len(query_ids)==12:break
 if len(query_ids)!=12:raise ValueError("Predeclared twelve-query metadata cohort unavailable")
 endpoint={
  "kind":"public_realdata_metadata_join_v1_frozen_observation_sensitivity_endpoint",
  "hypothesis_id":"H018","status":"design_frozen_real_genotypes_ARGs_models_not_run",
  "reference_subjects":selection,"reference_subject_count":8,"reference_haplotypes":16,
  "query_subjects":query_ids,"query_haplotypes":24,
  "reference_labels":"chronogeographic proxies from source Group label only; not proven pure ancestry",
  "selection":{"noflag_ID_and_Flag0_required":True,"minimum_global_coverage_x":.4,"minimum_global_average_GP":.98,
   "dates":"valid positive cal.BP mean within95CI; source SupplementaryData7; no negative TableI/II BP copied",
   "ranking":"coverage descending, averageGP descending, exact subjectID ascending",
   "reference_maximum_subjects_per_source_archaeological_site":1,"listed_degree0_1_2_edges_forbidden_across_selected_subjects":True,
   "reference_subjects_per_proxy":2,"query_group":"Denmark_Neolithic","query_subjects":12,
   "AP_consulted":False,"ancestry_mixture_tables_or_IBD_cluster_labels_consulted":False},
  "region":{"assembly":"GRCh37/b37","chromosome":"22","interval_bp0":[20000000,22000000],
   "coordinate_transformation":"absolute_bp0 = VCF_POS_bp1 - 1; original_region_bp0 = absolute_bp0 - 20000000",
   "normalization_length_bp":2000000,"prediction_grid_region_bp0":"50,150,...,1999950",
   "support":"retained center100bp bins clipped to actual inferred coverage; exact query-bp; no tail extrapolation",
   "tiles_bp":250000,"map":"real GRCh37 chromosome22 recombination map must be separately obtained/versioned/hashed; no fabricated constant map"},
  "observation_QC":{"SNP":"biallelic A/C/G/T transversions only","1000G_strict_mask":"require presence of MASK_1000G Flag",
   "INFO_minimum":.8,"reference_panel_MAF":"min(RAF,1-RAF)>=0.05, finite NumberA scalar; never target AF",
   "genotype":"all20 selected subjects callable diploid, GP triplet finite[0,1], rounded sum within0.002, maxGP>=0.8; sites failing any selected subject excluded identically for all HS runs",
   "HS":"all20 have valid 0<=HS<2**(2*NMAIN), NMAIN15, decoded fixed iteration index per individual across entire ligated region",
   "HS_draw_indexes":[0,1,2,3,4,5,6,7],"GT_baseline":"verify actual phased separators in entire admitted region; preserve source GT",
   "ancestral_state":"pan_troglodytes must encode one A/C/G/T equal to REF or ALT; exclude missing/conflicting/third allele; do not call REF ancestral",
   "minimum_retained_SNPs":1000,"failure":"fail cohort/region admission; do not tune thresholds, duplicate references, fabricate phase, or use AP to recover sites"},
  "primary_endpoint":{"name":"median_query_posterior_observation_Jensen_Shannon_sensitivity_nats",
   "definition":"For each query subject, average its two haplotype C4 probability vectors to a diploid mean. For each HSdraw0..7 compare with frozen phased-GT run by Jensen-Shannon divergence using natural logs and0log0=0. Integrate over identical represented100bp bins, divide by exact scored diploid-query bp, average over8 draws, then take median across12 subjects.",
   "falsifiable_claim":"a fixed method/checkpoint is observation-stable on this frozen cohort/region if median<=0.05nats; otherwise claim fails",
   "threshold_nats":.05,"HS_changes":"both genotype and phase posterior uncertainty; never labelled pure phase error",
   "all_methods_reported_separately":True,"checkpoint_selection":"simulation-trained checkpoint identities frozen before real-data extraction; no real-data tuning/selection"},
  "secondary_descriptive":["all12 per-subject sensitivities and query-bp coverage","raw GT vs HS heterozygosity/dosage changes from admitted observations",
   "masked SNP/call counts and source global coverage strata","optional GP>=0.9 condition only under a separately declared common-support report; no primary redefinition"],
  "ARGformer_adaptation":"if within-ARG unsupervised adaptation is used, separately disclose/freeze steps/objective/seed; use no proxy/query ancestry targets or AP and report separately from direct simulation-trained inference",
  "AP_policy":"remove FORMAT/AP and header definitions before any genotype/tree/model artifact; no AP in references,QC,features,tuning,labels. Any future concordance requires separate comparator namespace opened only after outputs are sealed; cannot establish biological accuracy.",
  "no_true_local_ancestry_labels":True,"biological_accuracy_from_concordance":False,
  "blockers":["target_region_not_downloaded_or_genotype_QC_qualified","full_chromosome_publisher_checksum_unavailable",
   "genetic_map_and_polarization_region_inputs_not_frozen","actual_GT_HS_ligation_validity_not_verified_beyond_first32_records",
   "new_R16_condition_and_runtime_integration_required; original_R96_not_supported_by_metadata_panel",
   "metadata_QC_and_absence_of_close_pair_listing_do_not_prove_proxy_purity_or_all_individual_independence",
   "no_real_data_ARG_or_inference_pipeline_executed"]}
 RESULTS.mkdir(parents=True,exist_ok=True)
 save(RESULTS/"subject_metadata_join.json",{"kind":"public_realdata_metadata_join_v1_subjects","subjects":joined,"close_relationships":relations,"metadata_source_sha256":FROZEN_METADATA,"AP_inputs":False,"inferred_ancestry_cluster_fields_excluded":"SuppData7 columns S–V"})
 save(RESULTS/"frozen_endpoint.json",endpoint)
 receipt={"kind":"public_realdata_metadata_join_v1_completion","hypothesis_id":"H018","status":"metadata_join_and_falsifiable_endpoint_frozen_runtime_region_QC_pending",
  "completed_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"script_sha256":sha(Path(__file__)),
  "source_sha256":FROZEN_METADATA,"original_header_sha256":sha(hp),"passing_IDs_sha256":sha(np),
  "subject_count":len(joined),"passing_IDs":len(passing),"all_passing_IDs_joined":passing.issubset(set(ids)),
  "ordinary_VCF_columns":len(ordinary),"pseudo_rnd_columns":len(samples)-len(ordinary),
  "ordinary_without_metadata":sorted(ordinary-set(ids)),"metadata_without_ordinary":sorted(set(ids)-ordinary),
  "normal_without_rnd":sorted(s for s in ordinary if s+"_rnd" not in samples),
  "rnd_without_normal":sorted(s for s in samples if s.endswith("_rnd") and s[:-4] not in ordinary),
  "valid_positive_BP_date_ranges":sum(r["valid_positive_BP_date_range"] for r in joined),
  "invalid_or_missing_date_IDs":[r["subject_id"] for r in joined if not r["valid_positive_BP_date_range"]],
  "metadata_QC_eligible_total":sum(r["metadata_QC_eligible"] for r in joined),
  "published_close_relationship_pairs":len(relations),"proxy_availability":avail,
  "frozen_reference_subjects":selection,"frozen_query_subjects":query_ids,
  "subject_join":{"path":str((RESULTS/"subject_metadata_join.json").relative_to(ROOT)),"sha256":sha(RESULTS/"subject_metadata_join.json")},
  "endpoint":{"path":str((RESULTS/"frozen_endpoint.json").relative_to(ROOT)),"sha256":sha(RESULTS/"frozen_endpoint.json")},
  "full_chromosome_downloaded":False,"large_archive_downloaded":False,"AP_paintings_consulted":False,
  "ancestry_analysis_executed":False,"GPUs_used":False,"truth_or_simulation_scores_accessed":False,
  "blockers":endpoint["blockers"]}
 save(RESULTS/"completion.json",receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in {"source_sha256","invalid_or_missing_date_IDs","proxy_availability"}}))
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--phase",choices=("fetch","inspect","probe","join"),required=True)
 a.add_argument("--attempt",default="v1");args=a.parse_args()
 if Path.cwd().resolve()!=ROOT:raise RuntimeError("Dedicated anogena project root required")
 if args.phase=="fetch":
  if args.attempt=="v2":
   URLS.update({f"MOESM{i}.xlsx":f"https://pmc.ncbi.nlm.nih.gov/articles/instance/10781627/bin/41586_2023_6865_MOESM{i}_ESM.xlsx" for i in (3,4,5,6)})
  if args.attempt=="v3":
   URLS={f"springer_MOESM{i}.xlsx":f"https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06865-0/MediaObjects/41586_2023_6865_MOESM{i}_ESM.xlsx" for i in (3,4,5,6)}
   URLS.update({name:"https://raw.githubusercontent.com/odelaneau/GLIMPSE/v1.0.0/"+source for name,source in {"GLIMPSE_phase_genotype_writer.cpp":"phase/src/io/genotype_writer.cpp","GLIMPSE_sample_algorithm.cpp":"sample/src/sampler/sampler_algorithm.cpp","GLIMPSE_ligate_algorithm.cpp":"ligate/src/ligater/ligater_algorithm.cpp"}.items()})
  fetch(args.attempt)
 elif args.phase=="join":join()
 elif args.phase=="probe":probe()
 else:inspect()
