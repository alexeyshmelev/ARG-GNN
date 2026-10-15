"""Root-owned bounded acquisition of the official prospective chr22 map asset."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,io,json,math,os,resource,sys,tarfile,traceback,urllib.request
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/ARGMix_prospective1611_map_root_20261016_v2"
SOURCE=REL+"/acquire_map_asset_v2.py"
OUT="data/ARGMix_prospective1611_HapMapII_GRCh37_chr22_v1"
CATALOG=".runtime/env/lib/python3.11/site-packages/stdpopsim/catalog/HomSap/genetic_maps.py"
URL="https://stdpopsim.s3-us-west-2.amazonaws.com/genetic_maps/HomSap/HapmapII_GRCh37_RecombinationHotspots.tar.gz"
ARCHIVE_SHA="80f22d9e6cb0e497074ed1bc277e765fa9d8e22f21b2f66c3b10286520f6b68f"
SOURCE_SHA="fe688fb1722c0f2aaeb691d813ad5d759fd033e345369e50694f5cb04b12d8a8"
INTERVAL=(16051347,51229805)
RENEWAL_CERTIFICATE_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json', 'sha256': '66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7'}
RENEWAL_SOURCE_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py', 'sha256': '491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468'}
RENEWAL_BOOTSTRAP_REF={'path': 'research/reviews/H039_marker_repair_renewal_source_20261016_v1/bootstrap_renewal_v1.py', 'sha256': 'f4ae97eeaf244373d1fcd7922e2235d08225468fe5006449d189cc461317fff2'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 need(p.resolve()==p and p.is_relative_to(ROOT),"Exact canonical project path")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,obj):
 raw=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
 with p.open("xb") as f:f.write(raw)
 return ref(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--execute",action="store_true");a=parser.parse_args()
 if not a.execute:
  print(json.dumps(dict(status="source_only",source_url=URL,archive_sha256=ARCHIVE_SHA,source_interval_bp=INTERVAL)));return 0
 need(Path.cwd()==ROOT,"Exact repository cwd")
 packet=ROOT/REL/"admitted";need(not packet.exists() and not (ROOT/OUT).exists(),"Fresh asset only, no replay")
 packet.mkdir()
 config=save(packet/"config.json",dict(schema="prospective_chr22_map_acquisition_v1",URL=URL,archive_sha256=ARCHIVE_SHA,source_sha256=SOURCE_SHA,source_interval_bp=INTERVAL,
  output=OUT,reserved_bytes=67108864,max_RSS_bytes=536870912,network_timeout_seconds=45,maximum_download_bytes=67108864))
 sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2")]
 renewal_path=ROOT/RENEWAL_SOURCE_REF["path"]
 need(ref(renewal_path)==RENEWAL_SOURCE_REF,"Exact reviewed renewal source")
 renewal_spec=importlib.util.spec_from_file_location("_fresh_map_renewal_v2",renewal_path)
 renewal=importlib.util.module_from_spec(renewal_spec);renewal_spec.loader.exec_module(renewal)
 renewal.install_renewal_v1(RENEWAL_CERTIFICATE_REF)
 import root_contract_v1 as c
 guard=c.original_guard()
 with guard.budget.v4.mutex(ROOT):state=guard._state_locked(c.CORE,c.BASE,fresh=True)
 bindings=list(state[3].values())+[ref(SOURCE),config,ref(CATALOG),c.GUARD_REF,c.POLICY_REF,c.REPAIR_REF,RENEWAL_SOURCE_REF,RENEWAL_BOOTSTRAP_REF,RENEWAL_CERTIFICATE_REF]
 bindings=list({(r["path"],r["sha256"]):r for r in bindings}.values())
 expiry=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
 request=dict(key="DATA/ARGMix-prospective1611-chr22-map-v2",output=str(ROOT/OUT),auxiliary_scopes=[str(packet/"worker_aux")],reserved_bytes=67108864,consumer_bindings=bindings)
 auth=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3]);auth.update(active=True,expires_at_utc=expiry)
 ar=save(packet/"reserve_authority.json",auth)
 reserved=guard.apply_reserve_v1(ar,expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
 save(packet/"reserve_receipt.json",reserved)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,job_id="ARGMix-prospective1611-chr22-map-v2",
  output_directory=OUT,reserved_new_bytes=67108864,reservation=reserved["value"],initial_ledger_pins=reserved["initial_ledger_pins"],
  expires_at_utc=expiry,completed_overlay_policy_ref=c.POLICY_REF,
  budget_components=dict(implementation=c.ref(guard.budget.SOURCE_REL),launcher=c.ref(guard.budget.LAUNCHER_REL),consumer_adapter=c.GUARD_REF))
 storage_ref=save(packet/"storage.json",storage)
 print(json.dumps(dict(event="map_asset_admitted",storage_ref=storage_ref,projected_bytes=reserved["projected_bytes"])),flush=True)
 catalog=(ROOT/CATALOG).read_text();need(URL.split("/")[-1] in catalog and ARCHIVE_SHA in catalog and "HapMapII_GRCh37" in catalog,"Installed official catalog association")
 with urllib.request.urlopen(URL,timeout=45) as r:archive=r.read(67108865)
 need(len(archive)<=67108864 and hashlib.sha256(archive).hexdigest()==ARCHIVE_SHA,"Exact complete official archive")
 with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as tar:
  members=[m for m in tar.getmembers() if m.name=="genetic_map_GRCh37_chr22.txt"]
  need(len(members)==1 and members[0].isfile() and members[0].size<=8000000,"Exact bounded regular chr22 member")
  raw=tar.extractfile(members[0]).read()
 need(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,"Exact authenticated chr22 source")
 lines=raw.decode().splitlines();need(lines[0]=="Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)","Explicit source columns")
 rows=[]
 for line in lines[1:]:
  chrom,pos,rate,cm=line.split()
  need(chrom=="chr22","Only chromosome22")
  rows.append((int(pos),float(cm)))
 need((rows[0][0],rows[-1][0])==INTERVAL,"Frozen full source support")
 need(all(rows[i+1][0]>p and rows[i+1][1]>=cm for i,(p,cm) in enumerate(rows[:-1])),"Monotone physical/cumulative coordinates")
 origin=rows[0][1];length=INTERVAL[1]-INTERVAL[0]
 text=["position COMBINED_rate(cM/Mb) Genetic_Map(cM)"]
 for i,(pos,cm) in enumerate(rows):
  rate=0.0 if i==len(rows)-1 else 1e6*(rows[i+1][1]-cm)/(rows[i+1][0]-pos)
  need(math.isfinite(rate) and rate>=0,"Finite nonnegative derived rate")
  text.append(f"{pos-INTERVAL[0]} {rate:.17g} {cm-origin:.17g}")
 rebased=("\n".join(text)+"\n").encode()
 need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=536870912,"512MiB peak RSS ceiling")
 output=ROOT/OUT;output.mkdir(exist_ok=True)
 guard.budget_check_v9(storage,boundary=True,required_write_bytes=len(archive)+len(raw)+len(rebased)+1048576)
 for name,data in [("HapmapII_GRCh37_RecombinationHotspots.tar.gz",archive),("genetic_map_GRCh37_chr22.txt",raw),("chr22.rebased.map",rebased)]:
  with (output/name).open("xb") as f:f.write(data)
 binding=dict(schema="prospective_chr22_GRCh37_map_binding_v1",chromosome="22",genome_build="GRCh37",length_bp=length,source_interval_bp=list(INTERVAL),source_url=URL,source_sha256=SOURCE_SHA,
  rebased_map_path=OUT+"/chr22.rebased.map",rebased_map_sha256=hashlib.sha256(rebased).hexdigest())
 br=save(output/"map_binding.json",binding)
 receipt=dict(schema="prospective_chr22_GRCh37_map_derivation_receipt_v1",root_authorizer="root_agent",root_approved=True,complete=True,
  source_url=URL,archive_path=OUT+"/HapmapII_GRCh37_RecombinationHotspots.tar.gz",archive_sha256=ARCHIVE_SHA,
  source_member_name="genetic_map_GRCh37_chr22.txt",chromosome="22",genome_build="GRCh37",source_map_path=OUT+"/genetic_map_GRCh37_chr22.txt",source_map_sha256=SOURCE_SHA,
  source_column_indices=dict(chromosome=0,position=1,rate=2,cumulative_cM=3),source_header=lines[0].split("\t"),
  rebased_column_indices=dict(position=0,rate=1,cumulative_cM=2),
  derivation_algorithm="linear_cumulative_interpolation_clip_rebase_v1",
  simulation_rate_semantics="piecewise_constant_between_breakpoints_from_cumulative_cM_slope",
  physical_coordinate_transform="rebased_bp=source_bp-interval_start_bp; no source coordinate shift",
  source_interval_bp=list(INTERVAL),length_bp=length,rebased_map_path=binding["rebased_map_path"],rebased_map_sha256=binding["rebased_map_sha256"],map_binding_ref=br,
  derivation="Full authenticated source support, retain all breakpoints, subtract first bp and first cumulative cM, outgoing rate=1e6*delta_cM/delta_bp, terminal rate0. Simulation rates=delta_cM/(100*delta_bp). No extrapolation.",
  first_rebased_bp=0,last_rebased_bp=length,genetic_length_cM=rows[-1][1]-origin,row_count=len(rows),official_catalog_ref=ref(CATALOG),source_ref=ref(SOURCE),storage_ref=storage_ref,renewal_certificate_ref=RENEWAL_CERTIFICATE_REF,
  original_paper_interval_reproduction_claimed=False,primary_prospective_interval=True,completed_utc=datetime.now(timezone.utc).isoformat())
 rr=save(output/"derivation_receipt.json",receipt)
 guard.budget_check_v9(storage,boundary=True)
 cr=save(output/"completed.json",dict(status="completed_prospective_chr22_map_acquisition",source_ref=ref(SOURCE),config_ref=config,storage_ref=storage_ref,derivation_receipt_ref=rr,root_pid=os.getpid(),no_workers_spawned=True,scientific_simulation_performed=False))
 print(json.dumps(dict(event="official_map_asset_completed",completed_ref=cr,derivation_receipt_ref=rr,source_interval_bp=INTERVAL,length_bp=length)),flush=True)
 return 0
if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception:
  traceback.print_exc(file=sys.stdout)
  raise SystemExit(1)
