"""Prospective full-panel observed-only Relate adapter. Root owns all execution."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, os, sys
from types import SimpleNamespace
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
GEN=ROOT/"research/reviews/ARGMix_prospective1611_generation_20261016_v1"
ADAPTER=ROOT/"research/reviews/ARGMix_prospective1611_native_adapters_20261016_v1"
PLAN=ROOT/"research/reviews/ARGMix_published_panel_fair_comparison_source_plan_20261016_v1/prospective_panel_config.proposed.inactive.json"
SEEDS=ROOT/"research/reviews/ARGMix_published_panel_next_source_20261015_v1/published_panel_plan.v1.json"
MAP_SHA="72453ecd3fc5940d4d3fd560f7897342b10e608cc81d6155b24f21f4d57e88e6"
N=3222;D=1611;L=35178458
PUBLIC={"observed.vcf","observed.haps","observed.sample","variants.tsv","haplotypes.tsv","roster","map_binding"}
REQUEST_FIELDS={"schema","root_approved","root_authorizer","registered_before_execution","parent_id","source_ref","source_pins_ref",
    "observed_receipt_ref","map_derivation_receipt_ref","runtime_binding_ref","caps_ref","resource_binding_ref","runner_factory_ref","output_directory"}
PARENTS=tuple("published_recon_train_%03d"%i for i in range(5))+("published_recon_val_000",)+tuple("published_recon_iid_%03d"%i for i in range(3))

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project path")
    return p

def ref(p):
    p=local(p);need(p.is_file() and not p.is_symlink(),"Regular bound file")
    return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))

def short(r):return {k:r[k] for k in ("path","sha256")}

def checked(r):
    need(isinstance(r,dict) and set(r) in ({"path","sha256"},{"path","sha256","bytes"}),"Exact file reference")
    p=local(r["path"]);actual=ref(p);need(actual["sha256"]==r["sha256"] and ("bytes" not in r or actual["bytes"]==r["bytes"]),"Bound bytes changed")
    return p

def same_ref(a,b):return local(a["path"])==local(b["path"]) and a["sha256"]==b["sha256"]

def read(r,limit=2097152):
    p=checked(r);need(p.stat().st_size<=limit,"Bounded JSON metadata")
    return json.loads(p.read_text())

def write(p,value):
    with Path(p).open("x") as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def load(r,name):
    p=checked(r);sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def verify_sources():
    pins=json.loads((HERE/"source_pins.v1.json").read_text())
    for r in pins["sources"]:checked(r)
    return pins

def panel_module():
    verify_sources();sys.path.insert(0,str(ADAPTER))
    import panel_adapter
    need(Path(panel_adapter.__file__).resolve()==ADAPTER/"panel_adapter.py","Exact frozen panel adapter")
    return panel_adapter

def registered(parent):
    need(parent in PARENTS,"Registered TRAIN/VAL/IID only; OOD not implemented")
    doc=json.loads(SEEDS.read_text());rows=[r for r in doc["parents"] if r["parent_id"]==parent]
    need(len(rows)==1,"Exact existing seed registry parent")
    return rows[0]

def sample_text(ids):return "ID_1 ID_2 missing\n0 0 0\n"+"".join(x+" "+x+" 0\n" for x in ids)

def canonical_sample(raw,ids):
    lines=[x.split() for x in raw.splitlines() if x.strip()]
    need(lines[:2]==[["ID_1","ID_2","missing"],["0","0","0"]] and len(lines)==len(ids)+2,"Native .sample headers/count")
    for fields,did in zip(lines[2:],ids):
        need(len(fields)==3 and fields[2]=="0" and ((fields[0]==did and fields[1] in (did,"NA")) or (fields[1]==did and fields[0]=="NA")),"Native sample identities/order changed")
    return sample_text(ids)

def roster_order(document):
    rows=document["rows"];need(len(rows)==N,"All3222 observed haplotypes")
    ids=[];ages=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        need(a["diploid_id"]==b["diploid_id"] and (a["phase"],b["phase"])==(0,1)
            and a["age_generations"]==b["age_generations"],"Adjacent paired phase0/1 roster")
        ids.append(a["diploid_id"]);ages.extend([a["age_generations"],b["age_generations"]])
    need(len(ids)==len(set(ids))==D and all(type(x) is int and 0<=x<=300 for x in ages),"1611 unique diploids and integer roster ages")
    return ids,ages

def shifted_map(rows,length):
    need(len(rows)>=2 and rows[0][0]==0 and rows[-1][0]==length and rows[0][2]==0,"Exact full0-based cumulative map support")
    previous=-1;cm=-1
    out=["pos COMBINED_rate Genetic_Map"]
    for x,rate,g in rows:
        need(type(x) is int and previous<x<=length and math.isfinite(rate) and rate>=0 and math.isfinite(g) and g>=cm,"Strict physical/nondecreasing cumulative map")
        out.append(f"{x+1} {rate:.17g} {g:.17g}");previous=x;cm=g
    return "\n".join(out)+"\n"

def parse_map(path):
    with Path(path).open() as f:
        need(f.readline().split()==["position","COMBINED_rate(cM/Mb)","Genetic_Map(cM)"],"Canonical persisted map header")
        rows=[]
        for line in f:
            fields=line.split();need(len(fields)==3,"Three map fields")
            x,rate,cm=map(float,fields);need(x==int(x),"Integral map physical coordinate");rows.append((int(x),rate,cm))
    return rows

def checked_haps(fields,n):
    need(len(fields)==5+n and fields[0]=="22" and fields[3:5]==["A","C"] and all(x in ("0","1") for x in fields[5:]),"Exact binary phased haps schema")
    p=int(fields[2]);need(1<=p<=L,"Haps POS1 in full original domain")
    return p

def validate_converted(original,converted,vcf,variants,ids):
    """Stream all observed rows/alleles; no row loss, flip, reorder or rescue."""
    count=0;previous=0;row_digest=hashlib.sha256()
    with Path(original).open() as a,Path(converted).open() as b,Path(vcf).open() as v,Path(variants).open() as vm:
        header=None
        for line in v:
            if line.startswith("#CHROM"):
                header=line.rstrip().split("\t");break
            need(line.startswith("#"),"VCF ordered header before data")
        need(header is not None and header[9:]==ids,"Exact1611 VCF diploid IDs/order")
        vr=csv.DictReader(vm,delimiter="\t")
        for left in a:
            x=left.split();right=b.readline();need(bool(right),"Native haps dropped rows");y=right.split()
            p=checked_haps(x,N);need(y==x and p>previous,"Native haps changed ordered fields, phase, allele or SNP")
            z=v.readline().rstrip().split("\t");need(len(z)==9+D and z[:5]==["22",str(p),x[1],"A","C"] and z[8]=="GT","VCF/hap exact variant join")
            gt=[s for g in z[9:] for s in g.split("|")]
            need(len(gt)==N and gt==x[5:] and all(g in ("0|0","0|1","1|0","1|1") for g in z[9:]),"All phased VCF calls match haps")
            m=next(vr,None);need(m is not None and m["variant_id"]==x[1] and int(m["position_bp_zero_based"])==p-1
                and int(m["vcf_position_one_based"])==p and int(m["source_position_bp"])==16051347+p-1
                and m["REF"]=="A" and m["ALT"]=="C" and int(m["observed_alt_haplotype_count"])==sum(map(int,x[5:])),"Exact variant coordinate/count manifest")
            need(0<sum(int(t) for t in x[5:])<N,"No additional invariant filtering allowed")
            row_digest.update((" ".join(x)+"\n").encode());previous=p;count+=1
        need(count>0 and not b.readline() and not v.readline() and next(vr,None) is None,"All native/VCF/variant rows retained exactly")
    return dict(variant_count=count,last_POS1=previous,canonical_haps_rows_sha256=row_digest.hexdigest(),all_N_phased_calls_verified=True)

def commands(tools,inputs,work,parent,caps):
    seed=registered(parent)["new_seed_values"]["relate_initial"]
    f=str(tools["RelateFileFormats"]);native=str(tools["Relate"])
    prefix=str(inputs/"input")
    return [
      ("convert_from_vcf",[f,"--mode","ConvertFromVcf","--haps",prefix+".haps","--sample",str(inputs/"converted.sample"),"-i",str(caps["observed_vcf_prefix"])]),
      ("annotate",[f,"--mode","GenerateSNPAnnotations","--haps",prefix+".haps","--sample",prefix+".sample","--poplabels",str(inputs/"input.poplabels"),"-o",prefix]),
      ("infer_all",[native,"--mode","All","-m","1.25e-8","-N","30000","--memory","2","--seed",str(seed),
        "--haps",prefix+".haps","--sample",prefix+".sample","--sample_ages",str(inputs/"sample_ages.generations"),
        "--map",str(inputs/"input.map"),"--annot",prefix+".annot","-o","inferred"]),
      ("convert_to_trees",[f,"--mode","ConvertToTreeSequence","-i","inferred","-o","converted"])
    ]

def invoke(runner,label,argv,work,caps,resource_ref,*,fixture=False):
    evidence=runner.run_stage_v1(label=label,argv=argv,cwd=str(work),caps=caps,resource_binding_ref=resource_ref)
    need(evidence.get("argv")==argv and evidence.get("cwd")==str(work) and evidence.get("stage")==label,"Exact executed command binding")
    need(evidence.get("actual_direct_native_wait") is True and evidence.get("returncode")==0
        and evidence.get("all_owned_processes_and_groups_drained") is True and evidence.get("operational_failure") is None,"Genuine native direct wait0/owned drain")
    if not fixture:
        need(evidence.get("fixture_only") is not True and type(evidence.get("actual_OS_wait_status")) is int
            and os.waitstatus_to_exitcode(evidence["actual_OS_wait_status"])==0
            and evidence.get("resource_binding_ref")==resource_ref,"Actual OS status and resource binding; fixture receipts forbidden")
    return evidence

def valid_caps(caps):
    need(caps["schema"]=="prospective1611_Relate_operational_caps_v1" and caps["cpu_threads"]==1,"One CPU root-bound Relate caps")
    for k in ("rss_bytes","wall_seconds","scope_bytes","input_bytes","native_bytes","export_bytes","receipt_bytes","log_bytes"):
        need(type(caps[k]) is int and caps[k]>0,"Positive explicit operational cap "+k)
    need(sum(caps[k] for k in ("input_bytes","native_bytes","export_bytes","receipt_bytes","log_bytes"))<=caps["scope_bytes"],"Contained storage categories")
    return caps

def input_request(record):
    verify_sources();r=read(record)
    need(set(r)==REQUEST_FIELDS and r["schema"]=="prospective1611_Relate_root_request_v1"
        and r["root_approved"] is True and r["root_authorizer"]=="root_agent" and r["registered_before_execution"] is True,"Exact root-approved observed-only request")
    need(short(ref(HERE/"relate_adapter_v1.py"))==r["source_ref"] and short(ref(HERE/"source_pins.v1.json"))==r["source_pins_ref"],"Exact source pins")
    registered(r["parent_id"]);plan=json.loads(PLAN.read_text())
    need(plan["relate"]["mode"]=="All" and plan["relate"]["N"]==30000 and plan["relate"]["mu"]==1.25e-8,"Frozen prospective All/N/mu condition")
    obs=read(r["observed_receipt_ref"]);need(obs["schema"]=="prospective1611_shared_observed_generation_v1" and obs["complete"] is True
        and obs["parent_id"]==r["parent_id"] and obs["diploids"]==D and obs["haplotypes"]==N
        and obs["contains_genealogy_or_founder_truth"] is False and set(obs["files"])==PUBLIC,"Actual complete observed-only full panel")
    bank=checked(r["observed_receipt_ref"]).parent
    paths={}
    for k,f in obs["files"].items():
        paths[k]=checked(f);need(paths[k].parent==bank,"Exact public bank whitelist; no private path capabilities")
    need(r["map_derivation_receipt_ref"]["sha256"]==MAP_SHA and obs["map_derivation_receipt_sha256"]==MAP_SHA,"Actual full primary map receipt")
    md=read(r["map_derivation_receipt_ref"])
    need(md["complete"] is True and md["length_bp"]==L and md["source_interval_bp"]==[16051347,51229805],"Full requested physical interval unchanged")
    checked(dict(path=md["rebased_map_path"],sha256=md["rebased_map_sha256"]))
    runtime=read(r["runtime_binding_ref"]);need(runtime["schema"]=="prospective1611_cpu_runtime_binding_v1" and runtime["root_qualified"] is True and runtime["cpu_only"] is True,"Actual qualified CPU runtime")
    need(sha(Path(sys.executable).resolve())==runtime["python_executable_sha256"],"Actual qualified Python")
    caps=valid_caps(read(r["caps_ref"]));res=read(r["resource_binding_ref"])
    output=local(r["output_directory"])
    need(res["schema"]=="prospective1611_Relate_root_resource_binding_v1" and res["root_authorizer"]=="root_agent" and res["root_enforced"] is True
        and res["outer_process_limits_enforced"] is True and res["output_directory"]==str(output)
        and res["caps_ref"]==r["caps_ref"] and res["runner_factory_ref"]==r["runner_factory_ref"] and res["adapter_source_ref"]==r["source_ref"],"Root admission/outer+native resource enforcement binding")
    admission=read(res["admission_receipt_ref"])
    need(admission["kind"]=="concurrent_completed_scope_reserve_receipt_v1" and admission["value"]["output"]==str(output)
        and admission["value"]["reserved_bytes"]==caps["scope_bytes"] and r["source_ref"] in admission["value"]["consumer_bindings"]
        and r["runner_factory_ref"] in admission["value"]["consumer_bindings"],"Actual original guard allocation already owned by root launcher")
    need(not output.exists(),"Fresh one-parent inference output; no rerun/resume/downscale")
    return r,obs,paths,md,runtime,caps,output

def normalize(raw,roster,expected):
    """Shift physical origin only; preserve all native node times/topologies."""
    need(raw.num_samples==N,"Native all3222 sample nodes retained")
    samples=list(map(int,raw.samples()));rows=roster["rows"]
    need(all(float(raw.node(n).time)==row["age_generations"] for n,row in zip(samples,rows)),"Native ordered haplotype ages retained exactly")
    positions=list(raw.tables.sites.position)
    need(len(positions)==expected["variant_count"] and positions and positions[-1]==expected["last_POS1"]
        and raw.sequence_length==expected["last_POS1"]+1,"Actual raw native last-POS support")
    need(all(float(x)==int(x) and 1<=x<=L for x in positions),"Raw integral POS1 sites")
    right=int(expected["last_POS1"]);need(0<right<=L,"Actual supported right endpoint")
    tables=raw.dump_tables();lefts=[];rights=[];parents=[];children=[]
    for edge in raw.edges():
        need(raw.node(edge.parent).time>raw.node(edge.child).time,"Positive native chronology; no adapter date repair")
        need(not edge.metadata,"Native edges expected without metadata")
        left=max(0.,float(edge.left)-1);end=float(edge.right)-1
        need(end<=right,"Raw native edge beyond measured support")
        if end>left:lefts.append(left);rights.append(end);parents.append(edge.parent);children.append(edge.child)
    tables.edges.clear();tables.edges.set_columns(left=lefts,right=rights,parent=parents,child=children)
    tables.sites.position=[float(x)-1 for x in positions];tables.sequence_length=right;tables.sort()
    ts=tables.tree_sequence()
    need(list(map(int,ts.samples()))==samples and ts.num_nodes==raw.num_nodes,"Actual sample IDs/all native nodes retained")
    for tree in ts.trees():
        need(tree.num_roots==1 and tree.num_samples(tree.root)==N and all(tree.is_leaf(s) for s in samples),"Every supported tree retains connected sampled-leaf topology")
    return ts,dict(schema="prospective1611_Relate_physical_support_v1",requested_interval_bp=[0,L],supported_scoring_interval_bp=[0,right],
        unobserved_terminal_interval_bp=[right,L],supported_bp=right,unobserved_terminal_bp=L-right,
        normalized_sequence_length_bp=right,full_requested_span_covered=right==L,
        terminal_topology_extended=False,first_artificial_prefix_clipped_bp=1,physical_origin_bp=0,
        transform="site=x_POS1-1; edge_left=max(0,raw_left-1); edge_right=raw_right-1; no new terminal edges",
        native_converter_chronology="author converter output preserved exactly; adapter applies no time repair")

def verify_native_genotypes(ts,haps):
    count=0
    with Path(haps).open() as f:
        for variant in ts.variants(isolated_as_missing=False):
            fields=f.readline().split();need(bool(fields),"Native additional inferred site")
            need(variant.site.position==int(fields[2])-1,"Normalized ordered site position differs")
            need(all(a in ("A","C") for a in variant.alleles),"Native nucleotide alleles differ")
            observed=[variant.alleles[int(g)] for g in variant.genotypes]
            need(observed==["A" if x=="0" else "C" for x in fields[5:]],"Native converter changed any observed haploid call")
            count+=1
        need(not f.readline(),"Native conversion lost observed sites")
    return count

def infer(record,runner=None):
    r,obs,paths,md,runtime,caps,output=input_request(record)
    p=panel_module();roster=json.loads(paths["roster"].read_text());p.Panel(roster,require_bound=False);ids,ages=roster_order(roster)
    need(paths["observed.sample"].read_text()==sample_text(ids),"Exact original paired same-ID .sample")
    with paths["haplotypes.tsv"].open() as f:
        records=list(csv.DictReader(f,delimiter="\t"))
    need(len(records)==N and all(int(x["haplotype_column_zero_based"])==i and x["haplotype_id"]==row["haplotype_id"]
        and x["diploid_id"]==row["diploid_id"] and int(x["phase"])==row["phase"] and int(x["age_generations"])==row["age_generations"]
        for i,(x,row) in enumerate(zip(records,roster["rows"]))),"Actual ordered haplotype/phase/age identity manifest")
    pins=json.loads((HERE/"source_pins.v1.json").read_text());tools={k:checked(v) for k,v in pins["native_binary_refs"].items()}
    if runner is None:runner=load(r["runner_factory_ref"],"_prospective1611_root_runner").create_runner_v1(r)
    output.mkdir();inputs=output/"inputs";work=output/"native";export=output/"observed"
    for d in (inputs,work,export):d.mkdir()
    (inputs/"sample_ages.generations").write_text("".join(str(x)+"\n" for x in ages))
    (inputs/"input.poplabels").write_text("sample population group sex\n"+"".join(f"{row['diploid_id']} {row['cohort']} {row['cohort']} NA\n" for row in roster["rows"][::2]))
    maptext=shifted_map(parse_map(local(md["rebased_map_path"])),L);(inputs/"input.map").write_text(maptext)
    stage_caps=dict(caps,observed_vcf_prefix=str(paths["observed.vcf"].with_suffix("")));cmd=commands(tools,inputs,work,r["parent_id"],stage_caps)
    journal=[]
    journal.append(invoke(runner,*cmd[0],work,caps,r["resource_binding_ref"]))
    canonical=canonical_sample((inputs/"converted.sample").read_text(),ids);(inputs/"input.sample").write_text(canonical)
    check=validate_converted(paths["observed.haps"],inputs/"input.haps",paths["observed.vcf"],paths["variants.tsv"],ids)
    need(check["variant_count"]==obs["observed_sites"],"All observed sites retained through native VCF conversion")
    for label,argv in cmd[1:]:journal.append(invoke(runner,label,argv,work,caps,r["resource_binding_ref"]))
    write(output/"commands.json",dict(schema="prospective1611_Relate_native_commands_v1",stages=journal,author_tools_unchanged=True,
        algorithm="Relate1.2.4 All; fixedN30000;mu1.25e-8;memory2GB;registered relate_initial seed; no size iteration/postprocessing"))
    import tskit
    check_module(tskit,runtime["modules"]["tskit"])
    raw=tskit.load(work/"converted.trees");ts,support=normalize(raw,roster,check)
    verify_native_genotypes(ts,inputs/"input.haps")
    ts.dump(export/"observed.trees")
    mapping=dict(schema="haplotype_node_mapping_v1",rows=[dict(haplotype_id=row["haplotype_id"],node_id=int(n)) for row,n in zip(roster["rows"],ts.samples())])
    bound=p.bind_roster(roster,mapping);write(export/"observed_node_mapping.json",mapping);write(export/"roster.bound.json",bound)
    transform=dict(schema="prospective1611_Relate_map_and_coordinate_transform_v1",observed_receipt_ref=r["observed_receipt_ref"],
        map_derivation_receipt_ref=r["map_derivation_receipt_ref"],shifted_native_map_ref=ref(inputs/"input.map"),
        original_haps_ref=ref(paths["observed.haps"]),converted_haps_ref=ref(inputs/"input.haps"),variant_validation=check,
        ages_ref=ref(inputs/"sample_ages.generations"),sample_ref=ref(inputs/"input.sample"),native_commands_ref=ref(output/"commands.json"),
        native_raw_tree_ref=ref(work/"converted.trees"),normalized_tree_ref=ref(export/"observed.trees"),support=support,
        source_ref=r["source_ref"],true_genealogy_or_labels_used=False)
    write(output/"transform_receipt.json",transform)
    bank=dict(schema="prospective1611_Relate_supported_bank_v1",complete=True,parent_id=r["parent_id"],requested_sequence_length_bp=L,
        normalized_sequence_length_bp=int(ts.sequence_length),support=support,observed_generation_receipt_ref=r["observed_receipt_ref"],
        map_derivation_receipt_ref=r["map_derivation_receipt_ref"],runtime_binding_ref=r["runtime_binding_ref"],source_ref=r["source_ref"],transform_receipt_ref=ref(output/"transform_receipt.json"),
        files=dict(observed_trees=ref(export/"observed.trees"),roster=ref(export/"roster.bound.json"),observed_node_mapping=ref(export/"observed_node_mapping.json")),
        contains_simulation_truth=False,legacy_full_L_receipt_compatibility=False,inference_condition="prospective All fixedN; native-supported common span")
    write(output/"bank_receipt.json",bank)
    write(output/"native_method_handles.json",dict(schema="prospective1611_common_supported_native_method_handles_v1",
        bank_receipt_ref=ref(output/"bank_receipt.json"),support=support,ARGMix=dict(trees=bank["files"]["observed_trees"],roster=bank["files"]["roster"],
        num_refs=15,num_ancs=0,full_supported_parent_context=True),AP=dict(trees=bank["files"]["observed_trees"],roster=bank["files"]["roster"],
        six_head_classifier_four_way_decoder=True),common_target_policy=json.loads(PLAN.read_text())["shared_targets"],truth_access=False))
    for original in obs["files"].values():checked(original)
    checked(r["observed_receipt_ref"])
    print(json.dumps(dict(bank_receipt_ref=ref(output/"bank_receipt.json"),support=support)),flush=True)
    return bank

def supported_bank(record):
    verify_sources();bank=read(record)
    need(bank["schema"]=="prospective1611_Relate_supported_bank_v1" and bank["complete"] is True and bank["contains_simulation_truth"] is False
        and bank["requested_sequence_length_bp"]==L and bank["support"]["terminal_topology_extended"] is False,"Genuine supported-only bank")
    right=bank["normalized_sequence_length_bp"]
    need(0<right<=L and bank["support"]["supported_scoring_interval_bp"]==[0,right] and bank["support"]["unobserved_terminal_interval_bp"]==[right,L],"Exact shared support and excluded terminal domain")
    for v in bank["files"].values():checked(v)
    return bank

def postprocess_resource(resource_ref,bank_ref,bank,output,phase,target_ref=None):
    res=read(resource_ref)
    need(res["schema"]=="prospective1611_Relate_postprocess_resource_root_v1" and res["root_authorizer"]=="root_agent"
        and res["phase"]==phase and res["root_enforced"] is True and res["outer_process_limits_enforced"] is True
        and same_ref(res["bank_receipt_ref"],bank_ref) and local(res["output_directory"])==local(output)
        and same_ref(res["adapter_source_ref"],bank["source_ref"]),"Actual root postprocessing resource binding")
    caps=valid_caps(read(res["caps_ref"]));admission=read(res["admission_receipt_ref"])
    need(admission["kind"]=="concurrent_completed_scope_reserve_receipt_v1" and admission["value"]["output"]==str(local(output))
        and admission["value"]["reserved_bytes"]==caps["scope_bytes"] and bank["source_ref"] in admission["value"]["consumer_bindings"],"Actual original postprocessing allocation")
    runtime=read(bank["runtime_binding_ref"]);need(runtime["root_qualified"] is True and runtime["cpu_only"] is True
        and sha(Path(sys.executable).resolve())==runtime["python_executable_sha256"],"Exact root-qualified postprocessing Python")
    if phase=="features":
        qualification=read(res["feature_runtime_qualification_ref"])
        need(res["feature_runtime_root_qualified"] is True and same_ref(res["targets_ref"],target_ref)
            and qualification["root_authorizer"]=="root_agent" and qualification["native_ABI_qualified"] is True
            and qualification["cpu_only"] is True,"Actual native ABI qualification and common target binding")
    return runtime

def check_module(module,binding):
    need(module.__version__==binding["version"] and Path(module.__file__).resolve()==Path(binding["file"]).resolve()
        and sha(module.__file__)==binding["sha256"],"Exact qualified numerical module identity")

def targets(bank_ref,output,resource_ref):
    """Exactly100000 distinct common pairs; observed supported domain only."""
    bank=supported_bank(bank_ref);runtime=postprocess_resource(resource_ref,bank_ref,bank,output,"targets")
    parent=registered(bank["parent_id"]);p=panel_module();panel=p.Panel(read(bank["files"]["roster"]))
    import numpy as np
    import tskit
    check_module(np,runtime["modules"]["numpy"]);check_module(tskit,runtime["modules"]["tskit"])
    ts=tskit.load(checked(bank["files"]["observed_trees"]))
    step=10 if parent["split"]=="train" else (100 if parent["split"]=="val" else 1)
    trees=[(t.index,float(t.interval.left),float(t.interval.right)) for t in ts.trees() if t.index%step==0 and t.interval.right>t.interval.left]
    queries=[r for r in panel.rows if r["cohort"] in p.QUERIES];need(len(queries)==2972,"All2972 query haplotypes")
    space=len(trees)*len(queries);need(space>=100000,"Insufficient common supported pair space; no replacement/resampling")
    rng=np.random.Generator(np.random.PCG64(parent["new_seed_values"]["sampler"]));chosen=rng.choice(space,size=100000,replace=False)
    rows=[];cohorts=set()
    for i,value in enumerate(chosen):
        ti,qi=divmod(int(value),len(queries));idx,left,right=trees[ti];q=queries[qi];cohorts.add(q["cohort"])
        rows.append(dict(row_id=bank["parent_id"]+":common:%06d"%i,parent_id=bank["parent_id"],haplotype_id=q["haplotype_id"],
            tree_index=idx,left=left,right=right,position=(left+right)/2))
    need(cohorts==set(p.QUERIES),"A required query cohort has zero selected rows; no resampling")
    p.validate_targets(rows,panel,parent_id=bank["parent_id"])
    directory=local(output);need(not directory.exists(),"Fresh common target scope");directory.mkdir();out=directory/"targets.jsonl"
    with out.open("x") as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"))+"\n")
    write(str(out)+".metadata.json",dict(schema="prospective1611_supported_common_targets_v1",complete=True,
        bank_receipt_ref=bank_ref,targets_ref=ref(out),parent_id=bank["parent_id"],rows=100000,pair_space=space,tree_step=step,
        registered_sampler_seed=parent["new_seed_values"]["sampler"],sampling="numpy Generator(PCG64(seed)).choice(space,100000,replace=False)",
        support=bank["support"],resource_binding_ref=resource_ref,labels_or_truth_accessed=False))
    return rows

def features(bank_ref,target_ref,output,resource_ref):
    """Unchanged native extractors for both methods, exact same supported targets."""
    bank=supported_bank(bank_ref);p=panel_module();panel=p.Panel(read(bank["files"]["roster"]));target_path=checked(target_ref)
    rows=[json.loads(line) for line in target_path.read_text().splitlines()];p.validate_targets(rows,panel,parent_id=bank["parent_id"])
    meta=json.loads(Path(str(target_path)+".metadata.json").read_text())
    need(same_ref(meta["bank_receipt_ref"],bank_ref) and same_ref(meta["targets_ref"],target_ref) and meta["rows"]==len(rows)==100000,"One exact common target serialization")
    runtime=postprocess_resource(resource_ref,bank_ref,bank,output,"features",target_ref)
    import tskit
    check_module(tskit,runtime["modules"]["tskit"])
    need(all(0<=float(row["left"])<float(row["right"])<=bank["normalized_sequence_length_bp"] for row in rows),"Every common target within genuine support")
    sys.path.insert(0,str(ADAPTER));import native_runtime
    need(Path(native_runtime.__file__).resolve()==ADAPTER/"native_runtime.py","Exact frozen native feature source")
    out=local(output);need(not out.exists(),"Fresh native feature representation scope");out.mkdir()
    refs={}
    for method in ("ap","argmix"):
        path=out/(method+".features.jsonl")
        native_runtime.extract(SimpleNamespace(roster=str(checked(bank["files"]["roster"])),targets=str(target_path),
            parent=bank["parent_id"],trees=str(checked(bank["files"]["observed_trees"])),method=method,output=str(path)))
        refs[method]=dict(features_ref=ref(path),metadata_ref=ref(str(path)+".metadata.json"))
    write(out/"representations.json",dict(schema="prospective1611_common_supported_native_representations_v1",complete=True,
        bank_receipt_ref=bank_ref,targets_ref=target_ref,representations=refs,source_ref=short(ref(ADAPTER/"native_runtime.py")),
        resource_binding_ref=resource_ref,support=bank["support"],all_supported_ARGMix_context_retained=True,truth_or_labels_accessed=False))
    return refs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("infer","targets","features"))
    p.add_argument("--execute",action="store_true");p.add_argument("--request");p.add_argument("--request-sha256")
    p.add_argument("--bank");p.add_argument("--bank-sha256");p.add_argument("--targets");p.add_argument("--targets-sha256")
    p.add_argument("--resource-binding");p.add_argument("--resource-sha256");p.add_argument("--output");a=p.parse_args()
    if not a.execute:print(json.dumps(dict(status="source_only_root_execution_deferred",command=a.command)));return 0
    need(Path.cwd()==ROOT,"Exact project cwd");sys.dont_write_bytecode=True
    if a.command=="infer":infer(dict(path=a.request,sha256=a.request_sha256))
    elif a.command=="targets":targets(dict(path=a.bank,sha256=a.bank_sha256),local(a.output),dict(path=a.resource_binding,sha256=a.resource_sha256))
    else:features(dict(path=a.bank,sha256=a.bank_sha256),dict(path=a.targets,sha256=a.targets_sha256),a.output,dict(path=a.resource_binding,sha256=a.resource_sha256))
    return 0

if __name__=="__main__":raise SystemExit(main())
