"""Source and synthetic-only observed2074 packet checks; no empirical arrays/processes."""
from pathlib import Path
import ast,copy,hashlib,json,struct,math,argparse,contextlib,io
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_observed2074_native_inference_source_v1"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def require(ok,message):
    if not ok:raise ValueError(message)
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError,struct.error):checks.append(name);return
    raise AssertionError(name+" accepted")
def fn(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
def pure(source,name,ns):
    exec(compile(ast.Module(body=[fn(source,name)],type_ignores=[]),"<source-only-"+name+">","exec"),ns)
    return ns[name]
def ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
driver=(HERE/"root_inference.py").read_text()
prior=(ROOT/"research/reviews/H018_native_Relate_qualification_source_v1/root_qualifier_v4.py").read_text()
core_source=(HERE/"inference_core.py").read_text()
for name,source in [("driver",driver),("core",core_source),("gate",(HERE/"qualification_gate.py").read_text())]:
    compile(source,name,"exec");checks.append("compile "+name)
for name in ("identity","bounded_file","write_bytes","group_absent","direct_wait_owned","native_limits",
 "_inventory_once_v4","file_inventory","owned_rss","supervise","generated_project_pycache_path_v2",
 "verify_runtime_packages","runtime_preflight_v2"):
    check("unchanged qualified supervisor primitive "+name,ast.dump(fn(driver,name),include_attributes=False)==ast.dump(fn(prior,name),include_attributes=False))
core={};exec(compile(core_source,"<pure-observed-core>","exec"),core)
original=(ROOT/"research/reviews/H018_native_Relate_qualification_source_v1/fixture_and_verify.py").read_text()
for name in ("rounded_age_token","parse_anc"):
    check("same qualified date/native-format primitive "+name,ast.dump(fn(core_source,name),include_attributes=False)==ast.dump(fn(original,name),include_attributes=False))
cfg=json.loads((HERE/"config.json").read_text());manifest=json.loads((HERE/"source_manifest.json").read_text())
request=json.loads((HERE/"root_request.inactive.json").read_text())
literal=next(n.value for n in ast.parse(driver).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SETTINGS" for t in n.targets))
check("literal exact reviewed caps/options",ast.literal_eval(literal)==cfg)
check("inert source request without qualification or registration",request["status"]=="inactive_source_plan" and request["qualification_refs"] is None and request["registration_ref"] is None and request["root_authorizer"] is None and request["native_execution_authorized"] is False)
check("all model HS GPU truth authorities false",all(request[k] is False for k in ("model_training_authorized","model_application_authorized","HS_execution_authorized","GPU_authorized","ancestry_truth_reads_authorized")))
check("all actual new source pins",request["driver_ref"]==manifest["source_refs"]["driver"]==ref(HERE/"root_inference.py") and request["config_ref"]==manifest["source_refs"]["config"]==ref(HERE/"config.json") and request["source_manifest_ref"]==ref(HERE/"source_manifest.json") and cfg["helper_ref"]==ref(HERE/"inference_core.py"))
direct=pure(driver,"direct_authority_refs",dict(Path=Path,require=require,DIRECT_BINDING_SUFFIXES=(".json",".py",".sh")))
refs=direct(manifest)
check("observed assets all transitively bound not authority slots",manifest["transitive_only_refs"]["data_refs"]==manifest["data_refs"] and not any(x in refs for x in manifest["data_refs"].values()))
check("source metadata closure excludes empirical arrays",not any(r["path"].startswith("data/H018_R16_observed_bridge_v2/") and not r["path"].endswith("/completed.json") for g in ("source_refs","metadata_refs") for r in manifest[g].values()))
check("native profile unchanged from planned baseline",cfg["scientific_options"]["mutation_rate"]=="1.25e-8" and cfg["scientific_options"]["effective_haploid_N"]==30000 and cfg["scientific_options"]["native_memory_GB"]=="2" and cfg["scientific_options"]["pre_size_only"] is True)
label=cfg["scientific_options"]["seed_derivation"]
check("prospective deterministic seed",cfg["scientific_options"]["seed"]==int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],"big")%2147483646+1)
parse=core["parse_headers"]
g=struct.pack("<iiidii",40,2074,1,.01,0,2074);c=struct.pack("<iiiii",40,2074,2,0,2074)
h=parse(g,{0:c});check("full original one-chunk header",h["sites"]==2074 and h["chunks"][0]["last_section"]==0)
g2=struct.pack("<iiidiiii",40,2074,2,.01,0,1000,1200,2074)
cs={0:struct.pack("<iiiiii",40,1200,3,0,600,1200),1:struct.pack("<iiiii",40,1074,2,0,1074)}
h2=parse(g2,cs);check("bounded overlapping variablechunk headers",h2["num_chunks"]==2 and h2["chunks"][0]["last_section"]==1)
for name,gg,cc in [
 ("wrong haplotypes",struct.pack("<iiidii",39,2074,1,.01,0,2074),{0:c}),
 ("wrong site count",struct.pack("<iiidii",40,2073,1,.01,0,2073),{0:c}),
 ("missing chunk",g2,{0:cs[0]}),
 ("chunk length mismatch",g,{0:struct.pack("<iiiii",40,2073,2,0,2073)}),
 ("bad section bounds",g,{0:struct.pack("<iiiiii",40,2074,3,0,2074,2000)}),
 ("appended global bytes",g+b"x",{0:c}),
 ("nonfinite metadata",struct.pack("<iiidii",40,2074,1,float("nan"),0,2074),{0:c}),
 ("gap in global coverage",struct.pack("<iiidiiii",40,2074,2,.01,0,1300,1200,2074),cs)]:
    reject("headers reject "+name,lambda gg=gg,cc=cc:parse(gg,cc))
binaries={"Relate":"/synthetic/Relate","RelateFileFormats":"/synthetic/Convert"}
first=core["cli_stages"](binaries,Path("/synthetic/input"),cfg["scientific_options"])
rest=core["cli_stages"](binaries,Path("/synthetic/input"),cfg["scientific_options"],h2)
check("one MakeChunks then full ordered chunks and converter",len(first)==1 and first[0][0]=="make_chunks" and len(rest)==12 and rest[-2][0]=="finalize" and rest[-1][0]=="convert")
check("full ranges maintain source RNG offsets",all("--first_section" in argv and argv[argv.index("--first_section")+1]=="0" for name,argv,_ in rest if name.startswith(("build_topology","infer_branch_lengths"))))
check("exact original map/date/haplotype args every Relate mode",all("--sample_ages" in argv and argv[argv.index("--map")+1].endswith("/qualified.regional.map") and argv[argv.index("--haps")+1].endswith("/baseline.haps") and argv[argv.index("--effectiveN")+1]=="30000" for _,argv,kind in first+rest if kind=="relate"))
# One compact full-count synthetic preparation exercises the exact public parser.
roster=[];names=["SYNTH_%02d"%i for i in range(20)]
for i in range(40):
    a=1000.125+7.1*(i//2)
    roster.append(dict(sample_index=i,haplotype=i%2,subject_id=names[i//2],
      role="reference" if i<16 else "query",date_zero="calBP1950",years_per_generation=28,
      age_mean_cal_BP=a,age_generations=a/28.,query_ancestry_truth=None,native_tree_node_id=None))
identities=[];audits=[];haps=[];mono=[];digest=hashlib.sha256()
for i in range(2074):
    pos=20000100+1000*i;refbase,alt="A","C";aa,da=(alt,refbase) if i%7==0 else (refbase,alt)
    original_codes=([i%2]*40) if i<277 else [int(j==i%40) for j in range(40)]
    codes=[v^int(aa==alt) for v in original_codes];nt="".join((refbase,alt)[v] for v in original_codes)
    identities.append(["22",pos,refbase,alt]);local=pos-20000000
    haps.append(" ".join(["22","H018_22_"+str(pos)+"_"+refbase+"_"+alt,str(local),aa,da]+list(map(str,codes))))
    audits.append(dict(source_POS_bp1=pos,haps_POS_bp1=local,regional_bp0=local-1,source_REF=refbase,
      source_ALT=alt,source_GT_haplotype_codes=original_codes,native_haps_codes=codes,
      nucleotide_equivalence_sha256=hashlib.sha256(nt.encode()).hexdigest(),
      exact_nucleotide_and_phase_order_equivalence=True,monomorphic_among_actual40=i<277))
    if i<277:mono.append(pos)
    digest.update((str(pos)+":"+nt+"\n").encode())
inputs={"baseline.haps":("\n".join(haps)+"\n").encode(),
 "baseline.sample":("ID_1 ID_2 missing\n0 0 0\n"+"".join(n+" "+n+" 0\n" for n in names)).encode(),
 "sample_ages.generations":"".join(format(r["age_generations"],".17g")+"\n" for r in roster).encode(),
 "frozen_site_identities.json":json.dumps(identities).encode(),"polarization_equivalence.json":json.dumps(audits).encode()}
readback=dict(original_nucleotide_vectors_sha256=digest.hexdigest(),
 monomorphic_source_POS_bp1_sha256=hashlib.sha256(json.dumps(mono,separators=(",",":")).encode()).hexdigest())
observed=core["expected_observations"](inputs,json.dumps(roster).encode(),readback)
check("synthetic2074/40/277 exact original nucleotide/date/phase parser",len(observed["rows"])==2074 and len(observed["monomorphic_absolute_POS"])==277)
bad=dict(inputs);bad["baseline.haps"]=("\n".join(haps[:-1])+"\n").encode()
reject("input site dropping rejected",lambda:core["expected_observations"](bad,json.dumps(roster).encode(),readback))
bad=dict(inputs);f=haps[500].split();f[5]=str(1-int(f[5]));rows=haps.copy();rows[500]=" ".join(f);bad["baseline.haps"]=("\n".join(rows)+"\n").encode()
reject("input phase nucleotide change rejected",lambda:core["expected_observations"](bad,json.dumps(roster).encode(),readback))
bad=dict(inputs);bad["sample_ages.generations"]=b"0\n"*40
reject("rounded or invented dates rejected",lambda:core["expected_observations"](bad,json.dumps(roster).encode(),readback))
gate={};exec(compile((HERE/"qualification_gate.py").read_text(),"<pure-qualified-evidence-gate>","exec"),gate)
stage_names=gate["STAGES"];d={"path":"synthetic/driver.py","sha256":"1"*64};m={"path":"synthetic/manifest.json","sha256":"2"*64}
qrefs={k:{"path":"synthetic/"+k+".json","sha256":"3"*64} for k in ("completed","controller_result","controller_wait","admitted","request","facts")}
qrefs["waits"]={s:{"path":"synthetic/"+s+".json","sha256":"4"*64} for s in stage_names}
facts=dict(status="qualified_tiny_synthetic_native_path_only",failures=[],all_fixture_sites_preserved=True,
 exact_six_decimal_age_tokens=True,combined_binary64_sixdecimal_age_bound=True,
 exact_parsed_anc_to_binary_sample_time=True,no_converter_chronology_adjustment_trigger=True,
 finite_strict_native_edge_chronology=True,native_direct_wait_refs=qrefs["waits"])
records=dict(completed=dict(schema="H018_Relate_synthetic_qualification_completed_v1",
 status="qualified_tiny_synthetic_native_path_only",synthetic_only=True,fixture_sites=512,haplotypes=40,
 constant_sites=6,sites_dropped=0,admitted_ref=qrefs["admitted"],facts_ref=qrefs["facts"],
 actual_native_direct_wait_refs=qrefs["waits"]),
 admitted=dict(request_ref=qrefs["request"],driver_ref=d),request=dict(driver_ref=d,source_manifest_ref=m),
 controller_result=dict(success=True,completed_ref=qrefs["completed"],facts_ref=qrefs["facts"],admitted_ref=qrefs["admitted"]),
 controller_wait=dict(actual_direct_controller_wait=True,returncode=0,group_absent_after_wait=True,controller_pid=123),
 facts=facts,waits={s:dict(stage=s,actual_direct_wait_observed=True,actual_exit_code=0,
 process_group_drained=True,owned_group_absent_after_wait=True,control_error=None,supervisor_pid=123) for s in stage_names})
check("complete successful prerequisite metadata accepted",gate["verify_v4"](records,qrefs,d,m)["empirical_native_execution_authority"] is False)
for label,alter in [
 ("failed controller",lambda r:r["controller_result"].update(success=False)),
 ("nonzero actual root wait",lambda r:r["controller_wait"].update(returncode=1)),
 ("nonzero native wait",lambda r:r["waits"]["finalize"].update(actual_exit_code=-15)),
 ("undrained group",lambda r:r["waits"]["finalize"].update(process_group_drained=False)),
 ("unknown controller",lambda r:r["controller_wait"].update(actual_direct_controller_wait=False)),
 ("unqualified chronology",lambda r:r["facts"].update(no_converter_chronology_adjustment_trigger=False)),
 ("dropped qualifier site",lambda r:r["completed"].update(sites_dropped=1))]:
    bad=copy.deepcopy(records);alter(bad)
    reject("prerequisite rejects "+label,lambda bad=bad:gate["verify_v4"](bad,qrefs,d,m))
launch=ast.get_source_segment(driver,fn(driver,"launch"));verify=ast.get_source_segment(driver,fn(driver,"verify_request"))
check("qualification gate required before admission","verify_qualification_v1(b,r[" in verify)
outer_gate=ast.get_source_segment(driver,fn(driver,"verify_qualification_v1"))
check("exact8 qualifier wait metadata cap before reads",outer_gate.index('set(refs["waits"])=={')<outer_gate.index('records={k:b.read'))

check("empirical assets only opened after first guard",launch.index("m.budget_check_v9(storage,first=True")<launch.index("payloads={name:bounded_file"))
check("perstage source inputs rehashed","for key in (" in launch and 'manifest["data_refs"][key]' in launch)
check("native header gate follows genuine wait0/drain",launch.index('observation["actual_exit_code"]==0')<launch.index('headers=read_native_headers_v1'))
check("native topology and support exact comparison", "Counter(edge_expected)==Counter(actual_edges)" in core_source)
check("genuine last SNP plusone and unrepresented tails", 'genuine_regional_bp0=[first-1,last]' in core_source and "full_10000000bp_coverage_claim=False" in core_source and "no_flanking_tree_extrapolation=True" in core_source)
check("no native tree manipulation APIs",all(token not in core_source for token in (".simplify(",".keep_intervals(",".set_columns(")))
main=pure(driver,"main",dict(argparse=argparse,json=json,require=require))
with contextlib.redirect_stdout(io.StringIO()):status=main([])
check("default inert",status==0)
print(json.dumps(dict(schema="H018_observed2074_inference_source_synthetic_checks_v1",status="passed",
 checks=len(checks),check_names=checks,empirical_arrays_or_truth_opened=False,
 model_native_GPU_guard_accounting_process_probe_executed=False,
 synthetic_input_sites=2074,synthetic_input_haplotypes=40),sort_keys=True,indent=2))
