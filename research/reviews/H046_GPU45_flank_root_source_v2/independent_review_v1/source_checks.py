"""Source/metadata-only fast H046 v2 review; never imports project code."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H046_GPU45_flank_root_source_v2/independent_review_v1"
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
def digest(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
base="research/reviews/H046_GPU45_flank_root_source_v2/"
old="research/reviews/H046_GPU45_flank_root_source_v1/"
pins={
"src/arggnn/H046_GPU45_flank_worker_v2.py":"91822ac6fb4c14656a075def4df9af749c7d18f8b3483da2dcc86afb626127c2",
base+"config_inactive.json":"13d1b2fa0503a452c3bee7ba3b47bf3d702ade18a796c0f4c6e562e6f129f5a8",
base+"source_manifest.json":"a2e11b2ef25c0c9316a296d17c28047941d5e98e3b5ea51e0eb62ed190904170",
base+"root_common_v1.py":"96e6f3979b3b302882ed63c66f873a5690476b6874ec3d6bbd472e54cd6fbdfc",
base+"root_driver.py":"ae1652757407cd7cb43f54e464298fd875a1e45587f2c1edaa2544a1102c9839",
base+"context_factory_v1.py":"01ab5b52721b1afa19f5f54c9351a85159c16dc514237f1f7a738a3006fbda99",
base+"integration_correction.json":"889391adce448e66207dbf5bede8faece161411e72ec7c77c1ec41459e7f8dcd",
base+"integration_only.diff":"9e44dccfa834ea99decc88349d7d16c4ad385cd52a1daf3c202215aefc78fbb2",
"src/arggnn/continuous_clade_flank_gnn_v1.py":"2c81b74c59a532b46ed75f7fd656ea526ec06fd9bf62af35855ab291f5451e20",
"src/arggnn/train_continuous_clade_flank_v1.py":"eda198ef04192d978433c64f921b865aa89a9304adcc76893fa2852a14d72ae4",
"research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py":"5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0",
"scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py":"d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d",
"scripts/launch_reserved_gpu_full_storage_v1.py":"d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff"}
for path,sha in pins.items():check("frozen_hash:"+path,digest(path)==sha)
module="arggnn.H046_GPU45_flank_worker_v2";cli="src/"+module.replace(".","/")+".py"
launcher=(ROOT/"scripts/launch_reserved_gpu_full_storage_v1.py").read_text()
check("actual_launcher_predicate_present",'if not args.module.startswith("arggnn.") or any(not part.isidentifier() for part in args.module.split(".")):' in launcher)
check("actual_predicate_accepts_canonical_module",module.startswith("arggnn.") and all(part.isidentifier() for part in module.split(".")))
check("actual_reserved_binding_mapping",cli=="src/arggnn/H046_GPU45_flank_worker_v2.py" and 'expected not in record["reservation"]["consumer_bindings"]' in launcher)
a=read(old+"config_inactive.json");b=read(base+"config_inactive.json")
check("config_only_manifest_ref_changed",{k:v for k,v in a.items() if k!="source_manifest_ref"}=={k:v for k,v in b.items() if k!="source_manifest_ref"})
check("config_binds_v2_manifest",b["source_manifest_ref"]==dict(path=base+"source_manifest.json",sha256=pins[base+"source_manifest.json"]))
s1=read(old+"source_manifest.json")["source_sha256"];s2=read(base+"source_manifest.json")["source_sha256"]
check("original77_pins_preserved",len(s1)==77 and all(s2.get(k)==v for k,v in s1.items()))
check("v2_adds_four_integration_sources",len(s2)==81 and set(s2)-set(s1)=={base+"root_common_v1.py",base+"root_driver.py",base+"context_factory_v1.py",cli})
check("all81_source_pins_match",all(digest(k)==v for k,v in s2.items()))
texts={p:(ROOT/p).read_text() for p in [base+"root_common_v1.py",base+"root_driver.py",base+"context_factory_v1.py",cli]}
check("four_sources_parse",all(isinstance(ast.parse(t),ast.Module) for t in texts.values()))
check("no_stale_v1_package_imports",all("research.reviews.H046_GPU45_flank_root_source_v1" not in t for t in texts.values()))
check("common_v2_here",'HERE="research/reviews/H046_GPU45_flank_root_source_v2"' in texts[base+"root_common_v1.py"])
d=texts[base+"root_driver.py"];w=texts[cli]
check("driver_canonical_module",'CLI_MODULE="arggnn.H046_GPU45_flank_worker_v2"' in d)
check("driver_cli_path_matches_launcher",'CLI="src/"+CLI_MODULE.replace(".","/")+".py"' in d)
check("mandatory_direct_cli_ref","ref(CONTEXT),ref(SOURCE),ref(CLI)" in d)
check("spec_direct_cli_ref","worker_cli_ref=ref(CLI)" in d)
check("launch_passes_same_module","BASE,CLI_MODULE," in d)
check("wrapper_checks_direct_cli_ref",'spec["worker_cli_ref"]==common.ref(SOURCE)' in w)
check("wrapper_delegates_unchanged_api","from arggnn.train_continuous_clade_flank_v1 import run_context_v1" in w and "run_context_v1(context,record)" in w)
check("wrapper_inert_branch_precedes_capability_reads",w.index("if not args.execute:")<w.index("spec=common.read(record)"))
check("fresh_v2_output_job","flank_%s_seed101_v2" in d and "seed101-v2-GPU%d" in d)
receipt=dict(schema="H046_v2_fast_independent_source_review_v1",status="PASS" if all(x["passed"] for x in checks) else "FAIL",checks=checks,check_count=len(checks),source_refs=[dict(path=k,sha256=v) for k,v in pins.items()],scope="Source hashes, metadata comparison and AST/text checks only. No project imports, runtime/data/checkpoint reads, admission or GPU execution.",finding="Canonical module, actual launcher predicate, reserved source binding and wrapper delegation agree. No scientific/API blocker found in this integration change.",limits="Source review does not establish runtime success, GPU fit or scientific outcome.")
(HERE/"review_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status=receipt["status"],checks=len(checks),failures=[x["name"] for x in checks if not x["passed"]])))
assert receipt["status"]=="PASS"
