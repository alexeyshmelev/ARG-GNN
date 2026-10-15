#!/usr/bin/env python3
"""Prospective1611 source-only metadata and native-role adapters.

No third-party import or data/model I/O occurs on import. Author sources stay frozen.
Native numeric extraction/fitting lives in native_runtime.py and is opt-in CLI work.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARGMIX = ROOT / ".runtime/author_sources/argmix_c2481248eb0d"
AP = ROOT / ".runtime/author_sources/ancestralpaths_77db3cfae100"
COUNTS = {"Ana":22,"CHG":13,"EHG":42,"WHG":48,"NF":159,"Yam":13,
          "PostNeolithic":654,"Present":660}
AGES = {"Ana":260,"CHG":300,"EHG":250,"WHG":250,"NF":180,"Yam":160,
        "PostNeolithic":135,"Present":0}
DONORS = frozenset(("Ana","CHG","EHG","WHG","NF","Yam"))
QUERIES = frozenset(("NF","Yam","PostNeolithic","Present"))
LABELS = ("Ana","CHG","WHG","EHG")
# Original AP training.py:122-137. Channel 0=post-Neolithic,1=BAA remain zero
# under this prospective donor policy; native missing-block padding remains -15.
AP_CHANNELS = {"PostNeolithic":0,"BAA":1,"Yam":2,"NF":3,"WHG":4,"EHG":5,"Ana":6,"CHG":7}
SPECIAL_TOKENS = ("<PAD>","<UNK>","<SAMPLE>","<ANC>")
ROW_KEYS = frozenset(("haplotype_id","diploid_id","phase","cohort","age_generations","node_id"))
TARGET_KEYS = frozenset(("row_id","parent_id","haplotype_id","tree_index",
                         "left","right","position"))

def load_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path=Path(path)
    with path.open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()

def integer(value, name):
    if isinstance(value,bool) or not isinstance(value,int):
        raise ValueError(name+" must be an integer")
    return value

def make_roster():
    """Only synthetic observed metadata, no simulated alleles/trees or real people."""
    rows=[]
    for cohort,n in COUNTS.items():
        for i in range(n):
            did="%s_%04d"%(cohort,i)
            age=185 if cohort=="NF" and i==158 else AGES[cohort]
            for phase in (0,1):
                rows.append({"haplotype_id":did+"_h"+str(phase),
                             "diploid_id":did,"phase":phase,"cohort":cohort,
                             "age_generations":age,"node_id":None})
    return {"schema":"prospective1611_roster_v1","identity":"synthetic",
            "date_policy":"new fixed integer generations; NF_0158 is a synthetic older-NF proxy, not recovered Iceman",
            "rows":rows}

class Panel:
    def __init__(self, document, *, full=True, require_bound=True):
        if document.get("schema")!="prospective1611_roster_v1":
            raise ValueError("wrong roster schema")
        self.rows=document["rows"]
        if not isinstance(self.rows,list) or not self.rows:
            raise ValueError("empty/non-list roster")
        self.by_hap={};self.by_node={};self.by_diploid=collections.defaultdict(list)
        for row in self.rows:
            if set(row)!=ROW_KEYS:
                raise ValueError("roster fields differ: "+str(set(row)^ROW_KEYS))
            hid=row["haplotype_id"];did=row["diploid_id"];cohort=row["cohort"]
            if not isinstance(hid,str) or not hid or hid in self.by_hap:
                raise ValueError("duplicate/invalid haplotype identity")
            if not isinstance(did,str) or not did or cohort not in COUNTS:
                raise ValueError("invalid diploid/cohort identity")
            if integer(row["phase"],"phase") not in (0,1):
                raise ValueError("phase must be 0/1")
            if integer(row["age_generations"],"age_generations")<0:
                raise ValueError("negative age")
            nid=row["node_id"]
            if nid is None:
                if require_bound:
                    raise ValueError("unbound node ID")
            else:
                if integer(nid,"node_id")<0 or nid in self.by_node:
                    raise ValueError("negative/duplicate node ID")
                self.by_node[nid]=row
            self.by_hap[hid]=row;self.by_diploid[did].append(row)
        for pair in self.by_diploid.values():
            if len(pair)!=2 or {r["phase"] for r in pair}!={0,1}:
                raise ValueError("every diploid needs distinct phases 0 and 1")
            if len({(r["cohort"],r["age_generations"]) for r in pair})!=1:
                raise ValueError("diploid mate cohort/age disagreement")
        counts=collections.Counter(r["cohort"] for r in self.rows)
        if full and counts!={c:2*n for c,n in COUNTS.items()}:
            raise ValueError("full panel must preserve all 3222 haplotypes/cohort counts")
        self.full=full
        self.bound=len(self.by_node)==len(self.rows)
        if require_bound and not self.bound:
            raise ValueError("partial mapping")

    def query(self,hid):
        row=self.by_hap[hid]
        if row["cohort"] not in QUERIES:
            raise ValueError("haplotype is not eligible as query: "+hid)
        return row

    def donor_rows(self,hid):
        q=self.query(hid)
        return sorted((r for r in self.rows
                       if r["cohort"] in DONORS and r["diploid_id"]!=q["diploid_id"]),
                      key=lambda r:r["haplotype_id"])

    def excluded_mates(self,hid):
        return sorted(r["haplotype_id"] for r in self.by_diploid[self.query(hid)["diploid_id"]])

    def summary(self):
        return {"diploids":len(self.by_diploid),"haplotypes":len(self.rows),
                "cohort_haplotypes":dict(sorted(collections.Counter(r["cohort"] for r in self.rows).items())),
                "donor_candidates":sum(r["cohort"] in DONORS for r in self.rows),
                "query_candidates":sum(r["cohort"] in QUERIES for r in self.rows),
                "bound":self.bound,"identity_exclusion":"both focal diploid haplotypes"}

def bind_roster(document,mapping):
    """Mapping is exact observed ID->node metadata, never guessed integer offsets."""
    Panel(document,full=True,require_bound=False)
    if mapping.get("schema")!="haplotype_node_mapping_v1":
        raise ValueError("wrong mapping schema")
    pairs=mapping.get("rows",[])
    if any(set(r)!={"haplotype_id","node_id"} for r in pairs):
        raise ValueError("mapping rows need exactly haplotype_id,node_id")
    ids=[r["haplotype_id"] for r in pairs]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate mapping identity")
    by_hap={r["haplotype_id"]:integer(r["node_id"],"node_id") for r in pairs}
    if set(by_hap)!={r["haplotype_id"] for r in document["rows"]}:
        raise ValueError("mapping must cover exactly the roster; no missing/extraneous IDs")
    result=json.loads(json.dumps(document))
    for r in result["rows"]:
        r["node_id"]=by_hap[r["haplotype_id"]]
    Panel(result)
    result["node_mapping_sha256"]=canonical_sha(mapping)
    return result

def token_vocab():
    vocab={s:i for i,s in enumerate(SPECIAL_TOKENS)}
    for p in sorted(DONORS|QUERIES):
        vocab[p]=len(vocab)
    return vocab

def argmix_request(panel,hid,*,num_refs=15,batch_size=256):
    """Existing Rust ABI, no author edit/recompile, one query per iterator.

    Unlike ShardDataset's auto-selected reference-population list, the Rust
    constructor already accepts explicit donor IDs. All true cohort metadata
    stays intact, including excluded NF/Yam focal leaves.
    """
    if not panel.bound:
        raise ValueError("bound node IDs required")
    q=panel.query(hid)
    donors=panel.donor_rows(hid)
    if len(donors)<num_refs or num_refs<1:
        raise ValueError("not enough allowed donors")
    metadata={r["node_id"]:(r["cohort"],r["age_generations"])
              for r in sorted(panel.rows,key=lambda r:r["haplotype_id"])}
    return {"max_rel_distance":14,"node_metadata":metadata,"num_refs":num_refs,
            "drop_top_m_nearest":None,"num_ancs":0,"max_count":0,
            "ref_node_ids":[r["node_id"] for r in donors],
            "predict_node_ids":[q["node_id"]],"exclude_edge_trees":False,
            "log_scale_tmrca":True,"subgraph_merge_enabled":False,
            "subgraph_merge_signature_mode":"ids_only","backend_name":"fast",
            "infer_min_bp":None,"infer_max_bp":None,"batch_size":batch_size}

def native_argmix_iterator(ancpath_module,tree_path,request):
    """Dependency injected for metadata tests; numeric library import is external."""
    names=("max_rel_distance","node_metadata","num_refs","drop_top_m_nearest",
           "num_ancs","max_count","ref_node_ids","predict_node_ids",
           "exclude_edge_trees","log_scale_tmrca","subgraph_merge_enabled",
           "subgraph_merge_signature_mode","backend_name","infer_min_bp",
           "infer_max_bp","batch_size")
    return ancpath_module.SampleBatchIterator(str(tree_path),
                                             *(request[k] for k in names))

def native_pop_token_lookup(panel):
    # Exactly tree_sampler.py:252-257; unique population IDs use three prefixes.
    vocab=token_vocab()
    pops=["<SAMPLE>","<ANC>","<PAD>"]+sorted({r["cohort"] for r in panel.rows})
    return [vocab.get(p,vocab["<UNK>"]) for p in pops]

def unpack_native_flat(payload,pop_lookup,expected_width=16):
    """Native flat-row decoding without label leakage; Rust labels are discarded."""
    if len(payload) not in (6,7):
        raise ValueError("unexpected native flat ABI")
    matrices,pops,node_ids,intervals,n,width=payload[:6]
    n=integer(n,"batch_len");width=integer(width,"seq_len")
    if n<0 or width!=expected_width:
        raise ValueError("native row width mismatch")
    if len(matrices)!=n*width*width or len(pops)!=n*width or len(node_ids)!=n or len(intervals)!=n:
        raise ValueError("native flat field length mismatch")
    out=[]
    for i in range(n):
        pids=pops[i*width:(i+1)*width]
        if any(integer(x,"pop_id")<0 or x>=len(pop_lookup) for x in pids):
            raise ValueError("native population ID outside lookup")
        matrix=matrices[i*width*width:(i+1)*width*width]
        left,right=map(float,intervals[i])
        if not math.isfinite(left+right) or not left<right:
            raise ValueError("invalid native interval")
        out.append({"node_id":int(node_ids[i]),"left":left,"right":right,
                    "input_ids":[pop_lookup[x] for x in pids],
                    "tmrca":[list(matrix[j*width:(j+1)*width]) for j in range(width)]})
    return out

def ap_features(tree,panel,hid,*,num_ancestors=5):
    """Author training.py:102-144 ancestor logic with explicit donor identities.

    Keep newly encountered allowed donor leaves at each ancestor; skip empty
    ancestors; normalize by donor count; int(age)/1500; missing blocks -15.
    The original training/paint paths differ on fractional ages. This adapter
    consistently uses the original training path's int(age)/1500.
    """
    q=panel.query(hid)
    allowed={r["node_id"]:r for r in panel.donor_rows(hid)}
    features=[];previous=set();u=tree.parent(q["node_id"]);visited=set()
    while u!=-1 and len(features)<num_ancestors:
        if u in visited:
            raise ValueError("cycle in tree parent metadata")
        visited.add(u)
        leaves={int(x) for x in tree.leaves(u)}
        if not leaves <= set(panel.by_node):
            raise ValueError("tree leaf absent from observed roster mapping")
        total=(leaves & set(allowed))-previous
        if total:
            age=float(tree.time(u))
            if not math.isfinite(age) or age<0:
                raise ValueError("invalid observed ancestor age")
            row=[0.0]*9
            row[8]=int(age)/1500.0
            for nid in sorted(total):
                row[AP_CHANNELS[allowed[nid]["cohort"]]]+=1.0/len(total)
            previous.update(total);features.append(row)
        u=tree.parent(u)
    features.extend([[-15.0]*9 for _ in range(num_ancestors-len(features))])
    return features

def encode_label(ancestry):
    if ancestry not in LABELS:
        raise ValueError("target is not one of the four founder classes")
    return LABELS.index(ancestry)

def project_ap_probabilities(p):
    if len(p)!=6 or any(not math.isfinite(float(x)) or x<0 for x in p):
        raise ValueError("AP output requires six finite nonnegative probabilities")
    if not math.isclose(sum(p),1.0,rel_tol=1e-5,abs_tol=1e-6):
        raise ValueError("AP probabilities must sum to one")
    return [p[0]+p[4],p[1]+p[5],p[2],p[3]]

def validate_targets(targets,panel,*,parent_id=None):
    ids=set();keys=set();tree_intervals={}
    for row in targets:
        if set(row)!=TARGET_KEYS:
            raise ValueError("target metadata fields differ; labels/truth forbidden here")
        if not isinstance(row["row_id"],str) or not row["row_id"] or row["row_id"] in ids:
            raise ValueError("duplicate/invalid target row ID")
        if not isinstance(row["parent_id"],str) or not row["parent_id"]:
            raise ValueError("invalid parent ID")
        if parent_id is not None and row["parent_id"]!=parent_id:
            raise ValueError("mixed/wrong parent")
        panel.query(row["haplotype_id"])
        if integer(row["tree_index"],"tree_index")<0:
            raise ValueError("negative tree index")
        left,right,pos=(float(row[k]) for k in ("left","right","position"))
        if not all(math.isfinite(x) for x in (left,right,pos)) or not 0<=left<right or pos!=(left+right)/2:
            raise ValueError("target must use midpoint of positive observed interval")
        tree_key=(row["parent_id"],row["tree_index"])
        if tree_key in tree_intervals and tree_intervals[tree_key]!=(left,right):
            raise ValueError("one inferred tree index has conflicting intervals")
        tree_intervals[tree_key]=(left,right)
        key=(row["parent_id"],row["haplotype_id"],row["tree_index"])
        if key in keys:
            raise ValueError("duplicate tree/query target")
        ids.add(row["row_id"]);keys.add(key)
    if not ids:
        raise ValueError("empty targets")
    return {"target_rows":len(ids),"parents":sorted({r["parent_id"] for r in targets}),
            "rows_sha256":canonical_sha(targets)}

def validate_labels(labels,targets):
    expected={r["row_id"] for r in targets}
    if any(set(r)!={"row_id","ancestry"} for r in labels):
        raise ValueError("supervision rows need row_id,ancestry only")
    if len(labels)!=len({r["row_id"] for r in labels}) or {r["row_id"] for r in labels}!=expected:
        raise ValueError("labels must join exactly once to all target IDs")
    return {r["row_id"]:encode_label(r["ancestry"]) for r in labels}

def compare_fit_summaries(argmix_fit,ap_fit):
    if argmix_fit.get("schema")!="prospective1611_native_fit_summary_v1" or ap_fit.get("schema")!="prospective1611_native_fit_summary_v1" or argmix_fit.get("method")!="argmix" or ap_fit.get("method")!="ap":
        raise ValueError("wrong paired fit summaries")
    fields=("train_shared_signatures","val_shared_signatures","logical_train_targets",
            "logical_val_targets","fit_config_sha256")
    if any(k not in argmix_fit or k not in ap_fit or canonical_sha(argmix_fit[k])!=canonical_sha(ap_fit[k]) for k in fields):
        raise ValueError("fit observations, targets, labels or truth provenance differ")
    for split in ("train_shared_signatures","val_shared_signatures"):
        if any(not {"labels_sha256","supervision_contract_sha256","truth_definition_sha256"}<=set(r)
               for r in argmix_fit[split]):
            raise ValueError("fit signatures omit biological truth identity")
    return {"same_supervised_data_verified":True,
            "shared_fit_signature_sha256":canonical_sha({k:argmix_fit[k] for k in fields}),
            "methods":["ARGMix current-release count/role adapter","AncestralPaths-derived C4 adapter"]}

def compare_paired_metadata(argmix_meta,ap_meta):
    if argmix_meta.get("schema")!="prospective1611_native_features_v1" or ap_meta.get("schema")!="prospective1611_native_features_v1":
        raise ValueError("wrong native feature metadata schema")
    if argmix_meta.get("method")!="argmix" or ap_meta.get("method")!="ap" or not argmix_meta.get("complete") or not ap_meta.get("complete"):
        raise ValueError("paired method metadata incomplete/wrong method")
    fields=("parent_id","target_rows","targets_sha256","roster_sha256",
            "observed_trees_sha256","source_pins_sha256","fit_config_sha256")
    if any(k not in argmix_meta or k not in ap_meta or argmix_meta[k]!=ap_meta[k] for k in fields):
        raise ValueError("native methods do not have identical shared observations/targets/config")
    shared={k:argmix_meta[k] for k in fields}
    return {"same_data_verified":True,"shared":shared,
            "shared_signature_sha256":canonical_sha(shared)}

def verify_sources():
    doc=load_json(HERE/"source_pins.v1.json")
    for row in doc["sources"]:
        if sha256(ROOT/row["path"])!=row["sha256"]:
            raise ValueError("pinned source changed: "+row["path"])
    return len(doc["sources"])

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    r=sub.add_parser("roster");r.add_argument("--output",required=True)
    b=sub.add_parser("bind");b.add_argument("--roster",required=True);b.add_argument("--mapping",required=True);b.add_argument("--output",required=True)
    s=sub.add_parser("summary");s.add_argument("--roster",required=True);s.add_argument("--unbound",action="store_true")
    a=sub.add_parser("requests");a.add_argument("--roster",required=True);a.add_argument("--query",required=True);a.add_argument("--output",required=True)
    v=sub.add_parser("validate-targets");v.add_argument("--roster",required=True);v.add_argument("--targets",required=True)
    m=sub.add_parser("paired-metadata");m.add_argument("--argmix-metadata",required=True);m.add_argument("--ap-metadata",required=True)
    f=sub.add_parser("paired-fit-summaries");f.add_argument("--argmix-summary",required=True);f.add_argument("--ap-summary",required=True)
    sub.add_parser("verify-sources")
    args=p.parse_args()
    if args.command=="paired-fit-summaries":
        print(json.dumps(compare_fit_summaries(load_json(args.argmix_summary),load_json(args.ap_summary)),sort_keys=True));return
    if args.command=="paired-metadata":
        print(json.dumps(compare_paired_metadata(load_json(args.argmix_metadata),load_json(args.ap_metadata)),sort_keys=True));return
    if args.command=="verify-sources":
        print(json.dumps({"verified_source_files":verify_sources()}));return
    if args.command=="roster":
        document=make_roster();Panel(document,require_bound=False)
        write_json(args.output,document);return
    if args.command=="bind":
        write_json(args.output,bind_roster(load_json(args.roster),load_json(args.mapping)));return
    panel=Panel(load_json(args.roster),require_bound=not getattr(args,"unbound",False))
    if args.command=="summary":
        print(json.dumps(panel.summary(),sort_keys=True))
    elif args.command=="requests":
        req=argmix_request(panel,args.query)
        write_json(args.output,{"query":args.query,"excluded_haplotypes":panel.excluded_mates(args.query),
                               "request":req,"ap_channels":AP_CHANNELS,"labels":LABELS,
                               "token_vocab":token_vocab()})
    else:
        print(json.dumps(validate_targets(read_jsonl(args.targets),panel),sort_keys=True))

if __name__=="__main__":
    main()
