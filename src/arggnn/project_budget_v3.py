"""Supplemental 1TB project accounting. Default CLI is source-only, no GPU/model imports."""
from pathlib import Path
import argparse, contextlib, copy, fcntl, hashlib, json, os, stat, uuid

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = ROOT / "src/arggnn/project_budget_v3.py"
CONFIG = ROOT / "configs/project_budget_v3.json"
LEGACY = ".runtime/gpup/storage_reservations.json"
SUPPLEMENTAL = ".runtime/storage/project_budget_v3.json"
LOCK = ".runtime/locks/scheduler.lock"
MAX_JSON = 32 * 1024 * 1024
FIXED_POLICY = {
    "kind": "project_budget_v3_policy",
    "root": str(ROOT),
    "human_authorization": {
        "path": "research/resource_authorizations/project_storage_1TB_20261011.json",
        "sha256": "f3fb4e4da30dae4ca4604e4e6ae3ff7d62799b01158fc6f86a7ea0239d2f43cd",
    },
    "legacy_pool_source": {
        "path": "src/arggnn/gpu_pool_v1.py",
        "sha256": "ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1",
    },
    "legacy_invariants": {
        "schema": "gpu_pool_storage_ledger_v1",
        "snapshot_sha256": "0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d",
        "baseline_bytes": 167336640512,
        "external_reserve_bytes": 130000000000,
        "limit_bytes": 450000000000,
    },
    "legacy_ledger": LEGACY,
    "supplemental_ledger": SUPPLEMENTAL,
    "scheduler_lock": LOCK,
    "scheduling_ceiling_bytes": 950000000000,
    "hard_limit_bytes": 1000000000000,
}

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("exact integer required: " + label)
    return value

def canonical(root, value):
    p = Path(value); p = p if p.is_absolute() else root / p
    if p == root or root not in p.parents or ".." in p.parts or p.resolve() != p:
        raise ValueError("canonical project path without symlinks required")
    return p

def file_bytes(root, path):
    p = canonical(root, path); before = p.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_JSON:
        raise ValueError("bounded singly linked regular metadata/source required")
    fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if fields(os.fstat(stream.fileno())) != fields(before):
            raise ValueError("metadata identity changed before read")
        raw = stream.read(MAX_JSON + 1); after = os.fstat(stream.fileno())
    if len(raw) > MAX_JSON or fields(before) != fields(after) or fields(after) != fields(p.lstat()):
        raise ValueError("metadata changed during read")
    return raw, hashlib.sha256(raw).hexdigest()

def parse(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def read_json(root, path):
    raw, sha = file_bytes(root, path); return parse(raw), sha

def reference(root, path):
    p = canonical(root, path); return {"path": str(p.relative_to(root)), "sha256": file_bytes(root, p)[1]}

def bound(root, ref, as_json=False):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("exact path/SHA reference required")
    sha = ref["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("exact SHA256 required")
    raw, actual = file_bytes(root, ref["path"])
    if actual != sha: raise ValueError("bound source or authority bytes changed")
    return parse(raw) if as_json else raw

def policy(root, expected_source_sha256, expected_config_sha256):
    c, sha = read_json(root, CONFIG)
    if encoded(c) != encoded(FIXED_POLICY) or sha != expected_config_sha256 or file_bytes(root, SOURCE)[1] != expected_source_sha256:
        raise ValueError("exact reviewed companion source/config required")
    if c["root"] != str(root): raise ValueError("exact project root required")
    human = bound(root, c["human_authorization"], True)
    if (human.get("kind") != "human_project_storage_authorization" or human.get("project_root") != str(root)
            or human.get("authorized_hard_limit_bytes") != c["hard_limit_bytes"]
            or human.get("future_scheduling_ceiling_bytes") != c["scheduling_ceiling_bytes"]):
        raise ValueError("exact human 1TB authorization required")
    bound(root, c["legacy_pool_source"])
    return c, {"implementation": reference(root, SOURCE), "config": reference(root, CONFIG),
               "human_authorization": c["human_authorization"]}

@contextlib.contextmanager
def mutex(root):
    p = canonical(root, LOCK); before = p.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("existing canonical scheduler mutex required")
    fd = os.open(p, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, "r+") as stream:
        current = os.fstat(stream.fileno())
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("mutex identity changed")
        fcntl.flock(stream, fcntl.LOCK_EX)
        after = p.lstat()
        if (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("mutex replaced")
        yield

def paths_overlap(a, b):
    return a == b or a in b.parents or b in a.parents

def accounting_paths(root):
    return [root/LEGACY,root/".runtime/storage",root/LOCK]

def legacy_key(value, label):
    # Pinned pool names use ASCII alphanumerics followed by alphanumerics/underscore/dot/hyphen.
    if not isinstance(value,str) or not value or len(value)>129:
        raise ValueError("bounded nonempty legacy key required: "+label)
    parts=value.split("/")
    if len(parts)>(2 if label=="job" else 1):
        raise ValueError("legacy key components differ: "+label)
    allowed="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
    for part in parts:
        if not 1<=len(part)<=64 or part[0] not in allowed[:62] or any(ch not in allowed for ch in part):
            raise ValueError("typed canonical legacy name required: "+label)

def legacy_state(root,c,legacy):
    fields={"schema","root","snapshot_sha256","baseline_bytes","external_reserve_bytes","limit_bytes","controllers","jobs"}
    if not isinstance(legacy,dict) or set(legacy)!=fields:
        raise ValueError("exact known legacy ledger header required")
    if any(encoded(legacy[k])!=encoded(v) for k,v in {**c["legacy_invariants"],"root":str(root)}.items()):
        raise ValueError("legacy schema/root/snapshot/baseline/external/450GB drift")
    if not isinstance(legacy["jobs"],dict) or not isinstance(legacy["controllers"],dict):
        raise ValueError("legacy reservation collections malformed")
    paths=[];charge=0
    for group in ["jobs","controllers"]:
        for key,entry in legacy[group].items():
            legacy_key(key,"job" if group=="jobs" else "controller")
            required={"reserved_bytes","output"} if group=="jobs" else {"reserved_bytes","pid"}
            if not isinstance(entry,dict) or set(entry)!=required:
                raise ValueError("exact known legacy "+group+" entry required")
            charge+=integer(entry["reserved_bytes"],"legacy charge")
            if group=="controllers":
                integer(entry["pid"],"legacy controller PID",1)
                continue
            output=entry["output"]
            if not isinstance(output,str) or not Path(output).is_absolute():
                raise ValueError("legacy output must be canonical absolute string")
            p=canonical(root,output)
            if str(p)!=output or any(paths_overlap(p,a) for a in accounting_paths(root)):
                raise ValueError("legacy output overlaps accounting/mutex or differs from canonical identity")
            if any(paths_overlap(p,other) for other in paths):
                raise ValueError("legacy outputs have pairwise ancestor/descendant overlap")
            paths.append(p)
    projected=legacy["baseline_bytes"]+legacy["external_reserve_bytes"]+charge
    if projected>=c["legacy_invariants"]["limit_bytes"]:
        raise ValueError("legacy 450GB guard no longer qualifies")
    return paths,charge,projected

def scope_spec(root, value, stored=False):
    fields = {"key", "output", "auxiliary_scopes", "reserved_bytes", "consumer_bindings"}
    if stored: fields.add("root_admission")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("exact supplemental reservation schema required")
    key = value["key"]
    if not isinstance(key, str) or not key or len(key) > 256 or any(x in key for x in ["..", "\\", "\0"]):
        raise ValueError("bounded reservation key required")
    integer(value["reserved_bytes"], "reservation charge", 1)
    if not isinstance(value["auxiliary_scopes"], list) or len(value["auxiliary_scopes"]) > 8:
        raise ValueError("bounded auxiliary scopes required")
    paths = [canonical(root, x) for x in [value["output"], *value["auxiliary_scopes"]]]
    if [str(p) for p in paths] != [value["output"], *value["auxiliary_scopes"]]:
        raise ValueError("canonical absolute reservation scopes required")
    if any(paths_overlap(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("reservation scopes overlap")
    for p in paths:
        if any(paths_overlap(p,a) for a in accounting_paths(root)):
            raise ValueError("product scope overlaps live accounting/mutex namespace")
    if not isinstance(value["consumer_bindings"], list) or not 1 <= len(value["consumer_bindings"]) <= 64:
        raise ValueError("source/runtime consumer metadata bindings required")
    for ref in value["consumer_bindings"]:
        if Path(ref.get("path","")).suffix not in {".py",".sh",".json"}:
            raise ValueError("consumer bindings must be source/runtime authority metadata")
        bound(root, ref)
    if stored:
        admission = bound(root, value["root_admission"], True)
        required = {"kind":"project_budget_v3_root_reserve_admission","root_authorizer":"root_agent","active":True,
                    "implementation":reference(root,SOURCE),"config":reference(root,CONFIG),
                    "human_authorization":FIXED_POLICY["human_authorization"],
                    "all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,
                    "GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False}
        if any(encoded(admission.get(k))!=encoded(v) for k,v in required.items()):
            raise ValueError("stored reservation root authority drift")
        request = {k:v for k,v in value.items() if k!="root_admission"}
        if not any(encoded(request)==encoded(r) for r in admission.get("reservations",[])):
            raise ValueError("stored reservation not present in its root admission")
        review = bound(root,admission["independent_review"],True)
        expected = {"kind":"project_budget_v3_independent_review","review_role":"independent_reviewer","verdict":"PASS",
                    "implementation":required["implementation"],"config":required["config"],
                    "human_authorization":required["human_authorization"],"actual_apply_performed":False}
        if any(encoded(review.get(k))!=encoded(v) for k,v in expected.items()):
            raise ValueError("stored independent companion review drift")
    return paths

def empty_ledger(root, c, bindings):
    return {"schema": "project_budget_v3_supplemental_ledger", "root": str(root), **bindings,
            "legacy_invariants": c["legacy_invariants"],
            "scheduling_ceiling_bytes": c["scheduling_ceiling_bytes"], "hard_limit_bytes": c["hard_limit_bytes"],
            "reservations": {}}

def current(root, c, bindings):
    legacy, lsha = read_json(root, LEGACY)
    legacy_paths, charge, legacy_projected = legacy_state(root,c,legacy)
    p = canonical(root, SUPPLEMENTAL)
    if p.exists(): supplemental, ssha = read_json(root, p)
    else: supplemental, ssha = empty_ledger(root, c, bindings), None
    header = {k:v for k,v in supplemental.items() if k != "reservations"}
    if encoded(header) != encoded({k:v for k,v in empty_ledger(root,c,bindings).items() if k != "reservations"}):
        raise ValueError("supplemental authority/header drift")
    reservations = supplemental.get("reservations")
    if not isinstance(reservations, dict) or len(reservations) > 4096:
        raise ValueError("bounded supplemental reservation map required")
    allpaths = list(legacy_paths)
    for key, value in reservations.items():
        if key != value.get("key") or key in legacy["jobs"]: raise ValueError("duplicate reservation key")
        paths = scope_spec(root, value, True)
        if any(paths_overlap(a,b) for a in paths for b in allpaths):
            raise ValueError("supplemental scope already charged or overlaps another scope")
        allpaths.extend(paths)
    total_supplemental = sum(integer(x["reserved_bytes"], "supplemental charge", 1) for x in reservations.values())
    projected = legacy_projected + total_supplemental
    if projected >= c["scheduling_ceiling_bytes"] or projected >= c["hard_limit_bytes"]:
        raise ValueError("combined project budget ceiling exceeded")
    return legacy, supplemental, {"legacy_ledger_sha256":lsha,"supplemental_ledger_sha256":ssha,
            "baseline_bytes":legacy["baseline_bytes"],"external_reserve_bytes":legacy["external_reserve_bytes"],
            "legacy_reserved_bytes":charge,"supplemental_reserved_bytes":total_supplemental,
            "projected_bytes":projected,"limit_bytes":c["scheduling_ceiling_bytes"],"hard_limit_bytes":c["hard_limit_bytes"]}

def measure_scopes(root, paths):
    """Stat only, no file content decode. Contractual boundary checks are not OS quotas."""
    total = 0; count = 0; records = {}
    fields = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns)
    def add(p,s):
        nonlocal total,count
        records[str(p)] = fields(s); total += max(s.st_size,s.st_blocks*512);count += 1
        if count>1000000: raise ValueError("scope stat inventory too large")
    def scan(p):
        p = canonical(root,p);before = p.lstat()
        if not stat.S_ISDIR(before.st_mode): raise ValueError("scope directory changed type")
        add(p,before)
        # Capture directory identity before listing so membership changes during listing fail closed.
        with os.scandir(p) as entries: names = sorted(e.name for e in entries)
        for name in names:
            child=canonical(root,p/name);s=child.lstat()
            if stat.S_ISDIR(s.st_mode): scan(child)
            elif stat.S_ISREG(s.st_mode) and s.st_nlink==1: add(child,s)
            else: raise ValueError("unsafe scope namespace/hardlink")
        if fields(p.lstat())!=fields(before): raise ValueError("scope directory membership changed during boundary")
    for scope in paths:
        p = canonical(root,scope)
        if not p.exists(): continue
        scan(p)
    for path,before in records.items():
        if fields(Path(path).lstat())!=before: raise ValueError("scope stat/membership changed during boundary")
    return {"conservative_scope_bytes":total,"entries":count,"scope_paths":[str(p) for p in paths],"stat_only":True}

def check_reservation(root, reservation, *, expected_source_sha256, expected_config_sha256,
                      initial_ledger_pins=None, boundary=False, required_write_bytes=0):
    """Read-only API. Exact own reservation stays bound while unrelated legitimate charges may change."""
    root = Path(root)
    with mutex(root):
        c, bindings = policy(root, expected_source_sha256, expected_config_sha256)
        legacy, supplemental, view = current(root, c, bindings)
        if initial_ledger_pins is not None:
            if set(initial_ledger_pins) != {"legacy_ledger_sha256","supplemental_ledger_sha256"} or any(view[k] != v for k,v in initial_ledger_pins.items()):
                raise ValueError("first-read exact ledger hash binding changed")
        own = supplemental["reservations"].get(reservation.get("key"))
        if own is None or encoded(own) != encoded(reservation): raise ValueError("own supplemental reservation changed or absent")
        paths = scope_spec(root, own, True)
        integer(required_write_bytes,"planned boundary write")
        if required_write_bytes and not boundary: raise ValueError("planned write requires scope boundary measurement")
        measurement = measure_scopes(root, paths) if boundary else None
        if measurement is not None and measurement["conservative_scope_bytes"] + required_write_bytes > own["reserved_bytes"]:
            raise ValueError("actual scope boundary exceeds declared reservation")
        return {**view,"reservation":copy.deepcopy(own),"reservation_sha256":hashlib.sha256(encoded(own)).hexdigest(),
                "scope_measurement":measurement,"required_write_bytes":required_write_bytes,
                "remaining_scope_bytes":own["reserved_bytes"]-measurement["conservative_scope_bytes"] if measurement else None,
                "projection_is_historical_accounting_not_global_remeasurement":True,
                "legacy_ledger_mutated":False,"supplemental_ledger_mutated":False}

def reserve(root, authority_ref, *, expected_source_sha256, expected_config_sha256):
    """Root-only exact authority. Sole mutation creates/appends supplemental reservations atomically."""
    root = Path(root)
    with mutex(root):
        c, bindings = policy(root, expected_source_sha256, expected_config_sha256)
        authority = bound(root, authority_ref, True)
        required = {"kind":"project_budget_v3_root_reserve_admission","root_authorizer":"root_agent","active":True,
                    **bindings,"all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,
                    "GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False}
        if any(encoded(authority.get(k)) != encoded(v) for k,v in required.items()):
            raise ValueError("exact activated root reserve authority required")
        review = bound(root, authority["independent_review"], True)
        expected_review = {"kind":"project_budget_v3_independent_review","review_role":"independent_reviewer",
                           "verdict":"PASS",**bindings,"actual_apply_performed":False}
        if any(encoded(review.get(k)) != encoded(v) for k,v in expected_review.items()):
            raise ValueError("fresh exact independent companion PASS required")
        legacy, supplemental, before = current(root, c, bindings)
        initial = authority["expected_before_ledger_pins"]
        if set(initial) != {"legacy_ledger_sha256","supplemental_ledger_sha256"} or any(before[k] != v for k,v in initial.items()):
            raise ValueError("root reserve initial ledger hashes changed")
        incoming = authority["reservations"]
        if not isinstance(incoming, list) or not 1 <= len(incoming) <= 64: raise ValueError("bounded new reservation batch required")
        proposed = copy.deepcopy(supplemental); occupied = [canonical(root,x["output"]) for x in legacy["jobs"].values()]
        for value in proposed["reservations"].values(): occupied += scope_spec(root,value,True)
        for request in incoming:
            paths = scope_spec(root, request)
            key = request["key"]
            if key in legacy["jobs"] or key in proposed["reservations"]: raise ValueError("permanent reservation key cannot be reused")
            if any(p.exists() for p in paths) or any(paths_overlap(a,b) for a in paths for b in occupied):
                raise ValueError("new reservation scopes must be fresh and exclusively charged")
            value = copy.deepcopy(request); value["root_admission"] = copy.deepcopy(authority_ref)
            proposed["reservations"][key] = value; occupied += paths
        newcharge = sum(x["reserved_bytes"] for x in proposed["reservations"].values())
        projected = before["projected_bytes"] - before["supplemental_reserved_bytes"] + newcharge
        if projected >= c["scheduling_ceiling_bytes"] or projected >= c["hard_limit_bytes"]:
            raise ValueError("new combined reservation exceeds project ceiling")
        # Recheck source/authority/legacy and existing supplemental bytes immediately before sole write.
        policy(root,expected_source_sha256,expected_config_sha256); bound(root,authority_ref,True)
        bound(root,authority["independent_review"],True)
        for value in proposed["reservations"].values(): scope_spec(root,value,True)
        if read_json(root,LEGACY)[1] != before["legacy_ledger_sha256"]: raise ValueError("legacy changed under mutex")
        p = canonical(root,SUPPLEMENTAL)
        actual = read_json(root,p)[1] if p.exists() else None
        if actual != before["supplemental_ledger_sha256"]: raise ValueError("supplemental changed under mutex")
        p.parent.mkdir(parents=True,exist_ok=True)
        data = encoded(proposed)+b"\n"
        if len(data)>MAX_JSON: raise ValueError("supplemental ledger too large")
        tmp = canonical(root,p.with_name("."+p.name+"."+uuid.uuid4().hex+".tmp"))
        fd = os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(fd,"wb") as stream: stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,p)
        finally: tmp.unlink(missing_ok=True)
        written, aftersha = read_json(root,p)
        if encoded(written) != encoded(proposed): raise ValueError("post-write supplemental uncertainty; no automatic rollback")
        if read_json(root,LEGACY)[1] != before["legacy_ledger_sha256"]: raise ValueError("legacy changed; preserve root uncertainty evidence")
        return {"kind":"project_budget_v3_root_reserve_receipt","before":before,"supplemental_ledger_sha256":aftersha,
                "legacy_ledger_sha256":before["legacy_ledger_sha256"],"projected_bytes":projected,
                "limit_bytes":c["scheduling_ceiling_bytes"],"hard_limit_bytes":c["hard_limit_bytes"],
                "admitted_reservations":{r["key"]:proposed["reservations"][r["key"]] for r in incoming},
                "legacy_ledger_mutated":False,"actual_scientific_execution":False}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reserve",action="store_true")
    p.add_argument("--expected-source-sha256");p.add_argument("--expected-config-sha256")
    p.add_argument("--admission");p.add_argument("--expected-admission-sha256")
    args = p.parse_args(argv)
    if not args.reserve:
        print(json.dumps({"kind":"project_budget_v3_source_report","implementation":reference(ROOT,SOURCE),
            "config":reference(ROOT,CONFIG),"human_authorization":FIXED_POLICY["human_authorization"],
            "live_ledger_or_scope_access":False,"actual_reserve_apply_performed":False},indent=2));return 0
    if not all([args.expected_source_sha256,args.expected_config_sha256,args.admission,args.expected_admission_sha256]):
        raise ValueError("root exact source/config/admission opt-in required")
    result = reserve(ROOT,{"path":args.admission,"sha256":args.expected_admission_sha256},
                     expected_source_sha256=args.expected_source_sha256,expected_config_sha256=args.expected_config_sha256)
    print(json.dumps(result,indent=2));return 0

if __name__ == "__main__": raise SystemExit(main())
