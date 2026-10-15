"""Additive, read-only marker-repair authority renewal for fresh interpreters.

No original file, allocation expiry, journal, clock, proof, credit or guard method
is changed. Install only before future exact guard imports. The import hook routes
one exact source path; its private _bound changes one exact authority's returned
expiry after revalidating a separate root certificate on every such read.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,hashlib,importlib.machinery,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H039_marker_repair_renewal_source_20261016_v1"
SOURCE=HERE+"/marker_repair_renewal_v1.py"
BOOTSTRAP=HERE+"/bootstrap_renewal_v1.py"
OLD_AUTHORITY_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",
 sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
OLD_SOURCE_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py",
 sha256="e4185dc51b285d8ad4eba224b8e1e15f3e1479f57c73b4ec53fff2b238d0ef2f")
OLD_BARRIER_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/old_guard_terminal_barrier_20261015.json",
 sha256="85a69aa907ed5c0c80ca5c51232a58ef67a212bceca71b367f1446acdaf8459f")
KEY="H039/directed-clade-exchange-train-stage-seed101-v1"
RESERVATION_SHA="e824b52b0bb9ff26e1b7d81b48cfe03c85ee58d5aab0120601b52f43f9c5acee"


def require(ok,message):
    if not ok:raise ValueError(message)


def _file(name):
    p=ROOT/name
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),
        "Canonical regular contained renewal binding")
    return p


def reference(name):
    return dict(path=name,sha256=hashlib.sha256(_file(name).read_bytes()).hexdigest())


def _json(record,limit=1024*1024):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact renewal path/SHA256 ref")
    p=_file(record["path"])
    with p.open("rb") as stream:raw=stream.read(limit+1)
    require(len(raw)<=limit and hashlib.sha256(raw).hexdigest()==record["sha256"],"Renewal bound bytes changed")
    return json.loads(raw)


def validate_certificate_v1(record):
    certificate=_json(record,65536)
    fields={"schema","active","root_authorizer","project_root","source_ref","bootstrap_ref",
        "old_authority_ref","old_repair_source_ref","old_activation_barrier_ref","reservation_sha256",
        "retained_allowance_bytes","accounting_credit_bytes","issued_at_utc","renewed_expires_at_utc",
        "read_only_marker_authority_expiry_renewal","fresh_admissions_and_existing_read_close_authorized",
        "all_allocation_expiries_unchanged","historical_journal_mutation_authorized",
        "historical_source_or_binding_mutation_authorized","global_rebase_authorized",
        "malformed_key_future_writes_or_relaunch_authorized","old_terminal_barrier_reused_without_new_absence_claim"}
    require(set(certificate)==fields and certificate.get("schema")=="H039_marker_repair_expiry_renewal_root_certificate_v1"
        and certificate.get("active") is True and certificate.get("root_authorizer")=="root_agent"
        and certificate.get("project_root")==str(ROOT),"Exact active root read-only renewal certificate")
    require(certificate["source_ref"]==reference(SOURCE) and certificate["bootstrap_ref"]==reference(BOOTSTRAP)
        and certificate["old_authority_ref"]==OLD_AUTHORITY_REF
        and certificate["old_repair_source_ref"]==OLD_SOURCE_REF
        and certificate["old_activation_barrier_ref"]==OLD_BARRIER_REF
        and reference(OLD_SOURCE_REF["path"])==OLD_SOURCE_REF,"Exact old/new renewal source and barrier pins")
    require(certificate["reservation_sha256"]==RESERVATION_SHA
        and certificate["retained_allowance_bytes"]==15000000 and certificate["accounting_credit_bytes"]==0,
        "Original full15MB charge and zero credit preserved")
    for name in ("read_only_marker_authority_expiry_renewal","fresh_admissions_and_existing_read_close_authorized",
        "all_allocation_expiries_unchanged","old_terminal_barrier_reused_without_new_absence_claim"):
        require(certificate[name] is True,"Explicit narrow read/close/new-admission renewal scope")
    for name in ("historical_journal_mutation_authorized","historical_source_or_binding_mutation_authorized",
        "global_rebase_authorized","malformed_key_future_writes_or_relaunch_authorized"):
        require(certificate[name] is False,"No historical mutation/rebase/old-key relaunch")
    issued=datetime.fromisoformat(certificate["issued_at_utc"].replace("Z","+00:00"))
    expires=datetime.fromisoformat(certificate["renewed_expires_at_utc"].replace("Z","+00:00"))
    now=datetime.now(timezone.utc)
    require(issued.tzinfo is not None and expires.tzinfo is not None and issued<=now<expires
        and expires-issued<=timedelta(days=7),"Current real-time bounded root renewal expiry")
    old=_json(OLD_AUTHORITY_REF);_json(OLD_BARRIER_REF)
    require(old.get("activation_barrier_ref")==OLD_BARRIER_REF and old.get("source_ref")==OLD_SOURCE_REF
        and old.get("reservation_sha256")==RESERVATION_SHA and old.get("retained_allowance_bytes")==15000000
        and old.get("accounting_credit_bytes")==0 and old.get("active") is True,"Exact original proved marker authority")
    original_expiry=datetime.fromisoformat(old["expires_at_utc"].replace("Z","+00:00"))
    require(original_expiry.tzinfo is not None and expires>original_expiry,"Explicit additive expiry extension only")
    return certificate,old


def _wrap_private_marker_v1(module,record):
    certificate,old=validate_certificate_v1(record)
    require(Path(module.__file__).resolve()==ROOT/OLD_SOURCE_REF["path"]
        and reference(OLD_SOURCE_REF["path"])==OLD_SOURCE_REF,"Only exact private original marker module")
    prior=getattr(module,"_root_expiry_renewal_ref",None)
    if prior is not None:
        require(prior==record,"No renewal authority swap in one interpreter")
        return module
    original_bound=module._bound
    def renewed_bound(bound_ref):
        value=original_bound(bound_ref)
        if bound_ref!=OLD_AUTHORITY_REF:return value
        current,proved=validate_certificate_v1(record)
        require(value==proved,"Original bound reader still returns exact original authority")
        renewed=copy.deepcopy(value)
        renewed["expires_at_utc"]=current["renewed_expires_at_utc"]
        return renewed
    module._bound=renewed_bound
    module._root_expiry_renewal_ref=copy.deepcopy(record)
    return module


def install_renewal_v1(record):
    """Install for future marker imports only; never patch an existing consumer."""
    validate_certificate_v1(record)
    loader=importlib.machinery.SourceFileLoader
    current=loader.exec_module
    prior=getattr(current,"_H039_expiry_renewal_ref",None)
    if prior is not None:
        require(prior==record,"No renewal certificate swap in one interpreter")
        return copy.deepcopy(record)
    target=ROOT/OLD_SOURCE_REF["path"]
    for loaded in tuple(sys.modules.values()):
        name=getattr(loaded,"__file__",None)
        require(name is None or Path(name).resolve()!=target,
            "Install renewal before importing any original marker consumer")
    def exact_marker_exec(loader_instance,module):
        name=Path(loader_instance.path)
        if name.resolve()!=target:
            return current(loader_instance,module)
        validate_certificate_v1(record)
        require(name==target and reference(OLD_SOURCE_REF["path"])==OLD_SOURCE_REF,
            "Exact source path/bytes at fresh marker load")
        current(loader_instance,module)
        _wrap_private_marker_v1(module,record)
    exact_marker_exec._H039_expiry_renewal_ref=copy.deepcopy(record)
    loader.exec_module=exact_marker_exec
    return copy.deepcopy(record)
