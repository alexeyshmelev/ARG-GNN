"""Exact8MiB H056 map scope hotpath. Original guards/accounting unchanged."""
from pathlib import Path
import hashlib,importlib.util,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/H056_map_acquisition_root_source_20261016_v1/map_scope_check_v1.py"
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")

def require(ok,message):
    if not ok:raise ValueError(message)

class ControllerMetadataScopeV1:
    """One full original guard; each later write rechecks only fully charged scope.

    Every call remains under original v4 mutex, authenticates direct consumers,
    source/policy/storage expiry and append-only overlay journal, rejects closure
    of its own reservation, and invokes original measure_scopes plus cap inequality.
    Other scopes are never credited, measured, borrowed, or treated as writable.
    Subsequent receipts explicitly do not claim a refreshed global projection.
    """
    def __init__(self,guard,storage):
        require(type(storage["reserved_new_bytes"]) is int and storage["reserved_new_bytes"]==8*2**20,
            "Only one exact8MiB fully charged H056 map reservation")
        require(storage["reservation"]["auxiliary_scopes"]==[],"Controller has exactly one isolated output scope")
        for r in (MEMO_REF,V4_REF):
            require(hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()==r["sha256"],"Exact reviewed memo2/originalv4")
        require(dict(path=SOURCE,sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()) in storage["reservation"]["consumer_bindings"],
            "Controller scope checker directly bound in actual reservation")
        # Memo2 caches only within the original v4 mutex transaction and clears
        # at exit. No source/auth/journal data survives between checks.
        sp=importlib.util.spec_from_file_location("_GPU45_controller_metadata_memo2_v1",ROOT/MEMO_REF["path"])
        memo=importlib.util.module_from_spec(sp);sys.modules[sp.name]=memo;sp.loader.exec_module(memo)
        self.memo=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
        self.guard=guard;self.storage=storage
        self.initial=guard.budget_check_v9(storage,first=True,boundary=True,required_write_bytes=0)
        self.initial_prefix=dict(self.initial["completed_overlay_prefix"]);self.bindings=dict(self.initial["bindings"])
        self.initial_monotonic=time.monotonic()

    def check(self,required=0):
        g=self.guard;s=self.storage;own=s["reservation"]
        with g.budget.v4.mutex(ROOT):
            g.budget.v4.integer(required,"planned isolated controller growth")
            g._expire(s);policy=g._policy()
            require(s["completed_overlay_policy_ref"]==g._POLICY_REF
                and s["budget_components"]["implementation"]==policy["implementation_ref"]
                and s["budget_components"]["launcher"]==policy["launcher_ref"]
                and s["budget_components"]["consumer_adapter"]==policy["source_ref"],"Exact unchanged controller source/policy/storage authority")
            for r in own["consumer_bindings"]:g._raw(r)
            events,hashes=g._events(self.bindings)
            g.check_prefix(self.initial_prefix,hashes)
            g.check_prefix(s["initial_ledger_pins"]["completed_overlay_prefix"],hashes)
            require(s["initial_ledger_pins"]["base_pins"]==policy["base_pins"],"Original immutable ancestor pins")
            reserves=[]
            for i,event in enumerate(events):
                auth=g._authority(event["authority"],event["operation"],self.bindings,fresh=False)
                expected=dict(base_pins=policy["base_pins"],completed_overlay_prefix=dict(events=i,tail_sha256=hashes[i-1] if i else None))
                require(auth["expected_before"]==expected,"Actual immutable ordered root journal authority")
                if event["operation"]=="reserve" and event["value"].get("key")==own["key"]:
                    require(event["value"]==dict(auth["request"],root_admission=event["authority"])==own,"Exact original controller reserve event")
                    reserves.append(i+1)
                require(not(event["operation"]=="close_completed" and auth["request"].get("key")==own["key"]),
                    "Closed/revoked controller scope cannot write")
            require(len(reserves)==1 and reserves[0]<=s["initial_ledger_pins"]["completed_overlay_prefix"]["events"],
                "Exactly one genuine controller reserve before pinned prefix")
            measured=g.budget.v4.measure_scopes(ROOT,g._paths(own))
            used=measured["conservative_scope_bytes"]
            require(used+required<=own["reserved_bytes"]==s["reserved_new_bytes"],"Full conservative isolated scope plus planned write fits charged cap")
            return dict(schema="GPU45_isolated_controller_metadata_scope_check_v1",accounting_mode=g.MODE,
                original_full_global_guard_passed_at_startup=True,global_projection_recomputed_this_check=False,
                global_credit_or_rebase_performed=False,original_v4_scope_measurement=True,
                no_cross_transaction_cache=True,completed_overlay_prefix=dict(events=len(events),tail_sha256=hashes[-1] if hashes else None),
                reservation_key=own["key"],reserved_scope_bytes=own["reserved_bytes"],
                scope_measurement=measured,required_write_bytes=required,remaining_scope_bytes=own["reserved_bytes"]-used)
