"""Transaction-local I/O memo for explicitly pinned future consumers.

No import/configure/install side effects. install() is explicit. This is not
an accounting authority: all original checks and journals remain in use.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib, marshal, stat, sys, threading

ADAPTER_SCHEMA = "resource_guard_transaction_memo_adapter_v1"
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ENTRIES = 4096
MAX_KEY_BYTES = 16384

def _identity(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def _implementation(fn):
    while hasattr(fn,"__wrapped__"):fn=fn.__wrapped__
    return fn

def _function_identity(fn):
    implementation=_implementation(fn)
    return dict(implementation_code_file=implementation.__code__.co_filename,implementation_first_line=implementation.__code__.co_firstlineno,module=fn.__module__,qualname=fn.__qualname__,
                code_file=fn.__code__.co_filename,first_line=fn.__code__.co_firstlineno,
                process_function_id=id(fn),
                code_sha256=hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest())

class TransactionMemoAdapter:
    """Install once before future consumer threads start; no live job mutation."""
    def __init__(self,module,*,expected_source_path,expected_source_sha256,
                 max_bytes=DEFAULT_MAX_BYTES,max_entries=DEFAULT_MAX_ENTRIES):
        if type(max_bytes) is not int or type(max_entries) is not int or not (0<max_bytes<=DEFAULT_MAX_BYTES and 0<max_entries<=DEFAULT_MAX_ENTRIES):
            raise ValueError("Bounded positive memo limits required")
        source=Path(expected_source_path)
        if source.resolve()!=source or Path(module.__file__).resolve()!=source:
            raise ValueError("Exact original source pathname required")
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_source_sha256:
            raise ValueError("Original source SHA changed")
        self.module=module;self.max_bytes=max_bytes;self.max_entries=max_entries
        self.original={name:getattr(module,name) for name in ("canonical","file_bytes","mutex")}
        for name,fn in self.original.items():
            if fn.__module__!=module.__name__ or Path(_implementation(fn).__code__.co_filename).resolve()!=source:
                raise ValueError("Unwrapped original function identities required: "+name)
        reference=getattr(module,"reference",None)
        if reference is None or reference.__module__!=module.__name__ or Path(reference.__code__.co_filename).resolve()!=source:
            raise ValueError("Original reference helper required")
        # Canonical calls from fingerprints, journal listing and scope inventories
        # are deliberately excluded. These two pinned read bodies alone may memo.
        self.namespace_callers=frozenset((self.original["file_bytes"].__code__,reference.__code__))
        self.tls=threading.local();self.active_count=0;self.lifecycle_lock=threading.Lock()
        self.receipt=dict(schema=ADAPTER_SCHEMA,original_source=dict(path=str(source),sha256=expected_source_sha256),
                          original_functions={k:_function_identity(v) for k,v in self.original.items()},
                          canonical_memo_callers=[_function_identity(reference),_function_identity(self.original["file_bytes"])],
                          max_retained_payload_and_key_charge_bytes=max_bytes,max_entries=max_entries,
                          cross_operation_cache=False,installed=False)
        self.canonical_wrapper=self._make_canonical()
        self.file_bytes_wrapper=self._make_file_bytes()
        self.mutex_wrapper=self._make_mutex()

    def _state(self):
        return getattr(self.tls,"state",None)

    def _key(self,root,path):
        # Do not broaden the original API's accepted argument types.
        if not isinstance(root,Path) or not isinstance(path,(str,Path)):
            return None
        p=Path(path);p=p if p.is_absolute() else root/p
        key=(type(root),str(root),str(p))
        charge=len(str(root).encode())+len(str(p).encode())+256
        return (key,charge) if charge<=MAX_KEY_BYTES else None

    def _room(self,state,charge):
        return state["entries"]<self.max_entries and state["charge"]+charge<=self.max_bytes

    def _make_canonical(self):
        original=self.original["canonical"]
        def canonical(root,path):
            state=self._state()
            if state is None or sys._getframe(1).f_code not in self.namespace_callers:
                return original(root,path)
            item=self._key(root,path)
            if item is None:return original(root,path)
            key,charge=item
            if key in state["canonical"]:
                state["canonical_hits"]+=1
                return state["canonical"][key][2]
            result=original(root,path);state["canonical_misses"]+=1
            if self._room(state,charge):
                state["canonical"][key]=(root,path,result)
                state["entries"]+=1;state["charge"]+=charge
            return result
        return canonical

    def _make_file_bytes(self):
        original=self.original["file_bytes"]
        def file_bytes(root,path):
            state=self._state()
            if state is None:return original(root,path)
            item=self._key(root,path)
            if item is None:return original(root,path)
            key,key_charge=item
            if key in state["raw"]:
                state["raw_hits"]+=1
                return state["raw"][key][3]
            # Original reader retains O_NOFOLLOW, before/after lstat/fstat,
            # size/type/link checks, read bound and SHA. No stat is memoized.
            p=self.original["canonical"](root,path);before=_identity(p.lstat())
            result=original(root,path);state["raw_misses"]+=1
            if _identity(p.lstat())!=before:
                raise ValueError("Memo read identity changed during source read")
            charge=len(result[0])+key_charge+512
            if self._room(state,charge):
                state["raw"][key]=(root,path,before,result)
                state["entries"]+=1;state["charge"]+=charge
            else:state["capacity_bypasses"]+=1
            return result
        return file_bytes

    def _validate_snapshot(self,state):
        # Recheck namespace/leaf identities uncached, still under the original
        # mutex, before a successful guard can authorize its caller's next write.
        for root,path,result in state["canonical"].values():
            if self.original["canonical"](root,path)!=result:
                raise ValueError("Memo namespace changed during storage operation")
        for root,path,before,_ in state["raw"].values():
            p=self.original["canonical"](root,path);after=p.lstat()
            if _identity(after)!=before or not stat.S_ISREG(after.st_mode) or after.st_nlink!=1:
                raise ValueError("Memo source/evidence identity changed during storage operation")

    def _make_mutex(self):
        original=self.original["mutex"]
        @contextmanager
        def mutex(root):
            if self._state() is not None:
                raise RuntimeError("Nested storage operations are rejected before relocking")
            with self.lifecycle_lock:
                if not self.receipt["installed"]:raise RuntimeError("Adapter is uninstalled")
                self.active_count+=1
            try:
              with original(root):
                state=dict(raw={},canonical={},entries=0,charge=0,
                           raw_hits=0,raw_misses=0,canonical_hits=0,canonical_misses=0,capacity_bypasses=0)
                self.tls.state=state
                success=False
                try:
                    yield
                    self._validate_snapshot(state)
                    success=True
                finally:
                    self.tls.last={k:v for k,v in state.items() if type(v) is int}
                    self.tls.last["success"]=success
                    state["raw"].clear();state["canonical"].clear()
                    self.tls.state=None
            finally:
                with self.lifecycle_lock:self.active_count-=1
        return mutex

    def install(self):
        if self.receipt["installed"]:raise RuntimeError("Adapter already installed")
        for name,fn in self.original.items():
            if getattr(self.module,name) is not fn:raise RuntimeError("Original function changed before install")
        self.module.canonical=self.canonical_wrapper
        self.module.file_bytes=self.file_bytes_wrapper
        self.module.mutex=self.mutex_wrapper
        self.receipt["installed"]=True
        return self

    def uninstall(self):
        # Explicit lifecycle only, before/after all consumer threads, never live.
        with self.lifecycle_lock:
            if self.active_count:raise RuntimeError("Cannot uninstall during a storage operation")
            for name,fn in (("canonical",self.canonical_wrapper),("file_bytes",self.file_bytes_wrapper),("mutex",self.mutex_wrapper)):
                if getattr(self.module,name) is not fn:raise RuntimeError("Adapter function identity drift")
            for name,fn in self.original.items():setattr(self.module,name,fn)
            self.receipt["installed"]=False

    def last_operation_counts(self):
        return dict(getattr(self.tls,"last",{}))

def install(module,**kwargs):
    return TransactionMemoAdapter(module,**kwargs).install()
