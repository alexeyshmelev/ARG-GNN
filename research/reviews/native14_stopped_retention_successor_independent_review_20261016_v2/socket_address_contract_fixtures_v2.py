"""Fake kernel catalogs and APIs only. No original filesystem, process or endpoint probes."""
import ast,copy,hashlib,json,os,posixpath,types
from pathlib import Path
HERE=Path(__file__).parent
REPORT=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def passed(name):REPORT.append(dict(case=name,passed=True))
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,OSError,KeyError):passed(name);return
    raise AssertionError("Expected rejection: "+name)
source=(HERE/"native14_stopped_retention_v1.py").read_text();tree=ast.parse(source)
names={"_parse_kernel_endpoints_v2","_native_kernel_path_absent_v2","_native_socket_absent","_native_targets"}
nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
env=dict(require=require,os=types.SimpleNamespace(fsencode=os.fsencode),posixpath=posixpath,Path=Path)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"<fake-kernel-source-contracts>","exec"),env)
parse=env["_parse_kernel_endpoints_v2"];lex=env["_native_kernel_path_absent_v2"]
header=b"Num       RefCount Protocol Flags    Type St Inode Path\n"
def row(address=b"/external/unrelated.sock",inode=12345,protocol=b"00000000",kind=b"0001",state=b"01"):
    suffix=b"" if address is None else b" "+address
    return b"0000000000000000: 00000002 "+protocol+b" 00010000 "+kind+b" "+state+b" "+str(inode).encode()+suffix+b"\n"
target="/fake/root/output/retained.sock"
valid=header+row()+row(b"@unrelated_abstract",12346)+row(None,12347)
catalog=parse(valid)
assert catalog==[b"/external/unrelated.sock",b"@unrelated_abstract"]
passed("complete_fake_catalog_parses_named_and_unnamed_rows")
assert lex(target,catalog) is True
passed("lexical_filter_accepts_unrelated_names_without_claiming_socket_inactivity")
for name,address in [
 ("canonical_absolute",b"/fake/root/output/retained.sock"),
 ("relative_basename",b"retained.sock"),
 ("relative_scoped",b"output/retained.sock"),
 ("redundant_slashes",b"/fake//root///output//retained.sock"),
 ("dot_components",b"/fake/root/./output/retained.sock"),
 ("dotdot_components",b"/fake/root/scratch/../output/retained.sock"),
 ("alias_parent_same_basename",b"/uninspected/alias/retained.sock"),
 ("relative_alias_parent",b"../alias/retained.sock"),
 ("double_leading_slash",b"//fake/root/output/retained.sock")]:
    endpoints=parse(header+row(address))
    reject("lexical_ambiguity_blocks_"+name,lambda endpoints=endpoints:lex(target,endpoints))
    env["_kernel_endpoints"]=lambda endpoints=endpoints:endpoints
    reject("production_socket_callback_blocks_"+name,lambda:env["_native_socket_absent"](target))
for name,address in [("renamed_different_basename",b"/external/original-name.sock"),("final_alias_different_basename",b"/external/link-name.sock")]:
    env["_kernel_endpoints"]=lambda address=address:parse(header+row(address))
    reject("production_callback_blocks_unqualified_"+name,lambda:env["_native_socket_absent"](target))
env["_kernel_endpoints"]=lambda:catalog
reject("production_callback_never_certifies_unmapped_socket_despite_raw_nonmatch",lambda:env["_native_socket_absent"](target))
for name,data in [
 ("empty",b""),("header_only",header),("blank_row",header+b"\n"),("no_header",row()),
 ("truncated_no_newline",(header+row())[:-1]),("unknown_header",header.replace(b"Protocol",b"Unknown")+row()),
 ("bad_numeric",header+row().replace(b"00010000",b"zzzzzzzz")),
 ("unknown_protocol",header+row(protocol=b"00000001")),
 ("unknown_type",header+row(kind=b"0099")),("unknown_state",header+row(state=b"99")),
 ("zero_inode",header+row(inode=0)),("duplicate_inode",header+row()+row(b"@different")),
 ("unnamed_only",header+row(None)),("leading_path_space",header+row(b" retained.sock")),
 ("embedded_path_space",header+row(b"/some path/socket")),
 ("trailing_path_space",header+row(b"/path/socket ")),("control_path",header+row(b"/path/\tsocket")),
 ("NUL_path",header+row(b"/path/\x00socket")),("bad_row",header+b"garbage\n")]:
    reject("catalog_fail_closed_"+name,lambda data=data:parse(data))
for name,unknown in [("empty",[]),("None",None),("unknown_type",[None]),("empty_address",[b""]),("terminal_dot",[b"."]),("terminal_dotdot",[b".."]),("whitespace",[b"foo bar"])]:
    reject("lexical_catalog_fail_closed_"+name,lambda unknown=unknown:lex(target,unknown))
# Run actual kernel-read wrapper only through fake Path/BytesIO, never real /proc.
wrapper=next(copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_kernel_endpoints")
class FakeFile:
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,count):
        assert count==16*2**20+1
        return valid
class FakePath:
    def __init__(self,path):assert path=="/proc/net/unix"
    def open(self,mode):
        assert mode=="rb"
        return FakeFile()
wrapper_env=dict(env,Path=FakePath)
exec(compile(ast.Module(body=[wrapper],type_ignores=[]),"<fake-bounded-catalog-read>","exec"),wrapper_env)
assert wrapper_env["_kernel_endpoints"]()==catalog
passed("actual_catalog_wrapper_uses_bounded_read_and_strict_parser_through_fake_API")
# The known-target path must reject preserved socket identity as unqualified; all
# process/group observations below are fake functions with no system operations.
metadata=types.SimpleNamespace(st_dev=1,st_ino=2,st_mode=0o140755,st_nlink=1,st_size=0,st_mtime_ns=3,st_ctime_ns=4)
known=dict(known_PID_start_rows=[dict(pid=99999,start_ticks="UNKNOWN")],known_groups=[99999],
    recorded_filesystem_endpoints=[dict(path="output/retained.sock",original_lstat=[1,2,0o140755,1,0,3,4])])
def absent_group(*args):raise ProcessLookupError()
helper=types.SimpleNamespace(lstat_no_follow_v1=lambda *args:metadata)
env.update(ROOT=Path("/fake/root"),_kernel_endpoints=lambda:catalog,
    _absent=lambda pid:dict(checked_utc="FAKE"),_utc=lambda:"FAKE",
    _native_helper=lambda:helper,stat=__import__("stat"),
    os=types.SimpleNamespace(fsencode=os.fsencode,killpg=absent_group))
reject("known_preserved_socket_identity_alone_cannot_qualify_inactivity",lambda:env["_native_targets"](known))
def missing(*args):raise FileNotFoundError()
helper.lstat_no_follow_v1=missing
facts=env["_native_targets"](known)
assert facts["endpoint_observations"]==[["output/retained.sock","absent_path","FAKE"]]
passed("known_absent_socket_path_can_be_observed_with_fake_targets_and_unrelated_catalog")
for name,address in [("relative",b"retained.sock"),("slash",b"/fake//root/output/retained.sock"),("dot",b"/fake/root/./output/retained.sock"),("alias",b"/alias/retained.sock")]:
    env["_kernel_endpoints"]=lambda address=address:parse(header+row(address))
    reject("known_absent_path_still_blocks_live_address_ambiguity_"+name,lambda:env["_native_targets"](known))
result=dict(schema="native14_v2_socket_address_fail_closed_fake_contract_results",status="PASS",cases=len(REPORT),results=REPORT,
    actual_kernel_catalog_read=False,actual_process_group_or_endpoint_probes=False,external_path_resolution_or_following=False,
    payload_reads_or_accounting=False,unqualified_retained_filesystem_sockets_preserved_and_blocked=True)
(HERE/"socket_address_contract_results.v2.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASS",socket_fake_cases=len(REPORT)),sort_keys=True))
