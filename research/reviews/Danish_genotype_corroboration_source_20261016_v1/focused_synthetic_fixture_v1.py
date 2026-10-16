"""Root-deferred tiny synthetic API checks. No network, real rows or metrics."""
from pathlib import Path
import argparse,gzip,importlib.util,math,struct
def run():
    p=Path(__file__).with_name("corroboration_v1.py")
    spec=importlib.util.spec_from_file_location("h055_reviewed_source",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.RECORD.size==160 and m.RANGE_RECEIPT.size==45
    assert [m.dosage_gp(x) for x in [b"1,0,0",b"0,1,0",b"0,0,1"]]==[0,1,2]
    assert abs(m.dosage_gp(b"0.1,0.2,0.7")-1.6)<1e-12
    assert abs(m.dosage_gp(b"0.1,0.2,0.701")-1.602/1.001)<1e-12
    assert all(m.dosage_gp(x) is None for x in [b".",b"0/1",b"0.2,0.2,0.2",b"-0.1,0.2,0.9",b"nan,0,1"])
    assert m.unpack_gzip(gzip.compress(b"first")+gzip.compress(b"second"),11)==b"firstsecond"
    try:m.unpack_gzip(gzip.compress(b"toolong"),2);raise AssertionError("expansion cap")
    except ValueError:pass
    fixed=[b"20",b"20000001",b".",b"A",b"C",b".",b"PASS",b"INFO=0.9;MASK_1000G;AF=0.99",b"GT:DS:GP:AP"]
    samples=[b"1|1:2:0.25,0.5,0.25:ANCESTRY_SENTINEL"]*3331
    row=b"\t".join(fixed+samples)
    result,reason=m.site_v1(row,list(range(18)),"20")
    assert reason=="metadata_eligible" and result[3]==(1<<18)-1 and result[4]==[1.]*18
    # Different GT/DS/AP with same GP must have identical parsed dosage.
    samples2=[b"0/0:0:0.25,0.5,0.25:OTHER_SENTINEL"]*3331
    assert m.site_v1(b"\t".join(fixed+samples2),list(range(18)),"20")== (result,reason)
    bad=samples.copy();bad[0]=b"0/1:1:.:NO_FALLBACK"
    r,_=m.site_v1(b"\t".join(fixed+bad),list(range(18)),"20")
    assert math.isnan(r[4][0]) and not r[3]&1
    trans=fixed.copy();trans[4]=b"G"
    assert m.site_v1(b"\t".join(trans+samples),list(range(18)),"20")[1]=="transition"
    index=b"TBI\x01"+struct.pack("<8i",1,2,1,2,0,35,0,3)+b"20\0"+struct.pack("<i",1)+struct.pack("<IiQQi",0,1,0,1000000<<16,0)
    assert m.chunks_tbi(gzip.compress(index[:20])+gzip.compress(index[20:]),"20")==[(0,1000000<<16)]
    return "synthetic API checks completed; no acquisition or empirical f4"
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--execute-synthetic-only",action="store_true");a=p.parse_args()
    if a.execute_synthetic_only:print(run())
    else:print("root-deferred; no checks executed")
