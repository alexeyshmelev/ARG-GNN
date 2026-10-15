"""Handmade H018 adapter checks. No remote calls or real genotype reads."""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,io,csv,ast
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/H018_R16_regional_extraction_source_v1'
S=ROOT/'src/arggnn/prepare_h018_r16_regional_extraction_v1.py'
ast.parse(S.read_text());sp=importlib.util.spec_from_file_location('H018_handmade_adapter',S)
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
cfg=json.loads((D/'config.json').read_text())
for r in cfg['frozen_refs'].values():m.bound(r)
runtime=m.runtime_report_v1(cfg)
subjects=(ROOT/cfg['frozen_refs']['subject_ids']['path']).read_text().splitlines()
original_header=(ROOT/cfg['frozen_refs']['public_VCF_header']['path']).read_bytes()
header_lines=[(i,x) for i,x in enumerate(original_header.splitlines())]
header,indexes,h=m.frozen_header(iter(header_lines),subjects)
assert b'##FORMAT=<ID=AP,'not in header and b'_rnd'not in header.split(b'#CHROM')[-1]
rows=list(csv.DictReader(io.StringIO(m.bound(cfg['frozen_refs']['subject_metadata']).decode()),delimiter='\t'))
assert [int(r['ordinary_VCF_column_index0']) for r in rows]==h['selected_ordinary_indexes0']
def record(pos,gt='0|0',gp='0.9,0.1,0',hs='0',ref='A',alt='C',extra=''):
 info='INFO=0.9;RAF=0.25;MASK_1000G;pan_troglodytes=A'+extra
 sample=gt+':APSECRET:'+gp+':'+hs+':0.1'
 return '\t'.join(['22',str(pos),'.',ref,alt,'.','PASS',info,'GT:AP:GP:HS:DS']+[sample]*3331).encode()
valid=m.clean_record(record(20000001),indexes)
assert valid[1]['H018_fixed_QC_pass'] and valid[1]['region_bp0']==0
assert b'APSECRET'not in valid[0] and valid[0].decode().split('\t')[8]=='GT:GP:HS:DS'
assert len(valid[0].rstrip().split(b'\t'))==29
assert m.clean_record(record(20000000),indexes)is None
assert m.clean_record(record(22000000),indexes)[1]['region_bp0']==1999999
assert m.clean_record(record(22000001),indexes)is None
assert not m.clean_record(record(20100000,gt='0/0'),indexes)[1]['all20_phased_diploid']
assert not m.clean_record(record(20100000,gp='.4,.4,.2'),indexes)[1]['all20_GP_max_ge_0_8']
assert not m.clean_record(record(20100000,hs=str(2**30)),indexes)[1]['all20_HS_valid30']
assert not m.clean_record(record(20100000,alt='G'),indexes)[1]['transversion']
assert m.decode_hs(0,7)==(0,0) and m.decode_hs(2**30-1,14)==(1,1)
chunks,plan=m.tbi_chunks(m.bound(cfg['frozen_refs']['public_TBI']))
assert len(chunks)==1 and plan['estimated_network_upper_bytes']<100_000_000
# HTTP tests use fake responses only. HTTP200 must be rejected before reading body.
class Reply:
 def __init__(self,status):
  self.status=status;self.url=m.VCF_URL;self.reads=0
  self.headers={'Content-Range':'bytes 0-3/'+str(m.SOURCE_BYTES),'Content-Length':'4','ETag':m.VCF_ETAG}
 def __enter__(self):return self
 def __exit__(self,*a):return False
 def read(self,n):self.reads+=1;return b'abcd'
good=Reply(206);net=m.RangeSource(opener=lambda *a,**kw:good)
assert net.fetch(m.VCF_URL,0,3,m.SOURCE_BYTES,m.VCF_ETAG)==b'abcd' and net.received==4
bad=Reply(200);net=m.RangeSource(opener=lambda *a,**kw:bad)
try:net.fetch(m.VCF_URL,0,3,m.SOURCE_BYTES,m.VCF_ETAG);raise AssertionError('No Range refusal')
except ValueError:assert bad.reads==0
# Exercise actual installed HTS on synthetic data only; raw AP fixture stays in RAM.
bgzip=ROOT/cfg['tools']['bgzip']['path'];tabix=ROOT/cfg['tools']['tabix']['path'];bcf=ROOT/cfg['tools']['bcftools']['path']
raw=original_header+b'\n'+b'\n'.join(record(p) for p in [19999999,20000001,20000002,21999999,22000000,22000001])+b'\n'
compressed=subprocess.run([str(bgzip),'-@','1','-c'],input=raw,capture_output=True,check=True).stdout
class Fixture(m.RangeSource):
 def bytes(self,start,n):
  assert 0<=start<=start+n<=len(compressed);return compressed[start:start+n]
fixture=Fixture();all_lines=list(fixture.lines(0));assert len(all_lines)>6
decoded=[line for _,line in all_lines if line and not line.startswith(b'#')]
assert len(decoded)==6 and decoded[0]==record(19999999)
second=next(vo for vo,line in all_lines if line==record(20000002))
assert next(fixture.lines(second))[1]==record(20000002)
synthetic=header+b''.join(m.clean_record(r,indexes)[0] for r in decoded if m.clean_record(r,indexes)is not None)
assert b'APSECRET'not in synthetic and b'##FORMAT=<ID=AP,'not in synthetic
toy=D/('handmade_no_empirical_data_'+hashlib.sha256(S.read_bytes()).hexdigest()[:8]);toy.mkdir(exist_ok=False)
vcf=toy/'synthetic.noAP.vcf';vcf.write_bytes(synthetic)
out=toy/'synthetic.noAP.vcf.gz'
out.write_bytes(subprocess.run([str(bgzip),'-@','1','-c',str(vcf)],capture_output=True,check=True).stdout)
subprocess.run([str(tabix),'-p','vcf',str(out)],capture_output=True,check=True)
assert m.tool_command(bcf,['query','-l',str(out)]).splitlines()==subjects
assert int(m.tool_command(bcf,['index','-n',str(out)]).strip())==4
checks=['AST and frozen metadata/tool pins','Exact20ordinary subject order and original-column join','AP header/value stripping before output',
'Fixed[20,22)Mb0based toVCFPOS1based bounds','Frozen H018 GP/GT/HS/transversion criteria',
'HS30bit decode','Public source TBI region byte plan','HTTP206 exact read accounting','HTTP200 refusal before body read',
'BGZF block/line/cross-block/virtual-offset reader in RAM','Actual bgzip/tabix/bcftools synthetic VCF round trip']
receipt=dict(schema='H018_R16_extraction_handmade_checks_v1',passed=True,source_ref=m.ref(S),config_ref=m.ref(D/'config.json'),
 runtime=runtime,checks=checks,source_index_plan=plan,actual_remote_requests=0,actual_empirical_genotype_reads=0,
 synthetic_variants=4,synthetic_selected_subjects=20,all_genotypes_handmade=True,raw_AP_fixture_saved=False,
 GPU_used=False,install=False,resource_admission=False,extraction_of_H018_region_executed=False)
(D/'handmade_checks.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
