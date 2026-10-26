"""Independent bounded parser review; fixtures and output stay in review scope."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import random
import time
import numpy as np
from research.preprocessing_optimization_20261026.observed import parser, reference_reader, adapter

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = ROOT/'research/preprocessing_optimization_20261026/review'

def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)

class RenameSiteCount(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'ts' and node.attr == 'num_sites':
            return ast.Name(id='n_sites', ctx=ast.Load())
        return self.generic_visit(node)

def ast_text(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)

original = function(ROOT/'src/arggnn/benchmarking/european_treeloom.py', 'load_relate_observation')
oracle = function(ROOT/'research/preprocessing_optimization_20261026/observed/reference_reader.py', 'read_observed_calls')
replacement = function(ROOT/'research/preprocessing_optimization_20261026/observed/adapter.py', 'load_relate_observation')
normalized = [RenameSiteCount().visit(copy.deepcopy(n)) for n in original.body[23:28]]
assert ast_text(normalized) == ast_text(oracle.body[:-1]), 'Copied oracle differs from actual production VCF statements'
assert ast_text(original.body[:23]) == ast_text(replacement.body[:23]), 'Adapter changed preceding validation'
assert ast_text(original.body[28:]) == ast_text(replacement.body[24:]), 'Adapter changed following validation or return'

rows = [dict(sample_id='a', haplotype=0),dict(sample_id='a',haplotype=1),dict(sample_id='b',haplotype=0),dict(sample_id='b',haplotype=1)]
header = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n'

def row(position, fmt='GT', a='0|1', b='1|0', ref='A', alt='C'):
    return '\t'.join(['1',str(position),'.',ref,alt,'.','.','.',fmt,a,b])+'\n'

def outcome(fn, path, events):
    try:
        arrays = fn(path, rows, {2:0,4:1}, 2, resource_check=events.append)
        return ('ok', tuple((a.dtype.str,a.shape,a.tobytes().hex()) for a in arrays))
    except Exception as exc:
        return ('error', type(exc).__name__, str(exc))

cases = []
base = header+row(1)+row(2)+row(3)+row(4)
for spelling in ['+0','-0','-1','00','01','0_0','0_1','\u0660','\uff11','\u00a01\u00a0','\v1','1\f',' 1 ','+0001','-0001','1_0','\x001','1e0','0x1','']:
    for fmt,a,b in [('GT',spelling+'|.',spelling+'|0'),('DP:GT:X','5:'+spelling+'|.:abc','3:'+spelling+'|0:'),('X:GT:GT',':'+spelling+'|0:broken',':0|1:broken')]:
        cases.append(('spelling '+repr((spelling,fmt)),header+row(2,fmt,a,b)+row(4)))
for position in ['+1','0001','1_0','\u0661','\uff11','1.0','0','-1','NaN','']:
    cases.append(('unretained position '+repr(position),header+row(position)+row(2)+row(4)))
for suffix in ['\t','\tEXTRA','\n','\t0|1\t','\r','\r\n']:
    cases.append(('unretained suffix '+repr(suffix),header+row(1).rstrip('\n')+suffix+'\n'+row(2)+row(4)))
for invalid in [row(3,fmt='BROKEN',a='x',b='x'),row(3,ref='N',alt='A'),row(1),row('bad')]:
    cases.append(('unretained row '+repr(invalid),header+row(2)+invalid+row(4)))
for tweak in [base.replace('\n','\r\n'),base.replace('\n','\r'),base.rstrip('\n'),base+'##tail\n',base+header,base.replace('#CHROM','#CHROM_compat')]:
    cases.append(('line conventions',tweak))
rng = random.Random(714413)
alphabet = ['','0|1','.|.','-1|0','0|0','0/0','0||0',':','::','1:0|1','1:0|1:','1::0|1','\u0660|1','0|1:many:more']
for trial in range(2500):
    fmt = rng.choice(['GT','DP:GT','GT:X','GT:GT','X:GT:X','X:X:GT','X',''])
    a,b = rng.choice(alphabet),rng.choice(alphabet)
    text = header+row(1,fmt='broken',a='ignored',b='ignored',ref='N')+row(2,fmt,a,b)+row(4)
    if trial % 9 == 0:
        text = text.replace('\t'+b+'\n','\t'+b+'\textra\n',1)
    cases.append(('independent delimiter fuzz '+str(trial),text))
started = time.perf_counter()
failures=[]
path=HERE/'parser_edge.vcf'
for name,text in cases:
    path.write_bytes(text.encode('utf-8'))
    a_events,b_events=[],[]
    a=outcome(reference_reader.read_observed_calls,path,a_events)
    b=outcome(parser.read_observed_calls,path,b_events)
    if a!=b or a_events!=b_events:
        failures.append(dict(name=name,reference=a,candidate=b,events=[a_events,b_events]))
path.unlink()
report=dict(status='pass' if not failures else 'fail',cases=len(cases),failures=failures,
    production_oracle_AST_identity=True,unchanged_full_loader_validation_AST_identity=True,
    seconds=time.perf_counter()-started,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/preprocessing_optimization_20261026/observed/parser.py',ROOT/'research/preprocessing_optimization_20261026/observed/adapter.py',ROOT/'research/preprocessing_optimization_20261026/observed/reference_reader.py',ROOT/'src/arggnn/benchmarking/european_treeloom.py']},CPU_only=True,no_training=True)
(HERE/'parser_adversarial.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','failures']},indent=2))
assert not failures, failures[:3]
