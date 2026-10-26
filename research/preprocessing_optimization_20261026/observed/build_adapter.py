"""Stage a readable isolated loader with one explicitly replaced parsing block."""
from pathlib import Path
import hashlib
root = Path('/workspace-SR003.nfs2/ARG-GNN')
source = root / 'src/arggnn/benchmarking/european_treeloom.py'
text = source.read_text()
start = text.index('def load_relate_observation(')
end = text.index('\n\ndef _sampled_records(', start)
loader = text[start:end]
parse_start = loader.index('    h = len(rows)')
parse_end = loader.index('    labels, queries, ages = [], [], []', parse_start)
loader = loader[:parse_start] + '''    genotype, allele_count = read_observed_calls(
        vcf_path, rows, wanted, ts.num_sites, resource_check=resource_check)
''' + loader[parse_end:]
preamble = '''"""Exact vectorized observed-call reader for our input adapter.

No simulation, native competitor, tree inference, or model operation changes.
Noncanonical integer spellings and unsupported fast layouts use the original
per-sample interpretation. The full public-input validation remains unchanged.
"""
from arggnn.benchmarking import european_treeloom as original
from .parser import read_observed_calls

_implementation = original._implementation
PROTOCOLS = original.PROTOCOLS
_need = original._need
_json_ref = original._json_ref
_bound = original._bound

'''
output = root / 'research/preprocessing_optimization_20261026/observed/adapter.py'
output.write_text(preamble + loader + '\n')
print({'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
       'adapter_sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
