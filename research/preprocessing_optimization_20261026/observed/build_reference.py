"""Copy only the original VCF loop into a readable differential oracle."""
from pathlib import Path
root = Path('/workspace-SR003.nfs2/ARG-GNN')
text = (root / 'src/arggnn/benchmarking/european_treeloom.py').read_text()
start = text.index('    h = len(rows)')
end = text.index('    labels, queries, ages = [], [], []', start)
block = text[start:end].replace('ts.num_sites', 'n_sites')
output = root / 'research/preprocessing_optimization_20261026/observed/reference_reader.py'
output.write_text('"""Unmodified original VCF loop, copied solely as a test oracle."""\nimport numpy as np\nfrom arggnn.benchmarking.european_treeloom import _need\n\ndef read_observed_calls(vcf_path, rows, wanted, n_sites, *, resource_check):\n' + block + '    return genotype, allele_count\n')
