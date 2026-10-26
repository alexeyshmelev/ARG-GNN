"""Copy the complete original preparation with three explicit module substitutions."""
from pathlib import Path
root=Path('/workspace-SR003.nfs2/ARG-GNN')
source=(root/'src/arggnn/benchmarking/european_treeloom.py').read_text()
start=source.index('def prepare_observation(')
end=source.find('\n\ndef ',start+1)
body=source[start:end if end>=0 else None]
replacements={
 'research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1':'research.preprocessing_optimization_20261026.tree.observations',
 'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.episode_preparation_v1':'research.preprocessing_optimization_20261026.tree.episodes',
 'research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.cut_preparation_v1':'research.preprocessing_optimization_20261026.tree.cut',
}
for old,new in replacements.items():
 assert body.count(old)==1
 body=body.replace(old,new)
preamble='''"""Complete original TreeLoom parent preparation using qualified exact kernels."""
import importlib
from arggnn.benchmarking import european_treeloom as original
_implementation=original._implementation
_need=original._need
PROTOCOLS=original.PROTOCOLS
_STATE_KEYS=original._STATE_KEYS
NATIVE_SIX_CLASSES=original.NATIVE_SIX_CLASSES
IDENTITY=original.IDENTITY
EuropeanWholeParent=original.EuropeanWholeParent
_sampled_records=original._sampled_records

'''
(root/'research/preprocessing_optimization_20261026/observed/preparation.py').write_text(preamble+body+'\n')
