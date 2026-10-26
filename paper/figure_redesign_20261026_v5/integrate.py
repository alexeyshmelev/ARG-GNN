"""Integrate accepted figures without changing any experimental table."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re
import shutil
import subprocess

root = Path.cwd()
paper = root / 'paper'
out = paper / 'figure_redesign_20261026_v5'
p = paper / 'current_full_panel_working.tex'
before = p.read_text()
head = subprocess.run(['git', 'show', 'HEAD:paper/current_full_panel_working.tex'], capture_output=True, text=True, check=True).stdout
acceptance = json.loads((out / 'reader_acceptance.json').read_text())
assert acceptance['accepted'] is True
for name in ['architecture', 'encoder']:
    path = out / f'TreeLoom_{name}.png'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == acceptance['image_sha256'][name]

captions = {
    'architecture': (out / 'caption_architecture.txt').read_text().strip(),
    'encoder': (out / 'caption_encoder.txt').read_text().strip(),
}
pattern = r'\\begin\{figure\*\}\[[^]]*\].*?\\end\{figure\*\}'
found = re.findall(pattern, before, re.S)
assert len(found) == 2
s = before
for name, caption in captions.items():
    old = next(block for block in found if '\\label{fig:' + name + '}' in block)
    block = '\\begin{figure*}[p]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/TreeLoom_' + name + '.pdf}\n\\caption{' + caption + '}\n\\label{fig:' + name + '}\n\\end{figure*}'
    (out / f'caption_{name}.tex').write_text(block + '\n')
    s = s.replace(old, block)
s = s.replace('Sample memory supplies a different kind of context (Figure~\\ref{fig:encoder}b).', 'Sample memory supplies a different kind of context (Figure~\\ref{fig:encoder}c).')
tabpattern = r'\\begin\{table\*?\}.*?\\end\{table\*?\}'
oldtables = re.findall(tabpattern, head, re.S)
newtables = re.findall(tabpattern, s, re.S)
assert len(newtables) == 8 and oldtables == newtables
assert len(re.findall(pattern, s, re.S)) == 2
p.write_text(s)
for name in captions:
    for ext in ['pdf', 'svg', 'png']:
        shutil.copy2(out / f'TreeLoom_{name}.{ext}', paper / 'figures' / f'TreeLoom_{name}.{ext}')
receipt = {
    'source_before_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'source_after_sha256': hashlib.sha256(s.encode()).hexdigest(),
    'table_source_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in newtables],
    'all_eight_result_tables_unchanged': True,
    'experiments_remain_stopped': True,
    'new_training_or_simulation': False,
    'main_figures': 2,
    'reader_acceptance': acceptance,
}
(out / 'integration.json').write_text(json.dumps(receipt, indent=2) + '\n')
(out / 'build').mkdir(exist_ok=True)
(out / 'overleaf_build').mkdir(exist_ok=True)
with ZipFile(paper / 'current_full_panel_overleaf_working.zip', 'w', ZIP_DEFLATED) as z:
    z.writestr('main.tex', s)
    for f in ['current_full_panel_refs.bib', 'aaai2027.sty', 'aaai2027.bst']:
        z.write(paper / f, f)
    for name in captions:
        f = f'figures/TreeLoom_{name}.pdf'
        z.write(paper / f, f)
    z.writestr('README.txt', 'Compile main.tex with pdfLaTeX and BibTeX. Two figures trace tensors through TreeLoom. This is a development manuscript, not a completed matched benchmark.\n')
with ZipFile(paper / 'current_full_panel_overleaf_working.zip') as z:
    z.extractall(out / 'overleaf_build')
print('Integrated exactly two independently read figures. All eight table source blocks match HEAD.')
