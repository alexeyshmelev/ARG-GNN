"""Inventory cached author release members without executing their source."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/efficiency_20261027/fidelity'
RELEASES = ['argmix_c2481248eb0d', 'ancestralpaths_77db3cfae100']
records = []
for release in RELEASES:
    folder = ROOT / '.runtime/author_sources' / release
    archive = folder.with_suffix('.tar.gz')
    archived = {}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            if member.isfile():
                rel = '/'.join(member.name.split('/')[1:])
                archived[rel] = hashlib.sha256(source.extractfile(member).read()).hexdigest()
    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(folder))
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode('utf-8')
            if '\0' in text:
                raise ValueError('binary null')
        except (UnicodeDecodeError, ValueError):
            text = None
        record = {'release': release, 'path': str(path.relative_to(ROOT)), 'bytes': len(data),
                  'sha256': digest, 'archive_member': rel in archived,
                  'archive_hash_matches': digest == archived.get(rel) if rel in archived else None,
                  'classification': 'binary_or_data' if text is None else 'text',
                  'semantic_review': 'not_read', 'source_executed': False}
        if text is not None:
            record['lines'] = len(text.splitlines())
            record['all_bytes_decoded_for_inventory'] = True
            if path.suffix in {'.tsv', '.csv'} or rel == 'example/mesoneo_ages_pop.txt':
                record['classification'] = 'textual_data'
            if path.name.endswith('.lock'):
                record['classification'] = 'dependency_lock_text'
            if path.suffix == '.py':
                try:
                    parsed = ast.parse(text)
                    record['declarations'] = [
                        {'line': node.lineno, 'end_line': node.end_lineno, 'name': node.name,
                         'kind': type(node).__name__}
                        for node in ast.walk(parsed)
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                    record['syntax_parse'] = 'PASS_no_execution'
                except SyntaxError as error:
                    record['syntax_parse'] = f'{type(error).__name__} at {error.lineno}'
            elif path.suffix == '.rs':
                record['declarations'] = [{'line': n, 'text': line.strip()} for n,line in enumerate(text.splitlines(),1)
                    if line.lstrip().startswith(('pub fn ', 'fn ', 'pub struct ', 'struct ', 'impl ', 'pub enum '))]
        records.append(record)
    for rel in sorted(set(archived) - {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}):
        records.append({'release':release,'path':str(folder.relative_to(ROOT)/rel),'archive_member':True,
                        'classification':'unavailable_extracted_member','semantic_review':'not_read'})
report = {'schema':'author_source_coverage_v1', 'scope':'Cached release archives, not a live Git tracked-file query',
          'claim':'Every extracted file inventoried; decoding and parsing do not mean every source line was semantically reviewed.',
          'records':records}
(OUT/'source_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
summary={name:{'files':sum(r['release']==name for r in records),
 'archive_members':sum(r['release']==name and r['archive_member'] for r in records),
 'mismatches':sum(r['release']==name and r.get('archive_hash_matches') is False for r in records),
 'bytes':sum(r.get('bytes',0) for r in records if r['release']==name)} for name in RELEASES}
print(json.dumps(summary,indent=2))
