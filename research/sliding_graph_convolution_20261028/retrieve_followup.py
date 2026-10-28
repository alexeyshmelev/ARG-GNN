"""Pinned author source and further primary papers, stored only as evidence."""
from pathlib import Path
import json
from retrieve_sources import retrieve, ROOT

records=[]
for label,repo in [('stsgcn','Davidham3/STSGCN'),('msg3d','kenziyuliu/MS-G3D')]:
 receipt=retrieve((label+'_commit.json','https://api.github.com/repos/'+repo+'/commits/master'))
 records.append(receipt)
 if 'error' in receipt: continue
 commit=json.loads((ROOT/'sources'/(label+'_commit.json')).read_text())['sha']
 paths=['README.md','models/stsgcn.py','utils.py'] if label=='stsgcn' else ['README.md','model/ms_gtcn.py','model/msg3d.py','model/ms_gcn.py']
 for path in paths:
  record=retrieve((label+'_'+path.replace('/','_'),'https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+path))
  record.update(repository=repo,commit=commit,source_path=path)
  records.append(record)
for item in [
 ('stsgcn.pdf','https://ojs.aaai.org/index.php/AAAI/article/download/5438/5294'),
 ('graph_time_arxiv_search.xml','https://export.arxiv.org/api/query?search_query=ti:%22Graph-Time%20Convolutional%20Neural%20Networks%22&max_results=3'),
 ('space_time_arxiv_search.xml','https://export.arxiv.org/api/query?search_query=ti:%22Space-Time%20Graph%20Neural%20Networks%22&max_results=3'),
 ('graph_time_repo_search.json','https://api.github.com/search/repositories?q=Graph-Time+Convolutional+Neural+Networks'),
]:
 records.append(retrieve(item))
(ROOT/'sources'/'followup_receipts.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
