"""Retrieve public primary source text as evidence; never import retrieved code."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import json

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN/research/sliding_graph_convolution_20261028')
SOURCES = [
 ('stgcn.pdf', 'https://arxiv.org/pdf/1709.04875'),
 ('msg3d.pdf', 'https://arxiv.org/pdf/2003.14111'),
 ('stsgcn.html', 'https://ojs.aaai.org/index.php/AAAI/article/view/5438'),
 ('stsgcn_repo.json', 'https://api.github.com/repos/Davidham3/STSGCN'),
 ('stsgcn_tree.json', 'https://api.github.com/repos/Davidham3/STSGCN/git/trees/master?recursive=1'),
 ('msg3d_repo.json', 'https://api.github.com/repos/kenziyuliu/MS-G3D'),
 ('msg3d_tree.json', 'https://api.github.com/repos/kenziyuliu/MS-G3D/git/trees/master?recursive=1'),
 ('graph_time_search.json', 'https://api.crossref.org/works?query.title=Graph-Time%20Convolutional%20Neural%20Networks&rows=4'),
 ('space_time_search.json', 'https://api.crossref.org/works?query.title=Space-Time%20Graph%20Neural%20Networks&rows=4'),
]

def retrieve(item):
    name, url = item
    record = {'name': name, 'requested_url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'ARG-GNN research primary-source audit'}), timeout=45) as response:
            data = response.read(12000000)
            record.update(status=response.status, final_url=response.url, bytes=len(data), sha256=sha256(data).hexdigest())
        (ROOT/'sources'/name).write_bytes(data)
    except Exception as error:
        record['error'] = str(error)
    return record

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(retrieve, SOURCES))
    (ROOT/'sources'/'retrieval_receipts.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))
