"""Check final compilation and render the complete paper for human visual review."""
from pathlib import Path
import hashlib
import json
import subprocess
from PIL import Image, ImageOps, ImageDraw

root=Path.cwd()
out=root/'paper/figure_redesign_20261026_v6'
bin_dir=root/'.runtime/env/bin'
receipt=json.loads((out/'integration.json').read_text())
texts={}
for name,pdf,log in [
 ('canonical',root/'paper/current_full_panel_working.pdf',out/'build/pass3.log'),
 ('overleaf',out/'overleaf_build/main.pdf',out/'overleaf_build/pass3.log'),
]:
 s=log.read_text()
 banned=['Overfull','undefined references','multiply defined','LaTeX Error','Float too large']
 assert not any(x in s for x in banned), (name,[x for x in banned if x in s])
 info=subprocess.run([str(bin_dir/'pdfinfo'),str(pdf)],capture_output=True,text=True,check=True).stdout
 pages=int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
 assert 1<=pages<=20
 target=out/(name+'_text.txt')
 subprocess.run([str(bin_dir/'pdftotext'),'-layout',str(pdf),str(target)],check=True)
 texts[name]=target.read_text()
 receipt[name+'_pages']=pages
 receipt[name+'_compile_success']=True
 receipt[name+'_underfull_diagnostics']=sum('Underfull' in line for line in s.splitlines())
assert texts['canonical']==texts['overleaf']
receipt['overleaf_rendered_text_identical']=True
render=out/'render'
render.mkdir(exist_ok=True)
subprocess.run([str(bin_dir/'pdftoppm'),'-r','70','-png',str(root/'paper/current_full_panel_working.pdf'),str(render/'page')],check=True)
files=sorted(render.glob('page-*.png'))
thumbs=[]
for i,p in enumerate(files,1):
 im=Image.open(p).convert('RGB')
 im.thumbnail((425,550))
 tile=Image.new('RGB',(445,580),'#e6e6e6')
 tile.paste(im,((445-im.width)//2,10))
 ImageDraw.Draw(tile).text((15,563),f'Page {i}',fill='black')
 thumbs.append(tile)
cols=4
rows=(len(thumbs)+cols-1)//cols
sheet=Image.new('RGB',(cols*445,rows*580),'white')
for i,t in enumerate(thumbs):sheet.paste(t,((i%cols)*445,(i//cols)*580))
sheet.save(render/'contact.png')
figure_pages=[]
for i,page in enumerate(texts['canonical'].split('\f'),1):
 if 'Figure 1:' in page or 'Figure 2:' in page:
  figure_pages.append(i)
  subprocess.run([str(bin_dir/'pdftoppm'),'-f',str(i),'-l',str(i),'-r','150','-png','-singlefile',str(root/'paper/current_full_panel_working.pdf'),str(render/f'figure-page-{i}')],check=True)
assert 1 <= len(figure_pages) <= 2
assert 'Figure 1:' in texts['canonical'] and 'Figure 2:' in texts['canonical']
receipt['figure_pages']=figure_pages
receipt['visual_inspection_complete']=False
paths=['paper/current_full_panel_working.tex','paper/current_full_panel_working.pdf','paper/current_full_panel_overleaf_working.zip']
paths += [f'paper/figures/TreeLoom_{name}.{ext}' for name in ['architecture','encoder'] for ext in ['pdf','svg','png']]
receipt['artifact_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
(out/'integration.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k.endswith('_pages') or k.endswith('_success') or k.endswith('identical')},indent=2))
