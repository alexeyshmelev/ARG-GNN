from pathlib import Path
import hashlib, io, json, os, re, shutil, subprocess, zipfile
from PIL import Image,ImageDraw
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
PAPER=ROOT/'paper'
REPORT=ROOT/'research/reviews/paper_root_integration_20261016_v1'
BUILD=PAPER/'_build_scientific_20261016_v1'
BUILD.mkdir(exist_ok=False)
TEX=ROOT/'.runtime/tex/.TinyTeX/bin/x86_64-linux'
env=os.environ.copy();env.update(TEXMFHOME=str(ROOT/'.runtime/texmf'),TEXMFVAR=str(ROOT/'.runtime/tex/var'),TEXMFCONFIG=str(ROOT/'.runtime/tex/config'),TEXMFCACHE=str(BUILD/'cache'),TMPDIR=str(BUILD),BIBINPUTS=str(PAPER),BSTINPUTS=str(PAPER))
actual=[]
commands=[('pdflatex1',[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'current_full_panel_working.tex']),('bibtex',[str(TEX/'bibtex'),str(BUILD/'current_full_panel_working')]),('pdflatex2',[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'current_full_panel_working.tex']),('pdflatex3',[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),'current_full_panel_working.tex'])]
for name,cmd in commands:
 r=subprocess.run(cmd,cwd=PAPER,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 (BUILD/(name+'.stdout')).write_bytes(r.stdout)
 actual.append(dict(stage=name,returncode=r.returncode,argv=cmd));print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
shutil.copyfile(BUILD/'current_full_panel_working.pdf',PAPER/'current_full_panel_working.pdf')
log=(BUILD/'current_full_panel_working.log').read_text()
warnings=[l for l in log.splitlines() if 'Warning' in l or 'Overfull' in l]
assert not warnings,warnings
info=subprocess.check_output(['/usr/bin/pdfinfo',str(PAPER/'current_full_panel_working.pdf')],text=True)
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages==9,pages
text=subprocess.check_output(['/usr/bin/pdftotext','-layout',str(PAPER/'current_full_panel_working.pdf'),'-'],text=True)
oldtext=subprocess.check_output(['/usr/bin/pdftotext','-layout',str(PAPER/'rewrite_20261016_v1/manuscript.pdf'),'-'],text=True)
assert text==oldtext,'Integrated paper content/layout text differs from reviewed rewrite'
# Archive contains only portable sources and instructions.
main=(PAPER/'current_full_panel_working.tex').read_text().replace('\\bibliography{current_full_panel_refs}','\\bibliography{references}')
assert '\\bibliography{references}' in main
zip_path=PAPER/'current_full_panel_overleaf_working.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('main.tex',main)
 for original,archive in [('current_full_panel_refs.bib','references.bib'),('aaai2027.sty','aaai2027.sty'),('aaai2027.bst','aaai2027.bst'),('figures/whole_tree_architecture.tex','figures/whole_tree_architecture.tex')]:z.write(PAPER/original,archive)
 z.writestr('README.txt','Working scientific manuscript. Select main.tex and compile with pdfLaTeX and BibTeX in Overleaf. Important research TODOs remain visible. Source code: https://github.com/alexeyshmelev/ARG-GNN\n')
check=BUILD/'overleaf_source_check';check.mkdir()
with zipfile.ZipFile(zip_path) as z:z.extractall(check)
env2=env.copy();env2.update(BIBINPUTS=str(check),BSTINPUTS=str(check))
for i,cmd in enumerate([[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','main.tex'],[str(TEX/'bibtex'),'main'],[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','main.tex'],[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error','main.tex']]):
 r=subprocess.run(cmd,cwd=check,env=env2,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 (BUILD/('overleaf_stage'+str(i)+'.stdout')).write_bytes(r.stdout)
 actual.append(dict(stage='overleaf_'+str(i),returncode=r.returncode,argv=cmd));print('overleaf',i,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
ziptext=subprocess.check_output(['/usr/bin/pdftotext','-layout',str(check/'main.pdf'),'-'],text=True)
assert text==ziptext,'Extracted Overleaf sources produce different document text/layout'
render=BUILD/'rendered';render.mkdir();thumbs=[]
for i in range(1,pages+1):
 raw=subprocess.check_output(['/usr/bin/pdftoppm','-r','85','-f',str(i),'-l',str(i),'-singlefile','-png',str(PAPER/'current_full_panel_working.pdf')],timeout=30)
 (render/('page_'+str(i).zfill(2)+'.png')).write_bytes(raw)
 im=Image.open(io.BytesIO(raw)).convert('RGB');im.thumbnail((612,792));thumbs.append(im)
for start in range(0,pages,3):
 ims=thumbs[start:start+3];contact=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)+25),'#e3e8ed');x=0
 for off,im in enumerate(ims):
  contact.paste(im,(x,25));ImageDraw.Draw(contact).text((x+8,5),'Page '+str(start+off+1),fill='black');x+=im.width
 contact.save(render/('contact_'+str(start+1)+'_'+str(start+len(ims))+'.jpg'),quality=85)
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
receipt=dict(schema='ROOT_scientific_paper_rewrite_integration_v1',actual_build_stages=actual,pages=pages,main_pages=[1,2,3,4,5,6],reference_pages=[6,7],supplement_pages=[8,9],warnings=warnings,all_Overleaf_build_stages_succeeded=True,reviewed_rewrite_text_layout_equal=True,extracted_Overleaf_text_layout_equal=True,artifacts=[ref(PAPER/x) for x in ['current_full_panel_working.tex','current_full_panel_working.pdf','current_full_panel_refs.bib','current_full_panel_overleaf_working.zip','figures/whole_tree_architecture.tex']],visual_review_pending=True,scientific_experiments_unchanged=True,scientific_goal_complete=False)
(REPORT/'integration_build_receipt.root_v1.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(pages=pages,warnings=warnings,Overleaf_build='PASS',visual_render_directory=str(render.relative_to(ROOT)))),flush=True)
