from pathlib import Path
import json,hashlib,subprocess,os,shutil,zipfile,datetime
import fitz
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/paper_junction_ROOT_20261017_v8'
BUILD=ROOT/'paper/_build_scientific_20261017_v8'
FIG=ROOT/'paper/figure_redesign_20261017_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
assert sha(FIG/'handoff_seal.json')=='246b8129b96147b51c62a6885ed48aa9d8baa216431b80612b15441e98e8ef6d'
assert sha(FIG/'TreeLoom_architecture.pdf')=='11f8ab071634dc018fcfda12b494c2d3c3124df617fec946730a0fcfbfd8d7e5'
source=ROOT/'paper/current_full_panel_working.tex'
source_ref=ref(source)
assert source_ref['sha256']=='37e2af5d6d1a4cafc125948f1235812c09e8d0514c93f60b1d48bddf523a2c12'
BUILD.mkdir(exist_ok=False)
shutil.copy2(ROOT/'paper/current_full_panel_working.pdf',BUILD/'prior_current_full_panel_working.pdf')
shutil.copy2(ROOT/'paper/current_full_panel_overleaf_working.zip',BUILD/'prior_current_full_panel_overleaf_working.zip')
for name in ('current_full_panel_working.tex','current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst'):
 shutil.copy2(ROOT/'paper'/name,BUILD/name)
(BUILD/'figures').mkdir()
shutil.copy2(FIG/'TreeLoom_architecture.pdf',BUILD/'figures/TreeLoom_architecture.pdf')
env=os.environ.copy();tex=ROOT/'.runtime/tex/.TinyTeX/bin/x86_64-linux'
env.update({'PATH':str(tex)+os.pathsep+env['PATH'],'TMPDIR':str(BUILD),'TEXMFVAR':str(ROOT/'.runtime/tex/var'),'TEXMFCONFIG':str(ROOT/'.runtime/tex/config'),'XDG_CACHE_HOME':str(ROOT/'.runtime/cache'),'CUDA_VISIBLE_DEVICES':''})
steps=[]
def compile(cwd,stem,prefix):
 for k,argv in enumerate((['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'],['bibtex',stem],['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'],['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'])):
  log=OUT/f'{prefix}_{k+1}.stdout.txt'
  with log.open('x') as f:r=subprocess.run(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
  steps.append({'argv':argv,'cwd':str(cwd.relative_to(ROOT)),'returncode':r.returncode,'stdout_ref':ref(log)})
  assert r.returncode==0,(argv,r.returncode)
compile(BUILD,'current_full_panel_working','canonical')
old=fitz.open(BUILD/'prior_current_full_panel_working.pdf');new=fitz.open(BUILD/'current_full_panel_working.pdf')
assert len(old)==len(new)==10
comparisons=[]
for i in range(len(new)):
 a=old[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6));b=new[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6))
 same=(a.width,a.height,a.samples)==(b.width,b.height,b.samples)
 comparisons.append({'page':i+1,'pixel_identical_to_previous':same})
 if i!=2:assert same,('unexpected page change',i+1)
 b.save(BUILD/f'page-{i+1:02d}.png')
assert not comparisons[2]['pixel_identical_to_previous']
bundle=BUILD/'current_full_panel_overleaf_working.zip'
with zipfile.ZipFile(BUILD/'prior_current_full_panel_overleaf_working.zip') as zin,zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as zout:
 for name in zin.namelist():zout.writestr(name,(FIG/'TreeLoom_architecture.pdf').read_bytes() if name=='figures/TreeLoom_architecture.pdf' else zin.read(name))
leaf=BUILD/'overleaf_extracted';leaf.mkdir()
with zipfile.ZipFile(bundle) as z:
 assert set(z.namelist())=={'main.tex','current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst','figures/TreeLoom_architecture.pdf','README.txt'}
 z.extractall(leaf)
compile(leaf,'main','overleaf')
other=fitz.open(leaf/'main.pdf');assert len(other)==len(new)
bundle_comparisons=[]
for i in range(len(new)):
 a=new[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6));b=other[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6))
 same=(a.width,a.height,a.samples)==(b.width,b.height,b.samples);assert same,i+1
 bundle_comparisons.append({'page':i+1,'pixel_identical':same})
assert ref(source)==source_ref
for ext in ('pdf','svg','png'):shutil.copy2(FIG/f'TreeLoom_architecture.{ext}',ROOT/f'paper/figures/TreeLoom_architecture.{ext}')
shutil.copy2(BUILD/'current_full_panel_working.pdf',ROOT/'paper/current_full_panel_working.pdf')
shutil.copy2(bundle,ROOT/'paper/current_full_panel_overleaf_working.zip')
rec={'schema':'ROOT_TreeLoom_junction_build_actual_v8','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_ref_unchanged':source_ref,'change':'One junction dot before clade update on the sample-memory branch. No manuscript text, equations, experimental metrics or tree layouts changed.','figure_handoff_ref':ref(FIG/'handoff_seal.json'),'figure_refinement_QA_ref':ref(FIG/'refinement_QA.json'),'independent_image_only_review_ref':ref(ROOT/'research/reviews/figure_fresh_reader_oct17_v8/image_only_review.md'),'compilation':steps,'pages':10,'only_page3_pixels_changed':True,'prior_page_comparisons':comparisons,'all10bundle_pages_pixel_identical':True,'bundle_page_comparisons':bundle_comparisons,'no_overfull_warnings':'Overfull' not in (BUILD/'current_full_panel_working.log').read_text(),'new_page3_visual_inspection_pending':True,'artifacts':[ref(ROOT/p) for p in ('paper/current_full_panel_working.tex','paper/current_full_panel_working.pdf','paper/current_full_panel_refs.bib','paper/current_full_panel_overleaf_working.zip','paper/figures/TreeLoom_architecture.pdf','paper/figures/TreeLoom_architecture.png','paper/figures/TreeLoom_architecture.svg')]}
(OUT/'build_and_bundle.root_actual_v1.json').write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps({k:rec[k] for k in ('pages','only_page3_pixels_changed','all10bundle_pages_pixel_identical','no_overfull_warnings','new_page3_visual_inspection_pending')}),flush=True)
