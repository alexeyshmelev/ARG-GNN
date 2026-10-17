from pathlib import Path
import re,hashlib,json,shutil,subprocess,os,zipfile,datetime,sys,traceback
from PIL import Image
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/paper_simplification_ROOT_20261017_v9';BUILD=ROOT/'paper/_build_scientific_20261017_v10';FIG=ROOT/'paper/figure_redesign_20261017_v10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def put(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2)
def main():
 BUILD.mkdir(exist_ok=False)
 assert sha(FIG/'TreeLoom_architecture.pdf')=='49c82738685bdf6bbfe16a87f57dad15c514df53a2e6401c491a009a0b0b9856'
 original=(ROOT/'paper/current_full_panel_working.tex').read_text()
 s=(ROOT/'paper/_build_scientific_20261017_v9/current_full_panel_working.tex').read_text()
 old_intro=s[s.index('We assess TreeLoom on a shared European development endpoint'):s.index('\n\\section{Related Work}')]
 new_intro='We evaluate TreeLoom against available pipelines based on genealogies and use the matched TreeLoom-NoMem control to examine its additional memory. The evaluation reports genomic span accuracy together with tract boundary recovery.\n'
 s=s.replace(old_intro,new_intro)
 begin=s.index('TreeLoom connects relationships within a local genealogy',s.index('\\section{Discussion}'))
 end=s.index('\n\n',begin)
 s=s[:begin]+'TreeLoom combines descendant and outside context within each genealogy with continuity along the chromosome. Its two routes for genomic history act before and after the second tree traversal. Both follow persistent sample identities without assuming correspondence between inferred ancestors.'+s[end:]
 begin=s.index('Complete local genealogies offer a structured representation for LAI',s.index('\\section{Discussion}'))
 end=s.index('\\paragraph{Code availability.}',begin)
 s=s[:begin]+s[end:]
 caption=(FIG/'caption.tex').read_text().strip();assert ';' not in caption
 start=s.index('\\caption{',s.index('\\begin{figure*}'));end=s.index('\n\\label{fig:architecture}',start)
 s=s[:start]+'\\caption{'+caption+'}'+s[end:]
 s='\n'.join(x.rstrip() for x in s.splitlines())+'\n';assert ';' not in s
 for e in ('table','table*','align','equation'):
  def blocks(t):return sorted(re.findall(r'\\begin\{'+re.escape(e)+r'\}(.*?)\\end\{'+re.escape(e)+r'\}',t,re.S))
  assert blocks(original)==blocks(s),e
 assert re.findall(r'\\todo\{[^\n]+',original)==re.findall(r'\\todo\{[^\n]+',s)
 (BUILD/'current_full_panel_working.tex').write_text(s)
 for name in ('current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst'):shutil.copy2(ROOT/'paper'/name,BUILD/name)
 (BUILD/'figures').mkdir();shutil.copy2(FIG/'TreeLoom_architecture.pdf',BUILD/'figures/TreeLoom_architecture.pdf')
 env=os.environ.copy();env.update(PATH=str(ROOT/'.runtime/tex/.TinyTeX/bin/x86_64-linux')+os.pathsep+env['PATH'],TMPDIR=str(BUILD),TEXMFVAR=str(ROOT/'.runtime/tex/var'),TEXMFCONFIG=str(ROOT/'.runtime/tex/config'),XDG_CACHE_HOME=str(ROOT/'.runtime/cache'),CUDA_VISIBLE_DEVICES='')
 steps=[]
 def compile(cwd,stem,prefix):
  for i,argv in enumerate((['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'],['bibtex',stem],['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'],['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',stem+'.tex'])):
   log=OUT/f'{prefix}_{i+1}.stdout.txt'
   with log.open('x') as f:r=subprocess.run(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
   steps.append(dict(argv=argv,cwd=str(cwd.relative_to(ROOT)),returncode=r.returncode,stdout_ref=ref(log)))
   assert r.returncode==0,(argv,r.returncode,log)
 def render(pdf,prefix):
  r=subprocess.run(['pdftoppm','-r','120','-png',str(pdf),str(prefix)],cwd=BUILD,env=env,capture_output=True,text=True);assert r.returncode==0,r.stderr
  return sorted(prefix.parent.glob(prefix.name+'-*.png'))
 compile(BUILD,'current_full_panel_working','final_canonical_candidate')
 pages=render(BUILD/'current_full_panel_working.pdf',BUILD/'page');assert 1<=len(pages)<=20
 bundle=BUILD/'current_full_panel_overleaf_working.zip'
 with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('main.tex',s)
  for name in ('current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst'):z.write(BUILD/name,name)
  z.write(BUILD/'figures/TreeLoom_architecture.pdf','figures/TreeLoom_architecture.pdf')
  z.writestr('README.txt','Upload to Overleaf and compile main.tex with pdfLaTeX and BibTeX. This AAAI2027 working draft reports development results and retains visible scientific TODOs.\n')
 leaf=BUILD/'overleaf_extracted';leaf.mkdir()
 with zipfile.ZipFile(bundle) as z:z.extractall(leaf)
 compile(leaf,'main','final_overleaf_candidate');bp=render(leaf/'main.pdf',BUILD/'bundle-page');assert len(bp)==len(pages)
 for a,b in zip(pages,bp):
  with Image.open(a) as x,Image.open(b) as y:assert x.size==y.size and x.mode==y.mode and x.tobytes()==y.tobytes(),str(a)
 log=(BUILD/'current_full_panel_working.log').read_text()
 rec=dict(schema='ROOT_TreeLoom_paper_simplification_final_build_v10',time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pages=len(pages),canonical_installed=False,all_tables_equation_bodies_and_TODOs_preserved=True,semicolon_count=0,figure_caption_words=len(caption.split()),main_words=len(s.split('\\appendix')[0].split()),all_bundle_pages_pixel_identical=True,overfull_warnings=[x for x in log.splitlines() if 'Overfull' in x],unresolved_reference_warnings=[x for x in log.splitlines() if 'undefined' in x.lower()],compilation=steps,figure_handoff_ref=ref(FIG/'handoff_seal.json'),figure_pdf_ref=ref(FIG/'TreeLoom_architecture.pdf'),source_ref=ref(BUILD/'current_full_panel_working.tex'),pdf_ref=ref(BUILD/'current_full_panel_working.pdf'),bundle_ref=ref(bundle),visual_inspection_pending=True,experimental_metrics_changed=False)
 put('build_and_bundle.root_actual_v2.json',rec)
 print(json.dumps({k:rec[k] for k in ('pages','canonical_installed','all_bundle_pages_pixel_identical','overfull_warnings','unresolved_reference_warnings','main_words','visual_inspection_pending')}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
