"""Build the actual revised manuscript and an extracted portable Overleaf bundle."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,zipfile,io
from PIL import Image,ImageDraw
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');PAPER=ROOT/'paper'
parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
BUILD=PAPER/('_build_TreeLoom_fig_naming_20261016_v2_attempt'+str(args.attempt));BUILD.mkdir(exist_ok=False)
TEX=ROOT/'.runtime/tex/.TinyTeX/bin/x86_64-linux'
env=os.environ.copy();env.update(TEXMFHOME=str(ROOT/'.runtime/texmf'),TEXMFVAR=str(ROOT/'.runtime/tex/var'),TEXMFCONFIG=str(ROOT/'.runtime/tex/config'),TEXMFCACHE=str(BUILD/'cache'),TMPDIR=str(BUILD),BIBINPUTS=str(PAPER),BSTINPUTS=str(PAPER))
actual=[]
def run(name,argv,cwd):
 localenv=env.copy();localenv.update(BIBINPUTS=str(cwd),BSTINPUTS=str(cwd))
 r=subprocess.run(argv,cwd=cwd,env=localenv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
 (BUILD/(name+'.stdout')).write_bytes(r.stdout);actual.append(dict(stage=name,argv=argv,cwd=str(cwd),returncode=r.returncode));print(name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
def compile_document(cwd,basename,prefix,output=None):
 options=[] if output is None else ['-output-directory='+output]
 run(prefix+'latex1',[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error',*options,basename+'.tex'],cwd)
 bib=basename if output is None else output+'/'+basename
 run(prefix+'bibtex',[str(TEX/'bibtex'),bib],cwd)
 for k in [2,3,4]:run(prefix+'latex'+str(k),[str(TEX/'pdflatex'),'-interaction=nonstopmode','-halt-on-error',*options,basename+'.tex'],cwd)
compile_document(PAPER,'current_full_panel_working','paper_',BUILD.name)
source=(PAPER/'current_full_panel_working.tex').read_text()
portable={'main.tex':source.replace('\\bibliography{current_full_panel_refs}','\\bibliography{references}').encode(),'references.bib':(PAPER/'current_full_panel_refs.bib').read_bytes(),'aaai2027.sty':(PAPER/'aaai2027.sty').read_bytes(),'aaai2027.bst':(PAPER/'aaai2027.bst').read_bytes(),'figures/whole_tree_architecture.tex':(PAPER/'figures/whole_tree_architecture.tex').read_bytes(),'figures/TreeLoom_architecture.pdf':(PAPER/'figures/TreeLoom_architecture.pdf').read_bytes(),'README.txt':b'Compile main.tex with pdfLaTeX and BibTeX. Official AAAI 2027 assets and the vector figure are included. This is a working scientific manuscript. Editable figure generator and toy ARG provenance are in the linked repository.\n'}
archive=BUILD/'current_full_panel_overleaf_working.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in portable.items():z.writestr(name,data)
extracted=BUILD/'overleaf_extracted';extracted.mkdir()
with zipfile.ZipFile(archive) as z:z.extractall(extracted)
compile_document(extracted,'main','overleaf_')
pdf=BUILD/'current_full_panel_working.pdf';opdf=extracted/'main.pdf'
def get_text(path):return subprocess.check_output(['/usr/bin/pdftotext','-layout',str(path),'-'],text=True)
text=get_text(pdf);assert text==get_text(opdf),'Extracted Overleaf text/layout differs'
pages=[t for t in text.split('\f') if t.strip()];assert len(pages)<=20,'User total page limit'
warnings=[]
for log in [BUILD/'current_full_panel_working.log',extracted/'main.log']:
 warnings.extend([line for line in log.read_text().splitlines() if 'Warning' in line or 'Overfull' in line])
render=BUILD/'rendered';render.mkdir();images=[]
for i in range(1,len(pages)+1):
 raw=subprocess.check_output(['/usr/bin/pdftoppm','-r','85','-f',str(i),'-l',str(i),'-singlefile','-png',str(pdf)])
 (render/('page_'+str(i).zfill(2)+'.png')).write_bytes(raw);im=Image.open(io.BytesIO(raw)).convert('RGB');im.thumbnail((612,792));images.append(im)
for start in range(0,len(images),3):
 ims=images[start:start+3];contact=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)+25),'#e3e8ed');x=0
 for k,im in enumerate(ims):contact.paste(im,(x,25));ImageDraw.Draw(contact).text((x+8,5),'Page '+str(start+k+1),fill='black');x+=im.width
 contact.save(render/('contact_'+str(start+1)+'_'+str(start+len(ims))+'.jpg'),quality=88)
def ref(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
receipt=dict(schema='TreeLoom_figure_names_actual_build_v2',actual_build_stages=actual,pages=len(pages),warnings=warnings,figure_caption_pages=[i+1 for i,t in enumerate(pages) if 'Figure 1:' in t],source_ref=ref(PAPER/'current_full_panel_working.tex'),figure_ref=ref(PAPER/'figures/TreeLoom_architecture.pdf'),candidate_pdf_ref=ref(pdf),candidate_Overleaf_ref=ref(archive),extracted_Overleaf_text_layout_equal=True,render_directory=str(render.relative_to(ROOT)),all_pages_visually_inspected=False,visual_review_pending=True,scientific_experiments_changed=False)
(BUILD/'build_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt),flush=True)
