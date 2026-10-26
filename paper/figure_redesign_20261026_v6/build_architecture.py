"""Compact TreeLoom overview. Read a saved toy, draw on CPU, run no model."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import json
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=Path(__file__).parent
TS=tskit.load(ROOT/'research/reviews/figure_v15_flow_revision_20261017_v1/v15r2/illustrative_recorded_ARG.trees')
INK='#26363b'
GOLD='#f8e6b8'
GOLD_EDGE='#8d6826'
GRAY='#f1f3f4'
TEAL=LinearSegmentedColormap.from_list('embedding',['#e5f4f5','#75b9c1','#256773'])
PURPLE=LinearSegmentedColormap.from_list('probability',['#faf6fc','#b58bc9','#582571'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7,3.05))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7),ylim=(3.05,0));ax.axis('off')
texts=[]
def txt(x,y,s,size=8.2,ha='center',bold=False):
    p=ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=INK,fontweight='bold' if bold else 'normal',zorder=10)
    texts.append(p)
def rect(x,y,w,h,face='white',edge=INK,lw=.65):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=face,edgecolor=edge,lw=lw,zorder=2))
def route(points,arrow=True,lw=.8):
    if len(points)>2:ax.plot(*zip(*points[:-1]),color=INK,lw=lw,zorder=4)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=6.5,shrinkA=0,shrinkB=0,lw=lw,color=INK,zorder=4))
    else:ax.plot(*zip(*points[-2:]),color=INK,lw=lw,zorder=4)
def grid(x,y,w,h,rows,cols,cmap=TEAL,key=0):
    vals=.4+.3*np.sin(np.arange(rows*cols).reshape(rows,cols)*1.2+key)
    for i in range(rows):
        for j in range(cols):rect(x+j*w/cols,y+i*h/rows,w/cols,h/rows,cmap(vals[i,j]),'white',.3)
def learned(x,y,w,h,label):
    rect(x,y,w,h,GOLD,GOLD_EDGE,.8);txt(x+w/2,y+h/2,label)
# Observed sequence. All leaf positions remain identical across the three trees.
txt(.12,.16,'a',10,bold=True)
for index,t in enumerate(TS.trees()):
    x0=.22+index*.49
    xs={u:x0+.094*u for u in range(5)}
    def px(u):
        if u not in xs:xs[u]=float(np.mean([px(c) for c in t.children(u)]))
        return xs[u]
    px(t.root)
    ys={u:1.08-.58*TS.node(u).time/675 for u in t.nodes()}
    for u in t.nodes():
        p=t.parent(u)
        if p>=0:route([(xs[u],ys[u]),(xs[u],ys[p]),(xs[p],ys[p])],False,.65)
        ax.add_patch(Circle((xs[u],ys[u]),.018 if u==0 else .012,facecolor='white',edgecolor=INK,lw=1.3 if u==0 else .55,zorder=5))
    txt(x0+.188,.31,f'T{index+1}')
    txt(xs[0],1.22,'q')
# Three genome segments are physically aligned with the three input trees.
for index in range(3):
    rect(.16+index*.49,1.40,.49,.075,GRAY,'white',.6)
route([(.16,1.55),(1.63,1.55)])
txt(.90,1.70,'genomic position')
# Compact concrete observation cards, kept subordinate to the model spine.
rect(.16,1.94,1.47,.65,GRAY,'#b5c0c4',.5)
alleles=TS.genotype_matrix()
for i in range(3):
    for j in range(5):
        rect(.23+j*.12,2.05+i*.12,.12,.12,'#dce2e5' if alleles[i,j] else 'white','white',.3)
        txt(.29+j*.12,2.11+i*.12,str(alleles[i,j]),8)
txt(.53,2.68,'alleles')
for j,v in enumerate(['?','A','C','W','E']):
    txt(.98+j*.115,2.10,v,8)
txt(1.23,2.38,'reference\nlabels',8)
# Inputs join at the shared encoder without introducing a second feature pipeline.
route([(1.63,1.13),(1.77,1.13),(1.77,1.35),(1.94,1.35)])
route([(1.63,2.22),(1.77,2.22),(1.77,1.35)])
txt(2.03,.62,'b',10,bold=True)
learned(1.94,.97,.91,.81,'TreeLoom\nencoder')
rect(1.94,1.78,.91,.33,GRAY,GOLD_EDGE,.7)
txt(2.395,1.945,'2 rounds · Fig. 2',8)
# All-node states for every tree. The same selected leaf row has a heavy outline.
route([(2.85,1.35),(3.03,1.35)])
txt(3.35,.73,'Node states')
for i in range(3):
    xx=3.03+i*.23
    txt(xx+.09,.93,f'T{i+1}',8)
    grid(xx,1.05,.18,.64,9,3,key=i)
    rect(xx,1.05,.18,.64/9,'none',INK,1.2)
txt(2.94,1.085,'q',8)
# A single leaf's sequence, with interval features physically appended.
txt(4.01,.16,'c',10,bold=True)
for i,ww in enumerate([.19,.25,.19]):
    xx=4.04+sum([.19,.25,.19][:i])
    rect(xx,.43,ww,.065,GRAY,INK,.5)
    route([(xx,.41),(xx,.52)],False,.5)
route([(4.67,.41),(4.67,.52)],False,.5)
txt(4.35,.27,'Intervals',8)
route([(4.61,.52),(4.61,.75),(4.61,1.05)])
txt(4.20,.89,'Leaf q')
for i in range(3):
    rect(4.02,1.05+i*.213,.17,.213,'white','white',.3)
    txt(4.105,1.156+i*.213,f'T{i+1}',8)
    grid(4.19,1.05+i*.213,.36,.213,1,4,key=i)
    for j in range(3):rect(4.55+j*.04,1.05+i*.213,.04,.213,'#c0c8cc','white',.3)
rect(4.02,1.05,.65,.64,'none',INK,.6)
route([(3.67,1.35),(4.02,1.35)])
# Contextual sequence and direct local embedding meet at the ancestry head.
txt(5.03,.62,'d',10,bold=True)
learned(5.01,1.10,.77,.50,'BiGRU')
route([(4.67,1.35),(5.01,1.35)])
learned(5.01,1.96,.77,.45,'Ancestry\nhead')
route([(5.395,1.60),(5.395,1.96)])
route([(4.37,1.69),(4.37,2.185),(5.01,2.185)])
# Output: per-interval class probabilities, not one global class for a tree set.
txt(6.43,1.60,'Ancestry')
for i in range(3):
    txt(6.89,1.975+.21*i,f'T{i+1}',8)
probs=np.array([[.78,.07,.08,.07],[.12,.72,.09,.07],[.10,.15,.65,.10]])
for i in range(3):
    for j in range(4):rect(6.14+j*.15,1.87+i*.21,.15,.21,PURPLE(probs[i,j]),'white',.4)
for j,v in enumerate(['A','C','W','E']):txt(6.215+j*.15,2.65,v,8)
route([(5.78,2.185),(6.14,2.185)])
# A single compact semantic legend. Node rings identify the selected query leaf.
for x,color,label in [(.20,GRAY,'observations'),(1.72,GOLD,'learned block'),(3.45,TEAL(.6),'embedding'),(4.97,PURPLE(.6),'probability')]:
    rect(x,2.89,.13,.075,color,INK,.45);txt(x+.19,2.9275,label,8,ha='left')
fig.canvas.draw();ren=fig.canvas.get_renderer()
overlaps=[]
for i,a in enumerate(texts):
    ba=a.get_window_extent(ren)
    for b in texts[i+1:]:
        bb=b.get_window_extent(ren)
        dx=min(ba.x1,bb.x1)-max(ba.x0,bb.x0);dy=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
        if dx>.5 and dy>.5:overlaps.append([a.get_text(),b.get_text()])
(OUT/'architecture_layout_qa.json').write_text(json.dumps({'min_font_pt':min(t.get_fontsize() for t in texts),'text_overlaps':overlaps},indent=2)+'\n')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'TreeLoom_architecture.{ext}',dpi=260,facecolor='white')
p=OUT/'TreeLoom_architecture.svg';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
print(json.dumps({'text_overlaps':overlaps,'figure_inches':[7,3.05]}))
