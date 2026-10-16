"""TreeLoom scientific schematic v4. Editable vector geometry, no model execution.

Uses the5-sample seed628 illustrative ARG made by make_illustrative_arg.py.
All writes stay beside this generator. Run with .runtime/env/bin/python -B.
"""
from pathlib import Path
import os, json, hashlib
OUT=Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"]=str(OUT/".mpl-cache")
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np
import tskit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.3,"mathtext.fontset":"dejavusans",
 "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
INK="#263545"; MUTED="#6b7a8c"; LIGHT="#d7dfe7"; FAINT="#eef2f6"
QUERY="#b12a8f"
SAMPLE=[QUERY]+["#7e8993"]*4
SEG=["#a2a9b0","#c0c6cc","#e0e3e7"]
HATCH=["","////","xxxx"]
ANCESTRY=["#72a4cb","#d69b76","#ad99c4","#88b5a1"]
NAMES={0:"q1",1:"r1",2:"r2",3:"r3",4:"r4"}
CLASS_NAMES=["Ana","CHG","WHG","EHG"]
full=tskit.load(OUT/"illustrative_full_arg.trees")
ts=tskit.load(OUT/"illustrative_tree_sequence.trees")
breaks=list(full.breakpoints())
assert full.num_trees==ts.num_trees==3 and full.num_samples==5
assert breaks==list(ts.breakpoints())
trees=[t.copy() for t in ts.trees()]
full_trees=[t.copy() for t in full.trees()]
TMAX=max(n.time for n in ts.nodes())
rec={n.id for n in full.nodes() if n.flags&131072}
# Exactly the same native trees, leaf order, node coordinates and ages in a and b.
layouts=[]
for t in trees:
    order=list(t.samples())
    xx={q:i/4 for i,q in enumerate(order)}
    def pos(u):
        if u not in xx:xx[u]=float(np.mean([pos(c) for c in t.children(u)]))
        return xx[u]
    pos(t.root)
    yy={u:ts.node(u).time/TMAX for u in t.nodes()}
    layouts.append((xx,yy,order))
    assert t.num_roots==1 and set(t.samples())=={0,1,2,3,4}
    for u in t.nodes():
        p=t.parent(u)
        assert p==-1 or ts.node(p).time>ts.node(u).time

W,H=7.,5.05
fig=plt.figure(figsize=(W,H))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis("off")
texts=[]
def txt(x,y,s,size=7.3,ha="center",va="center",c=INK,weight="normal",**kw):
    a=ax.text(x,y,s,fontsize=size,color=c,ha=ha,va=va,fontweight=weight,**kw);texts.append(a);return a
def line(points,c=INK,lw=.85,alpha=1,z=2):
    a=np.asarray(points);ax.plot(a[:,0],a[:,1],color=c,lw=lw,alpha=alpha,zorder=z)
def arrow(a,b,c=INK,lw=.8,ms=6,style="-|>",z=4):
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,color=c,lw=lw,shrinkA=0,shrinkB=0,zorder=z)
    ax.add_patch(p);return p
def route(points,c=INK,lw=.8,ms=6,z=4):
    line(points[:-1],c,lw,z=z)
    arrow(points[-2],points[-1],c,lw,ms,z=z)
def box(x,y,w,h,fc="white",ec=LIGHT,lw=.7,r=.04,z=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=z)
    ax.add_patch(p);return p
def dot(x,y,sample=None,r=.026,shape="o",z=5):
    color=INK if sample is None else SAMPLE[sample]
    if shape=="D":
        ax.add_patch(Polygon([(x,y+r),(x+r,y),(x,y-r),(x-r,y)],fc="white",ec=color,lw=.7,zorder=z))
    elif sample is not None and sample==0:
        ax.add_patch(Rectangle((x-r,y-r),2*r,2*r,fc="white",ec=color,lw=1.1,zorder=z))
    else:ax.add_patch(Circle((x,y),r,fc=color,ec="white",lw=.45,zorder=z))
def letter(x,y,s):txt(x,y,s,9.2,ha="left",weight="bold")
def tree(index,cx,base,width,height,c=INK,slant=0,features=False,labels=False,recombinations=True,lw=.75):
    t=trees[index];xx,yy,order=layouts[index]
    coords={u:(cx-width/2+width*xx[u],base+height*yy[u]+slant*xx[u]) for u in t.nodes()}
    # Native branch topology is preserved, including recorded unary event nodes.
    for u in t.nodes():
        p=t.parent(u)
        if p!=-1:
            a=coords[u];b=coords[p]
            # Same planar rectangular geometry in both panels before affine projection.
            bend=(a[0],base+height*yy[p]+slant*xx[u])
            line([a,bend,b],c,lw,z=3)
    for u,(x,y) in coords.items():
        if u<5:
            dot(x,y,u,r=.029 if labels else .023)
            if labels:txt(x,base-.115,NAMES[u],6.8,c=SAMPLE[u])
        elif len(t.children(u))>1:dot(x,y,r=.017)
    if recombinations:
        native=full_trees[index]
        clade_nodes={tuple(sorted(t.samples(v))):v for v in t.nodes()}
        for u in native.nodes():
            if u not in rec:continue
            clade=tuple(sorted(native.samples(u)))
            v=clade_nodes[clade]
            pp=t.parent(v);age=full.node(u).time
            if pp!=-1 and ts.node(v).time<=age<=ts.node(pp).time:
                x=coords[v][0];y=base+height*age/TMAX+slant*xx[v]
                dot(x,y,r=.020,shape="D",z=5)
    if features:
        for q in order:
            x,_=coords[q]
            # Observed genotype-derived vector available at reference AND query.
            fy=base-.36
            for j in range(3):
                val=int(full.genotype_matrix()[j,q])
                ax.add_patch(Rectangle((x-.071+j*.047,fy),.044,.082,fc="#8c9eb0" if val else "#e4eaf0",ec="white",lw=.3,zorder=4))
                ax.add_patch(Rectangle((x-.071+j*.047,fy+.088),.044,.024,
                  fc=SEG[j],ec="#969fa8",hatch=HATCH[j],lw=.22,zorder=4))
            if q>0:
                badge=CLASS_NAMES[q-1];col=ANCESTRY[q-1]
                ax.add_patch(Rectangle((x-.102,fy-.135),.204,.100,fc=col,ec=col,lw=.5,zorder=4))
                txt(x,fy-.086,badge,6.0,c=INK,weight="normal",zorder=5)
                for k in range(4):
                    ax.add_patch(Rectangle((x-.084+k*.042,fy-.215),.040,.040,
                      fc=ANCESTRY[q-1] if k==q-1 else "white",ec="#9ca8b2",lw=.25,zorder=4))
            else:
                ax.add_patch(Rectangle((x-.063,fy-.135),.126,.100,fc="white",ec=SAMPLE[q],lw=.6,zorder=4))
                txt(x,fy-.086,"?",6.4,c=SAMPLE[q],zorder=5)
                for k in range(4):
                    ax.add_patch(Rectangle((x-.084+k*.042,fy-.215),.040,.040,
                      fc="white",ec="#9ca8b2",lw=.25,zorder=4))
    return coords

# a: A wireframe ARG. Native tree slices are connected by exact native-node
# identity only where a lineage occurs in both intervals. No opaque surfaces.
letter(.13,4.86,"a")
txt(.48,4.85,"ARG",7.8,ha="left")
src_centers=[.62,1.46,2.30]
src=[]
ax.add_patch(Polygon([(.31,3.34),(1.99,3.34),(2.61,3.44),(.93,3.44)],
  fc="none",ec="#c8d0d8",lw=.50,zorder=.3))
txt(.12,3.31,"0",6.5,c=MUTED)
for i,cx in enumerate(src_centers):
    src.append(tree(i,cx,3.34,.62,1.10,c="#63717e",slant=.10,lw=.82))
    l=cx-.31;r=cx+.31
    ax.add_patch(Polygon([(l,3.285),(r,3.385),(r,3.415),(l,3.315)],
      fc=SEG[i],ec="#a0a9b2",hatch=HATCH[i],lw=.25,zorder=.4))
# Faint interval bridges provide the longitudinal dimension of the ARG.
for i in (0,1):
    shared=set(src[i])&set(src[i+1])
    for u in sorted(shared):
        a,b=src[i][u],src[i+1][u]
        # Longitudinal sample rails and coalescent lineage guides are context.
        line([a,b],"#b9c4cf",.38,alpha=.50,z=.8)
# Exact biological split detail, preserving the validated marginal views.
# Both paired event nodes9/10 have child1 and parents12/11 in full ARG.
assert full_trees[0].parent(9)==12 and full_trees[1].parent(10)==11
assert full_trees[0].children(9)==(1,) and full_trees[1].children(10)==(1,)
zoom=FancyBboxPatch((1.01,3.82),.94,.77,boxstyle="round,pad=0,rounding_size=.04",
  fc="none",ec="#d2d9e0",lw=.45,ls=(0,(2,3)),zorder=.3)
ax.add_patch(zoom)
rx,ry=1.48,4.22
dot(rx,ry,r=.030,shape="D")
txt(rx+.085,ry,"$R$",6.9,ha="left")
# Child and the two genuine older parents, in a compact exact-subgraph detail.
dot(rx,3.99,1,r=.023)
txt(rx,3.90,"r1",6.2,c=MUTED)
arrow((rx,4.03),(rx,ry-.035),INK,.85,5)
dot(1.12,4.50,r=.022);dot(1.84,4.48,r=.022)
arrow((rx-.025,ry+.025),(1.14,4.47),INK,.85,5)
arrow((rx+.025,ry+.025),(1.82,4.45),INK,.85,5)
# Dashed magnification leaders locate this same event's two interval copies.
event_positions=[]
for idx,event in [(0,9),(1,10)]:
    t=trees[idx];ft=full_trees[idx];xx,yy,_=layouts[idx]
    clade=tuple(sorted(ft.samples(event)))
    v=next(u for u in t.nodes() if tuple(sorted(t.samples(u)))==clade)
    point=(src[idx][v][0],3.34+1.10*full.node(event).time/TMAX+.10*xx[v])
    event_positions.append(point)
    txt(point[0]-.055,point[1]+.045,"$R$",5.6,c=MUTED)
    ax.plot([rx,point[0]],[ry-.035,point[1]+.025],ls=(0,(2,3)),
      color="#aeb9c3",lw=.45,zorder=.5)

# Exact genomic interval strip, unequal widths shown on a common genomic axis.
x0=.26;x1=2.67;y=3.04
for i,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    a=x0+(x1-x0)*l/10000;b=x0+(x1-x0)*r/10000
    ax.add_patch(Rectangle((a,y),b-a,.080,fc=SEG[i],ec="#8c969f",hatch=HATCH[i],lw=.45))
    txt((a+b)/2,y+.155,f"$I_{i+1}$",7.0,c=INK)
for j,site in enumerate(full.sites()):
    x=x0+(x1-x0)*site.position/10000
    ax.add_patch(Rectangle((x-.018,y+.093),.036,.052,fc=SEG[j],
      ec="#7e8994",hatch=HATCH[j],lw=.35))
for g in breaks:
    x=x0+(x1-x0)*g/10000
    line([(x,y-.02),(x,y-.06)],MUTED,.6)
    txt(x,y-.15,f"{g/1000:.2f}".rstrip("0").rstrip("."),6.6)
arrow((x0,y-.28),(x1,y-.28),MUTED,.6,5)
txt((x0+x1)/2,y-.42,"kb",6.9,c=MUTED)
arrow((.13,3.40),(.13,4.56),MUTED,.55,5)
txt(.035,3.98,"time",6.8,c=MUTED,rotation=90)
# Small event symbol stays in the caption rather than a prose legend.
dot(2.30,4.79,r=.022,shape="D")
txt(2.41,4.79,"recomb.",6.6,ha="left",c=MUTED)

# b: exact same native trees and planar coordinates, expanded for observations.
arrow((2.84,4.01),(3.10,4.01),MUTED,1.1,7)
letter(3.22,4.86,"b")
dst_centers=[3.85,5.03,6.21]
dst=[]
for i,cx in enumerate(dst_centers):
    txt(cx,4.78,f"$T_{i+1}$",8.0,c=INK)
    ax.add_patch(Rectangle((cx-.47,4.59),.94,.035,fc=SEG[i],ec="#939da7",hatch=HATCH[i],lw=.35))
    dst.append(tree(i,cx,3.39,.94,1.10,c="#63717e",features=True,labels=True,recombinations=False,lw=.8))
    txt(cx,2.72,f"$I_{i+1}$",7.0,c=INK)
# Observed ancestor vector attached to a genuine internal nonroot, not a tip.
ancestor=next(u for u in trees[1].nodes()
  if trees[1].parent(u)!=-1 and len(trees[1].children(u))>1)
assert ancestor not in set(ts.samples()) and trees[1].parent(ancestor)!=-1
root_xy=dst[1][ancestor]
line([root_xy,(root_xy[0]+.10,root_xy[1]+.13),(root_xy[0]+.12,root_xy[1]+.13)],MUTED,.5)
for j,label in enumerate([r"$\tau$","0","0"]):
    xx=root_xy[0]+.12+j*.075
    ax.add_patch(Rectangle((xx,root_xy[1]+.085),.072,.09,fc="#eef1f5",ec="#a2adb8",lw=.35))
    txt(xx+.036,root_xy[1]+.13,label,5.7)
# One compact input key. Vectors are actual observed alleles, badges show role.
txt(4.00,2.51,"alleles",6.9,ha="left",c=MUTED)
ax.add_patch(Rectangle((3.75,2.465),.16,.082,fc="#9babb9",ec="white",lw=.3))
txt(4.91,2.51,"ref.",6.9,ha="left",c=MUTED)
ax.add_patch(Rectangle((4.62,2.462),.23,.10,fc=ANCESTRY[0],ec="none"))
txt(4.735,2.512,"Ana",5.8,c=INK)
txt(5.84,2.51,"query",6.9,ha="left",c=MUTED)
ax.add_patch(Rectangle((5.61,2.462),.15,.10,fc="white",ec=SAMPLE[0],lw=.6))
txt(5.685,2.512,"?",6.2,c=SAMPLE[0])
# A single connector returns to the next row in clear whitespace.
route([(6.77,2.78),(6.88,2.78),(6.88,2.40),(.48,2.40),(.48,2.02)],MUTED,.65,5)

# c: full-tree graph encoding. All nodes remain in the first two traversals.
letter(.13,2.12,"c")
txt(.65,2.15,"MLP",7.2)
def matrix(x,y,w,h,rows=7,cols=4):
    for r in range(rows):
        q=r-(rows-5)
        col="#8c9eaf"
        for j in range(cols):
            ax.add_patch(Rectangle((x+j*w/cols,y+(rows-1-r)*h/rows),w/cols,h/rows,
             fc=col,alpha=.30+.12*((r+j)%4),ec="white",lw=.38,zorder=3))
    ax.add_patch(Rectangle((x,y),w,h,fc="none",ec=MUTED,lw=.65,zorder=4))
matrix(.29,1.44,.38,.49)
txt(.48,1.31,"$N_t\\times d$",6.8)
arrow((.71,1.69),(.93,1.69),ms=6)
# Two graph modules, each explicitly up then down over the whole tree.
def mini_message_tree(cx,base,width=.24,height=.40,direction="up"):
    qx=np.linspace(cx-width/2,cx+width/2,5)
    n=[(float(x),base) for x in qx]
    n+=[(cx-width*.38,base+height*.38),(cx+width*.14,base+height*.45),
        (cx+width*.37,base+height*.73),(cx,base+height)]
    edges=[(0,5),(1,5),(2,6),(3,6),(6,7),(4,7),(5,8),(7,8)]
    for a,b in edges:
        pa,pb=n[a],n[b]
        line([pa,pb],LIGHT,.65,z=2)
        axy=np.asarray(pa)*.72+np.asarray(pb)*.28
        bxy=np.asarray(pa)*.34+np.asarray(pb)*.66
        if direction=="down":axy,bxy=bxy,axy
        arrow(tuple(axy),tuple(bxy),MUTED,.7,3.8,z=3)
    for i,(x,y) in enumerate(n):dot(x,y,i if i<5 else None,r=.014,z=5)
def gnn(x,y):
    box(x,y,.76,.82,fc="#fbfcfe",ec="#c7d3de",r=.045)
    mini_message_tree(x+.205,y+.19,direction="up")
    mini_message_tree(x+.565,y+.19,direction="down")
    txt(x+.205,y+.085,"Up",6.7)
    txt(x+.565,y+.085,"Down",6.7)
    arrow((x+.31,y+.085),(x+.39,y+.085),INK,.75,4.4)
gnn(.96,1.26)
gnn(2.94,1.26)
txt(1.34,1.99,"GNN",7.2)
txt(3.32,1.99,"GNN",7.2)
# Shared weights are indicated as a small equality bracket, no layer-name jargon.
ax.plot([1.34,1.34,3.32,3.32],[2.08,2.27,2.27,2.08],
  color=MUTED,lw=.65,ls=(0,(3,2)),zorder=1)
txt(2.33,2.32,"same weights",6.5,c=MUTED)
arrow((1.76,1.69),(1.91,1.69),ms=5)
box(1.94,1.43,.43,.51,fc="#f5f7f9",ec="#c2ccd5",r=.03)
# A learned nodewise operation, with clade-change information defined in caption.
for yy in [1.56,1.69,1.82]:
    dot(2.055,yy,r=.018)
    arrow((2.10,yy),(2.22,yy),MUTED,.55,3.5)
txt(2.155,1.31,"Node update",6.7)
arrow((2.39,1.69),(2.55,1.69),ms=5)
ax.add_patch(Circle((2.66,1.69),.085,fc="white",ec=INK,lw=.8,zorder=5))
txt(2.66,1.69,"+",9,zorder=6)
arrow((2.77,1.69),(2.91,1.69),ms=5)
# Original node feature base is the direct residual input, not first-pass state.
route([(.67,1.89),(.79,1.89),(.79,2.14),(2.66,2.14),(2.66,1.79)],MUTED,.70,5)
dot(.67,1.89,r=.015)
# Sample memory: all5persistent sampled rows, q1 highlighted.
txt(1.89,1.02,"Sample memory",7.0)
memx=[1.16,1.83,2.50]
for j,x in enumerate(memx):
    txt(x,.91,f"$T_{j+1}$",6.2,c=INK)
    matrix(x-.08,.21,.16,.40,rows=5,cols=3)
    ax.add_patch(Rectangle((x-.09,.53),.18,.08,fc="none",ec=QUERY,lw=.75,zorder=5))
    # Current forcing context is available to both directional updates.
    force_x=x if j==0 else x-.15
    for k in range(3):
        ax.add_patch(Rectangle((force_x-.060+k*.040,.75),.037,.063,
          fc="#8c9eaf",alpha=.35+.16*k,ec="white",lw=.25))
    if j==0:arrow((x,.72),(x,.64),MUTED,.55,3.5)
    else:arrow((force_x,.72),(force_x,.635),MUTED,.55,3.5)
for a,b in zip(memx[:-1],memx[1:]):
    # All other sample rows persist independently, without ancestry colors.
    for yy in [.49,.41,.33,.25]:
        arrow((a+.09,yy),(b-.09,yy),"#bac4cd",.4,2.5,style="<->")
    gate=(a+b)/2-.04
    # Forward scan: decayed previous state + current forcing.
    arrow((a+.09,.60),(gate-.031,.60),QUERY,.75,3.5)
    ax.add_patch(Circle((gate,.57),.049,fc="white",ec=MUTED,lw=.55,zorder=5))
    txt(gate,.57,r"$\rho$",6.0,zorder=6)
    plus=b-.15
    arrow((gate+.033,.60),(plus-.035,.60),QUERY,.75,3.5)
    ax.add_patch(Circle((plus,.60),.030,fc="white",ec=MUTED,lw=.55,zorder=5))
    txt(plus,.60,"+",6.2,zorder=6)
    arrow((plus+.032,.60),(b-.09,.60),QUERY,.75,3.5)
    # The reverse scan uses the same physical-span decay, independently.
    arrow((b-.09,.54),(gate+.033,.54),QUERY,.65,3.5)
    arrow((gate-.029,.54),(a+.09,.54),QUERY,.65,3.5)
    # A physical interval-distance glyph defines the gate's genomic coordinate.
    line([(gate-.10,.16),(gate+.10,.16)],MUTED,.45)
    line([(gate-.10,.14),(gate-.10,.18)],MUTED,.45)
    line([(gate+.10,.14),(gate+.10,.18)],MUTED,.45)
    txt(gate,.10,r"$\Delta x$",5.7,c=MUTED)
txt(.95,.57,"$q_1$",6.2,c=QUERY,ha="right")
# Current cut-group mean and injection at the representative node.
arrow((2.60,.57),(2.71,.57),MUTED,.65,4)
ax.add_patch(Circle((2.76,.57),.045,fc="white",ec=MUTED,lw=.6,zorder=5))
txt(2.76,.57,r"$\mu$",6.3,zorder=6)
# Schematic representative-node location, distinct from the temporal scan.
line([(2.66,.89),(2.76,1.04),(2.87,.89)],LIGHT,.65)
dot(2.66,.89,0,r=.016);dot(2.87,.89,1,r=.016)
dot(2.76,1.04,r=.025)
route([(2.76,.63),(2.90,.63),(2.90,1.04),(2.80,1.04)],MUTED,.6,4)
route([(2.76,1.08),(2.76,1.15),(2.66,1.15),(2.66,1.59)],MUTED,.65,4.5)
route([(1.82,1.69),(1.82,1.14),(.96,1.14),(.96,.78),(1.08,.78)],MUTED,.65,4.5)

# Per-sample vectors in each tree, never pooled graph embeddings.
arrow((3.74,1.69),(3.92,1.69),ms=6)
embx=[3.99,4.23,4.47]
for j,x in enumerate(embx):
    txt(x,2.02,f"$T_{j+1}$",6.5,c=INK)
    matrix(x-.09,1.37,.18,.51,rows=5,cols=3)
    # Explicit persistent row selection; all rows still go through the GRU.
    ax.add_patch(Rectangle((x-.105,1.37+.51*4/5-.010),.21,.51/5+.02,fc="none",ec=SAMPLE[0],lw=.8,zorder=5))
txt(4.23,1.20,"$S\\times d$",6.9)
for a,b in zip(embx[:-1],embx[1:]):
    line([(a+.11,1.83),(b-.11,1.83)],SAMPLE[0],.7)
# Highlighted row is explicitly projected into the local per-sample vector.
line([(4.575,1.83),(4.65,1.83)],QUERY,.8)
for k in range(3):
    ax.add_patch(Rectangle((4.65+k*.047,1.785),.044,.09,
      fc="#8c9eaf",alpha=.35+.15*k,ec="white",lw=.25,zorder=4))
ax.add_patch(Rectangle((4.645,1.78),.15,.10,fc="none",ec=QUERY,lw=.7,zorder=5))
txt(4.72,1.98,"$e_{tq}$",6.8,c=QUERY)
line([(4.72,1.78),(4.72,1.69)],QUERY,.7)
dot(4.72,1.69,r=.018)
# d: one shared genomic recurrent layer processes each aligned haplotype.
letter(4.83,2.12,"d")
txt(5.40,2.13,"BiGRU",7.5)
box(4.91,1.24,.98,.76,fc="#f6f8fa",ec="#bdc9d4",r=.045)
txt(5.40,1.89,"1 layer",6.7,c=MUTED)
for j,x in enumerate([5.10,5.40,5.70]):
    box(x-.10,1.46,.20,.23,fc="#e2e8ee",ec="#9daebb",r=.025)
    txt(x,1.575,"h",7.0,c=INK)
for a,b in zip([5.10,5.40],[5.40,5.70]):
    arrow((a+.10,1.65),(b-.10,1.65),SAMPLE[0],.8,4)
    arrow((b-.10,1.50),(a+.10,1.50),SAMPLE[0],.65,4)
line([(4.56,1.69),(4.72,1.69)],INK,.8)
arrow((4.72,1.69),(4.87,1.69),ms=5)
# Interval geometry enters recurrence, local vector also reaches classifier.
for j,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    ax.add_patch(Rectangle((5.05+.70*l/10000,1.02),.70*(r-l)/10000,.065,
      fc=SEG[j],ec="#a0a9b2",hatch=HATCH[j],lw=.35))
for g in breaks:
    x=5.05+.70*g/10000
    line([(x,.99),(x,.96)],MUTED,.5)
arrow((5.4,1.13),(5.4,1.23),MUTED,.6,4)
route([(4.72,1.69),(4.72,.82),(6.02,.82),(6.02,1.61)],MUTED,.65,4)
# Local and recurrent contexts meet at an explicit concatenation node.
arrow((5.92,1.69),(5.95,1.69),ms=4)
ax.add_patch(Circle((6.02,1.69),.065,fc="white",ec=INK,lw=.7,zorder=5))
txt(6.02,1.69,r"$\Vert$",7.2,zorder=6)
arrow((6.10,1.69),(6.29,1.69),ms=5)
txt(6.19,1.94,"MLP",6.8)
# Schematic query probabilities and calls over the same three native intervals.
probs=np.asarray([[.08,.07,.78,.07],[.16,.08,.67,.09],[.07,.13,.74,.06]])
px0,px1=6.32,6.92
for i,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    a=px0+(px1-px0)*l/10000;b=px0+(px1-px0)*r/10000
    low=1.30
    for k,p in enumerate(probs[i]):
        ax.add_patch(Rectangle((a,low),b-a,p*.75,fc=ANCESTRY[k],ec="white",lw=.35))
        low+=p*.75
    ax.add_patch(Rectangle((a,1.07),b-a,.11,fc=ANCESTRY[int(np.argmax(probs[i]))],ec="white",lw=.35))
# The discrete strip is argmax of these schematic probabilities.
arrow((6.62,1.28),(6.62,1.20),MUTED,.6,4.0)
txt(6.565,2.19,"$p_{tq}$",8.2)
txt(6.565,.91,"$q_1$",7.1,c=SAMPLE[0])
arrow((6.23,.70),(6.90,.70),MUTED,.6,4)
txt(6.565,.58,"genome",6.7,c=MUTED)
for (x,y,k,name) in [(6.11,.35,0,"Ana"),(6.65,.35,1,"CHG"),(6.11,.16,2,"WHG"),(6.65,.16,3,"EHG")]:
    ax.add_patch(Rectangle((x,y-.035),.09,.07,fc=ANCESTRY[k],ec="none"))
    txt(x+.125,y,name,6.1,ha="left",c=MUTED)

# Deliver vector formats and a high-resolution PNG from the same geometry.
fig.savefig(OUT/"TreeLoom_architecture.svg")
fig.savefig(OUT/"TreeLoom_architecture.pdf")
fig.savefig(OUT/"TreeLoom_architecture.png",dpi=220)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
bounds=[]
for t in texts:
    bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    bounds.append({"text":t.get_text(),"fontsize":t.get_fontsize(),"bbox_figure":list(bb.bounds)})
outside=[b for b in bounds if b["bbox_figure"][0]<-.004 or b["bbox_figure"][1]<-.004
 or b["bbox_figure"][0]+b["bbox_figure"][2]>1.004 or b["bbox_figure"][1]+b["bbox_figure"][3]>1.004]
assert not outside, outside
(OUT/"text_layout_bounds.json").write_text(json.dumps(bounds,indent=2)+"\n")
source_files=[
 "src/arggnn/mutation_routed_whole_tree_gnn_v1.py",
 "src/arggnn/mutation_routed_pyg_gnn_v1.py",
 "src/arggnn/anonymous_clade_gnn_v1.py",
 "src/arggnn/anonymous_clade_flag_control_v1.py",
 "src/arggnn/ordered_clade_episode_gnn_v1.py",
 "research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py",
 "research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/direct_readout_cut_memory_v1.py"]
root=OUT.parent.parent
metadata={
 "revision":"v4_visual_two_row","figure_inches":[W,H],"titles":False,
 "native_ARG_source":"5-sample msprime seed628 full ARG; make_illustrative_arg.py",
 "topology_match":"a and b call the exact same tree() function using the same unary-collapsed tree layouts; only width/slant transform and recombination marker visibility differ",
 "ARG_intervals_bp":breaks,"display_tree_nodes":[len(list(t.nodes())) for t in trees],"full_native_tree_nodes":[len(list(t.nodes())) for t in full_trees],
 "unary_nodes_collapsed_in_both_panels":True,"recombination_markers":"ARG only, exact original ages on collapsed branches",
 "query_genotypes_observed":True,"query_ancestry_badge":"?","reference_ancestry_badges":CLASS_NAMES,"interval_encoding":"gray solid, diagonal hatch, crosshatch","ancestry_colors_reserved_for":"supplied reference badges and ancestry outputs only","identity_encoding":"neutral references, magenta outlined q1 and selected sample row",
 "node_shapes":{"reference":"neutral filled circle","query":"magenta open square","recombination":"open diamond"},
 "dimensions":{"node_states":"N_t × d","tree_sample_states":"S × d, d=32 in implementation","genomic_GRU":"1 bidirectional layer,32 per direction,35 input"},
 "no_whole_tree_graph_pooling":True,"sample_identity_preserved_in_each_row":True,
 "output_probs":"schematic WHG-dominant probabilities, varying confidence; WHGargmax in all3intervals, not fitted model predictions","feature_values":"schematic illustrative input values",
 "source_sha256":{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files},
 "illustrative_full_arg_sha256":hashlib.sha256((OUT/"illustrative_full_arg.trees").read_bytes()).hexdigest(),
 "text_count":len(texts),"minimum_font_size":min(t.get_fontsize() for t in texts),
 "validation":{"trees_connected":True,"parent_ages_strict":True,"exact_shared_layouts":True,"no_text_outside_canvas":True}}
(OUT/"figure_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps(metadata,indent=2))

