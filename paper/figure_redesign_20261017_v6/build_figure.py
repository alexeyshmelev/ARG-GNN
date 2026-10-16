"""TreeLoom scientific schematic v6. Editable vector geometry, no model execution.

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
    ax.add_patch(Rectangle((x-.018,y-.100),.036,.045,fc=SEG[j],
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
# A bracket groups all local trees and observed inputs before the node encoder.
line([(3.30,2.69),(3.30,2.63),(6.76,2.63),(6.76,2.69)],MUTED,.65)
route([(3.30,2.63),(3.17,2.63),(3.17,2.40),(.48,2.40),(.48,2.20)],MUTED,.65,5)

# c: full-tree graph encoding. All nodes remain in the first two traversals.
letter(.13,2.12,"c")
box(.29,2.05,.38,.14,fc="#f5f7f9",ec="#bdc9d4",r=.025)
txt(.48,2.12,"MLP",6.8)
arrow((.48,2.045),(.48,1.95),MUTED,.65,4)
def matrix(x,y,w,h,rows=7,cols=4):
    for r in range(rows):
        q=r-(rows-5)
        col="#8c9eaf"
        for j in range(cols):
            ax.add_patch(Rectangle((x+j*w/cols,y+(rows-1-r)*h/rows),w/cols,h/rows,
             fc=col,alpha=.30+.12*((r+j)%4),ec="white",lw=.38,zorder=3))
    ax.add_patch(Rectangle((x,y),w,h,fc="none",ec=MUTED,lw=.65,zorder=4))
matrix(.29,1.44,.38,.49,rows=len(list(trees[0].nodes())))
txt(.48,1.31,"$N_t\\times d$",6.8)
arrow((.71,1.69),(.93,1.69),ms=6)
# Two graph modules, each explicitly up then down over the whole tree.
def mini_message_tree(cx,base,width=.24,height=.40,direction="up"):
    # Genuine T1 node identities and topology, with rank-spaced icon heights.
    t=trees[0];xx,_,order=layouts[0]
    ranks={q:0 for q in order}
    def rank(u):
        if u not in ranks:ranks[u]=1+max(rank(c) for c in t.children(u))
        return ranks[u]
    rank(t.root)
    n={u:(cx-width/2+width*xx[u],base+height*rank(u)/ranks[t.root]) for u in t.nodes()}
    for u in t.nodes():
        p=t.parent(u)
        if p!=-1:
            a,b=n[u],n[p]
            line([a,(a[0],b[1]),b],MUTED,.50,z=2)
    # One external direction arrow avoids arrowheads crowding the native branches.
    axx=cx+width/2+.025
    a,b=(axx,base+.04),(axx,base+height-.04)
    if direction=="down":a,b=b,a
    arrow(a,b,MUTED,.65,4.0,z=3)
    for u,(x,y) in n.items():dot(x,y,u if u in order else None,r=.012,z=5)
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
box(1.91,1.43,.50,.51,fc="#f5f7f9",ec="#c2ccd5",r=.03)
# Presence of three genuine proper descendant sets across T1/T2/T3.
# Only current-node flags/states enter the adapter; no inter-tree state arrows.
structural_clades=[frozenset((0,3,4)),frozenset((0,1,3,4)),frozenset((0,3))]
clade_sets=[{frozenset(t.samples(u)) for u in t.nodes()
             if 1<len(list(t.samples(u)))<5} for t in trees]
for j in range(3):
    ax.add_patch(Rectangle((1.952+j*.137,1.868),.108,.025,
      fc=SEG[j],ec="#9ca8b3",hatch=HATCH[j],lw=.25,zorder=4))
for row,clade in enumerate(structural_clades):
    yy=1.788-row*.128
    for j in range(3):
        xx=2.006+j*.137
        present=clade in clade_sets[j]
        if present:
            # The fork is a clade-presence symbol, not a new genealogy.
            line([(xx-.042,yy-.030),(xx-.042,yy+.030),(xx+.042,yy+.030),
                  (xx+.042,yy-.030)],MUTED,.65)
            dot(xx,yy+.030,r=.012)
            dot(xx-.042,yy-.030,r=.012)
            dot(xx+.042,yy-.030,r=.012)
        else:
            line([(xx-.030,yy),(xx+.030,yy)],"#c4cdd6",.55)
txt(2.16,1.31,"Clade update",6.7)
arrow((2.43,1.69),(2.55,1.69),ms=5)
ax.add_patch(Circle((2.66,1.69),.085,fc="white",ec=INK,lw=.8,zorder=5))
txt(2.66,1.69,"+",9,zorder=6)
arrow((2.77,1.69),(2.91,1.69),ms=5)
# Original node feature base is the direct residual input, not first-pass state.
route([(.67,1.89),(.79,1.89),(.79,2.14),(2.66,2.14),(2.66,1.79)],MUTED,.70,5)
dot(.67,1.89,r=.015)
# Stable-sample memory. Current groups and history rows use the exact toy cut.
txt(1.84,1.04,"Sample memory",7.0)
memx=[1.12,1.81,2.50]
memory_order=[0,3,4,1,2]  # One fixed identity order, not each tree's tip order.
cut_groups=[]
for t in trees:
    group={}
    for q in memory_order:
        u=q
        while t.parent(u)!=-1 and ts.node(t.parent(u)).time<=350.:
            u=t.parent(u)
        group.setdefault(u,[]).append(q)
    cut_groups.append(list(group.values()))
assert cut_groups==[[[0,3,4],[1],[2]],[[0,3,4],[1],[2]],[[0,3],[4],[1],[2]]]
def current_vector(cx,cy,w=.15,h=.060):
    for k in range(3):
        ax.add_patch(Rectangle((cx-w/2+k*w/3,cy-h/2),w/3,h,
          fc="white",ec="#778b9d",lw=.55,zorder=4))
def history_vector(cx,cy,w=.18,h=.085):
    for k in range(3):
        ax.add_patch(Rectangle((cx-w/2+k*w/3,cy-h/2),w/3,h,
          fc="#8c9eaf",alpha=.38+.14*k,ec="white",lw=.3,zorder=4))
    ax.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fc="none",ec=MUTED,lw=.6,zorder=5))
memory_rows=[]
for j,x in enumerate(memx):
    txt(x,.91,f"$T_{j+1}$",6.2,c=INK)
    # Tiny groups contain actual sampled identities at the350-generation cut.
    leafx={q:x-.18+.09*k for k,q in enumerate(memory_order)}
    for group in cut_groups[j]:
        left=min(leafx[q] for q in group)-.036
        right=max(leafx[q] for q in group)+.036
        box(left,.724,right-left,.086,fc="white",ec="#9eacb9",lw=.5,r=.018,z=2)
        for q in group:dot(leafx[q],.767,q,r=.016,z=5)
    # The highlighted group's current forcing is open, history is filled.
    current_vector(x,.631)
    txt(x+.125,.631,"$f$",6.5,c=INK,ha="left")
    group_center=float(np.mean([leafx[q] for q in cut_groups[j][0]]))
    route([(group_center,.716),(group_center,.690),(x,.690),(x,.676)],MUTED,.55,3.4)
    arrow((x,.594),(x,.543),MUTED,.55,3.4)
    matrix(x-.08,.20,.16,.32,rows=5,cols=3)
    rows={q:.20+(.5+4-k)*.32/5 for k,q in enumerate(memory_order)}
    memory_rows.append(rows)
    # Light outlines show current pooling groups, without merging sample states.
    for group in cut_groups[j]:
        low=min(rows[q] for q in group)-.031
        high=max(rows[q] for q in group)+.031
        box(x-.104,low,.208,high-low,fc="none",ec="#9eacb9",lw=.55,r=.018,z=5)
    ax.add_patch(Rectangle((x-.092,rows[0]-.031),.184,.062,
      fc="none",ec=QUERY,lw=.75,zorder=6))
    # Interval span sets the scan's retention, defined in the caption.
    line([(x-.12,.142),(x+.12,.142)],MUTED,.45)
    line([(x-.12,.125),(x-.12,.159)],MUTED,.45)
    line([(x+.12,.125),(x+.12,.159)],MUTED,.45)
    txt(x,.084,f"$w_{j+1}$",5.9,c=MUTED)
for a,b in zip(memx[:-1],memx[1:]):
    for k,q in enumerate(memory_order):
        yy=memory_rows[0][q]
        if q==0:
            arrow((a+.105,yy+.014),(b-.105,yy+.014),QUERY,.75,3.5)
            arrow((b-.105,yy-.014),(a+.105,yy-.014),QUERY,.65,3.5)
        else:
            arrow((a+.105,yy),(b-.105,yy),"#b7c2cd",.45,2.7,style="<->")
txt(.94,memory_rows[0][0],"$q_1$",6.3,c=QUERY,ha="right")
txt(.94,.360,"$z$",6.6,c=INK,ha="right")
# Average both directions and pool within the present group, then center.
# The two vector styles repeat the current/history visual key from the inset.
line([(2.63,.392),(2.67,.392),(2.67,.520),(2.63,.520)],MUTED,.6)
arrow((2.67,.456),(2.88,.456),MUTED,.65,3.6)
history_vector(3.01,.456)
txt(3.01,.562,r"$\bar z$",6.8)
txt(3.245,.456,"-",9.2)
current_vector(3.48,.456,w=.18,h=.085)
txt(3.48,.562,"$f$",6.8)
arrow((3.245,.505),(3.245,.701),MUTED,.65,3.8)
box(3.015,.722,.46,.13,fc="#f5f7f9",ec=MUTED,lw=.55,r=.018)
txt(3.245,.787,"Linear",6.0)
arrow((3.245,.86),(3.245,1.105),MUTED,.65,4)
route([(3.245,1.105),(2.66,1.105),(2.66,1.59)],MUTED,.65,4.5)
route([(1.82,1.69),(1.82,1.16),(.87,1.16),(.87,.631),(1.035,.631)],
      MUTED,.65,4.5)

# Per-sample vectors in each tree, with one persistent sampled row traced.
arrow((3.74,1.69),(3.92,1.69),ms=6)
embx=[3.99,4.23,4.47]
for j,x in enumerate(embx):
    txt(x,2.12,f"$T_{j+1}$",6.3,c=INK)
    txt(x,1.98,f"$e_{{{j+1}q}}$",6.4,c=QUERY)
    matrix(x-.09,1.37,.18,.51,rows=5,cols=3)
    ax.add_patch(Rectangle((x-.105,1.37+.51*4/5-.010),.21,.51/5+.02,fc="none",ec=QUERY,lw=.8,zorder=5))
txt(4.23,1.20,"$S\\times d$",6.9)
for a,b in zip(embx[:-1],embx[1:]):
    for row in range(5):
        yy=1.37+(row+.5)*.51/5
        line([(a+.11,yy),(b-.11,yy)],QUERY if row==4 else "#c1cbd4",.7 if row==4 else .4)
# This is an aligned sampled row, not a query-only input selection.
line([(4.575,1.83),(5.0,1.83)],QUERY,.7)
dot(4.72,1.69,r=.018)
# d: one shared genomic recurrent layer processes each aligned haplotype.
letter(4.83,2.12,"d")
txt(5.40,2.16,"BiGRU",7.5)
box(4.91,1.24,.98,.76,fc="#f6f8fa",ec="#bdc9d4",r=.045)
txt(5.40,2.06,"1 layer",6.0,c=MUTED)
for x in [5.10,5.40,5.70]:
    box(x-.10,1.37,.20,.51,fc="#e2e8ee",ec="#9daebb",r=.025)
    for row in range(5):
        yy=1.37+(row+.5)*.51/5
        if row:line([(x-.10,yy-.51/10),(x+.10,yy-.51/10)],"white",.5)
        txt(x,yy,"h",5.5,c=INK if row==4 else "#8796a5")
    ax.add_patch(Rectangle((x-.104,1.37+.51*4/5-.006),.208,.51/5+.012,
      fc="none",ec=QUERY,lw=.65,zorder=5))
for a,b in zip([5.10,5.40],[5.40,5.70]):
    for row in range(4):
        yy=1.37+(row+.5)*.51/5
        arrow((a+.10,yy),(b-.10,yy),"#aebbc7",.4,2.4,style="<->")
    arrow((a+.10,1.85),(b-.10,1.85),QUERY,.7,3.5)
    arrow((b-.10,1.81),(a+.10,1.81),QUERY,.6,3.5)
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
txt(5.62,1.165,r"$\log(1+\cdot)$",5.3,c=MUTED)
txt(5.40,.91,r"$(w_t,\Delta c_t^-,\Delta c_t^+)$",5.7,c=MUTED)
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
    ax.add_patch(Rectangle((a,1.01),b-a,.11,fc=ANCESTRY[int(np.argmax(probs[i]))],ec="white",lw=.35))
# The discrete calls follow an explicit interval-wise argmax operator.
arrow((6.62,1.29),(6.62,1.271),MUTED,.6,3.0)
box(6.455,1.175,.33,.09,fc="white",ec=MUTED,lw=.45,r=.015)
txt(6.62,1.22,"argmax",5.7)
arrow((6.62,1.168),(6.62,1.127),MUTED,.6,3.0)
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
fig.savefig(OUT/"TreeLoom_architecture_2x.png",dpi=440)
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
 "revision":"v6_visual_sample_memory","figure_inches":[W,H],"titles":False,
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
 "cut_groups_350":cut_groups,"memory_identity_order":memory_order,
 "structural_clades":[sorted(x) for x in structural_clades],
 "text_count":len(texts),"minimum_font_size":min(t.get_fontsize() for t in texts),
 "validation":{"trees_connected":True,"parent_ages_strict":True,"exact_shared_layouts":True,"no_text_outside_canvas":True}}
(OUT/"figure_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps(metadata,indent=2))

