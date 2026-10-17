"""TreeLoom v9: editable, static scientific schematic. No model execution.
All outputs and caches stay in this directory. Runtime: existing repository venv.
"""
from pathlib import Path
import os, json, hashlib
OUT=Path(__file__).resolve().parent
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[k]="1"
os.environ["MPLCONFIGDIR"]=str(OUT/".mpl-cache")
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
import numpy as np
import tskit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon

INK="#293746"; MUTED="#788796"; LIGHT="#c5d0da"; FAINT="#f3f6f9"
QUERY="#b52b91"; GENO="#dce6ee"
ANC=["#71a5ce","#dc9b73","#ad98c5","#86b7a1"]
ANAMES=["Ana","CHG","WHG","EHG"]
SEG=["#b4bcc4","#c6cdd3","#dce1e5"]; HATCH=["","///","xxx"]
NAMES={0:"q1",1:"r1",2:"r2",3:"r3",4:"r4"}
ROW=[0,1,2,3,4]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.2,
 "mathtext.fontset":"dejavusans","svg.fonttype":"none","pdf.fonttype":42,
 "ps.fonttype":42,"savefig.facecolor":"white"})
full=tskit.load(OUT/"illustrative_full_arg.trees")
ts=tskit.load(OUT/"illustrative_tree_sequence.trees")
trees=[t.copy() for t in ts.trees()]
fts=[t.copy() for t in full.trees()]
breaks=list(ts.breakpoints())
assert len(trees)==3 and ts.num_samples==5 and breaks==list(full.breakpoints())
TMAX=max(n.time for n in full.nodes())
layouts=[]
for t in trees:
    order=list(t.samples()); xx={q:i/4 for i,q in enumerate(order)}
    def calc(u):
        if u not in xx: xx[u]=float(np.mean([calc(c) for c in t.children(u)]))
        return xx[u]
    calc(t.root)
    yy={u:ts.node(u).time/TMAX for u in t.nodes()}
    layouts.append((xx,yy,order))
    assert set(order)==set(ROW) and t.num_roots==1
    assert all(t.parent(u)==-1 or ts.node(t.parent(u)).time>ts.node(u).time for u in t.nodes())

W,H=7.,6.60
fig=plt.figure(figsize=(W,H));ax=fig.add_axes([0,0,1,1])
ax.set(xlim=(0,W),ylim=(0,H));ax.axis("off")
texts=[];routes=[]
def txt(x,y,s,size=7.2,ha="center",c=INK,weight="normal",**kw):
    t=ax.text(x,y,s,fontsize=size,ha=ha,va="center",color=c,fontweight=weight,**kw)
    texts.append(t);return t
def line(points,c=INK,lw=.85,alpha=1,z=2,ls="-"):
    a=np.asarray(points);ax.plot(a[:,0],a[:,1],color=c,lw=lw,alpha=alpha,zorder=z,ls=ls)
def arrow(a,b,c=INK,lw=.85,ms=5.5,style="-|>",z=4):
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,lw=lw,
       color=c,shrinkA=0,shrinkB=0,zorder=z);ax.add_patch(p);return p
def route(points,c=INK,lw=.85,ms=5.5):
    line(points[:-1],c,lw);arrow(points[-2],points[-1],c,lw,ms);routes.append(points)
def box(x,y,w,h,fc="white",ec=LIGHT,lw=.8,r=.035):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                    fc=fc,ec=ec,lw=lw,zorder=1);ax.add_patch(p)
def dot(x,y,q=None,r=.022,diamond=False,z=5):
    if diamond:
        p=Polygon([(x,y+r),(x+r,y),(x,y-r),(x-r,y)],fc="white",ec=INK,lw=.75,zorder=z)
    elif q==0:
        p=Rectangle((x-r,y-r),2*r,2*r,fc="white",ec=QUERY,lw=1.0,zorder=z)
    else:
        p=Circle((x,y),r,fc=MUTED if q is not None else INK,ec="white",lw=.35,zorder=z)
    ax.add_patch(p)
def panel(x,y,s):txt(x,y,s,9.5,ha="left",weight="bold")
def small_vector(cx,cy,w=.20,h=.115,query=False,cols=3):
    for j in range(cols):
        ax.add_patch(Rectangle((cx-w/2+j*w/cols,cy-h/2),w/cols,h,
          fc=["#a5b5c3","#c6d1da","#8fa4b5"][j%3],ec="white",lw=.45,zorder=4))
    ax.add_patch(Rectangle((cx-w/2,cy-h/2),w,h,fc="none",
      ec=QUERY if query else "#a0aeba",lw=1.0 if query else .55,zorder=5))
def display_tree(i,cx,base,width,height,labels=True):
    t=trees[i];xx,yy,order=layouts[i]
    pp={u:(cx-width/2+width*xx[u],base+height*yy[u]) for u in t.nodes()}
    for u in t.nodes():
        p=t.parent(u)
        if p!=-1:
            a,b=pp[u],pp[p];line([a,(a[0],b[1]),b],MUTED,.83)
    for u,(x,y) in pp.items():
        if u in ROW:dot(x,y,u,r=.024)
        elif len(t.children(u))>1:dot(x,y,r=.016)
    if labels:
        for q in order:
            x,_=pp[q];txt(x,base-.115,NAMES[q],7.0,c=QUERY if q==0 else MUTED)
    return pp

# a. Genuine recorded ARG. Native parent-child edges are translucent ribbons
# over their ACTUAL genomic spans, not arbitrary bridges between local trees.
panel(.12,6.40,"a")
xbase=.25;gx=1.65;depth=.43;ybase=4.95;ageh=1.05
global_leaf={q:i/4 for i,q in enumerate([1,2,4,0,3])}
# A constant coordinate for every recorded ARG node: union of its true sampled
# descendants across intervals, a layout choice independent of edge existence.
desc={u:set() for u in range(full.num_nodes)}
for ft in fts:
    for u in ft.nodes(): desc[u].update(ft.samples(u))
global_x={u:float(np.mean([global_leaf[q] for q in ds])) if ds else .5 for u,ds in desc.items()}
def proj(u,g):
    z=global_x[u]
    return xbase+gx*g/10000+depth*z, ybase+ageh*full.node(u).time/TMAX+.14*z
# Draw only native tskit edges, each bounded by its observed left/right interval.
native_polys=[]
for e in sorted(full.edges(),key=lambda e:full.node(e.parent).time,reverse=True):
    poly=[proj(e.child,e.left),proj(e.parent,e.left),proj(e.parent,e.right),proj(e.child,e.right)]
    ax.add_patch(Polygon(poly,closed=True,fc="#8e9eac",ec="#8e9eac",lw=.33,alpha=.115,zorder=.5))
    native_polys.append({"child":int(e.child),"parent":int(e.parent),"left":float(e.left),"right":float(e.right)})
# Three exact native marginal slices at interval centers, including unary events.
recs={n.id for n in full.nodes() if n.flags&131072}
for i,ft in enumerate(fts):
    mid=(breaks[i]+breaks[i+1])/2
    for u in ft.nodes():
        p=ft.parent(u)
        if p!=-1:line([proj(u,mid),proj(p,mid)],"#617282",.66,z=2)
    for u in ft.nodes():
        x,y=proj(u,mid)
        if u in ROW:dot(x,y,u,r=.019)
        elif u in recs:dot(x,y,r=.021,diamond=True)
        elif len(ft.children(u))>1:dot(x,y,r=.013)
# Time and genomic axes, with exact unequal spans.
arrow((.13,5.01),(.13,6.16),MUTED,.65,5)
txt(.045,5.60,"time",7.1,c=MUTED,rotation=90)
stripx=.27;stripw=2.05
for i,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    a=stripx+stripw*l/10000;b=stripx+stripw*r/10000
    ax.add_patch(Rectangle((a,4.72),b-a,.065,fc=SEG[i],ec="#8f9daa",hatch=HATCH[i],lw=.45))
    txt((a+b)/2,4.88,f"$I_{i+1}$",7.2)
for g in breaks:
    x=stripx+stripw*g/10000
    line([(x,4.70),(x,4.66)],MUTED,.65)
    txt(x,4.54,f"{g/1000:.2f}".rstrip("0").rstrip("."),7.0)
arrow((stripx,4.38),(stripx+stripw,4.38),MUTED,.65,5)
txt(1.30,4.24,"kb",7.1,c=MUTED)
# Recombination key is a symbol only; explanation in the caption.
dot(.38,6.22,r=.023,diamond=True)
txt(.54,6.22,"R",7.2,ha="left")

# b. Exact unary-collapsed marginals with explicit observed 0/1 allele cells.
panel(2.67,6.40,"b")
arrow((2.39,5.58),(2.64,5.58),MUTED,1.0,6)
centers=[3.29,4.65,6.01]; coords=[]
for i,cx in enumerate(centers):
    txt(cx,6.32,f"$T_{i+1}$",8.0)
    ax.add_patch(Rectangle((cx-.50,6.13),1.0,.04,fc=SEG[i],ec="#8f9daa",hatch=HATCH[i],lw=.4))
    pp=display_tree(i,cx,5.16,1.0,.88)
    coords.append(pp)
    for q in layouts[i][2]:
        x,_=pp[q]
        ax.add_patch(Rectangle((x-.055,4.82),.11,.145,fc=GENO,ec="#a6b4c1",lw=.45,zorder=4))
        txt(x,4.893,str(int(full.genotype_matrix()[i,q])),7.0,zorder=5)
    txt(cx,4.70,f"$g_{{{i+1},q}}$",7.2,c=MUTED)
# One ancestry vector per persistent sample, shared by all three trees.
axorder=[1,2,4,0,3]; axcenters=[3.06,3.85,4.64,5.43,6.22]
for q,cx in zip(axorder,axcenters):
    txt(cx,4.51,NAMES[q],7.2,c=QUERY if q==0 else MUTED)
    if q==0:
        ax.add_patch(Rectangle((cx-.20,4.26),.40,.16,fc="white",ec=QUERY,lw=.85,zorder=4))
        txt(cx,4.34,"?",8.0,c=QUERY,zorder=5)
    else:
        for k in range(4):
            x=cx-.20+k*.10
            ax.add_patch(Rectangle((x,4.26),.10,.16,fc=ANC[k] if k==q-1 else "white",ec="#a3b0bc",lw=.4,zorder=4))
            txt(x+.05,4.34,"1" if k==q-1 else "0",7.0,zorder=5)
# True internal-node observed vector, attached to a genuine nonroot internal node.
ancnode=next(u for u in trees[1].nodes() if trees[1].parent(u)!=-1 and len(trees[1].children(u))>1)
p=coords[1][ancnode]
line([p,(p[0]+.11,p[1]+.17),(p[0]+.16,p[1]+.17)],MUTED,.6)
for j,s in enumerate([r"$\tau$","0","0"]):
    x=p[0]+.16+j*.11
    ax.add_patch(Rectangle((x,p[1]+.095),.11,.15,fc=FAINT,ec=LIGHT,lw=.45))
    txt(x+.055,p[1]+.17,s,7.0)
# All observations, not just a chosen tip, feed the encoder.
line([(2.77,4.20),(2.77,4.15),(6.80,4.15),(6.80,4.20)],MUTED,.7)
route([(2.77,4.15),(.52,4.15),(.52,3.82)],MUTED,.75,5)

# c. Two-tier model, with literal complete Up/Down modules and no graph pooling.
panel(.12,3.91,"c")
box(.25,3.47,.55,.35,FAINT)
txt(.525,3.645,"MLP",8.0)
arrow((.80,3.645),(.91,3.645),ms=5)
# Node-state vector is just a symbol, not a fake whole-graph pooled embedding.
for j in range(3):
    ax.add_patch(Rectangle((.94+j*.067,3.43),.067,.43,fc=["#a5b5c3","#c6d1da","#8fa4b5"][j],ec="white",lw=.4))
txt(1.04,3.28,r"$b_v$",7.5)
arrow((1.16,3.645),(1.35,3.645),ms=5)
def message_tree(cx,base,width=.34,height=.39,direction="up"):
    t=trees[0];xx,_,order=layouts[0];rank={q:0 for q in order}
    def rr(u):
        if u not in rank:rank[u]=1+max(rr(c) for c in t.children(u))
        return rank[u]
    rr(t.root)
    p={u:(cx-width/2+width*xx[u],base+height*rank[u]/rank[t.root]) for u in t.nodes()}
    for u in t.nodes():
        par=t.parent(u)
        if par!=-1:
            a,b=p[u],p[par];bend=(a[0],b[1]);line([a,bend,b],MUTED,.55)
            # Every native branch has a message direction; arrowheads placed
            # inside the vertical segment leave branch junctions readable.
            lo=np.array(a);hi=np.array(bend)
            start=lo+.18*(hi-lo);end=lo+.78*(hi-lo)
            if direction=="down":start,end=end,start
            arrow(start,end,INK,.55,2.6)
    for u,(x,y) in p.items():dot(x,y,u if u in ROW else None,r=.0115)
def gnn(x,y):
    box(x,y,1.16,.84,"#fbfcfe",ec="#bacbd8",lw=.85)
    txt(x+.58,y+.72,r"GNN$_\theta$",8.3)
    message_tree(x+.28,y+.24,direction="up")
    message_tree(x+.86,y+.24,direction="down")
    txt(x+.28,y+.10,"Up",7.3)
    txt(x+.86,y+.10,"Down",7.3)
    arrow((x+.42,y+.10),(x+.66,y+.10),INK,.75,4.8)
gnn(1.37,3.19)
line([(2.53,3.645),(2.79,3.645)],INK,.85)
dot(2.79,3.645,r=.020)
# The original base, not the first pass output, is the residual source.
route([(1.04,3.86),(1.04,4.07),(5.99,4.07),(5.99,3.745)],MUTED,.8,5)
dot(1.04,3.86,r=.016)
# Local structural adapters use current-node state, no foreign-tree state arrow.
box(3.18,3.44,1.0,.40,FAINT)
txt(3.68,3.64,r"$\Delta_{\rm str}$",8.0)
arrow((2.79,3.645),(3.18,3.645),ms=5)
arrow((4.18,3.645),(5.88,3.645),ms=5)
# Independent same-sample scans before pass two.
box(3.14,2.18,1.95,.94,FAINT,ec="#bacbd8",lw=.85)
txt(4.115,3.015,"Sample memory",8.0)
memcols=[3.68,4.23,4.78]
for j,x in enumerate(memcols):txt(x,2.82,f"$T_{j+1}$",7.0)
memys=[2.66-.105*k for k in range(5)]
for k,q in enumerate(ROW):
    yy=memys[k];txt(3.42,yy,NAMES[q],7.0,ha="right",c=QUERY if q==0 else MUTED)
    for x in memcols:small_vector(x,yy,w=.18,h=.078,query=q==0)
    for a,b in zip(memcols[:-1],memcols[1:]):
        arrow((a+.10,yy+.014),(b-.10,yy+.014),QUERY if q==0 else "#97a9b9",.65 if q==0 else .45,3.2 if q==0 else 2.6)
        arrow((b-.10,yy-.014),(a+.10,yy-.014),QUERY if q==0 else "#97a9b9",.65 if q==0 else .45,3.2 if q==0 else 2.6)
route([(2.79,3.645),(2.79,2.65),(3.14,2.65)],MUTED,.8,5)
# A real learned output projection maps the centered memory to node width.
box(5.30,2.48,.58,.31,FAINT)
txt(5.59,2.635,"Linear",7.5)
arrow((5.09,2.635),(5.30,2.635),MUTED,.8,5)
route([(5.88,2.635),(5.99,2.635),(5.99,3.545)],MUTED,.8,5)
ax.add_patch(Circle((5.99,3.645),.095,fc="white",ec=INK,lw=.9,zorder=5))
txt(5.99,3.645,"+",10,zorder=6)
# Orthogonal return lane, kept clear of all labels and scan rows.
route([(6.085,3.645),(6.77,3.645),(6.77,1.98),(.84,1.98),(.84,1.62)],INK,.85,5.5)
# Second complete shared traversal.
gnn(.26,.78)
arrow((1.42,1.245),(1.60,1.245),ms=5)

# Readout retains every sample; rows follow the same identity permutation.
ecols=[1.98,2.35,2.72];eys=[1.48-.135*k for k in range(5)]
for k,q in enumerate(ROW):
    txt(1.74,eys[k],NAMES[q],7.0,ha="right",c=QUERY if q==0 else MUTED)
for j,x in enumerate(ecols):
    txt(x,1.70,f"$T_{j+1}$",7.2)
    for k,q in enumerate(ROW):small_vector(x,eys[k],w=.21,h=.112,query=q==0)
for a,b in zip(ecols[:-1],ecols[1:]):
    for k,q in enumerate(ROW):line([(a+.11,eys[k]),(b-.11,eys[k])],QUERY if q==0 else "#c0ccd5",.7 if q==0 else .4)
txt(2.35,.73,r"$e_{tq}$",7.6)
arrow((2.86,1.245),(3.17,1.245),ms=5)
dot(3.02,1.245,r=.020)
# Standard final genomic recurrent module, visibly distinct from sample memory.
box(3.18,.78,1.36,.92,FAINT,ec="#bacbd8",lw=.85)
txt(3.86,1.57,"BiGRU",8.3)
txt(3.86,1.41,"1 layer",7.0,c=MUTED)
hcols=[3.43,3.86,4.29]
hys=[1.25-.095*k for k in range(5)]
for k,q in enumerate(ROW):
    for x in hcols:
        box(x-.08,hys[k]-.035,.16,.07,"#d7e1e9",ec=QUERY if q==0 else "#b8c7d2",lw=.7 if q==0 else .45,r=.012)
    for a,b in zip(hcols[:-1],hcols[1:]):
        arrow((a+.08,hys[k]+.012),(b-.08,hys[k]+.012),QUERY if q==0 else "#97a9b9",.6 if q==0 else .4,3 if q==0 else 2.4)
        arrow((b-.08,hys[k]-.012),(a+.08,hys[k]-.012),QUERY if q==0 else "#97a9b9",.6 if q==0 else .4,3 if q==0 else 2.4)
# Exact-span geometry is visibly a second input, defined in the caption/Methods.
for i,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    ax.add_patch(Rectangle((3.47+.78*l/10000,.52),.78*(r-l)/10000,.065,
      fc=SEG[i],ec="#95a5b3",hatch=HATCH[i],lw=.4))
txt(3.86,.38,r"$I_t$",7.3,c=MUTED)
arrow((3.86,.60),(3.86,.77),MUTED,.7,4.6)
# Explicit same-sample local skip to classifier concatenation.
route([(3.02,1.245),(3.02,.27),(4.82,.27),(4.82,1.15)],MUTED,.8,5)
arrow((4.54,1.245),(4.725,1.245),ms=5)
ax.add_patch(Circle((4.82,1.245),.09,fc="white",ec=INK,lw=.85,zorder=5))
txt(4.82,1.245,r"$\Vert$",8.5,zorder=6)
arrow((4.91,1.245),(5.10,1.245),ms=5)
box(5.10,1.07,.54,.35,FAINT)
txt(5.37,1.245,"MLP",8.0)
arrow((5.64,1.245),(5.87,1.245),ms=5)
# Illustrative probabilities, not fitted predictions.
probs=np.asarray([[.08,.07,.78,.07],[.16,.08,.67,.09],[.07,.13,.74,.06]])
px0,px1=5.90,6.79
for i,(l,r) in enumerate(zip(breaks[:-1],breaks[1:])):
    a=px0+(px1-px0)*l/10000;b=px0+(px1-px0)*r/10000;low=.82
    for k,p in enumerate(probs[i]):
        ax.add_patch(Rectangle((a,low),b-a,p*.86,fc=ANC[k],ec="white",lw=.35));low+=p*.86
    ax.add_patch(Rectangle((a,.35),b-a,.12,fc=ANC[int(np.argmax(probs[i]))],ec="white",lw=.35))
txt(6.34,1.84,r"$p_{tq}$",8.2)
box(6.08,.57,.53,.19,"white",ec=MUTED,lw=.5,r=.02)
txt(6.345,.665,"argmax",7.0)
arrow((6.345,.81),(6.345,.77),MUTED,.7,4.4)
arrow((6.345,.56),(6.345,.48),MUTED,.7,4.4)
txt(6.345,.19,r"$q_1$",7.6,c=QUERY)
# Shared class key, with ancestry hues reserved for known labels and outputs.
for j,name in enumerate(ANAMES):
    x=.33+j*.67
    ax.add_patch(Rectangle((x,.075),.11,.075,fc=ANC[j],ec="none"))
    txt(x+.15,.1125,name,7.0,ha="left",c=MUTED)

fig.savefig(OUT/"TreeLoom_architecture.svg")
fig.savefig(OUT/"TreeLoom_architecture.pdf")
fig.savefig(OUT/"TreeLoom_architecture.png",dpi=220)
fig.savefig(OUT/"TreeLoom_architecture_2x.png",dpi=440)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
bounds=[]
for t in texts:
    bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    bounds.append({"text":t.get_text(),"fontsize":t.get_fontsize(),"bbox_figure":list(bb.bounds)})
outside=[b for b in bounds if min(b["bbox_figure"][:2])<-.001 or b["bbox_figure"][0]+b["bbox_figure"][2]>1.001 or b["bbox_figure"][1]+b["bbox_figure"][3]>1.001]
assert not outside,outside
assert min(t.get_fontsize() for t in texts)>=7
(OUT/"text_layout_bounds.json").write_text(json.dumps(bounds,indent=2)+"\n")
metadata={"revision":"v9_substantial_two_tier_redesign","figure_inches":[W,H],
 "minimum_font_pt":min(t.get_fontsize() for t in texts),"panel_titles":False,
 "panels":["a","b","c"],"trees":3,"samples":5,"genomic_breakpoints_bp":breaks,
 "ARG_native_edge_surfaces":native_polys,"ARG_edge_surface_count":len(native_polys),
 "ARG_layout":"fixed per-node union-descendant coordinate; actual native edge spans; three exact marginal slices",
 "marginal_trees":"exact source unary-collapsed trees; native topology/times/leaf identity preserved",
 "query_identity":"magenta open square and magenta sample row; all others neutral",
 "feature_values":"actual hypothetical 0/1 genotypes on toy ARG, one site per interval; supplied schematic known source labels; query ancestry unknown",
 "memory_display_order":[NAMES[q] for q in ROW],"sample_readout_display_order":[NAMES[q] for q in ROW],
 "weight_sharing":"both complete Up/Down blocks use theta; not two unrelated graph layers",
 "residual":"original base + current-node structural residual + sample-memory residual before second traversal",
 "memory_abstraction":"five independent same-sample forward/reverse rows; cut pooling, centering, decay defined in Methods; displayed Linear is real memory output projection",
 "readout":"all sample embeddings -> one-layer BiGRU plus local skip -> concat -> MLP -> four probabilities; q1 selected only for displayed track",
 "geometry_input":"native interval strip abbreviates log-scaled span and neighboring center distances",
 "no_graph_pooling":True,"output_probs":"illustrative, not a model run","operation_marker":"root executed once; not duplicated",
 "validation":{"exact_trees":True,"native_ARG_edges_only":True,"strict_parent_age":True,"minimum_font_7pt":True,"no_text_outside_canvas":True}}
(OUT/"figure_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps(metadata,indent=2))
