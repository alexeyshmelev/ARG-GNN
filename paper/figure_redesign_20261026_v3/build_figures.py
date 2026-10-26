"""Scientific vector diagrams from a preserved toy tree sequence. No model jobs."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json, hashlib
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'research/reviews/figure_v15_flow_revision_20261017_v1/v15r2/illustrative_recorded_ARG.trees'
ts=tskit.load(SOURCE)
trees=[t.copy() for t in ts.trees()]
assert len(trees)==3 and ts.num_samples==5
INK='#26323a';WHITE='#ffffff';TRAIN='#e7ddf4';TRAIN_EDGE='#705193'
FIX='#f6f7f8';VALUE=LinearSegmentedColormap.from_list('values',['#f5f3f8','#aa8ec1','#4f2869'])
ANC=['#0072b2','#e69f00','#009e73','#cc79a7']
NAMES=['q','r1','r2','r3','r4']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
fig=ax=None
texts=[]

def canvas(height,physical_height):
    global fig,ax,texts,H
    H=height;texts=[]
    fig=plt.figure(figsize=(7,physical_height));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,14),ylim=(H,0));ax.axis('off')

def txt(x,y,s,size=8,ha='center',bold=False,color=INK):
    t=ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,fontweight='bold' if bold else 'normal',zorder=20)
    texts.append(t)
    return t

def line(points,lw=.8,style='-',color=INK):
    ax.plot(*zip(*points),color=color,lw=lw,ls=style,zorder=2)

def arr(points,lw=.8,both=False,style='-'):
    if len(points)>2:line(points[:-1],lw,style)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='<->' if both else '-|>',mutation_scale=6,color=INK,lw=lw,shrinkA=0,shrinkB=0,zorder=3))

def box(x,y,w,h,label='',learn=False,size=7.6,rounding=.06,container=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=TRAIN if learn else FIX,ec=TRAIN_EDGE if learn else INK,lw=.85,zorder=0 if container else 4))
    if label:txt(x+w/2,y+h/2,label,size)

def op(x,y,s,r=.14,size=9):
    ax.add_patch(Circle((x,y),r,fc=WHITE,ec=INK,lw=.8,zorder=5));txt(x,y,s,size)

def vec(x,y,w=.50,h=.18,n=5,key=0,zero=False,values=None):
    vals=np.full(n,.5) if zero else .12+.82*(.5+.5*np.sin(np.arange(n)*1.91+key*.81))
    if values is not None:
        assert len(values)==n
        vals=.5+.5*np.asarray(values)
    for j,v in enumerate(vals):
        ax.add_patch(Rectangle((x+j*w/n,y),w/n,h,fc=VALUE(float(v)),ec=WHITE,lw=.35,zorder=6))
    ax.add_patch(Rectangle((x,y),w,h,fc='none',ec=INK,lw=.55,zorder=7))

def sample(x,y,j,r=.045):
    ax.add_patch(Circle((x,y),r,fc=WHITE if j==0 else ANC[j-1],ec=INK,lw=.55,zorder=7))

def panel(letter,y):txt(.13,y,letter,11,ha='left',bold=True)

def bracket(x1,x2,y,label=None,above=False):
    dy=-.10 if not above else .10
    line([(x1,y+dy),(x1,y),(x2,y),(x2,y+dy)],.65)
    if label:txt((x1+x2)/2,y+(.19 if not above else -.19),label,7.1)

def tree(i,cx,base,w=2.0,h=1.65,direction=None,states=False,state_key=0,cut=False,groups=False,labels=False,selected=None):
    t=trees[i];xp={u:cx-w/2+w*u/4 for u in range(5)}
    def xx(u):
        if u not in xp:xp[u]=float(np.mean([xx(v) for v in t.children(u)]))
        return xp[u]
    xx(t.root);yp={u:base-h*ts.node(u).time/725 for u in t.nodes()}
    for u in t.nodes():
        p=t.parent(u)
        if p<0:continue
        line([(xp[u],yp[u]),(xp[u],yp[p]),(xp[p],yp[p])],.70)
        if direction:
            a=np.array((xp[u],yp[u]));b=np.array((xp[u],yp[p]));start=a+.23*(b-a);end=a+.76*(b-a)
            if direction=='down':start,end=end,start
            arr([start,end],.75)
    for u in t.nodes():
        if states:
            vec(xp[u]-.13,yp[u]-.05,.26,.10,4,key=state_key+u,zero=direction=='down' and u==t.root)
            if u<5:sample(xp[u],yp[u]+.09,u,.031)
            if direction=='down' and u==t.root:txt(xp[u],yp[u]-.20,'0',6.3)
        elif u<5:sample(xp[u],yp[u],u)
        else:ax.add_patch(Circle((xp[u],yp[u]),.027,fc=INK,ec='none',zorder=6))
        if labels and u<5:txt(xp[u],base+.20,NAMES[u],7.5)
    if selected is not None:
        ax.add_patch(Rectangle((xp[selected]-.09,yp[selected]-.09),.18,.18,fc='none',ec=INK,lw=1.1,zorder=9))
    if cut:
        yy=base-h*350/725
        line([(cx-w*.57,yy),(cx+w*.57,yy)],.65,'--')
    if groups:
        rep={}
        for q in range(5):
            u=q
            while t.parent(u)>=0 and ts.node(t.parent(u)).time<=350:u=t.parent(u)
            rep.setdefault(u,[]).append(q)
        for j,(u,qs) in enumerate(rep.items()):
            bracket(xp[min(qs)]-.08,xp[max(qs)]+.08,base+.46,'$G_'+str(j+1)+'$')
            ax.add_patch(Rectangle((xp[u]-.07,yp[u]-.07),.14,.14,fc='none',ec=INK,lw=.65,zorder=8))
    return {u:(xp[u],yp[u]) for u in t.nodes()}

def key(y,ancestry=True):
    line([(.22,y),(13.78,y)],.55)
    box(.30,y+.16,.44,.24,learn=True);txt(.90,y+.28,'Learned layer / parameter',7.0,ha='left')
    box(5.30,y+.16,.44,.24);txt(5.88,y+.28,'Fixed operation',7.0,ha='left')
    for j,v in enumerate(np.linspace(0,1,12)):
        ax.add_patch(Rectangle((9.05+j*.12,y+.16),.12,.24,fc=VALUE(float(v)),ec='none'))
    txt(8.88,y+.28,'−',7);txt(10.65,y+.28,'+',7)
    txt(11.0,y+.28,'Feature values',7.0,ha='left')
    if ancestry:
        for c,(x,name) in zip(ANC,[(.30,'Anatolian farmer'),(7.0,'Caucasus hunter-gatherer'),(.30,'Western hunter-gatherer'),(7.0,'Eastern hunter-gatherer')]):
            yy=y+.65 if c in ANC[:2] else y+.96
            ax.add_patch(Rectangle((x,yy-.08),.22,.16,fc=c,ec='none'));txt(x+.34,yy,name,7.1,ha='left')
        sample(.40,y+1.28,0,.05);txt(.62,y+1.28,'Query',7.0,ha='left')
        ax.add_patch(Rectangle((3.12,y+1.20),.16,.16,fc='none',ec=INK,lw=.8));txt(3.42,y+1.28,'Group ancestor',7.0,ha='left')
        line([(7.00,y+1.28),(7.25,y+1.28)],.8);txt(7.40,y+1.28,'Tree edge / data flow',7.0,ha='left')

def save(name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in texts:
        b=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert b.x0>=-.003 and b.y0>=-.003 and b.x1<=1.003 and b.y1<=1.003,t.get_text()
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=210)
    p=OUT/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)

# FIGURE 1: biological input, all node states, then one query's recurrent readout.
canvas(13.9,7.1)
panel('a',.25)
for i,cx in enumerate([1.55,4.1,6.65]):
    txt(cx,.26,'$T_'+str(i+1)+'$',9)
    tree(i,cx,2.15,w=2.0,h=1.52,labels=True)
    bracket(cx-1,cx+1,2.57)
arr([(.55,2.80),(7.68,2.80)]);txt(4.1,3.02,'Genomic position',7.5)
arr([(.24,2.04),(.24,.60)]);txt(.24,.43,'Age',7)
txt(10.9,.25,'Observed alleles',8.5)
for j,n in enumerate(NAMES):txt(9.0+j*.83,.70,n,7.5)
for i,row in enumerate(ts.genotype_matrix()):
    txt(8.40,1.06+i*.40,'$T_'+str(i+1)+'$',7.5)
    for j,v in enumerate(row):
        x=8.66+j*.83;y=.91+i*.40
        ax.add_patch(Rectangle((x,y),.65,.30,fc=WHITE,ec=INK,lw=.5));txt(x+.325,y+.15,str(v),8)
txt(10.9,2.35,'Reference ancestry',7.6)
for j in range(5):sample(9.0+j*.83,2.73,j,.07)
txt(9.0,2.98,'?',8)
# Whole-tree state propagation, no redundant direction labels.
panel('b',3.46)
arr([(.64,3.18),(.64,3.60)])
box(.16,3.60,1.12,.54,'Site MLP',True,7.0)
arr([(.72,4.14),(.72,4.31)])
box(.16,4.31,1.12,.42,'Site pool',False,7.0)
arr([(.72,4.73),(.72,4.98)])
box(.16,4.98,1.12,.60,'Feature\nMLPs',True,7.0)
arr([(1.28,5.28),(1.42,5.28),(1.42,4.71),(1.58,4.71)])
for cx,di,k in [(2.40,'up',1),(4.40,'down',2),(8.70,'up',3),(10.70,'down',4)]:
    box(cx-.85,3.95,1.70,2.12,learn=True,container=True)
    tree(1,cx,5.77,w=1.44,h=1.52,direction=di,states=True,state_key=k)
    txt(cx,6.24,'$u_v$' if di=='up' else '$o_v$',7.5)
arr([(3.25,4.71),(3.55,4.71)])
arr([(9.55,4.71),(9.85,4.71)])
bracket(1.55,5.25,6.51,'Round 1');bracket(7.85,11.55,6.51,'Round 2 · shared weights')
txt(3.40,3.68,'32 features at every node',7.0)
# Grouping and corrections are expanded in their own linked figures.
arr([(5.25,4.71),(5.57,4.71)]);vec(5.59,4.54,.36,.34,4,13);txt(5.77,4.31,'$u,o$',7.3)
line([(5.77,4.88),(5.77,5.30)])
arr([(5.77,5.30),(5.94,5.30)]);box(5.94,4.98,1.12,.64,'Memory\nFig. 2',True,6.6)
arr([(5.77,5.30),(5.77,6.02),(5.94,6.02)]);box(5.94,5.70,1.12,.64,'Structure\nFig. 3c',True,6.6)
op(7.48,4.71,'+',.15,11)
arr([(7.06,5.30),(7.19,5.30),(7.19,4.71),(7.33,4.71)])
arr([(7.06,6.02),(7.48,6.02),(7.48,4.86)])
line([(1.42,4.71),(1.42,3.43),(7.48,3.43)])
arr([(7.48,3.43),(7.48,4.56)]);txt(6.80,3.25,'$b$',7.5)
arr([(7.63,4.71),(7.85,4.71)]);txt(7.72,4.40,"$b'$",7.3)
# Separate upward state is retained through the downward traversal.
line([(9.69,4.71),(9.69,3.55),(11.88,3.55)])
arr([(11.88,3.55),(11.88,4.56)])
arr([(11.55,4.71),(11.73,4.71)]);op(11.88,4.71,'∥',.15,10)
arr([(12.03,4.71),(12.20,4.71)]);box(12.20,4.19,1.60,.64,'Linear\n64 → 32',True,7.2)
arr([(13.0,4.83),(13.0,5.06)]);box(12.48,5.06,1.04,.42,'tanh',False,7.2)
arr([(13.0,5.48),(13.0,5.69)])
for j in range(3):vec(12.18+j*.55,5.72,.43,.24,5,21+j)
txt(13.0,6.20,'Per haplotype\nper tree',6.7)
# Physical geometry is visibly derived from interval width and centers.
panel('c',7.10)
lengths=np.array([t.interval.span for t in trees])
x=(.65+3.20*np.r_[0,np.cumsum(lengths)]/lengths.sum()).tolist();cy=7.87
for i in range(3):
    line([(x[i],cy),(x[i+1],cy)],2.0);line([(x[i],cy-.08),(x[i],cy+.08)],.7)
    txt((x[i]+x[i+1])/2,cy-.26,'$T_'+str(i+1)+'$',7.5)
line([(x[-1],cy-.08),(x[-1],cy+.08)],.7)
c=[(x[i]+x[i+1])/2 for i in range(3)]
for xx in c:ax.add_patch(Circle((xx,cy),.035,fc=INK,ec='none'))
arr([(c[0],8.22),(c[1],8.22)],both=True,lw=.6);txt((c[0]+c[1])/2,8.40,'$d_-$',7.3)
arr([(c[1],8.22),(c[2],8.22)],both=True,lw=.6);txt((c[1]+c[2])/2,8.40,'$d_+$',7.3)
bracket(x[1],x[2],7.38,'$L_2$',above=True)
arr([(3.95,7.87),(4.25,7.87)]);box(4.25,7.57,1.62,.60,'log scale\nper interval',False,7.0)
arr([(5.87,7.87),(6.15,7.87)])
for j in range(3):
    distances=[lengths[j],(lengths[j-1]+lengths[j])/2 if j>0 else 0,(lengths[j]+lengths[j+1])/2 if j<2 else 0]
    geometry=np.log1p(distances)/np.log(2000001)
    vec(6.18,7.48+j*.25,.66,.18,3,values=geometry)
    for k,distance in enumerate(distances):
        if distance==0:txt(6.18+(k+.5)*.22,7.57+j*.25,'0',5.4)
    txt(7.10,7.57+j*.25,'$T_'+str(j+1)+'$',7.0)
txt(6.51,7.15,'3 features per interval',7.0)
# A geometry vector is attached to every haplotype embedding, not pooled trees.
for j in range(3):vec(.68+j*.73,9.35,.55,.30,5,40+j)
txt(1.67,9.04,'$e_{1q}\quad e_{2q}\quad e_{3q}$',7.5)
txt(1.67,9.92,'One query q · 32 each',7.1)
arr([(2.78,9.50),(3.10,9.50)]);box(3.10,9.15,1.48,.70,'Concat per tree\n32 + 3',False,6.8)
arr([(6.51,8.16),(6.51,8.61),(3.84,8.61),(3.84,9.15)])
arr([(4.58,9.50),(4.89,9.50)]);box(4.89,9.10,1.83,.80,'BiGRU\n1 layer · 35 → 64',True,6.8)
arr([(6.72,9.50),(7.08,9.50)]);box(7.08,9.15,1.45,.70,'Concat\n32 + 64',False,7.1)
line([(2.96,9.50),(2.96,10.35),(7.81,10.35)]);arr([(7.81,10.35),(7.81,9.85)])
arr([(8.53,9.50),(8.81,9.50)]);box(8.81,9.10,1.40,.80,'Linear\n96 → 32',True,7)
arr([(10.21,9.50),(10.50,9.50)]);box(10.50,9.26,1.02,.48,'GELU',False,7)
arr([(11.52,9.50),(11.82,9.50)]);box(11.82,9.10,1.75,.80,'Linear\n32 → 4',True,7)
arr([(12.70,9.90),(12.70,10.48)]);box(11.95,10.48,1.50,.48,'Softmax',False,7.1)
arr([(11.95,10.72),(11.50,10.72)])
probs=[[.62,.12,.14,.12],[.17,.16,.53,.14],[.11,.16,.21,.52]]
for i,row in enumerate(probs):
    lo=11.18
    for j,v in enumerate(row):
        ax.add_patch(Rectangle((8.90+i*.82,lo-.80*v),.82,.80*v,fc=ANC[j],ec=WHITE,lw=.4));lo-=.80*v
    ax.add_patch(Rectangle((8.90+i*.82,11.40),.82,.16,fc=ANC[int(np.argmax(row))],ec=WHITE,lw=.4))
txt(7.20,10.80,'Ancestry probabilities',7.2)
txt(7.20,11.48,'Ancestry calls',7.2)
arr([(11.28,11.18),(11.28,11.40)],lw=.6)
key(12.14)
save('TreeLoom_architecture')

# FIGURE 2: memory, with a connected magnification of an actual track update.
canvas(15.8,7.7)
panel('a',.25)
coords=tree(1,1.80,2.45,w=2.50,h=1.95,cut=True,groups=True,labels=True,selected=8)
txt(1.80,.22,'$T_2$',9)
cut_y=2.45-1.95*350/725
txt(.29,cut_y-.17,'350',7.0)
txt(2.30,3.20,'Groups at 350 generations',7.0)
# Only the chosen ancestor's states feed this example.
node=coords[8]
line([node,(3.20,node[1]),(3.20,.83),(3.75,.83)])
vec(3.75,.73,.55,.20,5,2);txt(4.03,.45,'$u$ · 32',7.2)
vec(3.75,1.66,.55,.20,5,3);txt(4.03,2.13,'$o$ · 32',7.2)
line([(3.20,node[1]),(3.20,1.76),(3.75,1.76)])
arr([(4.30,.83),(4.65,.83),(4.65,1.17)]);arr([(4.30,1.76),(4.65,1.76),(4.65,1.47)])
op(4.65,1.32,'+',.15,10)
arr([(4.80,1.32),(5.12,1.32)]);box(5.12,.93,1.75,.78,'Linear\n32 → 8',True,8)
arr([(6.87,1.32),(7.22,1.32)]);vec(7.25,1.18,.65,.28,8,50)
txt(7.575,.85,'$f_2$',8);txt(7.575,1.90,'Group context',7.1)
# Explicit copying to the two current members.
line([(7.90,1.32),(8.20,1.32),(8.20,.73)])
arr([(8.20,.73),(8.57,.73)]);vec(8.60,.62,.55,.22,8,50);txt(9.40,.73,'r1',7.5)
arr([(8.20,1.32),(8.20,2.09),(8.57,2.09)]);vec(8.60,1.98,.55,.22,8,50);txt(9.40,2.09,'r2',7.5)
txt(8.20,2.55,'Copy',7.4)
# A local residual returns only to the representative, never every leaf.
box(10.47,.69,3.04,1.67,'',False,container=True)
txt(12.0,.98,'Before round 2',7.3)
vec(10.70,1.49,.45,.20,5,1);txt(10.92,1.23,'$b$',7.2)
vec(11.65,1.49,.45,.20,5,5);txt(11.88,1.23,'$s$',7.2)
vec(12.60,1.49,.45,.20,5,7);txt(12.83,1.23,r'$\delta$',7.2)
txt(11.40,1.59,'+',9);txt(12.35,1.59,'+',9)
txt(12.0,2.09,"$b'$ at the group ancestor",7.0)
# Sample tracks, three intervals. Compact cells share one actual expanded update M.
panel('b',3.67)
for x,lab in [(2.50,'$T_1$'),(6.42,'$T_2$'),(10.00,'$T_3$')]:txt(x,3.43,lab,8)
rows=[(4.42,'r1',True),(5.52,'r1',False),(6.92,'r2',True),(8.02,'r2',False)]
for idx,(y,label,forward) in enumerate(rows):
    txt(1.16,y,label,7.7)
    vec(2.10,y-.12,.68,.24,8,60+idx)
    vec(9.65,y-.12,.68,.24,8,71 if not forward else 70+idx)
    box(5.70,y-.24,1.44,.48,'EMA  M',False,7.2)
    vec(6.13,y-.73,.58,.20,8,50);txt(5.83,y-.63,'$f_2$',7.4)
    arr([(6.42,y-.53),(6.42,y-.24)],lw=.65)
    if forward:
        arr([(2.78,y),(5.70,y)]);arr([(7.14,y),(7.53,y)])
        vec(7.56,y-.12,.64,.24,8,80+idx);arr([(8.20,y),(9.65,y)])
        tap=(7.88,y+.12)
        txt(7.88,y-.30,r'$m_{2}^\rightarrow$',6.8)
    else:
        arr([(9.65,y),(7.14,y)]);arr([(5.70,y),(5.30,y)])
        vec(4.64,y-.12,.64,.24,8,81);arr([(4.64,y),(2.78,y)])
        tap=(4.96,y-.12)
        txt(4.96,y+.29,r'$m_{2}^\leftarrow$',6.8)
    if forward:
        arr([tap,(tap[0],y+.33),(11.10,y+.33),(11.10,y+.37)],lw=.6)
    else:
        arr([tap,(tap[0],y-.33),(11.10,y-.33),(11.10,y-.37)],lw=.6)
# Mean axes are explicit named operations, not unexplained half-sum symbols.
for y in [4.97,7.47]:
    box(10.10,y-.18,2.00,.36,'Mean directions',False,6.8)
arr([(12.10,4.97),(12.54,4.97),(12.54,6.29)])
arr([(12.10,7.47),(12.54,7.47),(12.54,6.99)])
box(11.86,6.29,1.36,.70,'Mean\nhaplotypes',False,6.7)
# Group centering and explicit learned output projection.
arr([(13.22,6.64),(13.63,6.64),(13.63,8.85),(10.45,8.85)])
op(10.30,8.85,'−',.15,11)
txt(10.67,8.67,'+',7.4);txt(10.03,9.16,'−',7.4)
vec(9.98,9.31,.64,.22,8,50);txt(10.90,9.42,'$f_2$',7.2)
arr([(10.30,9.31),(10.30,9.00)],lw=.65)
arr([(10.15,8.85),(8.92,8.85)]);box(7.02,8.46,1.90,.78,'Linear\n8 → 32',True,7.7)
arr([(7.02,8.85),(6.40,8.85)]);vec(5.75,8.72,.62,.26,5,7);txt(5.43,8.85,r'$\delta$',8)
# Return correction uses a perimeter route ending at the explicit residual input.
line([(6.06,8.98),(6.06,9.70),(13.86,9.70),(13.86,1.59)])
arr([(13.86,1.59),(13.05,1.59)],lw=.65)
# Direct callout from the first M cell to its expanded arithmetic below.
line([(5.70,4.42),(5.44,4.42),(5.44,3.96),(.34,3.96),(.34,10.40)],.65,'--')
arr([(.34,10.40),(.80,10.40)],lw=.65)
box(.80,10.03,12.47,3.50,'',False,container=True)
txt(1.13,10.30,'M',9,bold=True)
txt(3.14,10.30,'Exponential average',8)
# A planar cell: vector paths surround the coefficient calculation.
vec(1.19,10.80,.70,.24,8,50);txt(1.54,10.56,'$f_t$',7.6)
arr([(1.89,10.92),(3.40,10.92)]);box(3.40,10.65,1.35,.54,'Scale ×',False,7.4)
vec(1.19,12.21,.70,.24,8,60);txt(1.54,11.93,r'$m_{t-1}$',7.6)
arr([(1.89,12.33),(3.40,12.33)]);box(3.40,12.06,1.35,.54,'Scale ×',False,7.4)
box(6.25,11.25,2.12,.75,r'$\alpha=e^{-L_t/\ell}$'+'\n'+r'$1-\alpha$',False,8)
box(9.00,11.10,1.75,.40,'Scale ℓ',True,7.3)
box(9.00,11.70,1.75,.40,'Length Lₜ',False,7.3)
arr([(9.00,11.30),(8.37,11.44)],lw=.65)
arr([(9.00,11.90),(8.37,11.83)],lw=.65)
arr([(6.25,11.44),(4.075,11.44),(4.075,11.19)],lw=.7)
txt(5.10,11.22,r'$1-\alpha_t$',7.2)
arr([(6.25,11.83),(4.075,11.83),(4.075,12.06)],lw=.7)
txt(5.10,12.02,r'$\alpha_t$',7.2)
arr([(4.75,10.92),(11.62,10.92),(11.62,12.18)],lw=.7)
arr([(4.75,12.33),(11.47,12.33)],lw=.7)
op(11.62,12.33,'+',.15,11)
arr([(11.77,12.33),(12.25,12.33)]);vec(12.28,12.21,.70,.24,8,80);txt(12.63,12.67,'$m_t$',7.6)
txt(4.05,13.17,'Same update in the reverse direction',7.1)
txt(6.9,13.74,'EMA: exponential moving average · one memory per sampled haplotype',7.0)
key(14.05,True)
save('TreeLoom_memory')

# FIGURE 3: explicit learned layers in the exact upward/downward and structural cells.
canvas(14.25,7.1)
panel('a',.24)
# Origin cartoon and visible child messages.
coords=tree(1,1.15,2.20,w=1.50,h=1.62,direction='up',states=True,state_key=1)
arr([(1.95,1.37),(2.40,1.37)])
for y,k in [(1.07,2),(1.49,3)]:vec(2.43,y,.45,.18,5,k)
line([(2.88,1.16),(3.08,1.16),(3.08,1.58),(2.88,1.58)])
arr([(3.08,1.37),(3.35,1.37)]);box(3.35,.98,1.30,.78,'Linear\n32 → 32',True,7.1)
txt(3.98,.72,'Per child',6.8)
arr([(4.65,1.37),(4.96,1.37)]);op(5.12,1.37,'⊙',.16,11)
vec(4.84,.50,.56,.20,5,21);txt(5.12,.26,'Edge gate',7.2);arr([(5.12,.70),(5.12,1.21)])
arr([(5.28,1.37),(5.60,1.37)]);box(5.60,1.045,1.14,.65,'Mean over\nchildren',False,6.5)
arr([(6.74,1.37),(7.03,1.37)]);op(7.18,1.37,'∥',.15,10)
vec(6.91,.51,.54,.20,5,8);txt(7.18,.26,'$b$',7.5);arr([(7.18,.71),(7.18,1.22)])
arr([(7.33,1.37),(7.63,1.37)]);box(7.63,.98,1.30,.78,'Linear\n64 → 32',True,7.1)
arr([(8.93,1.37),(9.22,1.37)]);box(9.22,1.11,1.05,.52,'tanh',False,7.3)
arr([(10.27,1.37),(10.52,1.37)]);op(10.67,1.37,'+',.15,10)
line([(7.45,.61),(10.67,.61)]);arr([(10.67,.61),(10.67,1.22)])
arr([(10.82,1.37),(11.10,1.37)]);box(11.10,.98,1.30,.78,'LayerNorm\n32',True,6.9)
arr([(12.40,1.37),(12.72,1.37)]);vec(12.74,1.23,.60,.28,5,12);txt(13.04,1.96,'$u_v$',8)
# Downward context from outside and parent, excluding the recipient contribution.
panel('b',2.75)
coords=tree(1,1.15,4.57,w=1.50,h=1.62,direction='down',states=True,state_key=2)
for x,label,k in [(2.40,'$o_p$',11),(3.40,'$b_p$',12),(4.40,'$a_p$',13),(5.40,r'$m_{q\to p}$',14)]:
    vec(x,3.40,.52,.22,5,k);txt(x+.26,3.14,label,7.4)
for x,s in [(3.15,'+'),(4.15,'+'),(5.15,'−')]:txt(x,3.51,s,10)
arr([(5.92,3.51),(6.30,3.51)]);box(6.30,3.12,1.25,.78,'Linear\n32 → 32',True,7)
arr([(7.55,3.51),(7.84,3.51)]);op(8.00,3.51,'⊙',.16,11)
vec(7.73,2.63,.54,.20,5,22);txt(8.00,2.37,'Edge gate',7);arr([(8.00,2.83),(8.00,3.35)])
arr([(8.16,3.51),(8.44,3.51)]);box(8.44,3.12,1.25,.78,'Linear\n32 → 32',True,7)
arr([(9.69,3.51),(9.95,3.51)]);box(9.95,3.25,.92,.52,'tanh',False,7.2)
arr([(10.87,3.51),(11.10,3.51)]);box(11.10,3.12,1.30,.78,'LayerNorm\n32',True,6.9)
arr([(12.40,3.51),(12.72,3.51)]);vec(12.74,3.37,.60,.28,5,13);txt(13.04,4.10,'$o_q$',8)
# Gate generation shown as its actual learned and fixed layers.
vec(2.45,4.58,.55,.22,5,5);txt(2.72,4.32,'Edge features',7)
arr([(3.00,4.69),(3.33,4.69)]);box(3.33,4.35,1.40,.68,'Linear\n9 → 32',True,7)
arr([(4.73,4.69),(5.08,4.69)]);box(5.08,4.44,1.22,.50,'Sigmoid',False,7.1)
arr([(6.30,4.69),(6.64,4.69)]);vec(6.67,4.58,.55,.22,5,21)
txt(9.50,4.69,'Separate gates for the two directions',7)
# Structural changes are shown by presence values, not ambiguous X symbols.
panel('c',5.63)
for j,lab in enumerate(['$T_1$','$T_2$','$T_3$']):txt(1.12+j*.62,5.88,lab,7.6)
patterns=[([1,0,0],'lost',r'$l_1=1$'),([0,1,1],'gained',r'$g_2=1$'),([1,1,1],'persists',r'$e=1$')]
for i,(row,name,flag) in enumerate(patterns):
    y=6.37+i*.62
    pair=[(0,1),(1,2),(3,4)][i]
    line([(.23,y+.08),(.23,y-.10),(.57,y-.10),(.57,y+.08)],.6)
    sample(.23,y+.08,pair[0],.045)
    sample(.57,y+.08,pair[1],.045)
    for j,v in enumerate(row):
        box(.90+j*.62,y-.18,.44,.36,str(v),False,7.6)
    txt(2.82,y,name,6.8,ha='left');txt(4.05,y,flag,7.0)
txt(2.20,8.45,'1: descendant group present',7.1)
txt(2.20,8.76,'0: absent',7.1)
# Current-node input fans out to named learned bottleneck projections.
vec(4.98,5.90,.58,.22,5,14);txt(5.27,5.65,'$u_v$',8)
for y,name in [(6.63,'V'),(7.58,'U'),(8.53,'W')]:
    line([(5.27,6.12),(5.27,y),(5.63,y)])
    box(5.63,y-.32,1.40,.64,'Linear '+name+'\n32 → 8',True,6.8)
    arr([(7.03,y),(7.34,y)]);vec(7.37,y-.11,.48,.22,8,int(y*10))
arr([(7.85,6.63),(8.25,6.63)]);op(8.40,6.63,'⊙',.15,10)
arr([(7.85,8.53),(8.25,8.53)]);op(8.40,8.53,'⊙',.15,10)
line([(7.85,7.58),(8.40,7.58)]);arr([(8.40,7.58),(8.40,6.78)]);arr([(8.40,7.58),(8.40,8.38)])
for y,flag in [(6.63,'g'),(8.53,'l')]:
    arr([(8.55,y),(8.94,y)]);box(8.94,y-.25,1.15,.50,'Scale ×',False,7)
    txt(9.515,y-.68,flag,8);arr([(9.515,y-.52),(9.515,y-.25)],lw=.6)
arr([(10.09,6.63),(10.52,6.63),(10.52,7.43)]);arr([(10.09,8.53),(10.52,8.53),(10.52,7.73)])
op(10.52,7.58,'−',.15,10);txt(10.28,7.21,'+',7.4);txt(10.28,7.96,'−',7.4);arr([(10.67,7.58),(11.02,7.58)])
box(11.02,7.22,1.63,.72,'Linear O\n8 → 32',True,7.1)
arr([(12.65,7.58),(13.02,7.58)]);vec(13.04,7.47,.48,.22,5,91);txt(13.28,7.04,'$s_f$',7.7)
# Persistent correction: same current state, independent named learned maps.
vec(.80,10.49,.58,.22,5,14);txt(1.09,10.20,'$u_v$',8)
for y,name in [(9.63,'Uₚ'),(10.60,'Vₚ'),(11.57,'Wₚ')]:
    line([(1.38,10.60),(1.66,10.60),(1.66,y),(1.95,y)])
    box(1.95,y-.33,1.48,.66,'Linear '+name+'\n32 → 8',True,7)
    arr([(3.43,y),(3.80,y)]);vec(3.83,y-.11,.48,.22,8,int(y*10))
arr([(4.31,9.63),(4.85,9.63),(4.85,10.45)])
arr([(4.31,10.60),(4.70,10.60)]);op(4.85,10.60,'⊙',.15,10)
arr([(5.00,10.60),(5.55,10.60),(5.55,10.93)])
arr([(4.31,11.57),(5.55,11.57),(5.55,11.23)]);op(5.55,11.08,'+',.15,10)
arr([(5.70,11.08),(6.08,11.08)]);box(6.08,10.71,1.70,.74,'Linear Oₚ\n8 → 32',True,7.1)
arr([(7.78,11.08),(8.22,11.08)]);box(8.22,10.83,1.20,.50,'Scale ×',False,7)
txt(8.82,10.32,'e',8);arr([(8.82,10.49),(8.82,10.83)],lw=.6)
arr([(9.42,11.08),(9.80,11.08)]);vec(9.83,10.97,.56,.22,5,92);txt(10.11,11.55,'$s_p$',7.5)
arr([(10.39,11.08),(11.50,11.08),(11.50,10.08),(11.95,10.08)])
arr([(13.28,7.69),(13.28,10.08),(12.25,10.08)]);op(12.10,10.08,'+',.15,10)
arr([(12.10,10.23),(12.10,11.65)]);vec(11.79,11.68,.62,.24,5,93);txt(12.10,12.15,'$s$',8)
txt(6.9,12.50,'⊙  elementwise product       Scale ×  scalar times vector       ∥  concatenate',7.1)
key(12.78,True)
save('TreeLoom_operators')
(OUT/'generation.json').write_text(json.dumps({'source_tree_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'tree_intervals':[[t.interval.left,t.interval.right] for t in trees],'figures':['TreeLoom_architecture','TreeLoom_memory','TreeLoom_operators'],'activations':'Illustrative values, not model output','model_or_simulation_run':False},indent=2)+'\n')
print('Rendered three CPU-only vector diagrams.')
