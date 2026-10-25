"""Editable TreeLoom schematic. Reads a preserved toy tree sequence, no training.
Palette is semantic: black structure, gray features, four ancestry classes.
"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import hashlib
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'research/reviews/figure_v15_flow_revision_20261017_v1/v15r2/illustrative_recorded_ARG.trees'
ts = tskit.load(SOURCE)
trees = [t.copy() for t in ts.trees()]
assert ts.num_trees == 3 and ts.num_samples == 5
INK = '#202020'
FEATURE = '#bdbdbd'
WHITE = '#ffffff'
ANC = ['#0072b2', '#e69f00', '#009e73', '#cc79a7']
NAMES = ['q', 'r1', 'r2', 'r3', 'r4']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'savefig.facecolor': WHITE})
W, H = 7.0, 9.08
fig = plt.figure(figsize=(W, 7.65))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
texts = []
def text(x, y, s, size=8, ha='center', bold=False):
    t = ax.text(x, y, s, ha=ha, va='center', color=INK, fontsize=size,
                fontweight='bold' if bold else 'normal', zorder=10)
    texts.append(t)
    return t

def line(points, lw=.8, style='-'):
    ax.plot(*zip(*points), color=INK, lw=lw, ls=style, zorder=3)

def arrow(points, both=False, lw=.8):
    if len(points) > 2:
        line(points[:-1], lw)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1],
                 arrowstyle='<->' if both else '-|>', color=INK,
                 mutation_scale=6, lw=lw, shrinkA=0, shrinkB=0, zorder=4))

def box(x, y, w, h, label=None, size=8, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0,rounding_size=.035', fc=WHITE,
                 ec=INK, lw=.8, ls='--' if dashed else '-', zorder=2))
    if label:
        text(x+w/2, y+h/2, label, size)

def vector(x, y, w=.29, h=.10, n=4):
    for j in range(n):
        ax.add_patch(Rectangle((x+j*w/n, y), w/n, h, fc=FEATURE, ec=WHITE, lw=.55, zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=INK, lw=.55, zorder=6))

def dot(x, y, sample=None, r=.025):
    color = INK if sample is None or sample == 0 else ANC[sample-1]
    ax.add_patch(Circle((x, y), r, fc=WHITE if sample == 0 else color,
                 ec=INK, lw=.55, zorder=6))

def panel(x, y, s):
    text(x, y, s, 11, ha='left', bold=True)

# (a) Only one genomic representation: three exact trees from a saved toy ARG.
panel(.12, .18, 'a')
centers = [.86, 2.12, 3.38]
for i, (tree, cx) in enumerate(zip(trees, centers)):
    text(cx, .20, '$T_'+str(i+1)+'$', 9)
    line([(cx-.51, 1.34-350/725*.95), (cx+.51, 1.34-350/725*.95)], .55, '--')
    xp = {u: cx-.48+.24*u for u in range(5)}
    def node_x(u):
        if u not in xp:
            xp[u] = np.mean([node_x(v) for v in tree.children(u)])
        return xp[u]
    node_x(tree.root)
    yp = {u: 1.34-ts.node(u).time/725*.95 for u in tree.nodes()}
    for u in tree.nodes():
        p = tree.parent(u)
        if p != -1:
            line([(xp[u], yp[u]), (xp[u], yp[p]), (xp[p], yp[p])], .75)
    for u in tree.nodes():
        dot(xp[u], yp[u], u if u<5 else None, .027 if u<5 else .016)
    for u in range(5):
        text(xp[u], 1.47, NAMES[u], 7.5)
    line([(cx-.48, 1.62), (cx+.48, 1.62)], 1.3)
    for x in [cx-.48, cx+.48]:
        line([(x, 1.58), (x, 1.66)], .7)
arrow([(.38, 1.82), (3.87, 1.82)])
text(2.12, 1.97, 'Genomic position', 8)
arrow([(.16, 1.30), (.16, .45)], lw=.7)
text(.16, .34, 'Age', 7.5)
text(.29, .88, '$t_c$', 7.5)
# Observed alleles use actual hypothetical mutations on this saved tree sequence.
text(5.52, .22, 'Observed alleles', 9)
for j, name in enumerate(NAMES):
    text(4.72+.40*j, .48, name)
for i, row in enumerate(ts.genotype_matrix()):
    text(4.30, .75+i*.27, '$T_'+str(i+1)+'$', 8)
    for j, value in enumerate(row):
        x = 4.55+.40*j
        ax.add_patch(Rectangle((x, .63+i*.27), .34, .24, fc=WHITE, ec=INK, lw=.45))
        text(x+.17, .75+i*.27, str(value), 8)
# Ancestry input is visibly known at references and unknown at q.
text(5.52, 1.57, 'Reference ancestry', 8)
for j in range(5):
    dot(4.72+.40*j, 1.82, j, .048)
text(4.72, 1.97, '?', 9)

# Small multiples retain the exact saved topology and sample order.
def tree_glyph(i, cx, base, w=.68, h=.66, direction=None, cut=False, rep=None):
    tree=trees[i]
    xp={u:cx-w/2+w*u/4 for u in range(5)}
    def xx(u):
        if u not in xp: xp[u]=float(np.mean([xx(v) for v in tree.children(u)]))
        return xp[u]
    xx(tree.root)
    yp={u:base-h*ts.node(u).time/725 for u in tree.nodes()}
    for u in tree.nodes():
        p=tree.parent(u)
        if p<0:continue
        a=(xp[u],yp[u]);b=(xp[u],yp[p]);c=(xp[p],yp[p])
        line([a,b,c],.65)
        if direction:
            lo=np.array(a);hi=np.array(b)
            start=lo+.15*(hi-lo);end=lo+.82*(hi-lo)
            if direction=='down':start,end=end,start
            arrow([start,end],lw=.7)
    for u in tree.nodes():dot(xp[u],yp[u],u if u<5 else None,.021 if u<5 else .011)
    if cut:
        yy=base-h*350/725
        line([(cx-w*.57,yy),(cx+w*.57,yy)],.5,'--')
    if rep is not None:
        ax.add_patch(Rectangle((xp[rep]-.040,yp[rep]-.040),.080,.080,fc='none',ec=INK,lw=.85,zorder=7))
    return {u:(xp[u],yp[u]) for u in tree.nodes()}

def op(x,y,s,r=.075,size=10):
    ax.add_patch(Circle((x,y),r,fc=WHITE,ec=INK,lw=.7,zorder=5))
    text(x,y,s,size)

def mat(x,y,w=.23,h=.26,label=None):
    for i in range(4):
        for j in range(3):
            ax.add_patch(Rectangle((x+j*w/3,y+i*h/4),w/3,h/4,fc=FEATURE,ec=WHITE,lw=.35,zorder=6))
    ax.add_patch(Rectangle((x,y),w,h,fc='none',ec=INK,lw=.6,zorder=7))
    if label:text(x+w/2,y+h+.13,label,7.5)

def save(name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in texts:
        bb=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert bb.x0>=-.002 and bb.y0>=-.002 and bb.x1<=1.002 and bb.y1<=1.002,t.get_text()
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=220)
    p=OUT/(name+'.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')

# (b) Four explicit directed traversals replace the opaque GNN boxes.
panel(.12,2.26,'b')
line([(.44,2.11),(6.58,2.11)],.6)
arrow([(.36,2.11),(.36,2.65)])
box(.08,2.65,.56,.68,'MLPs\n+ pool',7.5)
arrow([(.64,2.99),(.81,2.99)])
for cx,mode,idx in [(1.18,'up',1),(2.12,'down',1),(4.43,'up',2),(5.37,'down',2)]:
    box(cx-.39,2.59,.78,.79)
    tree_glyph(0,cx,3.28,w=.64,h=.65,direction=mode)
    text(cx,2.44,(r'$\uparrow_' if mode=='up' else r'$\downarrow_')+str(idx)+'$',10)
arrow([(1.57,2.99),(1.73,2.99)])
arrow([(4.82,2.99),(4.98,2.99)])
for left,right,num in [(.79,2.51,'1'),(4.04,5.76,'2')]:
    line([(left,3.43),(left,3.49),(right,3.49),(right,3.43)],.6)
    text((left+right)/2,3.64,'Round '+num,8)
# First-sweep states enter two explicitly referenced corrections.
arrow([(2.51,2.99),(2.68,2.99)])
vector(2.70,2.86,.20,.27,3)
text(2.80,2.67,'$u,o$',7.5)
line([(2.80,3.13),(2.80,3.43)],.65)
arrow([(2.80,3.43),(3.04,3.43)])
vector(3.06,3.37,.19,.12,3)
text(3.15,3.66,'s (Fig. 2c)',7.5)
arrow([(2.80,3.43),(2.80,3.87),(3.50,3.87),(3.50,3.51)])
vector(3.405,3.37,.19,.12,3)
text(3.70,3.66,r'$\delta$ (c)',7.5)
# The corrected initial features, not the first-pass hidden state, enter pass two.
line([(.72,2.99),(.72,2.27),(3.47,2.27)],.65)
arrow([(3.47,2.27),(3.47,2.88)])
text(2.80,2.17,'b',8)
op(3.47,2.99,'+',.105,12)
arrow([(3.15,3.37),(3.15,3.18),(3.39,3.04)],lw=.65)
arrow([(3.50,3.37),(3.50,3.10)],lw=.65)
arrow([(3.575,2.99),(4.04,2.99)])
text(3.81,2.81,"$b'$",8)
# Combine upward and outside states at each sampled node, then project.
arrow([(5.76,2.99),(5.835,2.99)],lw=.6)
op(5.91,2.99,'∥',.075,9)
line([(4.90,2.99),(4.90,2.22),(5.91,2.22)],.6)
arrow([(5.91,2.22),(5.91,2.915)],lw=.6)
arrow([(5.985,2.99),(6.08,2.99)],lw=.6)
box(6.08,2.71,.80,.56,'Linear\n+ tanh',7.2)
text(6.48,2.55,'64→32',7.1)
arrow([(6.48,3.27),(6.48,3.40)],lw=.6)
for j in range(3):vector(6.05+j*.29,3.42,.22,.16,3)
text(6.45,3.74,r'$T_1\quad T_2\quad T_3$',7.5)
text(6.45,3.95,'Query embeddings',7.1)
# The count is represented by all four tree copies, not an ambiguous depth label.
text(1.65,3.88,r'$\theta_{\uparrow 1}=\theta_{\uparrow 2}\quad\theta_{\downarrow 1}=\theta_{\downarrow 2}$',7.7)

# (c) Memory is visibly indexed by sampled haplotype and genomic interval.
# The tracks show stored states. The small arithmetic cell expands one update.
panel(.12,4.17,'c')
left=tree_glyph(1,.57,5.18,w=.77,h=.83,cut=True,rep=8)
text(.57,4.23,'$T_2$',8.5)
x1=.57-.77/2+.77/4;x2=.57
line([(x1,5.26),(x1,5.31),(x2,5.31),(x2,5.26)],.65)
text((x1+x2)/2,5.46,'r1  r2',7.5)
# The memory signal combines both representative states before compression.
line([left[8],(.99,left[8][1])],.55)
line([(.99,4.52),(.99,5.17)],.55)
for y,lab in [(4.52,'u'),(5.17,'o')]:
    arrow([(.99,y),(1.05,y)],lw=.55)
    vector(1.05,y-.045,.15,.09,3)
    text(1.27,y,lab,7)
arrow([(1.125,4.565),(1.125,4.765)],lw=.55)
arrow([(1.125,5.125),(1.125,4.915)],lw=.55)
op(1.125,4.84,'+',.075,9)
arrow([(1.20,4.84),(1.36,4.84)],lw=.6)
mat(1.36,4.70,.22,.28,'32→8')
arrow([(1.58,4.84),(1.75,4.84)],lw=.65)
vector(1.77,4.78,.22,.12,3)
text(1.88,4.58,'$f_2$',8)
# Three positions and two directions show that histories are propagated.
cols=[2.65,3.65,4.65]
for j,x in enumerate(cols):text(x,4.24,'$T_'+str(j+1)+'$',8)
for y,name in [(4.64,'r1'),(5.36,'r2')]:
    text(2.18,y+.19,name,8)
    for yy,forward in [(y,True),(y+.38,False)]:
        for x in cols:vector(x-.11,yy-.055,.22,.11,3)
        for a,b in zip(cols[:-1],cols[1:]):
            points=[(a+.11,yy),(b-.11,yy)] if forward else [(b-.11,yy),(a+.11,yy)]
            arrow(points,lw=.7)
        # The outer copies enter the first and last state rows.
        if (name=='r1' and forward) or (name=='r2' and not forward):
            fy=yy-.20 if forward else yy+.20
            vector(3.58,fy-.035,.14,.07,2)
            text(3.38,fy,'$f_2$',7)
            arrow([(3.65,fy+(.035 if forward else -.035)),(3.65,yy+(-.055 if forward else .055))],lw=.55)
    # Current directional states are averaged before group pooling.
    arrow([(3.65,y+.055),(3.65,y+.09),(3.98,y+.09),(3.98,y+.115)],lw=.55)
    arrow([(3.65,y+.325),(3.65,y+.30),(3.98,y+.30),(3.98,y+.265)],lw=.55)
    op(3.98,y+.19,'½Σ',.075,7)
    arrow([(4.055,y+.19),(5.10,y+.19),(5.10,5.24 if name=='r1' else 5.42)],lw=.6)
# A single visible copy supplies the two adjacent inner rows.
vector(3.58,5.155,.14,.07,2);text(3.38,5.19,'$f_2$',7)
arrow([(3.65,5.155),(3.65,5.075)],lw=.55)
arrow([(3.65,5.225),(3.65,5.305)],lw=.55)
op(5.10,5.33,'½Σ',.09,7.5)
arrow([(5.19,5.33),(5.50,5.33)],lw=.65)
op(5.59,5.33,'−',.09,10)
vector(5.48,5.77,.22,.11,3);text(5.84,5.825,'$f_2$',8)
arrow([(5.59,5.77),(5.59,5.42)],lw=.6)
arrow([(5.68,5.33),(6.00,5.33)],lw=.65)
mat(6.00,5.17,.23,.32,'8→32')
arrow([(6.23,5.33),(6.56,5.33),(6.56,5.72)],lw=.6)
vector(6.44,5.74,.24,.12,3);text(6.80,5.80,r'$\delta$',8)
right=tree_glyph(1,6.29,6.96,w=.77,h=.83,cut=True,rep=8)
arrow([(6.56,5.86),(6.56,6.13),(right[8][0],6.13),(right[8][0],right[8][1]-.04)],lw=.6)
# One magnified update gives the operation used at every step of both tracks.
# This avoids four repeated arithmetic circuits and retains visible recurrence.
box(.20,6.03,4.63,.95,dashed=True)
vector(.34,6.54,.25,.11,3);text(.47,6.37,r'$m_{t-1}^{\to}$',7.5)
arrow([(.59,6.595),(.90,6.595)],lw=.65)
op(.98,6.595,'×',.075,9);text(.98,6.37,r'$\alpha_t$',7.5)
arrow([(1.055,6.595),(2.20,6.595)],lw=.65)
op(2.29,6.595,'+',.09,11)
vector(1.35,6.22,.22,.10,3);text(1.23,6.27,'$f_t$',7.5)
arrow([(1.57,6.27),(1.80,6.27)],lw=.6)
op(1.88,6.27,'×',.075,9);text(1.88,6.13,r'$1-\alpha_t$',7)
arrow([(1.955,6.27),(2.29,6.27),(2.29,6.505)],lw=.6)
arrow([(2.38,6.595),(2.64,6.595)],lw=.65)
vector(2.66,6.54,.27,.11,3);text(2.795,6.37,r'$m_t^{\to}$',7.5)
arrow([(2.93,6.595),(3.18,6.595)],lw=.65)
text(3.45,6.595,r'$T_{t+1}$',8)
text(2.50,6.86,r'$\alpha_t=\exp(-|I_t|/\ell)$',8)
# Readout retains panel d's familiar operations, arranged along one short line.
panel(.12,7.22,'d')
for j in range(3):vector(.34+j*.29,7.58,.22,.25,3)
text(.74,7.41,'$T_1\;T_2\;T_3$',8)
arrow([(1.16,7.70),(1.42,7.70)])
box(1.42,7.43,1.13,.55,'BiGRU\n1 layer',8)
text(1.99,7.22,'Interval geometry',7.5)
arrow([(1.99,7.31),(1.99,7.43)],lw=.65)
arrow([(2.55,7.70),(2.83,7.70)])
box(2.83,7.43,1.06,.55,'Concat\n+ MLP',8)
line([(1.27,7.70),(1.27,8.13),(3.36,8.13)],.65)
arrow([(3.36,8.13),(3.36,7.98)],lw=.65)
arrow([(3.89,7.70),(4.12,7.70)])
box(4.12,7.49,.79,.43,'Softmax',7.7)
arrow([(4.91,7.70),(5.17,7.70)])
probs=[[.62,.12,.14,.12],[.17,.16,.53,.14],[.11,.16,.21,.52]]
breaks=list(ts.breakpoints());x0,x1=5.22,6.91
for i,row in enumerate(probs):
    a=x0+(x1-x0)*breaks[i]/ts.sequence_length;b=x0+(x1-x0)*breaks[i+1]/ts.sequence_length;lo=7.99
    for k,v in enumerate(row):
        ax.add_patch(Rectangle((a,lo-.49*v),b-a,.49*v,fc=ANC[k],ec=WHITE,lw=.4));lo-=.49*v
    ax.add_patch(Rectangle((a,8.18),b-a,.10,fc=ANC[int(np.argmax(row))],ec=WHITE,lw=.4))
text(6.06,7.31,'$p(q)$',8)
text(5.02,8.23,'argmax',7.5,ha='right')
arrow([(6.84,7.99),(6.84,8.17)],lw=.6)
# Palette key shared by every panel.
line([(.18,8.34),(6.91,8.34)],.55)
legend=[('Anatolian farmer (Ana)',.24,8.70),('Caucasus hunter-gatherer (CHG)',3.55,8.70),
        ('Western hunter-gatherer (WHG)',.24,8.91),('Eastern hunter-gatherer (EHG)',3.55,8.91)]
for color,(name,x,y) in zip(ANC,legend):
    ax.add_patch(Rectangle((x,y-.05),.14,.10,fc=color,ec='none'));text(x+.21,y,name,7.5,ha='left')
# The separate neutral legend is above the ancestry key to stay within the page.
vector(.24,8.44,.14,.08,2);text(.45,8.48,'Features / weights',7.0,ha='left')
line([(2.00,8.48),(2.14,8.48)],.8);text(2.22,8.48,'Structure / operations',7.0,ha='left')
dot(4.02,8.48,0,.035);text(4.14,8.48,'Query',7.3,ha='left')
ax.add_patch(Rectangle((4.88,8.44),.08,.08,fc='none',ec=INK,lw=.7));dot(4.92,8.48,None,.010)
text(5.06,8.48,'Group ancestor',7.0,ha='left')
save('TreeLoom_architecture')

# A separate full-width detail figure gives exact operators without shrinking
# the biological overview or hiding operations behind custom module names.
plt.close(fig)
texts=[];fig=plt.figure(figsize=(7,6.55));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7),ylim=(7.48,0));ax.axis('off')
panel(.12,.16,'a')
# Upward cell: gated arithmetic mean, concatenation, affine map, tanh, residual, LN.
for y in [.59,.73,.87]:vector(.28,y,.23,.09,3)
line([(.51,.635),(.59,.635),(.59,.915),(.51,.915)],.6)
arrow([(.59,.77),(.77,.77)])
box(.77,.57,.54,.40,'Linear',7.5)
arrow([(1.31,.77),(1.48,.77)])
op(1.56,.77,'⊙',.08,10)
vector(1.43,.29,.26,.10,3);text(1.56,.14,'gate ↑',7.5);arrow([(1.56,.39),(1.56,.69)],lw=.6)
arrow([(1.64,.77),(1.82,.77)])
box(1.82,.57,.55,.40,'Mean',7.5)
arrow([(2.37,.77),(2.55,.77)])
op(2.63,.77,'∥',.08,9)
vector(2.50,.29,.26,.10,3);text(2.63,.14,'b',8)
arrow([(2.63,.39),(2.63,.69)],lw=.6)
arrow([(2.71,.77),(2.91,.77)])
box(2.91,.57,.56,.40,'Linear',7.5)
arrow([(3.47,.77),(3.66,.77)])
box(3.66,.57,.48,.40,'tanh',7.5)
arrow([(4.14,.77),(4.33,.77)])
op(4.41,.77,'+',.08,11)
line([(2.76,.34),(4.41,.34)],.6);arrow([(4.41,.34),(4.41,.69)],lw=.6)
arrow([(4.49,.77),(4.69,.77)])
box(4.69,.57,.58,.40,'LN',8)
arrow([(5.27,.77),(5.52,.77)])
vector(5.54,.64,.30,.26,4);text(5.69,1.11,'u',8)
# Visual child-to-parent origin for this operator.
tree_glyph(0,6.43,1.03,w=.62,h=.77,direction='up')
text(1.04,1.14,'32→32',7.5)
text(3.19,1.14,'64→32',7.5)

panel(.12,1.44,'b')
# Four explicitly named feature bars show subtraction of the recipient message.
xs=[.25,.64,1.03,1.42];labs=[r'$o_p$',r'$b_p$',r'$a_p$',r'$m_{q\to p}$']
for x,lab in zip(xs,labs):vector(x,1.91,.23,.12,3);text(x+.115,1.74,lab,7.5)
for x,sym in [(.55,'+'),(.94,'+'),(1.33,'−')]:text(x,1.97,sym,9)
arrow([(1.65,1.97),(1.84,1.97)])
box(1.84,1.77,.54,.40,'Linear',7.5)
arrow([(2.38,1.97),(2.56,1.97)])
op(2.64,1.97,'⊙',.08,10)
vector(2.51,1.43,.26,.10,3);text(2.64,1.28,'gate ↓',7.5);arrow([(2.64,1.53),(2.64,1.89)],lw=.6)
arrow([(2.72,1.97),(2.91,1.97)])
box(2.91,1.77,.56,.40,'Linear',7.5)
arrow([(3.47,1.97),(3.66,1.97)])
box(3.66,1.77,.48,.40,'tanh',7.5)
arrow([(4.14,1.97),(4.39,1.97)])
box(4.39,1.77,.58,.40,'LN',8)
arrow([(4.97,1.97),(5.22,1.97)])
vector(5.24,1.84,.30,.26,4);text(5.39,2.31,'$o_q$',8)
# The recipient's own contribution is visibly excluded.
line([(5.95,2.18),(6.38,1.48),(6.81,2.18)],.7)
dot(6.38,1.48);dot(5.95,2.18,0);dot(6.81,2.18,1)
arrow([(6.76,2.08),(6.46,1.60)],lw=.8)
arrow([(6.30,1.62),(6.02,2.08)],lw=.8)
text(5.95,2.37,'q',8);text(6.38,1.33,'p',8)
# Shared gate generation is shown once, as a separate explicit small chain.
vector(.25,2.56,.25,.11,3);text(.375,2.82,'edge',7.5)
arrow([(.50,2.615),(.70,2.615)],lw=.65);box(.70,2.46,.52,.31,'Linear',7.5)
arrow([(1.22,2.615),(1.40,2.615)],lw=.65);box(1.40,2.46,.53,.31,'sigmoid',7.3)
arrow([(1.93,2.615),(2.10,2.615)],lw=.65);vector(2.12,2.56,.27,.11,3)
text(2.56,2.615,'gate',7.5,ha='left')
text(4.47,2.615,'LN: layer normalization',7.5)

# (c) Presence patterns explain the selectors, and all state inputs originate
# from the current node u. There are no arrows from neighboring hidden states.
panel(.12,3.09,'c')
for j in range(3):text(.53+j*.34,3.22,'$T_'+str(j+1)+'$',7.5)
for row,(pair,vals) in enumerate([((0,1),[1,0,0]),((1,2),[0,1,1]),((3,4),[1,1,1])]):
    y=3.55+row*.36
    for j,v in enumerate(vals):
        x=.53+j*.34
        if v:
            line([(x-.10,y+.07),(x,y-.08),(x+.10,y+.07)],.6)
            dot(x-.10,y+.07,pair[0],.018);dot(x+.10,y+.07,pair[1],.018)
        else:text(x,y,'×',10)
text(.86,4.65,'descendant groups',7.5)
text(1.12,4.88,r'$g,l,e\in\{0,1\}$',7.5)
# Three schematic switches derived from these topology changes.
for x,y,lab,sym in [(1.66,3.52,'lost',r'$l_1$'),(1.66,3.90,'gained',r'$g_2$'),(1.66,4.28,'persists',r'$e_{1:3}$')]:
    text(x,y,lab,7.5)
    text(x+.42,y,sym,8)
# Exact factorization displayed as parallel vector calculations with matrices.
# Flag correction: O_f[g(Uu ⊙ Vu) - l(Uu ⊙ Wu)].
vector(2.30,3.20,.24,.12,3);text(2.42,3.04,'u',8)
for y,tag in [(3.57,'V'),(3.94,'U'),(4.31,'W')]:
    line([(2.42,3.32),(2.42,y),(2.61,y)],.6)
    mat(2.61,y-.11,.19,.22)
    text(2.70,y+.23,tag,7.5)
    arrow([(2.80,y),(2.98,y)],lw=.6)
    vector(3.00,y-.05,.22,.10,3)
arrow([(3.22,3.57),(3.60,3.57)],lw=.6);op(3.68,3.57,'⊙',.075,9)
arrow([(3.22,4.31),(3.60,4.31)],lw=.6);op(3.68,4.31,'⊙',.075,9)
line([(3.22,3.94),(3.68,3.94)],.6)
arrow([(3.68,3.94),(3.68,3.645)],lw=.6)
arrow([(3.68,3.94),(3.68,4.235)],lw=.6)
arrow([(3.755,3.57),(3.99,3.57)],lw=.6);op(4.07,3.57,'×',.075,9);text(4.07,3.35,'g',8)
arrow([(3.755,4.31),(3.99,4.31)],lw=.6);op(4.07,4.31,'×',.075,9);text(4.07,4.53,'l',8)
arrow([(4.145,3.57),(4.40,3.57),(4.40,3.86)],lw=.6)
arrow([(4.145,4.31),(4.40,4.31),(4.40,4.02)],lw=.6)
op(4.40,3.94,'−',.08,10)
arrow([(4.48,3.94),(4.65,3.94)],lw=.6);mat(4.65,3.81,.23,.26,'O')
arrow([(4.88,3.94),(5.06,3.94)],lw=.6);vector(5.08,3.88,.25,.12,3)
text(5.20,4.25,'$s_f$',8)
# Persistence has its own three projections of this SAME current node.
vector(2.30,4.85,.24,.12,3);text(2.42,4.69,'u',8)
for y,tag in [(5.23,'$U_p$'),(5.60,'$V_p$'),(5.97,'$W_p$')]:
    line([(2.42,4.97),(2.42,y),(2.61,y)],.6)
    mat(2.61,y-.11,.19,.22)
    text(2.70,y+.23,tag,7.5)
    arrow([(2.80,y),(2.98,y)],lw=.6)
    vector(3.00,y-.05,.22,.10,3)
arrow([(3.22,5.23),(3.68,5.23),(3.68,5.525)],lw=.6)
arrow([(3.22,5.60),(3.605,5.60)],lw=.6)
op(3.68,5.60,'⊙',.075,9)
arrow([(3.755,5.60),(4.08,5.60),(4.08,5.71)],lw=.6)
arrow([(3.22,5.97),(4.08,5.97),(4.08,5.87)],lw=.6)
op(4.08,5.79,'+',.08,10)
arrow([(4.16,5.79),(4.36,5.79)],lw=.6);mat(4.36,5.66,.23,.26,'$O_p$')
arrow([(4.59,5.79),(4.81,5.79)],lw=.6);op(4.89,5.79,'×',.075,9);text(4.89,5.56,'e',8)
arrow([(4.965,5.79),(5.18,5.79)],lw=.6);vector(5.20,5.73,.25,.12,3)
text(5.325,6.07,'$s_p$',8)
# Both local corrections enter s before the second complete tree sweep.
arrow([(5.33,3.94),(5.83,3.94),(5.83,4.87)],lw=.6)
arrow([(5.45,5.79),(5.83,5.79),(5.83,5.05)],lw=.6)
op(5.83,4.96,'+',.09,11)
arrow([(5.92,4.96),(6.14,4.96)],lw=.6);vector(6.16,4.89,.30,.14,3);text(6.31,5.23,'s',8)
text(3.78,6.42,'U,V,W: 32→8       O: 8→32',7.5)
# Every color is defined on this figure independently.
line([(.18,6.60),(6.91,6.60)],.55)
for color,(name,x,y) in zip(ANC,[('Anatolian farmer',.23,6.79),('Caucasus hunter-gatherer',3.55,6.79),('Western hunter-gatherer',.23,7.00),('Eastern hunter-gatherer',3.55,7.00)]):
    ax.add_patch(Rectangle((x,y-.05),.12,.10,fc=color,ec='none'));text(x+.20,y,name,7.3,ha='left')
vector(.23,7.23,.13,.08,2);text(.45,7.27,'Features / weights',7.3,ha='left')
line([(2.30,7.27),(2.44,7.27)],.8);text(2.52,7.27,'Structure / operations',7.3,ha='left')
dot(4.90,7.27,0,.035);text(5.02,7.27,'Query',7.3,ha='left')
save('TreeLoom_operators')
print('Rendered overview and operator detail in',OUT)
