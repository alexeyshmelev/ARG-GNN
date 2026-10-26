"""Draw the TreeLoom input and readout as connected tensor operations.

Only the preserved illustrative tree sequence is read. This program neither
simulates data nor constructs or executes any trained model.
"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = Path(__file__).parent
SOURCE = ROOT / 'research/reviews/figure_v15_flow_revision_20261017_v1/v15r2/illustrative_recorded_ARG.trees'
ts = tskit.load(SOURCE)
trees = [tree.copy() for tree in ts.trees()]
INK = '#26363b'
GOLD = '#f8e6b8'
GOLD_EDGE = '#8d6826'
FIXED = '#f5f6f6'
EMB = LinearSegmentedColormap.from_list('emb', ['#e6f5f7', '#67b7c4', '#175e68'])
OBS = LinearSegmentedColormap.from_list('obs', ['#ffffff', '#afb8bd', '#34434a'])
PROB = LinearSegmentedColormap.from_list('prob', ['#faf6fc', '#ba92ce', '#582571'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(7, 8.10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 7), ylim=(8.10, 0))
ax.axis('off')
labels = []

def text(x, y, s, size=7, ha='center', weight='normal', color=INK):
    t = ax.text(x, y, s, ha=ha, va='center', fontsize=size, color=color, fontweight=weight, zorder=20)
    labels.append(t)
    return t

def box(x, y, w, h, face='white', edge=INK, lw=.6):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=lw, zorder=2))

def line(points, lw=.6, color=INK):
    ax.plot(*zip(*points), color=color, lw=lw, zorder=4)

def arrow(points, lw=.6):
    if len(points) > 2:
        line(points[:-1], lw)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>', mutation_scale=5, lw=lw, color=INK, shrinkA=0, shrinkB=0, zorder=5))

def panel(letter, y):
    text(.14, y, letter, 9, weight='bold')

def operator(x, y, w, h, label, learned=False, size=7):
    box(x, y, w, h, GOLD if learned else FIXED, GOLD_EDGE if learned else INK)
    text(x+w/2, y+h/2, label, size)

def stack(x, y, w, specifications):
    heights = [.23 if '\n' in s else .17 for s in specifications]
    for index, (s, h) in enumerate(zip(specifications, heights)):
        operator(x, y, w, h, s, s.startswith('Linear'))
        y += h
        if index < len(specifications)-1:
            arrow([(x+w/2, y), (x+w/2, y+.045)])
            y += .045
    return y

def grid(x, y, w, h, r, c, key=1, kind='emb', values=None, numbers=False, digits=2):
    vals = .48 + .31*np.sin(np.arange(r*c).reshape(r,c)*1.8+key) if values is None else np.asarray(values)
    cmap = {'emb': EMB, 'obs': OBS, 'prob': PROB}[kind]
    for i in range(r):
        for j in range(c):
            v = float(vals[i,j])
            box(x+j*w/c, y+i*h/r, w/c, h/r, cmap(float(np.clip(v,0,1))), 'white', .3)
            if numbers:
                lab = str(int(v)) if v.is_integer() else f'{v:.{digits}f}'
                text(x+(j+.5)*w/c, y+(i+.5)*h/r, lab, 7, color='white' if v>.65 else INK)
    box(x,y,w,h,'none')

def tensor(x,y,w,h,label,rows,cols=4,key=1,kind='emb',values=None, row_names=None):
    grid(x,y,w,h,rows,cols,key,kind,values)
    text(x+w/2,y-.12,label)
    if row_names:
        for i,name in enumerate(row_names):
            text(x-.065,y+(i+.5)*h/rows,name,7,ha='right')

def port(x,y,label,w=.57):
    box(x,y,w,.19,FIXED,lw=.5)
    text(x+w/2,y+.095,label,7)
    return (x+w/2,y+.19)

def bracket(x1,x2,y,label,up=False):
    dy=-.045 if up else .045
    line([(x1,y+dy),(x1,y),(x2,y),(x2,y+dy)])
    text((x1+x2)/2,y+(-.10 if up else .11),label)

# a: one preserved marginal tree, its exact node rows, and observed leaf rows.
panel('a', .15)
text(.85,.16,r'$T_2$: 3000–7000 bp')
t = trees[1]
node_order = [0,1,2,3,4,6,8,9,12]
node_names = ['q','r1','r2','r3','r4','a','b','c','root']
xx = {u:.45+.205*i for i,u in enumerate(range(5))}
def xpos(u):
    if u not in xx:
        xx[u] = np.mean([xpos(v) for v in t.children(u)])
    return xx[u]
xpos(t.root)
ypos = {u:1.22-.80*ts.node(u).time/675 for u in t.nodes()}
line([(.32,.42),(.32,1.22)])
for age in [0,350,675]:
    yy=1.22-.80*age/675
    line([(.30,yy),(.34,yy)])
    text(.27,yy,str(age),7,ha='right')
text(.28,.30,'age',7)
cut_y=1.22-.80*350/675
line([(.35,cut_y),(1.30,cut_y)],.45,color='#8a979c')
for u in t.nodes():
    p=t.parent(u)
    if p>=0:
        line([(xx[u],ypos[u]),(xx[u],ypos[p]),(xx[p],ypos[p])])
    ax.add_patch(Circle((xx[u],ypos[u]),.012,facecolor='white',edgecolor=INK,lw=.5,zorder=6))
    name=node_names[node_order.index(u)]
    if u<5:
        text(xx[u],ypos[u]+.09,name)
    else:
        text(xx[u]-.03,ypos[u]-.08,name,ha='right')
line([(1.32,.42),(1.35,.42),(1.35,1.22),(1.32,1.22)])
arrow([(1.35,1.04),(1.40,1.04),(1.40,.38),(1.57,.38),(1.57,.43)])
# The first matrix has one row for every depicted node.
x0,y0,w0,h0=1.62,.41,1.12,.945
text(2.15,.16,r'$X_2$: 9×3')
for j,lab in enumerate(['age′','root','leaf']):
    text(x0+(j+.5)*w0/3,.31,lab)
for i,(u,lab) in enumerate(zip(node_order,node_names)):
    text(x0-.07,y0+(i+.5)*h0/9,lab,ha='right')
    vals=[np.log1p(ts.node(u).time)/np.log(10001),int(u==12),int(u<5)]
    for j,val in enumerate(vals):
        box(x0+j*w0/3,y0+i*h0/9,w0/3,h0/9,OBS(float(val)*.3),'white',.25)
        text(x0+(j+.5)*w0/3,y0+(i+.5)*h0/9,f'{val:.3f}' if j==0 else str(val))
box(x0,y0,w0,h0,'none')
# This formula is bound to the actual age column.
line([(x0+w0/6,1.355),(x0+w0/6,1.43),(2.30,1.43)])
text(2.38,1.43,'log(1+age) / log(10001)',ha='left')
arrow([(2.74,.83),(2.84,.83),(2.84,.665),(2.95,.665)])
stack(2.95,.55,.62,['Linear\n3→32','tanh'])
arrow([(3.57,.91),(3.67,.91),(3.67,.83),(3.77,.83)])
tensor(3.77,.43,.62,.87,r'$N_2$: 9×32',9)
port(3.79,1.33,'a:N')
# Four ancestry entries, a missing-label entry, and a normalized sample age.
ref=np.zeros((5,6));ref[0,4]=1
for i in range(1,5):ref[i,i-1]=1
text(5.68,.16,r'$R$: 5×6')
for j,lab in enumerate(['Ana','CHG','WHG','EHG','unknown','age′']):
    text(4.79+(j+.5)*1.98/6,.23 if lab=='unknown' else .35,lab)
    if lab=='unknown':line([(4.79+(j+.5)*1.98/6,.29),(4.79+(j+.5)*1.98/6,.45)])
for i,lab in enumerate(['q','r1','r2','r3','r4']):
    text(4.72,.49+(i+.5)*.70/5,lab,ha='right')
grid(4.79,.49,1.98,.70,5,6,kind='obs',values=ref,numbers=True)
port(5.50,1.30,'a:R')
# b: one called site in T2, then pooling and repeated leaf-row assembly.
panel('b',1.66)
text(.83,1.69,'alleles')
for j,name in enumerate(['q','r1','r2','r3','r4']):
    text(.37+(j+.5)*1.05/5,1.88,name)
site_labels=[f'{site.position/1000:g}k' for site in ts.sites()]
for i,name in enumerate(site_labels):
    text(.31,2.02+(i+.5)*.51/3,name,ha='right')
grid(.37,2.02,1.05,.51,3,5,kind='obs',values=ts.genotype_matrix(),numbers=True)
box(.37,2.02+.17,1.05/5,.17,'none',lw=1.2)
arrow([(1.42,2.27),(1.57,2.27)])
operator(1.57,2.04,.59,.46,'allele\nlookup')
px,py=port(1.58,1.74,'a:R')
arrow([(px,py),(px,2.04)])
# This tensor is a grouped display of its 19 channels, not extra site rows.
text(3.73,1.68,r'$x_{5000,q}$: 19')
text(2.98,1.80,'reference sampling age')
line([(2.61,1.86),(2.61,2.02)])
box(2.38,1.87,2.64,.79,'none')
text(3.10,1.96,'log freq.')
text(4.00,1.96,'available')
for i,lab in enumerate(['<350','≥350']):
    text(2.61,2.15+i*.20,lab)
logs=np.array([[-1.39,-1.39,-.29,-.29],[0,0,0,0]])
for i in range(2):
    for j in range(4):
        xx0=2.83+j*.24
        box(xx0,2.05+i*.20,.24,.20,FIXED,'white',.3)
        text(xx0+.12,2.15+i*.20,f'{logs[i,j]:.1f}' if logs[i,j] else '0')
grid(3.86,2.05,.64,.40,2,4,kind='obs',values=np.array([[1]*4,[0]*4]),numbers=True)
for lab,val,ly,vy in [('panel\nlog p','−.51',1.98,2.16),('entropy','.67',2.26,2.36),('called','1',2.47,2.57)]:
    text(4.76,ly,lab)
    text(4.76,vy,val)
text(3.56,2.55,'8 + 8 + 3 channels')
arrow([(2.16,2.27),(2.38,2.27)])
arrow([(5.02,2.27),(5.11,2.27),(5.11,1.915),(5.20,1.915)])
stack(5.20,1.80,.65,['Linear\n19→32','GELU','Linear\n32→32','tanh'])
arrow([(5.85,2.65),(5.97,2.65),(5.97,2.27),(6.10,2.27)])
tensor(6.10,2.20,.66,.14,r'$Z_{2q}$: S×32',1)
text(6.43,2.49,'S=1\ncalled site')
port(6.12,2.64,'b:Z')
# A named continuation port avoids crossing the input-feature illustration.
port(.41,2.89,'b:Z')
grid(.41,3.14,.57,.14,1,4)
text(.695,3.42,'called sites')
arrow([(.98,3.21),(1.14,3.21)])
operator(1.14,3.00,.52,.43,'mean\nmax')
# Its pooled vector is drawn in two grouped strips, with explicit channel counts.
text(2.59,2.90,r'$v_{2q}$: 67')
grid(1.84,3.08,.72,.14,1,4,key=1)
grid(2.56,3.08,.72,.14,1,4,key=1)
text(2.20,3.36,'mean 32')
text(2.92,3.36,'max 32')
for j,(name,value) in enumerate([('count′','.060'),('empty','0'),('span′','.572')]):
    xx0=1.84+j*.48
    box(xx0,3.49,.48,.16,FIXED)
    text(xx0+.24,3.57,value)
    text(xx0+.24,3.47,name)
line([(1.84,3.08),(1.79,3.08),(1.79,3.65),(1.84,3.65)])
arrow([(1.66,3.21),(1.79,3.21)])
line([(3.28,3.08),(3.35,3.08),(3.35,3.65),(3.28,3.65)])
arrow([(3.35,3.21),(3.44,3.21),(3.44,3.075),(3.53,3.075)])
stack(3.53,2.96,.61,['Linear\n67→32','tanh'])
# R[q] is a true second concat operand, connected to the observed columns.
port(4.37,2.88,'a:R[q]',w=.73)
grid(4.37,3.13,.73,.13,1,6,kind='obs',values=ref[0:1],numbers=True)
text(4.74,3.42,'32 | 6')
grid(4.36,3.54,.55,.13,1,4,key=2)
grid(4.91,3.54,.19,.13,1,6,kind='obs',values=ref[0:1])
arrow([(4.14,3.32),(4.24,3.32),(4.24,3.605),(4.36,3.605)])
arrow([(4.735,3.26),(4.735,3.32),(5.01,3.32),(5.01,3.54)])
arrow([(5.10,3.605),(5.17,3.605),(5.17,3.375),(5.25,3.375)])
stack(5.25,3.26,.61,['Linear\n38→32','tanh'])
tensor(6.10,3.12,.66,.55,r'$F_t^{leaf}$: 5×32',5)
for i,name in enumerate(['q','r1','r2','r3','r4']):text(6.84,3.12+(i+.5)*.55/5,name,ha='left')
arrow([(5.86,3.62),(5.94,3.62),(5.94,3.175),(6.10,3.175)])
# This repeat scope ends at the assembled leaf matrix, not at a hidden state.
text(6.43,3.77,'b:F · repeat leaves')
# c: gate inputs, physical scatter, and a single matched ENC expansion.
panel('c',4.07)
text(1.54,4.10,'branch b→c · one of 8')
box(.26,4.19,2.31,.69,FIXED,lw=.45)
box(.30,4.23,2.31,.69,FIXED,lw=.45)
feature_names=['child age','parent age','duration','leaf share','exact fit','closest fit','discrepancy','split count','empty']
feature_values=['.582','.651','.569','.4','1','1','0','.095','0']
for k,(name,value) in enumerate(zip(feature_names,feature_values)):
    i,j=divmod(k,3)
    xx0=.34+j*.77;yy0=4.27+i*.23
    box(xx0,yy0,.77,.23,FIXED)
    text(xx0+.385,yy0+.070,name)
    text(xx0+.385,yy0+.176,value)
arrow([(2.65,4.615),(2.72,4.615),(2.72,4.405),(2.83,4.405)])
box(2.78,4.20,.71,.82,'none',GOLD_EDGE)
stack(2.83,4.29,.61,['Linear\n9→32','sigmoid'])
text(3.13,4.88,'2 maps\nshared ×8')
tensor(3.66,4.31,.36,.55,'',8,key=4)
text(3.84,4.00,r'$G^{↑}$')
text(3.84,4.19,'8×32')
tensor(4.10,4.31,.36,.55,'',8,key=5)
text(4.28,4.00,r'$G^{↓}$')
text(4.28,4.19,'8×32')

arrow([(3.44,4.65),(3.55,4.65),(3.55,4.25),(3.84,4.25),(3.84,4.31)])
arrow([(3.55,4.25),(4.28,4.25),(4.28,4.31)])
# Base matrices, with identical nine-node row order.
tensor(.34,5.38,.49,.54,'a:N',9)
tensor(1.02,5.38,.49,.30,'b:F',5)
scatter=np.zeros((9,4));scatter[:5]=.48+.31*np.sin(np.arange(20).reshape(5,4)*1.8+1)
tensor(1.78,5.38,.49,.54,'9×32',9,values=scatter)
text(2.025,5.38+7*.54/9,'0')
arrow([(1.51,5.53),(1.78,5.53)])
operator(2.49,5.55,.20,.20,'+')
arrow([(2.27,5.65),(2.49,5.65)])
arrow([(.585,5.92),(.585,6.01),(2.59,6.01),(2.59,5.75)])
tensor(2.92,5.38,.49,.54,r'$B_t$: 9×32',9,key=3)
arrow([(2.69,5.65),(2.92,5.65)])
# The outline and literal ENC label match Figure 2, which expands this block.
box(4.72,5.28,.81,.64,GOLD,GOLD_EDGE,1.0)
text(5.125,5.48,'ENC',8,weight='bold')
text(5.125,5.73,'Figure 2')
arrow([(3.41,5.65),(4.72,5.65)])
arrow([(3.84,4.86),(3.84,5.20),(4.91,5.20),(4.91,5.28)])
arrow([(4.28,4.86),(4.28,5.14),(5.32,5.14),(5.32,5.28)])
tensor(5.92,5.28,.85,.27,r'$U_t^2$: 9×32',9,key=4)
tensor(5.92,5.74,.85,.27,r'$O_t^2$: 9×32',9,key=6)
arrow([(5.53,5.60),(5.71,5.60),(5.71,5.415),(5.92,5.415)])
arrow([(5.71,5.60),(5.71,5.875),(5.92,5.875)])
port(6.05,6.09,'c:U,O',w=.68)
# d: leaf selection and assembly across trees, with the true interval distances.
panel('d',6.20)
line([(.34,6.265),(.34,6.22),(3.26,6.22),(3.26,6.265)])
text(1.80,6.10,'T1, T2, T3 · same projection')
port(.34,6.26,'c:U,O',w=.68)
grid(.34,6.45,.27,.36,9,4,key=4)
grid(.69,6.45,.27,.36,9,4,key=6)
box(.34,6.45,.27,.20,'none',lw=1.0);box(.69,6.45,.27,.20,'none',lw=1.0)
operator(1.13,6.41,.58,.43,'leaf rows\nconcat')
arrow([(.96,6.63),(1.13,6.63)])
grid(1.88,6.44,.30,.40,5,4,key=4)
grid(2.18,6.44,.30,.40,5,4,key=6)
text(2.18,6.32,'5×(32 | 32)')
arrow([(1.71,6.63),(1.88,6.63)])
stack(2.66,6.37,.60,['Linear\n64→32','tanh'])
arrow([(2.48,6.63),(2.56,6.63),(2.56,6.485),(2.66,6.485)])
for i,x in enumerate([3.48,4.25,5.02]):
    tensor(x,6.59,.55,.40,rf'$E_{i+1}$: 5×32',5,key=i+2)
    box(x,6.59,.55,.08,'none',lw=1.2)
arrow([(3.26,6.73),(3.37,6.73),(3.37,6.78),(3.48,6.78)])
line([(3.37,6.78),(3.37,7.02),(4.91,7.02)])
arrow([(4.14,7.02),(4.14,6.79),(4.25,6.79)])
arrow([(4.91,7.02),(4.91,6.79),(5.02,6.79)])
box(4.25,6.59,.55,.40,'none',lw=1.05)
eq_values=np.stack([.48+.31*np.sin(np.arange(4)*1.8+i+2) for i in range(3)])
tensor(6.17,6.59,.62,.42,r'$E_q$: 3×32',3,values=eq_values)
for i,x in enumerate([3.48,4.25,5.02]):
    yy=6.30+.06*i;xxend=6.06-.06*i
    arrow([(x+.55,6.63),(x+.60,6.63),(x+.60,yy),(xxend,yy),(xxend,6.66+.14*i),(6.17,6.66+.14*i)])
# Geometry is drawn directly below the sequence assembly.
geo_y=7.27
positions=np.array([0,3000,7000,10000]);coords=.34+2.10*positions/10000
centers=(coords[:-1]+coords[1:])/2
line([(coords[0],geo_y),(coords[-1],geo_y)],1.0)
for x,num in zip(coords,positions):
    line([(x,geo_y-.025),(x,geo_y+.025)])
    text(x,geo_y-.12,str(num))
for x,num in zip(centers,[1500,5000,8500]):
    ax.add_patch(Circle((x,geo_y),.015,facecolor=INK,zorder=6))
    text(x,geo_y+.12,str(num))
bracket(coords[1],coords[2],geo_y-.24,r'$L_2$',True)
bracket(centers[0],centers[1],geo_y+.24,r'$d_-$')
bracket(centers[1],centers[2],geo_y+.24,r'$d_+$')
raw=np.array([[3000,0,3500],[4000,3500,3500],[3000,3500,0]])
for j,name in enumerate([r'$L$',r'$d_-$',r'$d_+$']):text(2.89+j*.40,7.08,name)
for i,row in enumerate(raw):
    for j,value in enumerate(row):
        box(2.69+j*.40,7.19+i*.14,.40,.14,FIXED)
        text(2.89+j*.40,7.26+i*.14,str(value))
arrow([(2.44,7.27),(2.69,7.27)])
operator(4.08,7.20,1.22,.39,'log(1 + entries)\n÷ log(2000001)')
arrow([(3.89,7.40),(4.08,7.40)])
normalized=np.log1p(raw)/np.log(2000001)
grid(5.52,7.19,1.26,.42,3,3,kind='obs',values=normalized,numbers=True,digits=2)
text(6.15,7.07,r'$D$: 3×3')
arrow([(5.30,7.40),(5.52,7.40)])
# Compress only inter-band gutters. Text sizes and tensor channel counts stay fixed.
from matplotlib.transforms import Transform
class CloseGutters(Transform):
    input_dims=output_dims=2
    is_separable=True
    has_inverse=False
    def transform_non_affine(self, values):
        out=np.array(values,copy=True)
        y=out[...,1]
        y=y-np.clip((y-1.52)/.14,0,1)*.10
        y=y-np.clip((y-3.64)/.33,0,1)*.32
        y=y-np.clip((y-4.56)/.37,0,1)*.22
        out[...,1]=y
        return out
layout_transform=CloseGutters()+ax.transData
for artist in list(ax.patches)+list(ax.lines)+list(ax.texts):
    artist.set_transform(layout_transform)
# e: both GRU directions receive every input row and produce explicit 32-vectors.
panel('e',7.12)
port(.34,7.10,'d:E_q',w=.43)
port(.91,7.05,'d:D',w=.43)
grid(.34,7.36,.34,.51,3,4,values=eq_values)
grid(.68,7.36,.10,.51,3,3,kind='obs',values=normalized)
arrow([(.555,7.29),(.555,7.36)])
arrow([(1.125,7.24),(1.125,7.25),(.73,7.25),(.73,7.36)])
text(.56,7.99,'X: 3×35')
text(2.20,7.20,'same X')
grid(2.00,7.36,.34,.51,3,4,values=eq_values)
grid(2.34,7.36,.10,.51,3,3,kind='obs',values=normalized)
for i in range(3):
    cy=7.445+i*.17
    operator(.96,cy-.055,.42,.11,'GRU',True)
    operator(2.60,cy-.055,.42,.11,'GRU',True)
    arrow([(.78,cy),(.96,cy)])
    arrow([(2.44,cy),(2.60,cy)])
    grid(1.55,cy-.055,.30,.11,1,4,key=10+i)
    grid(3.19,cy-.055,.30,.11,1,4,key=14+i)
    arrow([(1.38,cy),(1.55,cy)])
    arrow([(3.02,cy),(3.19,cy)])
    if i<2:
        arrow([(1.17,cy+.055),(1.17,cy+.115)])
        arrow([(2.81,cy+.115),(2.81,cy+.055)])
text(1.17,7.32,'forward')
text(2.81,7.20,'reverse')
text(1.70,7.31,'32')
text(3.34,7.31,'32')
forward_values=np.stack([.48+.31*np.sin(np.arange(4)*1.8+10+i) for i in range(3)])
reverse_values=np.stack([.48+.31*np.sin(np.arange(4)*1.8+14+i) for i in range(3)])
grid(3.66,7.36,.25,.51,3,4,values=forward_values)
grid(3.91,7.36,.25,.51,3,4,values=reverse_values)
text(3.91,7.20,'H: 3×64')
line([(1.85,7.39),(1.90,7.39),(1.90,7.84),(1.85,7.84)])
arrow([(1.90,7.84),(1.90,7.985),(2.15,7.985)])
line([(3.49,7.39),(3.54,7.39),(3.54,7.84),(3.49,7.84)])
arrow([(3.54,7.84),(3.54,7.87),(3.40,7.87),(3.40,7.90)])
for i in range(3):
    cy=7.445+i*.17
    line([(1.85,cy),(1.90,cy)])
    line([(3.49,cy),(3.54,cy)])
operator(2.15,7.90,1.40,.17,'stack rows, concat')
arrow([(3.55,7.985),(3.91,7.985),(3.91,7.87)])
port(4.33,7.10,'d:E_q',w=.44)
grid(4.33,7.36,.17,.51,3,4,values=eq_values)
grid(4.50,7.36,.33,.51,3,8,values=np.concatenate([forward_values,reverse_values],axis=1))
arrow([(4.55,7.29),(4.55,7.32),(4.415,7.32),(4.415,7.36)])
arrow([(4.16,7.615),(4.25,7.615),(4.25,7.96),(4.66,7.96),(4.66,7.87)])
text(4.58,7.99,'3×96')
arrow([(4.83,7.615),(4.92,7.615),(4.92,7.295),(5.01,7.295)])
stack(5.01,7.18,.61,['Linear\n96→32','GELU','Linear\n32→4'])
arrow([(5.62,7.785),(5.73,7.785),(5.73,7.46),(5.86,7.46)])
tensor(5.86,7.36,.44,.21,'logits',3,cols=4,key=7)
operator(5.84,7.70,.48,.20,'softmax')
arrow([(6.08,7.57),(6.08,7.70)])
illustrative_logits=.48+.31*np.sin(np.arange(12).reshape(3,4)*1.8+7)
probabilities=np.exp(illustrative_logits)/np.exp(illustrative_logits).sum(1,keepdims=True)
tensor(6.51,7.36,.34,.51,'P: 3×4',3,cols=4,kind='prob',values=probabilities)
for j,name in enumerate(['A','C','W','E']):text(6.51+(j+.5)*.34/4,7.97,name)
arrow([(6.32,7.80),(6.42,7.80),(6.42,7.615),(6.51,7.615)])
box(4.83,3.695,2.06,.875,'none',edge='#a7b1b4',lw=.45)
for yy,col,label in [(3.72,GOLD,'learned'),(3.86,FIXED,'fixed'),(4.00,EMB(.60),'embedding / logits'),(4.14,OBS(.40),'observed'),(4.28,PROB(.6),'probability')]:
    box(4.92,yy,.14,.075,col)
    text(5.12,yy+.037,label,ha='left')
text(4.92,4.50,'A=Ana  C=CHG  W=WHG  E=EHG',ha='left')
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
import json
boxes=[(t,t.get_window_extent(renderer)) for t in labels if t.get_text()]
overlap=[]
for i,(ta,ba) in enumerate(boxes):
    for tb,bb in boxes[i+1:]:
        x=min(ba.x1,bb.x1)-max(ba.x0,bb.x0)
        y=min(ba.y1,bb.y1)-max(ba.y0,bb.y0)
        if x>.35 and y>.35:
            overlap.append({'a':ta.get_text(),'b':tb.get_text(),'area_px':round(x*y,2)})
(OUT/'architecture_layout_qa.json').write_text(json.dumps({'min_font_pt':min(t.get_fontsize() for t in labels),'text_overlaps':overlap},indent=2)+'\n')
print(json.dumps({'overlap_count':len(overlap),'overlaps':overlap}))
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/('TreeLoom_architecture.'+ext),dpi=240,facecolor='white')
p=OUT/'TreeLoom_architecture.svg'
p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
print('Generated connected input and readout tensors from preserved illustration. No training or simulation.')
