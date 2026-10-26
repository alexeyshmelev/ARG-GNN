"""Two conceptual figures. Read saved illustrative trees, never run a model."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import hashlib,json
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
INK='#26363b'; LEARN='#f8e6b8'; LEARN_EDGE='#8d6826'; FIX='#f5f6f6'
EMB=LinearSegmentedColormap.from_list('embedding',['#e7f6f8','#63b9c6','#145965'])
PROB=LinearSegmentedColormap.from_list('probability',['#f7f4fa','#bb94d0','#582571'])
OBS=LinearSegmentedColormap.from_list('observed',['#fafafa','#a8b0b6','#34434a'])
ANC=['#0072b2','#e69f00','#009e73','#cc79a7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42})
texts=[]
DY=0
def canvas(h,ph):
 global fig,ax,texts,DY
 DY=0
 texts=[];fig=plt.figure(figsize=(7,ph));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,14),ylim=(h,0));ax.axis('off')
def text(x,y,s,size=7,ha='center',bold=False):
 texts.append(ax.text(x,y+DY,s,ha=ha,va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=INK,zorder=20))
def line(ps,lw=.75,style='-'):
 ax.plot(*zip(*[(x,y+DY) for x,y in ps]),color=INK,lw=lw,ls=style,zorder=2)
def arrow(ps,lw=.75,both=False,style='-'):
 if len(ps)>2:line(ps[:-1],lw,style)
 ax.add_patch(FancyArrowPatch((ps[-2][0],ps[-2][1]+DY),(ps[-1][0],ps[-1][1]+DY),arrowstyle='<->' if both else '-|>',mutation_scale=6,color=INK,lw=lw,shrinkA=0,shrinkB=0,zorder=3))
def block(x,y,w,h,s,learn=False,size=7,frame=False):
 ax.add_patch(FancyBboxPatch((x,y+DY),w,h,boxstyle='round,pad=0,rounding_size=.055',facecolor=LEARN if learn else FIX,edgecolor=LEARN_EDGE if learn else INK,lw=.85,zorder=0 if frame else 4))
 if s:text(x+w/2,y+h/2,s,size)
def matrix(x,y,w,h,r=3,c=5,key=0,kind='emb',values=None,row_names=None,col_names=None,highlight=None):
 vals=np.clip(.45+.36*np.sin(np.arange(r*c).reshape(r,c)*1.73+key),0,1) if values is None else np.asarray(values)
 cmap={'emb':EMB,'prob':PROB,'obs':OBS}[kind]
 for i in range(r):
  for j in range(c):
   ax.add_patch(Rectangle((x+w*j/c,y+h*i/r+DY),w/c,h/r,facecolor=cmap(float(vals[i,j])),edgecolor='white',lw=.4,zorder=6))
 ax.add_patch(Rectangle((x,y+DY),w,h,fill=False,edgecolor=INK,lw=.65,zorder=7))
 if highlight is not None:ax.add_patch(Rectangle((x,y+h*highlight/r+DY),w,h/r,fill=False,edgecolor=INK,lw=1.2,zorder=8))
 if row_names:
  for i,s in enumerate(row_names):text(x-.12,y+(i+.5)*h/r,s,6.2,ha='right')
 if col_names:
  for j,s in enumerate(col_names):text(x+(j+.5)*w/c,y-.18,s,6.5)
def leaf(x,y,j):ax.add_patch(Circle((x,y+DY),.042,facecolor='white' if j==0 else ANC[j-1],edgecolor=INK,lw=.5,zorder=7))
def tree(i,cx,base,w=1.2,h=.8,labels=False,cut=False,colored=True):
 t=trees[i];xp={u:cx-w/2+w*u/4 for u in range(5)}
 def xx(u):
  if u not in xp:xp[u]=np.mean([xx(v) for v in t.children(u)])
  return xp[u]
 xx(t.root);yp={u:base-h*ts.node(u).time/725 for u in t.nodes()}
 for u in t.nodes():
  p=t.parent(u)
  if p>=0:line([(xp[u],yp[u]),(xp[u],yp[p]),(xp[p],yp[p])],.6)
  if u<5:
   leaf(xp[u],yp[u],u if colored else 0)
   if labels:text(xp[u],base+.18,['q','r1','r2','r3','r4'][u],5.7)
  else:ax.add_patch(Circle((xp[u],yp[u]+DY),.022,facecolor=INK,zorder=5))
 if cut:
  yy=base-h*350/725;line([(cx-w*.58,yy),(cx+w*.58,yy)],.6,'--');text(cx-w*.60,yy,'350',6,ha='right')
  reps=set()
  for sample in ts.samples():
   u=int(sample)
   if ts.node(u).time<=350:
    while t.parent(u)>=0 and ts.node(t.parent(u)).time<=350:u=t.parent(u)
   reps.add(u)
  for u in reps:
   ax.add_patch(Rectangle((xp[u]-.065,yp[u]-.065+DY),.13,.13,fill=False,edgecolor=INK,lw=.85,zorder=8))
 return xp,yp
def badge(x,y,label):
 ax.add_patch(Circle((x,y+DY),.13,facecolor=INK,zorder=8));text(x,y,label,7,bold=True)
 # White labels need contrast on the matching parent marker.
 texts[-1].set_color('white')
def legend(x,y,w=7.7,ancestry=True):
 line([(x,y),(x+w,y)],.55)
 block(x,y+.15,.34,.19,'',True);text(x+.45,y+.25,'Learned layer',6.4,ha='left')
 block(x+2.6,y+.15,.34,.19,'');text(x+3.05,y+.25,'Fixed operation',6.4,ha='left')
 matrix(x,y+.53,.65,.16,1,7,kind='emb');text(x+.78,y+.61,'Embedding values',6.4,ha='left')
 matrix(x+3.4,y+.53,.65,.16,1,7,kind='obs');text(x+4.18,y+.61,'Observed features',6.4,ha='left')
 matrix(x,y+.91,.65,.16,1,7,kind='prob',values=np.linspace(0,1,7)[None]);text(x+.78,y+.99,'Probability: 0 to 1',6.4,ha='left')
 if ancestry:
  for i,(name,c) in enumerate(zip(['Ana','CHG','WHG','EHG'],ANC)):
   xx=x+i*1.40;leaf(xx+.04,y+1.38,i+1);text(xx+.17,y+1.38,name,6.4,ha='left')
  text(x,y+1.73,'Ana: Anatolian farmer    CHG: Caucasus hunter-gatherer',6,ha='left')
  text(x,y+2.02,'WHG: Western hunter-gatherer    EHG: Eastern hunter-gatherer',6,ha='left')
def save(name):
 fig.canvas.draw();ren=fig.canvas.get_renderer()
 for t in texts:
  b=t.get_window_extent(ren).transformed(fig.transFigure.inverted())
  assert b.x0>=-.003 and b.y0>=-.003 and b.x1<=1.003 and b.y1<=1.003,t.get_text()
 for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=230,facecolor='white')
 p=OUT/(name+'.svg');p.write_text('\n'.join(z.rstrip() for z in p.read_text().splitlines())+'\n')
 plt.close(fig)

# Figure 1: observed inputs -> all-node encoder -> leaf sequences -> probabilities.
canvas(12.7,6.35)
text(.16,.18,'a',10,bold=True)
ax.add_patch(Rectangle((.36,.05),3.98,1.54,fill=False,edgecolor='#a8b0b6',lw=.5,zorder=0))
for i,cx in enumerate([.95,2.35,3.75]):
 text(cx,.23,'$T_'+str(i+1)+'$',7)
 tree(i,cx,1.20,1.06,.77)
text(2.35,1.45,'Inferred trees + node ages',6.9)
text(6.05,.20,'Observed variants',7)
matrix(5.13,.48,1.85,.65,3,5,kind='obs',values=ts.genotype_matrix())
text(6.05,1.40,'Allele / frequency / callability',6.1)
text(9.15,.20,'Reference ancestry',7)
for j in range(5):leaf(8.35+j*.4,.73,j)
text(8.35,1.03,'?',7);text(9.40,1.24,'Query labels masked',6.2)
text(12.15,.20,'Sampling ages',7)
matrix(11.4,.62,1.5,.25,1,5,kind='obs',values=np.array([[0,.3,.5,.2,.7]]))
text(12.15,1.23,'One value per leaf',6.2)
# Feature rows are formed separately for all nodes and sampled leaves.
arrow([(2.35,1.62),(2.35,1.85)])
block(.66,1.85,3.38,.58,'Node MLP → 32\nAge / root / leaf indicators',True,6.2)
arrow([(6.05,1.62),(6.05,1.85)])
block(4.65,1.85,3.50,.82,'Site MLP → pool → Linear\nMean/max of called sites\nPer interval, per leaf → 32',True,6.0)
arrow([(8.15,2.26),(9.62,2.26)])
arrow([(9.15,1.45),(9.15,1.65),(10.55,1.65),(10.55,1.93)])
arrow([(12.15,1.45),(12.15,1.93)])
block(9.62,1.93,3.27,.66,'Leaf MLP → 32',True,6.8)
arrow([(2.35,2.43),(2.35,3.02)])
arrow([(11.25,2.59),(11.25,3.29),(4.03,3.29)])
text(7.30,3.08,'Encoded leaf features',6.3)
block(.67,3.02,3.36,.55,'Add only to leaf rows',False,6.6)
arrow([(2.35,3.57),(2.35,4.48)])
# Separate tree and variant inputs explicitly form learned edge gates.
arrow([(4.34,.80),(4.43,.80),(4.43,3.82),(4.91,3.82)])
arrow([(5.13,1.00),(4.59,1.00),(4.59,4.00),(4.91,4.00)])
# White underpasses mark crossings with the returning leaf-feature line.
for xx in [4.43,4.59]:ax.plot([xx,xx],[3.20,3.38],color='white',lw=2.6,zorder=3)
ax.plot([4.72,4.30],[3.29,3.29],color=INK,lw=.75,zorder=4)
block(4.91,3.59,3.50,.61,'Edge gates\nBranch geometry + allele agreement',True,5.9)
arrow([(5.08,4.20),(5.08,4.36),(4.05,4.36),(4.05,4.54)])
DY=1.6
text(.16,2.22,'b',10,bold=True)
matrix(1.93,2.90,.84,.76,4,5,key=1,row_names=['q','r1','…','ancestor'])
text(2.35,3.89,'$B_t$: all nodes × 32',6.5)
arrow([(2.77,3.28),(3.13,3.28)])
block(3.13,2.94,1.83,.68,'Tree encoder\n2 rounds · Fig. 2',True,6.8)

arrow([(4.96,3.28),(5.30,3.28)])
matrix(5.32,2.90,.51,.76,4,3,key=2);matrix(5.91,2.90,.51,.76,4,3,key=3)
text(5.58,2.74,'$U_t$',6.8);text(6.17,2.74,'$O_t$',6.8)
text(5.87,3.89,'All nodes × 32 each',6.3)
arrow([(6.42,3.28),(6.75,3.28)])
block(6.75,2.87,1.88,.82,'Select leaf rows\nConcat: 32 + 32',False,6.5)
arrow([(8.63,3.28),(8.92,3.28)])
block(8.92,2.94,1.60,.68,'Linear 64 → 32\ntanh',True,6.4)
arrow([(10.52,3.28),(10.82,3.28)])
# A per-tree leaf matrix retains its row identities through the selection.
matrix(10.84,2.9,1.30,.76,3,5,key=4,highlight=0)
for yy,lab in [(3.03,'q'),(3.28,'r1'),(3.53,'…')]:text(12.26,yy,lab,6.2,ha='left')
text(11.49,2.64,'$E_t$',7);text(12.45,3.91,'Leaves × 32',6.2)
arrow([(11.49,3.66),(11.49,4.16)])
block(9.72,4.16,3.55,.55,'Same leaf q in every tree → stack rows',False,6.3)
arrow([(11.49,4.71),(11.49,5.00)])
matrix(10.64,5.03,1.70,.75,3,6,key=5,row_names=['$T_1$','$T_2$','$T_3$'])
text(10.23,6.02,'$E_q$: trees × 32',6.2,ha='right')
# Geometry has the same row order, with every column visibly named.
text(.16,4.53,'c',10,bold=True)
x=[.62,1.58,2.86,3.82];cy=5.23
for i in range(3):
 line([(x[i],cy),(x[i+1],cy)],1.9);line([(x[i],cy-.07),(x[i],cy+.07)],.6)
 text((x[i]+x[i+1])/2,cy-.28,'$T_'+str(i+1)+'$',6.8)
line([(x[-1],cy-.07),(x[-1],cy+.07)],.6)
centers=[(x[i]+x[i+1])/2 for i in range(3)]
for xx in centers:ax.add_patch(Circle((xx,cy+DY),.027,facecolor=INK))
line([(x[1],4.79),(x[1],4.67),(x[2],4.67),(x[2],4.79)])
text((x[1]+x[2])/2,4.47,'Length $L_2$',6.5)
arrow([(centers[0],5.59),(centers[1],5.59)],both=True);text((centers[0]+centers[1])/2,5.80,'Previous $d_-$',6.3)
arrow([(centers[1],5.59),(centers[2],5.59)],both=True);text((centers[1]+centers[2])/2,5.80,'Next $d_+$',6.3)
arrow([(3.98,5.23),(4.35,5.23)])
block(4.35,4.98,1.08,.50,'log(1+x)',False,5.9)
lengths=np.array([t.interval.span for t in trees]);raw=np.array([[lengths[t],(lengths[t-1]+lengths[t])/2 if t else 0,(lengths[t]+lengths[t+1])/2 if t<2 else 0] for t in range(3)])
geom=np.log1p(raw)/np.log(2000001)
arrow([(5.43,5.23),(5.79,5.23)])
matrix(5.82,4.89,1.44,.75,3,3,kind='obs',values=geom,col_names=['$L$','$d_-$','$d_+$'])
for i in range(3):text(7.40,5.015+i*.25,'$T_'+str(i+1)+'$',6.2,ha='left')
text(6.54,5.87,'Trees × 3',6.4)
for row,col in [(0,1),(2,2)]:text(5.82+(col+.5)*.48,4.89+(row+.5)*.25,'0',6)
text(3.85,6.53,'Length and adjacent center distances (bp)\nScale by log(2000001). Missing neighbor → 0',6.6)
# Sequence assembly is shown as actual adjacent columns, not an unexplained wire.
arrow([(6.54,5.64),(6.54,6.04),(8.07,6.04),(8.07,6.46),(10.24,6.46)])
arrow([(11.49,5.78),(11.49,6.20)])
block(10.24,6.20,2.50,.52,'Append columns: 32 + 3',False,6.5)
arrow([(11.49,6.72),(11.49,6.99)])
matrix(10.49,7.02,1.55,.57,3,6,key=5);matrix(12.04,7.02,.45,.57,3,3,kind='obs',values=geom)
text(10.24,7.81,'Trees × 35',6.2,ha='right')
arrow([(11.49,7.59),(11.49,8.06)])
block(10.24,8.06,2.50,.53,'BiGRU · 1 layer · 35 → 64',True,6.4)
arrow([(11.49,8.59),(11.49,8.85)])
block(10.24,8.85,2.50,.50,'Concat: local 32 + history 64',False,6.1)
# Direct local embedding skip, visibly anchored to the sequence matrix.
line([(12.34,5.40),(13.59,5.40),(13.59,9.10)])
arrow([(13.59,9.10),(12.74,9.10)])
text(13.23,7.0,'$E_q$',6.7)
arrow([(11.49,9.35),(11.49,9.60)])
block(9.69,9.60,3.60,.55,'Linear 96 → 32 · GELU · Linear 32 → 4',True,5.9)
arrow([(11.49,10.15),(11.49,10.36),(10.40,10.36),(10.40,10.64),(10.64,10.64)])
text(10.18,10.46,'Softmax',6.4,ha='right')
prob=np.array([[.62,.12,.14,.12],[.17,.16,.53,.14],[.11,.16,.21,.52]])
matrix(10.64,10.40,1.70,.48,3,4,kind='prob',values=prob,)
for j,lab in enumerate(['Ana','CHG','WHG','EHG']):text(10.64+(j+.5)*.425,11.00,lab,5.8)
# Compact key occupies otherwise unused space, not an extra full-width panel.
legend(.48,7.17,7.45)
text(.48,10.37,'q: query leaf     Each sampled leaf represents one haplotype.',6.2,ha='left')
text(.48,10.73,'One ancestry distribution per query leaf and interval.',6.5,ha='left')
save('TreeLoom_architecture')

# Figure 2: parent-tagged encoder with two full rounds and downward context flow.
canvas(11.9,5.95)
text(.15,.20,'a',10,bold=True)
text(.53,.25,'Tree encoder from Fig. 1b',7.4,ha='left')
text(8.40,.25,'Round 1',7.3)
matrix(.63,.83,.67,.61,4,4,key=1);text(.96,.61,'$B_t$',7);text(2.72,1.72,'All nodes × 32',6)
arrow([(1.30,1.14),(1.62,1.14)])
block(1.62,.76,2.38,.76,'Upward message passing\nGated mean of children',True,6.6)
arrow([(4,1.14),(4.33,1.14)]);matrix(4.35,.83,.65,.61,4,4,key=2);text(4.68,.61,'$U^1_t$',7)
arrow([(5,1.14),(5.32,1.14)])
block(5.32,.76,2.60,.76,'Downward message passing\nParent + other branches',True,6.5)
arrow([(7.92,1.14),(8.23,1.14)]);matrix(8.25,.83,.65,.61,4,4,key=3);text(8.57,.61,'$O^1_t$',7)
text(6.61,1.72,'Root outside state = 0',6.1)
line([(4.68,1.44),(4.68,1.94),(4.47,1.94),(9.70,1.94)])
line([(8.57,1.44),(8.57,1.94)])
# Original node features descend along a dedicated outside path, never a return wire.
line([(.96,1.44),(.96,1.94),(.33,1.94),(.33,9.20),(5.82,9.20)])
arrow([(5.82,9.20),(6.14,9.20)])
text(.47,8.73,'$B_t$',6.8,ha='left')
# Memory context from selected representatives, with a compact biological example.
text(.15,2.35,'b',10,bold=True)
xp,yp=tree(1,1.78,3.65,w=1.84,h=1.28,labels=True,cut=True,colored=False)
text(1.78,2.16,'$T_2$',7)
line([(1.29,3.98),(1.29,4.08),(1.82,4.08),(1.82,3.98)])
text(1.55,4.28,'Group: r1, r2',6.3)
text(1.80,4.52,'□ group ancestor',6.4)
text(1.80,4.77,'Highest ancestor ≤ 350 generations',5.5)
text(1.80,5.04,'Older leaf → singleton group',5.8)
arrow([(2.82,3.08),(3.24,3.08)])
block(3.24,2.76,2.45,.64,'Select group ancestors\nSum $U^1 + O^1$',False,6.5)
arrow([(4.47,1.94),(4.47,2.76)])
arrow([(5.69,3.08),(6.03,3.08)])
block(6.03,2.76,1.58,.64,'Linear 32 → 8',True,6.5)
arrow([(7.61,3.08),(7.91,3.08)]);matrix(7.94,2.97,.80,.22,1,6,key=6)
text(8.34,2.70,'$f_{t,g}$',7);text(8.34,3.44,'Group context',6.4)
arrow([(8.34,3.19),(8.34,3.70),(5.72,3.70),(5.72,3.94)])
block(3.90,3.94,3.65,.52,'Copy to group leaves · stack across trees',False,6.3)
arrow([(5.72,4.46),(5.72,4.74)])
matrix(5.22,4.77,1.00,.60,3,6,key=6,row_names=['$T_1$','$T_2$','$T_3$'],highlight=1)
text(7.90,4.99,'$F_q$: trees × 8\nOne sequence per leaf',6.4)
# The two scan directions live within one downward stage.
line([(5.72,5.37),(5.72,5.57),(4.45,5.57)])
arrow([(4.45,5.57),(4.45,5.78)])
arrow([(5.72,5.57),(7.15,5.57),(7.15,5.78)])
block(3.37,5.78,2.17,.48,'Forward memory →',False,6.4)
block(6.07,5.78,2.17,.48,'← Reverse memory',False,6.4)
text(1.90,5.39,r'$m_t=\alpha_t m_{\rm previous}$',6.9)
text(1.90,5.73,r'$+(1-\alpha_t)f_{t,g(q)}$',6.9)
text(1.90,6.15,r'$\alpha_t=e^{-L_t/\ell}$',6.9)
block(.96,6.46,1.90,.45,'Learned scale $\ell$',True,6.4)
text(1.91,7.16,'$L_t$: interval length (bp)',6.1)
text(1.91,7.57,'First state = current context',5.8)
# Outputs align by interval rather than by scan order.
arrow([(4.45,6.26),(4.45,7.21)]);arrow([(7.15,6.26),(7.15,7.21)])
matrix(4.07,7.23,.77,.53,3,5,key=9);matrix(6.77,7.23,.77,.53,3,5,key=10)
text(3.58,7.48,'$T_1$\n$T_2$\n$T_3$',5.9);text(7.98,7.48,'$T_1$\n$T_2$\n$T_3$',5.9)
arrow([(4.45,7.76),(4.45,8.02),(4.87,8.02)])
arrow([(7.15,7.76),(7.15,8.02),(6.65,8.02)])
block(4.87,7.81,1.78,.42,'Mean directions',False,6.1)
# Current-group leaf mean is not a second directional mean.
arrow([(5.76,8.23),(5.76,8.38),(3.07,8.38),(3.07,8.51)])
matrix(2.68,8.54,.78,.36,2,5,key=15,row_names=['r1','r2'])
text(1.85,8.18,'At $T_2$, select group rows',5.8)
arrow([(3.46,8.72),(3.78,8.72)])
block(3.78,8.47,4.22,.48,'Mean group leaves − current context $f_{t,g}$',False,5.9)
arrow([(8,8.71),(8.35,8.71)])
block(8.35,8.47,1.60,.48,'Linear 8 → 32',True,6.3)
arrow([(9.95,8.71),(10.25,8.71)]);matrix(10.27,8.60,.70,.22,1,5,key=11);text(10.62,8.35,r'$\Delta_t$',7)
# Structural branch: local states, flags from neighboring topology, no hidden-state transport.
text(10.02,2.35,'c',10,bold=True)
arrow([(9.70,1.94),(11.65,1.94),(11.65,2.69)])
block(10.17,2.69,3.10,.55,'Select current upward state $U^1_t$',False,6.7)
text(11.72,3.62,'Match descendant leaf sets',6.3)
text(11.72,3.91,'Presence in adjacent trees → flags',5.8)
for j,label in enumerate(['$T_{t-1}$','$T_t$','$T_{t+1}$']):text(10.29+j*.53,4.24,label,5.9)
text(11.65,4.24,'→',6.5)
for j,label in enumerate(['$g$','$l$','$p$']):text(12.04+j*.47,4.24,label,6.7)
for i,(row,flags) in enumerate([([0,1,1],[1,0,1]),([1,1,0],[0,1,1]),([1,1,1],[0,0,1]),([0,1,0],[1,1,0])]):
 for j,value in enumerate(row):text(10.29+j*.53,4.53+i*.29,str(value),6.5)
 for j,value in enumerate(flags):text(12.04+j*.47,4.53+i*.29,str(value),6.5)
text(11.72,5.76,'g: gain   l: loss   p: persistence',5.7)
text(11.72,6.00,'p = 1 for ≥ 2 consecutive trees',5.8)
arrow([(11.72,3.24),(13.55,3.24),(13.55,6.63),(13.29,6.63)])
arrow([(11.72,6.16),(11.72,6.30)])
block(10.12,6.30,3.17,.70,'Nonlinear factored maps of $u$\n32 → 8 → 32',True,6.2)
arrow([(11.72,7.00),(11.72,7.16)])
block(10.12,7.16,3.17,.63,'$S_v=gG(u)-lL(u)+pP(u)$\nAt this clade’s node row',False,6.2)
arrow([(11.72,7.79),(11.72,8.12)])
matrix(11.32,8.14,.80,.24,1,5,key=12);text(12.40,8.26,'$S_t$',7)
# Both branches converge only after all corrections are formed.
arrow([(10.62,8.82),(10.62,9.31),(8.84,9.31)])
arrow([(11.72,8.38),(11.72,9.51),(8.84,9.51)])
block(6.14,9.03,2.70,.72,r'$B^\prime_t=B_t+S_t+\Delta_t$',False,7.1)
text(3.50,9.64,r'$\Delta_t$: only at group ancestors',6.1)
# Second round is below the feedback, with an explicit return to the parent figure.
text(.15,10.16,'d',10,bold=True)
text(3.18,10.21,'Round 2 · same weights as round 1',6.2)
arrow([(7.49,9.75),(7.49,10.03),(1.11,10.03),(1.11,10.36)])
matrix(.76,10.39,.70,.62,4,4,key=13);text(1.11,11.23,"$B'_t$",6.8)
arrow([(1.46,10.70),(1.78,10.70)])
block(1.78,10.36,2.25,.68,'Upward\nmessage passing',True,6.6)
arrow([(4.03,10.70),(4.35,10.70)]);matrix(4.38,10.39,.64,.62,4,4,key=14);text(4.15,11.23,'$U^2_t$',6.8)
arrow([(5.02,10.70),(5.34,10.70)])
block(5.34,10.36,2.25,.68,'Downward\nmessage passing',True,6.6)
arrow([(7.59,10.70),(7.90,10.70)]);matrix(7.93,10.39,.64,.62,4,4,key=15);text(8.25,11.23,'$O^2_t$',6.8)
line([(4.70,11.01),(4.70,11.46),(9.40,11.46)])
arrow([(9.40,11.46),(9.40,10.70),(9.75,10.70)])
arrow([(8.57,10.70),(9.75,10.70)])
block(9.75,10.34,3.43,.72,'All-node $U^2_t,O^2_t$ → Fig. 1b\nSelect leaf rows',False,6.7)
block(.55,11.63,.30,.16,'',True);text(.96,11.72,'Learned layer',5.8,ha='left')
block(3.08,11.63,.30,.16,'');text(3.48,11.72,'Fixed operation',5.8,ha='left')
matrix(6.2,11.63,.57,.16,1,6);text(6.90,11.72,'Embedding values',5.8,ha='left')
text(10.42,11.72,'Every matrix row retains its node or leaf identity.',5.6)
save('TreeLoom_encoder')
(OUT/'generation.json').write_text(json.dumps({'source_tree_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'figures':['TreeLoom_architecture','TreeLoom_encoder'],'simulations_or_training':False,'hidden_values_and_probabilities':'illustrative'},indent=2)+'\n')
print('Rendered two figures without model or simulation jobs.')
