"""Original editable TreeLoom architecture figure; illustrative data only.

Run with the existing project Python. No experiment data, model weights, truth,
model execution, network service, or GPU is used. Every output is colocated.
"""
from pathlib import Path
import hashlib, json, os
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / '.mpl-cache')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'mathtext.fontset':'dejavusans',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,
 'savefig.facecolor':'white','savefig.transparent':False})
INK='#223246'; MUTED='#526276'; LINE='#8b99a9'; PALE='#f5f8fc'
BLUE='#cfe3f7'; TEAL='#d4eee8'; PURPLE='#e3ddf5'; PEACH='#f7dfca'; YELLOW='#f5ecc8'
SAMPLE=['#247ea9','#d7873d','#8d67b0','#4a9a78']
CLASSES=['#5b9bc6','#cc9470','#9a85bd','#77a894']

# A tiny actual fully recorded ARG, chosen for a readable 3-tree illustration.
# This bounded seed search is illustrative simulation only, with no fitted model.
import msprime
search=[]
for seed in range(1,257):
 arg=msprime.sim_ancestry(samples=4,ploidy=1,population_size=1000,
  sequence_length=10000,recombination_rate=4e-8,model='hudson',
  record_full_arg=True,random_seed=seed)
 display=arg.simplify(keep_unary=False,record_provenance=False)
 clades=[tuple(sorted(tuple(sorted(t.samples(u))) for u in t.nodes()
  if 1<len(list(t.samples(u)))<4)) for t in display.trees()]
 maxage=max(n.time for n in display.nodes())
 lengths=[display.node(t.parent(u)).time-display.node(u).time
  for t in display.trees() for u in t.nodes() if u>=4 and t.parent(u)!=-1]
 accepted=(display.num_trees==3 and arg.num_trees==3
  and min(t.span for t in display.trees())>=1200
  and len(set(clades))==3 and min(lengths)/maxage>.07)
 search.append({'seed':seed,'num_trees':arg.num_trees,'accepted':accepted})
 if accepted: break
else: raise RuntimeError('No illustrative ARG matched the fixed display criteria')
# Add three hypothetical observed biallelic sites on real branches.
tables=arg.dump_tables()
for i,tree in enumerate(arg.trees()):
 choices=[]
 for u in tree.nodes():
  desc=tuple(sorted(tree.samples(u)))
  if 1<len(desc)<4 and len(tree.children(u))>=2: choices.append((desc,u))
 choices.sort()
 desc,node=choices[i%len(choices)]
 position=(tree.interval.left+tree.interval.right)/2
 site=tables.sites.add_row(position,ancestral_state='0')
 tables.mutations.add_row(site=site,node=node,derived_state='1')
tables.sort(); full_arg=tables.tree_sequence()
ts,node_map=full_arg.simplify(keep_unary=False,record_provenance=False,map_nodes=True)
assert full_arg.num_trees==ts.num_trees==3 and ts.num_samples==4
assert list(full_arg.breakpoints())==list(ts.breakpoints())
for tree in ts.trees():
 assert tree.num_roots==1 and len(list(tree.nodes()))==7
 assert set(tree.samples())=={0,1,2,3}
 for u in tree.nodes():
  if tree.parent(u)!=tskit.NULL: assert ts.node(tree.parent(u)).time>ts.node(u).time
for tree in full_arg.trees():
 assert tree.num_roots==1
 for u in tree.nodes():
  if tree.parent(u)!=tskit.NULL:assert full_arg.node(tree.parent(u)).time>full_arg.node(u).time
assert np.array_equal(full_arg.genotype_matrix(),ts.genotype_matrix())
full_arg.dump(OUT/'illustrative_full_arg.trees')
ts.dump(OUT/'illustrative_tree_sequence.trees')
TIME_MAX=max(n.time for n in full_arg.nodes())
recombination_nodes=[n.id for n in full_arg.nodes() if n.flags&msprime.NODE_IS_RE_EVENT]
assert len(recombination_nodes)==4
simulation={'engine':'msprime','version':msprime.__version__,'model':'hudson',
 'samples':4,'ploidy':1,'population_size':1000,'sequence_length':10000,
 'recombination_rate':4e-8,'record_full_arg':True,'selected_seed':seed,
 'maximum_search_seed':256,'seeds_examined':len(search),
 'selection_criteria':'3 native intervals, each >=1200bp, 3 distinct proper-clade sets, minimum internal branch >7% of maximum time',
 'seed_search':search,'mutation_policy':'one hypothetical observed mutation on an actual proper-clade branch per interval'}
toy={'purpose':'Tiny illustrative simulated ARG and hypothetical observed alleles, not an experiment or population-history inference',
 'simulation':simulation,'representation':'fully recorded ARG with real paired recombination nodes; unary nodes collapsed only in extracted display trees',
 'sequence_length_bp':full_arg.sequence_length,'breakpoints_bp':list(full_arg.breakpoints()),
 'sample_ids':[0,1,2,3],'sample_names':['q1','q2','q3','q4'],
 'recombination_nodes':recombination_nodes,'display_node_map':node_map.tolist(),
 'nodes':[{'id':u.id,'time_generations':u.time,'flags':u.flags,'is_sample':u.is_sample()} for u in full_arg.nodes()],
 'edges':[{'left':e.left,'right':e.right,'parent':e.parent,'child':e.child} for e in full_arg.edges()],
 'sites':[{'position':v.site.position,'alleles':list(v.alleles),'genotypes':v.genotypes.tolist()} for v in full_arg.variants()],
 'trees':[{'index':t.index,'interval_bp':list(t.interval),'parent':{str(u):t.parent(u) for u in t.nodes()},
  'newick':t.as_newick(node_labels={u:f'q{u+1}' for u in ts.samples()})} for t in ts.trees()]}
(OUT/'illustrative_tree_sequence.json').write_text(json.dumps(toy,indent=2)+'\n')

W,H=7.2,7.55
fig=plt.figure(figsize=(7.0,8.15)); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(0,H)); ax.axis('off')
texts=[]
def text(x,y,s,size=8.2,weight='normal',color=INK,ha='center',va='center',**kw):
 t=ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,**kw); texts.append(t); return t

def box(x,y,w,h,label='',fc=PALE,ec=LINE,lw=.8,rounding=.055,size=8.2,weight='normal',z=2):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.015,rounding_size={rounding}',
  linewidth=lw,edgecolor=ec,facecolor=fc,zorder=z); ax.add_patch(p)
 if label: text(x+w/2,y+h/2,label,size,weight,zorder=z+1)
 return p

def arrow(a,b,color=INK,lw=.85,style='-|>',ms=7,connectionstyle='arc3',z=4):
 p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,linewidth=lw,color=color,
  connectionstyle=connectionstyle,zorder=z,shrinkA=0,shrinkB=0);ax.add_patch(p);return p

def route(points,color=INK,lw=.85,style='-|>',ms=7,z=4):
 for a,b in zip(points[:-2],points[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,zorder=z)
 return arrow(points[-2],points[-1],color,lw,style,ms,z=z)

def panel_label(letter,title,y):
 box(.16,y-.105,.24,.23,fc=INK,ec=INK,size=9,weight='bold')
 text(.28,y+.01,letter,9,weight='bold',color='white',zorder=5)
 text(.50,y+.01,title,size=10.2,weight='bold',ha='left')

def matrix(x,y,w,h,rows=4,cols=4,colors=None,stack=True):
 if stack:
  for k in (2,1): ax.add_patch(Rectangle((x+k*.035,y+k*.035),w,h,fc='#edf1f6',ec=LINE,lw=.5,zorder=1))
 ax.add_patch(Rectangle((x,y),w,h,fc='white',ec=INK,lw=.7,zorder=2))
 for r in range(rows):
  col=colors[r] if colors else BLUE
  for c in range(cols):
   ax.add_patch(Rectangle((x+c*w/cols,y+(rows-1-r)*h/rows),w/cols,h/rows,
    fc=col,ec='white',lw=.45,alpha=.4+.1*((r+c)%4),zorder=3))

panel_label('A','Observed genealogy, persistent samples and input features',7.31)
# First panel generous boundary and section header.
text(1.88,6.99,'A fully recorded illustrative ARG',9,weight='bold')
text(5.43,6.99,'Ordered marginal trees',9,weight='bold')

# The sheets are generated directly from each valid tskit edge interval.
def proj(g,s,t): return np.array([.43+2.46*g/10000+.17*s,5.43+1.24*t/TIME_MAX+.104*s])
# axes and subtle interval floor planes
for i,(l,r) in enumerate(zip(list(ts.breakpoints())[:-1],list(ts.breakpoints())[1:])):
 floor=[proj(l,0,0),proj(r,0,0),proj(r,3,0),proj(l,3,0)]
 ax.add_patch(Polygon(floor,fc=['#e9f3fb','#f5efe4','#ece8f5'][i],ec='none',alpha=.5,zorder=.2))
# Draw back layers first. Each sheet corresponds to one actual edge over an exact interval.
for tree in full_arg.trees():
 desc={u:tuple(tree.samples(u)) for u in tree.nodes()}
 depth={u:float(np.mean(desc[u])) for u in tree.nodes()}
 for child in sorted(tree.nodes(),key=lambda u:depth[u],reverse=True):
  parent=tree.parent(child)
  if parent==tskit.NULL:continue
  l,r=tree.interval; tc=full_arg.node(child).time;tp=full_arg.node(parent).time
  dc,dp=depth[child],depth[parent]
  samples=desc[child]
  col=SAMPLE[samples[0]] if len(samples)==1 else (TEAL if len(samples)==2 else PURPLE)
  poly=[proj(l,dc,tc),proj(r,dc,tc),proj(r,dp,tp),proj(l,dp,tp)]
  ax.add_patch(Polygon(poly,closed=True,fc=col,alpha=.19,ec=col,lw=.45,zorder=1+dp*.03))
  for g in [l,r]:
   a,b=proj(g,dc,tc),proj(g,dp,tp)
   ax.plot([a[0],b[0]],[a[1],b[1]],color='#4b6077',lw=.65,alpha=.72,zorder=2)
  a,b=proj(l,dp,tp),proj(r,dp,tp)
  ax.plot([a[0],b[0]],[a[1],b[1]],color='#536578',lw=.65,zorder=2)
  if parent in recombination_nodes:
   mid=proj((l+r)/2,dp,tp)
   ax.plot(mid[0],mid[1],marker='D',markersize=2.1,mfc='#765e9c',mec='white',mew=.3,zorder=4)
for q in [3,2,1,0]:
 a,b=proj(0,q,ts.node(q).time),proj(10000,q,ts.node(q).time)
 ax.plot([a[0],b[0]],[a[1],b[1]],c=SAMPLE[q],lw=1.55,zorder=3)
 ax.add_patch(Circle(tuple(b),.025,fc=SAMPLE[q],ec='white',lw=.35,zorder=4))
 text(b[0]+.09,b[1],f'$q_{q+1}$',8.2,ha='left',color=SAMPLE[q])
# reference axes
arrow((.31,5.42),(.31,6.90),MUTED,.65,ms=6)
text(.16,6.25,'Time (generations)',8.1,rotation=90)
text(.36,5.43+1.24*1000/TIME_MAX,'1k',8,ha='left')
text(.26,5.39,'0',8,ha='right')
arrow((.43,5.34),(3.42,5.34),MUTED,.65,ms=6)
text(1.61,5.03,'Genomic position (kb)',8.1)
for g in list(ts.breakpoints()):
 label=f'{g/1000:g}'
 x=proj(g,0,0)[0];ax.plot([x,x],[5.31,5.37],c=MUTED,lw=.6);text(x,5.21,label,8)
# Diagonal depth axis attached to front-right floor.
a=proj(10000,0,0); b=proj(10000,3,0)
arrow(tuple(a+np.array([.12,-.02])),tuple(b+np.array([.12,-.02])),MUTED,.65,ms=6)
text(3.41,5.02,'Sample depth',8)
# Straight extraction arrow, away from sheet polygons.
arrow((3.72,6.30),(3.98,6.30),MUTED,1,ms=8)

# Exactly the three marginal trees, using planar leaf order and persistent colors.
def draw_tree(tree,cx,base,width=.80,height=1.28):
 order=list(tree.samples());xx={q:cx-width/2+j*width/3 for j,q in enumerate(order)}
 def setx(u):
  if u not in xx:xx[u]=np.mean([setx(c) for c in tree.children(u)])
  return xx[u]
 setx(tree.root)
 yy={u:base+height*ts.node(u).time/TIME_MAX for u in tree.nodes()}
 for u in tree.nodes():
  p=tree.parent(u)
  if p!=tskit.NULL:
   ax.plot([xx[u],xx[u],xx[p]],[yy[u],yy[p],yy[p]],c='#5a6979',lw=.85,zorder=2)
 for u in tree.nodes():
  if u<4:
   ax.add_patch(Circle((xx[u],yy[u]),.036,fc=SAMPLE[u],ec='white',lw=.5,zorder=4))
   text(xx[u],base-.12,f'$q_{u+1}$',8.1,color=SAMPLE[u])
  else:ax.add_patch(Circle((xx[u],yy[u]),.022,fc='#4c5b70',ec='white',lw=.35,zorder=4))
for tree,cx in zip(ts.trees(),[4.42,5.50,6.58]):
 draw_tree(tree,cx,5.53,width=.76,height=1.16)
 text(cx,6.79,f'$T_{tree.index+1}$',8.8,weight='bold')
 l,r=tree.interval
 text(cx,5.23,f'[{l/1000:.2f}, {r/1000:.2f}) kb',8)
text(5.50,5.02,'Unary nodes collapsed for display',8.0,color=MUTED)
# input feature strip
box(.17,4.27,6.86,.63,fc='#fafbfd',ec='#cbd4df',rounding=.06)
text(.29,4.75,'Inputs',8.3,weight='bold',ha='left')
# phased matrix, factual genotypes from toy
matrix(.32,4.38,.47,.25,4,3,SAMPLE,False)
text(.555,4.32,'$S\\times M$',8.0)
text(1.10,4.53,'Phased alleles + ages\nKnown ref. labels\nQueries masked',8.1,ha='left')
arrow((2.29,4.56),(2.43,4.56),MUTED,.8,ms=7)
text(3.26,4.57,'Site MLP · 2 linear layers\nMean / max interval pooling',8.1)
arrow((4.24,4.56),(4.44,4.56),MUTED,.8,ms=7)
matrix(4.53,4.39,.45,.34,7,4,['#a6afbb']*3+SAMPLE,True)
text(4.755,4.32,'$N_t\\times d$',8.0)
text(5.99,4.58,'Node ages / roles → $B_t$\nBranch ages + concordance\n→ observed sigmoid gates',8.0)

# Separate lower panel; encoder main path across left, genomic path on right.
ax.plot([.17,7.03],[4.10,4.10],c='#c8d2df',lw=.7)
panel_label('B','TreeLoom: full-tree encoding and ancestry readout',3.91)
# Main top encoder path.
box(.18,1.18,4.12,2.47,fc='#fbfcfe',ec='#b9c7d7',rounding=.07,lw=.8)
text(.34,3.48,'Complete node states in every tree',8.5,weight='bold',ha='left')
matrix(.36,2.89,.38,.40,7,4,['#a6afbb']*3+SAMPLE,True)
text(.56,2.79,'$B_t: N_t\\times d$',8.0)
box(.98,2.89,.79,.43,'Sweep 1',BLUE,size=8.7,weight='bold')
arrow((.79,3.10),(.95,3.10),ms=7)
# Explicit base bypass into addition; node base has not been replaced by h.
route([(.56,3.30),(.56,3.36),(3.05,3.36),(3.05,3.17)],MUTED,.7,ms=6)
text(2.28,3.31,'same $B_t$',8.0,va='top')
# state fanout to residuals (short lower bus)
arrow((1.79,3.10),(2.02,3.10),ms=7)
text(2.29,3.10,'$h^{(1)},o^{(1)}$',8.1)
route([(2.27,2.99),(2.27,2.74),(1.91,2.74),(1.91,2.64)],MUTED,.7,ms=6)
# Residuals, parallel not serial. Each represents actual modules.
box(.34,1.82,1.14,.70,fc=PEACH,size=8)
text(.91,2.34,'Clade flags',8.1,weight='bold')
text(.91,2.07,'Gained / lost\nLow-rank residual',8.0)
box(1.62,1.82,1.14,.70,fc=YELLOW,size=8)
text(2.19,2.34,'Local episode',8.1,weight='bold')
text(2.19,2.07,'Own-state residual\nEligible episodes',8.0)
box(2.91,1.82,1.15,.70,fc=TEAL,size=8)
text(3.485,2.34,'Sample memory',8.1,weight='bold')
text(3.485,2.07,'Cut context → scan\nGroup mean − force\nProject residual',8.0)
# input conditioning via shared state bus
ax.plot([.91,3.485],[2.64,2.64],c=MUTED,lw=.65)
for x in [.91,2.19,3.485]:arrow((x,2.64),(x,2.54),MUTED,.7,ms=5)
# all 3 residuals sum into explicit addition along clear lower bus
for x in [.91,2.19,3.485]:ax.plot([x,x],[1.80,1.66],c=MUTED,lw=.65)
ax.plot([.91,3.485],[1.66,1.66],c=MUTED,lw=.65)
route([(3.485,1.66),(4.15,1.66),(4.15,2.70),(3.05,2.70),(3.05,3.03)],MUTED,.75,ms=6)
text(2.16,1.43,'$B_t + \Delta_t^{\mathrm{clade}} + \Delta_t^{\mathrm{local}} + \Delta_t^{\mathrm{memory}}$',8.1)
# adder and sweep2
ax.add_patch(Circle((3.05,3.10),.075,fc='white',ec=INK,lw=.8,zorder=5));text(3.05,3.10,'+',9.2,zorder=6)
box(3.28,2.89,.79,.43,'Sweep 2',BLUE,size=8.7,weight='bold')
arrow((3.13,3.10),(3.25,3.10),ms=6)
text(3.67,2.79,'shared weights',8.0)
# right sequence area
box(4.48,1.18,2.55,2.47,fc='#fcfbfe',ec='#c8bddc',rounding=.07,lw=.8)
# The implemented sampled-node readout is learned, after the second sweep.
box(4.65,3.31,2.22,.28,fc='#eeebf6',ec='#b9afcd',lw=.65,rounding=.035)
text(5.76,3.51,'Sampled concat [upward, outside]',8.0)
text(5.76,3.37,'Linear ($2d→d$) → tanh → $E_t$ ($S\\times d$)',8.0)
route([(4.09,3.10),(4.41,3.10),(4.41,3.45),(4.63,3.45)],INK,.85,ms=7)
for x,lab in [(4.68,'$E_1$'),(5.50,'$E_2$'),(6.32,'$E_3$')]:
 matrix(x,2.89,.43,.29,4,5,SAMPLE,True)
 text(x+.215,2.83,lab,8.4)
arrow((5.76,3.29),(5.76,3.22),MUTED,.75,ms=5)
# readout happens after full recurrent sequence
route([(6.75,3.18),(6.94,3.18),(6.94,1.59),(6.85,1.59)],MUTED,.65,ms=5)
text(6.83,2.59,'local',8.0,rotation=90)
# Highlight the same q1 row, then collect all three vectors in genomic order.
for x in [4.68,5.50,6.32]:
 ax.add_patch(Rectangle((x,2.89+.29*.75),.43,.29*.25,fc='none',ec=SAMPLE[0],lw=1.0,zorder=4))
 tap=x+.50
 ax.plot([x+.43,tap],[3.144,3.144],c=SAMPLE[0],lw=.7,zorder=4)
 ax.plot([tap,tap],[3.144,2.74],c=SAMPLE[0],lw=.7,zorder=4)
text(4.56,3.145,'$q_1$',8.0,color=SAMPLE[0])
ax.plot([5.18,6.82],[2.74,2.74],c=SAMPLE[0],lw=.7,zorder=4)
arrow((5.77,2.74),(5.77,2.65),SAMPLE[0],.7,ms=5)
text(5.77,2.56,'$e_{1q},e_{2q},e_{3q}$ + interval geometry',8.1)
arrow((5.77,2.48),(5.77,2.43),ms=5)
box(4.72,2.00,2.10,.40,'Bidirectional GRU · 1 layer',PURPLE,size=8.1,weight='bold')
arrow((4.86,2.10),(6.67,2.10),MUTED,.6,style='<->',ms=5)
# label placed upper in BiGRU block so directional line doesn't intersect text.
for t in texts:
 if t.get_text()=='Bidirectional GRU · 1 layer':t.set_position((5.77,2.27))
arrow((5.77,1.98),(5.77,1.83),ms=6)
box(4.72,1.38,2.10,.42,'Concatenate local + history\nLinear → GELU → Linear',PEACH,size=8.0)
# Give the expanded lower inset space, preserving two clear main panels.
from matplotlib.transforms import Affine2D
SHIFT=.90
for artist in list(ax.patches):artist.set_transform(Affine2D().translate(0,SHIFT)+ax.transData)
for artist in list(ax.lines):artist.set_ydata(np.asarray(artist.get_ydata())+SHIFT)
for artist in list(ax.texts):artist.set_y(artist.get_position()[1]+SHIFT)
ax.set_ylim(0,H+SHIFT)
# Exploded sweep inset. The two columns are traversal directions, not extra layers.
box(.18,.30,4.12,1.59,fc='#f4f7fb',ec='#b9c7d7',rounding=.055,lw=.75)
text(.33,1.74,'Inside each shared-weight sweep · every node',8.4,weight='bold',ha='left')
text(1.24,1.53,'↑ Bottom-up',8.3,weight='bold')
text(3.23,1.53,'↓ Top-down / outside',8.3,weight='bold')
up_labels=['Gated linear child messages','Mean over children','Concat [base, child mean]',
 'Linear → tanh','Add base → layer norm']
down_labels=['Parent outside + base + sum','Subtract recipient message','Gated linear message',
 'Linear → tanh','Layer norm · root outside = 0']
for labels,x,col in [(up_labels,.34,BLUE),(down_labels,2.31,TEAL)]:
 for i,label in enumerate(labels):
  y=1.24-i*.215
  box(x,y,1.80,.155,label,fc=col,ec='#b0bece',lw=.55,rounding=.025,size=8.0)
  if i<4:arrow((x+.90,y-.012),(x+.90,y-.047),MUTED,.65,ms=4)
# Clear softmax output and schematic tract, no numeric results.
box(4.48,.30,2.55,1.59,fc='#f8f5fc',ec='#c8bddc',rounding=.055,lw=.75)
text(5.76,1.71,'Softmax → ancestry probabilities',8.2,weight='bold')
arrow((5.77,1.36+SHIFT),(5.77,1.90),ms=7)
# Symbolic 4-channel probability vectors for each of the 3 native intervals.
for r in range(3):
 for c,col in enumerate(CLASSES):
  ax.add_patch(Rectangle((4.74+c*.67/4,.92+(2-r)*.48/3),.67/4,.48/3,
   fc=col,alpha=.40+.12*((r+c)%3),ec='white',lw=.6,zorder=3))
ax.add_patch(Rectangle((4.74,.92),.67,.48,fc='none',ec=INK,lw=.6,zorder=4))
text(5.075,1.51,'$p_{tq}$',8.5)
for c,label in enumerate(['A','C','W','E']):text(4.74+(c+.5)*.67/4,.85,label,8.0)
text(5.075,.70,'Tree rows 1–3',8.0)
text(6.16,1.30,'Sample $q$ ancestry tract',8.1)
for l,r,c in zip(list(ts.breakpoints())[:-1],list(ts.breakpoints())[1:],[CLASSES[0],CLASSES[1],CLASSES[1]]):
 x=5.54+(l/10000)*1.21; w=(r-l)/10000*1.21
 ax.add_patch(Rectangle((x,1.01),w,.17,fc=c,ec='white',lw=.7,zorder=3))
arrow((5.53,.94),(6.78,.94),MUTED,.6,ms=5)
text(6.16,.79,'Genomic position',8.0)
for i,(c,label) in enumerate(zip(CLASSES,['Ana','CHG','WHG','EHG'])):
 x=4.76+(i%2)*1.05; y=.56-(i//2)*.19
 ax.add_patch(Rectangle((x,y-.045),.12,.09,fc=c,ec='none',zorder=3))
 text(x+.20,y,label,8.0,ha='left')
text(.20,.07,'Toy ARG: diamonds mark recorded recombination nodes. Output tracts are schematic.',8.0,ha='left',color=MUTED)

text(.20,.21,'Shapes = rows × columns: $S$ sampled haplotypes, $M$ markers, $N_t$ tree nodes, $d$ features.',8.0,ha='left',color=MUTED)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
text_bounds=[{'text':t.get_text(),'font_points':t.get_fontsize(),'bbox_pixels':list(t.get_window_extent(renderer).bounds)} for t in texts]
(OUT/'text_layout_bounds.json').write_text(json.dumps(text_bounds,indent=2)+'\n')
fig.savefig(OUT/'TreeLoom_architecture.pdf')
fig.savefig(OUT/'TreeLoom_architecture.svg')
fig.savefig(OUT/'TreeLoom_architecture.png',dpi=220)

source_map={
 'site_encoder':{'file':'src/arggnn/mutation_routed_whole_tree_gnn_v1.py','symbol':'MutationRoutedWholeTreeGNNV1.__init__, _site_chunk, encode_sites','operation':'19→32 GELU→32 tanh; interval mean/max plus count, empty, span; 67→32 tanh'},
 'tree_sweep':{'file':'src/arggnn/anonymous_clade_gnn_v1.py','symbol':'AnonymousCladeTensorGNNV1.prepare_tree_base_v1, sweep_tree_state_v1, leaf_context_v1','operation':'gated linear child messages divided by child count; sum; base residual+tanh+layernorm; outside recipient exclusion; gated linear→linear→tanh→layernorm; sampled concat64→32tanh'},
 'clade_flags':{'file':'src/arggnn/anonymous_clade_flag_control_v1.py','symbol':'RankEightCladeFlagAdapterV1','operation':'O[gained(Uh⊙Vh)−lost(Uh⊙Wh)] at proper clade representatives'},
 'own_state_episode':{'file':'src/arggnn/ordered_clade_episode_gnn_v1.py','symbol':'RankEightCladeEpisodeAdapterV1.forward','operation':'O[Uh⊙Vh+Wh] where episode size>=2; own-state control used by H050'},
 'stable_sample_memory':{'file':'research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py','symbol':'CutMemoryV1','operation':'rank8 forcing U(h_up+h_out) at cut representative; stable sample forward/reverse span-decay; average, current cut-group mean, subtract forcing, O back to32'},
 'TreeLoom_encoder_splice':{'file':'research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/direct_readout_cut_memory_v1.py','symbol':'DirectReadoutCutMemoryGNNV1.encode_trees','operation':'first sweep; three parallel residual injections added to original base; second sweep with identical weights and gates; leaf projection'},
 'sampled_projection':{'file':'src/arggnn/anonymous_clade_gnn_v1.py','symbol':'AnonymousCladeTensorGNNV1.leaf_context_v1','operation':'concatenate sampled second-sweep upward and outside states; inherited Linear(2d,d) then tanh'},
 'genomic_head':{'file':'src/arggnn/mutation_routed_whole_tree_gnn_v1.py','symbol':'MutationRoutedWholeTreeGNNV1.__init__','operation':'one-layer BiGRU 35 input,32 per direction; concat32+64; Linear96→32,GELU,Linear32→4; softmax at prediction'},
}
root=OUT.parent.parent
for entry in source_map.values():entry['sha256']=hashlib.sha256((root/entry['file']).read_bytes()).hexdigest()
metadata={'diagram_revision':'v3_symbolic_shapes_and_sample_projection','symbolic_shapes':{'input':'S × M','all_native_node_states':'B_t: N_t × d','sampled_tree_embeddings':'E_t: S × d','projection':'sampled concat(upward,outside): 2d → Linear → tanh → d'},'axis_meaning':{'S':'sampled haplotypes','M':'observed markers','N_t':'all native nodes in tree t','d':'hidden feature width','convention':'rows × columns'},'title':'TreeLoom: full-tree encoding and ancestry readout','vector_outputs':['TreeLoom_architecture.pdf','TreeLoom_architecture.svg'],
 'editable_source':'build_figure.py','figure_size_inches':[7.0,8.15],'minimum_font_points':min(t.get_fontsize() for t in texts),'publication_width_inches':7.0,'AAAI_textwidth_inches':7.0,'AAAI_textheight_inches':9.0,
 'toy_validation':{'num_trees':ts.num_trees,'num_samples':ts.num_samples,'all_trees_connected':True,'strict_parent_time':True,'tskit_version':tskit.__version__,'msprime_version':msprime.__version__,'full_ARG_nodes':full_arg.num_nodes,'full_ARG_edges':full_arg.num_edges,'recombination_nodes':recombination_nodes,'simulation_seed':seed,'genotypes':ts.genotype_matrix().tolist()},
 'source_map':source_map,'scientific_scope':'Architecture schematic only; no experiment, data bank, truth, model checkpoint, fitted probability, or GPU read or execution',
 'recorded_ARG_recombination_nodes':True,'sample_colors_denote':'persistent sample identity only, not ancestry',
 'output_class_colors_denote':'ancestry class columns A=Ana,C=CHG,W=WHG,E=EHG and hypothetical tract; no numerical model predictions are plotted','gray_rows_denote':'internal ancestors in all-node B_t; sample-only E_t has no gray rows','highlighted_blue_row':'persistent sample q1 in each E_t, fed through an ordered bus to the GRU','query_reference_labels_masked':True}
(OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'pdf':str(OUT/'TreeLoom_architecture.pdf'),'num_trees':ts.num_trees,'min_font':metadata['minimum_font_points']}))
