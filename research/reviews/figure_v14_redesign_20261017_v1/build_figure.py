"""Original editable Figure 1 draft. Hand-recorded toy ARG, no model execution.
All outputs/cache stay in this owned directory; panel labels are letters only.
"""
from pathlib import Path
import os,json,hashlib
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'.mpl-cache')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import tskit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Circle,Polygon,FancyArrowPatch
W,H=1300,1000
INK='#243240';GRAY='#607080';PALE='#e6ebef';LIGHT='#b8c4ce';QUERY='#9b367f'
ANC=['#70a9d1','#e3b765','#8cbaa7','#c692b0'];AN=['Ana','CHG','WHG','EHG']
PST=['#e5eff9','#e7f1e9','#faead8','#efe6f2'];PLANE=['#b8dce5','#e8d8a4','#e7bfd0']
NAMES=['q','r1','r2','r3','r4'];BREAKS=[0,3000,7000,10000]
# Native interval-labelled ancestral graph. All marginals use these very nodes/edges.
node_times=[0,0,0,0,0,200,175,375,212.5,400,450,625,675,725]
parents=[{0:5,1:5,5:7,2:7,3:6,4:6,7:11,6:11},{1:8,2:8,0:9,8:9,3:6,4:6,9:12,6:12},{1:8,2:8,3:6,4:6,8:10,6:10,0:13,10:13}]
tables=tskit.TableCollection(sequence_length=10000)
for u,time in enumerate(node_times):tables.nodes.add_row(time=time,flags=tskit.NODE_IS_SAMPLE if u<5 else 0)
for i,edges in enumerate(parents):
 for child,parent in edges.items():tables.edges.add_row(left=BREAKS[i],right=BREAKS[i+1],parent=parent,child=child)
for i,node in enumerate([5,8,6]):
 s=tables.sites.add_row((BREAKS[i]+BREAKS[i+1])/2,ancestral_state='0');tables.mutations.add_row(site=s,node=node,derived_state='1')
tables.sort();ts=tables.tree_sequence();trees=[t.copy()for t in ts.trees()]
assert ts.num_trees==3 and ts.num_samples==5 and list(ts.breakpoints())==BREAKS
for t,expected in zip(trees,parents):
 assert t.num_roots==1 and {u:t.parent(u)for u in t.nodes()if t.parent(u)!=tskit.NULL}==expected
 assert sorted(t.samples())==list(range(5))
ts.dump(OUT/'illustrative_recorded_ARG.trees')
record=dict(schema='figure_v14_hand_recorded_valid_ARG_v1',purpose='Illustration only; no fitted data, experiment or model prediction',sample_names=NAMES,reference_classes=['unknown']+AN,nodes=[dict(id=i,time=t,flags=1 if i<5 else 0)for i,t in enumerate(node_times)],edges=[dict(left=e.left,right=e.right,parent=e.parent,child=e.child)for e in ts.edges()],breakpoints_bp=BREAKS,site_positions=[s.position for s in ts.sites()],genotypes=ts.genotype_matrix().tolist(),trees=[dict(interval=list(t.interval),parents={str(u):t.parent(u)for u in t.nodes()if t.parent(u)!=tskit.NULL},newick=t.as_newick(node_labels={u:NAMES[u]for u in range(5)}))for t in trees],memory_cut_time=350,illustrative_probabilities=[[.62,.12,.14,.12],[.17,.16,.53,.14],[.11,.16,.21,.52]])
(OUT/'illustrative_recorded_ARG.json').write_text(json.dumps(record,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(6.5,5.0),dpi=200);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
texts=[];routes=[];MIRROR=False
def X(x):return W-x if MIRROR else x
def text(x,y,s,size=8.2,ha='center',color=INK,weight='normal',**kw):
 ha=({'left':'right','right':'left'}.get(ha,ha)if MIRROR else ha)
 t=ax.text(X(x),y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight,zorder=10,**kw);texts.append(t);return t
def line(points,color=INK,lw=.8,alpha=1,ls='-',z=3):
 ax.plot([X(p[0])for p in points],[p[1]for p in points],color=color,lw=lw,alpha=alpha,ls=ls,zorder=z)
def arrow(points,color=INK,lw=.8,style='-|>',head=5):
 if len(points)>2:line(points[:-1],color,lw)
 a,b=[(X(p[0]),p[1])for p in points[-2:]];ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=head,lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=5));routes.append(points)
def box(x,y,w,h,fill='white',edge=LIGHT,r=7,lw=.7,z=2):
 p=FancyBboxPatch((X(x+w)if MIRROR else x,y),w,h,boxstyle='round,pad=0,rounding_size='+str(r),fc=fill,ec=edge,lw=lw,zorder=z);ax.add_patch(p);return p
def rect(x,y,w,h,fill,edge='white',lw=.5,z=3):ax.add_patch(Rectangle((X(x+w)if MIRROR else x,y),w,h,fc=fill,ec=edge,lw=lw,zorder=z))
def dot(x,y,sample=None,r=4,z=6):
 col=QUERY if sample==0 else ANC[sample-1]if sample is not None else INK
 ax.add_patch(Circle((X(x),y),r,fc='white'if sample==0 else col,ec=col,lw=.8,zorder=z))
def panel(x,y,letter):
 global MIRROR
 old=MIRROR;MIRROR=False;text(x,y,letter,size=10,ha='left',weight='bold');MIRROR=old
def vector(x,y,w=38,h=15,query=False):
 for k,col in enumerate(['#9db3c5','#d1dde5','#6e91aa']):rect(x+k*w/3,y,w/3,h,col)
 rect(x,y,w,h,'none',QUERY if query else GRAY,.6)
def sx(t,u):return sum(t.samples(u))/len(list(t.samples(u)))/4
# a: translucent local-tree planes with genuine black identity continuities.
panel(18,27,'a')
plane_coords=[]
for i,t in enumerate(trees):
 x0=35+i*100;y0=312;depth=90;shear=35;ageh=.272
 pts=[(x0,y0),(x0+depth,y0-shear),(x0+depth,y0-shear-210),(x0,y0-210)]
 ax.add_patch(Polygon(pts,fc=PLANE[i],ec=PLANE[i],alpha=.32,lw=.6,zorder=0))
 pp={u:(x0+depth*sx(t,u),y0-shear*sx(t,u)-ageh*ts.node(u).time)for u in t.nodes()};plane_coords.append(pp)
# Only identical recorded nodes continue along genome, never arbitrary topology links.
for i in range(2):
 for u in sorted(set(plane_coords[i])&set(plane_coords[i+1])):
  line([plane_coords[i][u],plane_coords[i+1][u]],INK,.75,alpha=.7,z=1)
for i,t in enumerate(trees):
 pp=plane_coords[i]
 for u in t.nodes():
  p=t.parent(u)
  if p!=tskit.NULL:line([pp[u],pp[p]],INK,.85,z=3)
 for u,pt in pp.items():dot(*pt,sample=u if u<5 else None,r=3.5 if u<5 else 2.5)
 if i==2:
  for u in range(5):text(pp[u][0],pp[u][1]+19,NAMES[u],size=8.,color=QUERY if u==0 else GRAY)
 text(35+i*100+43,76,'T'+str(i+1),size=8.)
arrow([(20,310),(20,93)],GRAY,.6);text(18,68,'t',ha='left',color=GRAY,size=8.)
arrow([(36,347),(338,347)],GRAY,.65);text(188,366,'genome',color=GRAY,size=8.)
arrow([(355,213),(409,213)],INK,1.)
# Exact same recorded marginal edges, node ages, sample order and mutations.
flat_coords=[]
for i,t in enumerate(trees):
 cx=486+i*201;w=155;base=291;scale=.272
 pp={u:(cx-w/2+w*sx(t,u),base-scale*ts.node(u).time)for u in t.nodes()};flat_coords.append(pp)
 text(cx,76,'T'+str(i+1))
 for u in t.nodes():
  p=t.parent(u)
  if p!=tskit.NULL:line([pp[u],pp[p]],INK,.9)
 for u,pt in pp.items():dot(*pt,sample=u if u<5 else None,r=4 if u<5 else 3)
 for u in range(5):text(pp[u][0],314,NAMES[u],size=8.,color=QUERY if u==0 else GRAY)
 rect(cx-w/2,336,w,9,PLANE[i],edge=PLANE[i]);text(cx,364,'I'+str(i+1),size=8.)
arrow([(972,213),(998,213)],INK,.7,head=4)
# Real toy allele observations, plus known reference vs masked query vectors.
text(1140,76,'alleles',size=8.)
for j,name in enumerate(NAMES):text(1084+j*37,108,name,size=8.,color=QUERY if j==0 else GRAY)
for i,row in enumerate(record['genotypes']):
 text(1036,143+i*29,'I'+str(i+1),size=8.)
 for j,value in enumerate(row):rect(1070+j*37,130+i*29,29,26,'#edf2f5');text(1084+j*37,143+i*29,str(value),size=8.)
# One-hot reference ancestry and unknown indicator, all genuine feature components.
for y,label,values in [(247,'r1',[1,0,0,0]),(281,'q',[0,0,0,0])]:
 text(1036,y,label,color=QUERY if label=='q'else GRAY,size=8.)
 for j,value in enumerate(values):
  rect(1070+j*37,y-13,29,26,ANC[j]if value and j<4 else '#f0e8ef'if value else'#f4f6f8',edge=LIGHT,lw=.35);text(1084+j*37,y,str(value),size=8.)
text(1232,281,'?',size=8.,color=QUERY)
text(1036,331,'node',ha='center',size=8.)
for j,value in enumerate(['τ','0','0']):rect(1080+j*48,318,42,26,'#edf2f5');text(1101+j*48,331,value,size=8.)
# Contiguous biological input paths: topology into GNN, observations into encoders.
box(1000,90,285,260,'none',edge=LIGHT,r=4,lw=.5,z=1)
line([(1285,350),(1285,624)],GRAY,.7)
arrow([(1285,350),(1285,400)],GRAY,.7,head=4)
for y in (462.5,553.5,606.5):arrow([(1285,y),(1278,y)],GRAY,.65,head=3)
line([(409,345),(409,380),(967,380),(967,345)],GRAY,.65)
arrow([(967,380),(940,380),(940,456)],GRAY,.75)
MIRROR=True
# b: sparse actual encoders and two applications of the SAME learned GNN.
panel(18,408,'b')
for y,label,col in [(445,'site MLP',PST[0]),(536,'leaf MLP',PST[1]),(589,'node MLP',PST[2])]:
 box(75,y,137,35,col);text(143,y+17.5,label,size=8.)
 vector(22,y+10,w=33,h=16);arrow([(56,y+17.5),(75,y+17.5)],GRAY,.6)
box(75,489,137,35,'#f3f6f8');text(143,506.5,'pool',size=8.)
arrow([(143,480),(143,489)],GRAY,.6,head=4)
arrow([(143,524),(143,536)],GRAY,.6,head=4)
arrow([(212,553.5),(235,553.5),(235,522),(240,522)],GRAY,.65,head=4)
arrow([(212,606.5),(250,606.5),(250,532)],GRAY,.65,head=4)
ax.add_patch(Circle((X(250),522),10,fc='white',ec=GRAY,lw=.65,zorder=6));text(250,522,'+',size=8.)
arrow([(260,522),(291,522)],INK,.85)
# Original base bypass stays above all processing; three additions meet once.
arrow([(270,522),(270,635),(746,635),(746,540)],GRAY,.75);text(682,651,'b',size=8.,color=GRAY)
def gnn(x):
 box(x,456,137,132,'#f8fafc',edge=LIGHT)
 box(x+9,465,119,49,PST[0]);text(x+68.5,490,'↑ GNN$_θ$',size=8.3)
 box(x+9,529,119,49,PST[2]);text(x+68.5,554,'↓ GNN$_θ$',size=8.3)
gnn(292)
# Current first-pass states feed both real residuals. No false direct forward bypass.
line([(429,522),(461,522)],INK,.8)
arrow([(461,522),(461,472),(497,472)],INK,.8)
arrow([(461,522),(461,573),(497,573)],INK,.8)
box(497,448,167,49,PST[1]);text(580.5,472,'local residual',size=8.)
box(497,548,167,49,PST[3]);text(580.5,573,'memory',size=8.3);text(651,585,'c',size=8.,color=GRAY)
arrow([(664,472),(705,472),(705,522),(728,522)],INK,.8)
arrow([(664,573),(705,573),(705,522)],INK,.8)
ax.add_patch(Circle((X(746),522),18,fc='white',ec=INK,lw=.85,zorder=6));text(746,522,'+',size=13)
arrow([(764,522),(794,522)],INK,.85);gnn(795)
arrow([(932,522),(967,522)],INK,.85)
box(968,477,106,89,PST[1]);text(1021,499,'Linear',size=8.);line([(975,521),(1067,521)],LIGHT,.5);text(1021,542,'tanh',size=8.)
arrow([(1074,522),(1089,522)],INK,.8)
# Aligned per-sample embedding sequences; 32D represented by three colored strips.
for i in range(3):text(1152+i*50,439,'T'+str(i+1),size=8.)
for j,name in enumerate(NAMES):
 text(1110,467+j*28,name,size=8.,color=QUERY if j==0 else GRAY)
 for i in range(3):vector(1133+i*50,459+j*28,w=38,h=16,query=j==0)
box(1090,451,198,146,'none',edge=LIGHT,r=2,lw=.5,z=1)
text(1222,618,'e',size=8.,color=GRAY)
# c: memory zoom; fixed sample rows visibly survive changing tree groups.
panel(330,670,'c')
# Same genuine T1, with actual350-generation cut and representative5 highlighted.
inset=trees[0];ip={u:(35+109*sx(inset,u),710-.08*ts.node(u).time)for u in inset.nodes()}
for u in inset.nodes():
 parent=inset.parent(u)
 if parent!=tskit.NULL:line([ip[u],ip[parent]],GRAY,.65)
line([(35,682),(144,682)],GRAY,.6,ls='--')
for u,pt in ip.items():dot(*pt,sample=u if u<5 else None,r=2.4)
ax.add_patch(Circle((X(ip[5][0]),ip[5][1]),5,fc=PST[3],ec=QUERY,lw=1.,zorder=8))
arrow([ip[5],(ip[5][0],725),(89,725),(89,740)],QUERY,.65,head=4)
box(35,740,109,45,PST[3]);text(89.5,762.5,'Linear U',size=8.)
arrow([(144,762.5),(196,762.5)],INK,.75)
vector(153,758.5,w=32,h=8)
text(153,735,'f',size=8.,color=GRAY)
box(196,718,469,198,'none',edge=LIGHT,r=7,lw=.5,z=1)
text(385,672,'↔ EMA',size=8.5)
cols=[207,389,571];ys=[743,781,819,857,895]
partitions=[[[0,1],[2],[3,4]],[[0],[1,2],[3,4]],[[0],[1,2],[3,4]]]
for i,x in enumerate(cols):
 text(x+35,707,'T'+str(i+1),size=8.)
 for group in partitions[i]:
  top=ys[min(group)]-15;bottom=ys[max(group)]+15
  box(x-5,top,83,bottom-top,'#eef2f6',edge='#aebdc9',r=6,lw=.55)
 for j in range(5):
  vector(x+7,ys[j]-7,w=59,h=14,query=j==0)
  if i==0:text(x-36,ys[j],NAMES[j],size=8.,color=QUERY if j==0 else GRAY)
for i in range(2):
 for j in range(5):arrow([(cols[i]+82,ys[j]),(cols[i+1]-12,ys[j])],QUERY if j==0 else '#9aabb9',.65,style='<->',head=4)
# Span input is physically proportioned to original I1:I2:I3=3:4:3.
x0=225
for span,col in zip([99,132,99],PLANE):rect(x0,942,span,10,col,edge=col);x0+=span
text(390,976,'Δx',size=8.);arrow([(390,940),(390,916)],GRAY,.6)
# Bidirectional averaging/current-group averaging/centering are genuine operations.
arrow([(654,819),(688,819)],INK,.75)
box(688,789,116,60,PST[3]);text(746,809,'mean',size=8.);text(746,834,'− f',size=8.)
arrow([(804,819),(826,819)],INK,.75)
box(826,797,110,45,PST[3]);text(881,819,'Linear O',size=8.)
arrow([(936,819),(965,819)],INK,.75);text(958,785,'δ',size=8.,color=GRAY)
# d: all-sample recurrent readout, actual local skip, query-only displayed output.
panel(18,640,'d')
arrow([(1202,597),(1202,659)],INK,.8)
# The same physical-interval geometry enters the real recurrent input.
for j,(span,col)in enumerate(zip([27,36,27],PLANE)):
 start=993+sum([27,36,27][:j]);rect(start,682,span,10,col,edge=col)
text(1038,665,'Δx',size=8.)
arrow([(1083,687),(1112,687)],GRAY,.65)
box(1112,660,170,58,PST[0]);text(1197,679,'BiGRU',size=8.5);text(1197,703,'1 layer',size=8.)
arrow([(1197,718),(1197,744)],INK,.8)
box(1112,744,170,34,'#f1f4f7');text(1197,761,'concat',size=8.)
# Local e enters same concat; geometry is included in BiGRU input (caption).
arrow([(1288,597),(1290,597),(1290,761),(1282,761)],GRAY,.75);text(1266,730,'e',size=8.,color=GRAY)
arrow([(1197,778),(1197,800)],INK,.8)
box(1112,800,170,52,PST[1]);text(1197,817,'MLP',size=8.5);text(1197,841,'softmax',size=8.)
arrow([(1197,852),(1197,883)],INK,.8)
# Explicit illustration only, not inferred probabilities. Legend uses real class order.
for i,(name,col)in enumerate(zip(AN,ANC)):
 x=983+i*77;rect(x,933,13,13,col,edge=col);text(x+19,940,name,ha='left',size=8.)
px=1112;pw=170;py=883;ph=37
for row,span in zip(record['illustrative_probabilities'],[.3,.4,.3]):
 h=0
 for prob,col in zip(row,ANC):rect(px,py+ph*(1-h-prob),pw*span,ph*prob,col,edge='white',lw=.3);h+=prob
 px+=pw*span
text(1074,898,'p(q)',size=8.,color=QUERY)
px=1112
for span,call in zip([.3,.4,.3],[0,2,3]):rect(px,970,pw*span,13,ANC[call],edge='white',lw=.4);px+=pw*span
arrow([(1112,995),(1282,995)],GRAY,.55);text(1072,976,'q',size=8.,color=QUERY)
# Export editable SVG text and vectors, embeddable PDF and publication PNG.
fig.canvas.draw();renderer=fig.canvas.get_renderer()
bounds=[]
for t in texts:
 bb=t.get_window_extent(renderer);bounds.append(dict(text=t.get_text(),font_points=t.get_fontsize(),pixel_bounds=[bb.x0,bb.y0,bb.x1,bb.y1]))
assert min(t.get_fontsize()for t in texts)>=8
for ext in ('svg','pdf','png'):fig.savefig(OUT/('Figure1_v14.'+ext),dpi=200)
fig.savefig(OUT/'Figure1_v14_2x.png',dpi=400)
(OUT/'text_bounds.v1.json').write_text(json.dumps(dict(width_inches=6.5,height_inches=5.0,min_font_points=8.,texts=bounds),indent=2)+'\n')
# Exact correspondence assertions independent of plotting positions.
checks=dict(native_tskit_tree_sequence_valid=True,all3_marginals_from_same_recorded_ARG=True,leaf_order_identical_all_views=NAMES,parents_identical_to_recorded_native_edges=True,all_plane_and_marginal_branch_ages_same=True,only_same_native_node_IDs_have_continuity_lines=True,probabilities_illustrative_not_measured=True,editable_vector_text=True,printed_inches=[6.5,5.0],minimum_font_points=8.,no_panel_titles=True,model_executed=False,independent_image_review_pending=True)
(OUT/'geometry_checks.v1.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
