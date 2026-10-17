"""Disjoint v15 flow revision of the exact recorded Figure 1 ARG. Hand-recorded toy ARG, no model execution.
All outputs/cache stay in this owned directory; panel labels are letters only.
"""
from pathlib import Path
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
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
PST=['#e5eff9','#e7f1e9','#faead8','#efe6f2'];PLANE=['#e9eef2','#e0e7ed','#d6e0e7'];STRIP=['#e5e9ed','#cfd8df','#b8c6d0'];HATCH=['///','...','\\\\'];REP='#a05f12'
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
def rect(x,y,w,h,fill,edge='white',lw=.5,z=3,hatch=None):ax.add_patch(Rectangle((X(x+w)if MIRROR else x,y),w,h,fc=fill,ec=edge,lw=lw,zorder=z,hatch=hatch))
def interval_strip(x,y,w,h,i):rect(x,y,w,h,STRIP[i],edge='#82909b',lw=.3,hatch=HATCH[i])
def dot(x,y,sample=None,r=4,z=6):
 col=QUERY if sample==0 else ANC[sample-1]if sample is not None else INK
 ax.add_patch(Circle((X(x),y),r,fc='white'if sample==0 else col,ec=col,lw=.8,zorder=z))
def panel(x,y,letter):
 global MIRROR
 old=MIRROR;MIRROR=False;text(x,y,letter,size=10,ha='left',weight='bold');MIRROR=old
def vector(x,y,w=38,h=15,query=False,outline=None):
 for k,col in enumerate(['#9db3c5','#d1dde5','#6e91aa']):rect(x+k*w/3,y,w/3,h,col)
 rect(x,y,w,h,'none',outline or(QUERY if query else GRAY),.6)
def sx(t,u):return sum(t.samples(u))/len(list(t.samples(u)))/4

# a: exact recorded ARG and marginals; all genomic views run left to right.
panel(18,25,'a')
# Evidence table, placed directly above the three encoder streams.
box(20,62,250,285,'none',edge=LIGHT,r=4,lw=.5,z=1)
text(156,77,'alleles',size=8.)
for j,name in enumerate(NAMES):text(95+j*34,106,name,size=8.,color=QUERY if j==0 else GRAY)
for i,row in enumerate(record['genotypes']):
 text(48,137+i*27,'I'+str(i+1),size=8.)
 for j,value in enumerate(row):
  rect(81+j*34,125+i*27,28,24,'#edf2f5');text(95+j*34,137+i*27,str(value),size=8.)
text(156,217,'ancestry',size=8.)
for y,label,values in [(246,'r1',[1,0,0,0]),(277,'q',[0,0,0,0])]:
 text(48,y,label,color=QUERY if label=='q'else GRAY,size=8.)
 for j,value in enumerate(values):
  rect(81+j*34,y-12,28,24,ANC[j]if value else'#f4f6f8',edge=LIGHT,lw=.35)
  text(95+j*34,y,str(value),size=8.)
text(232,277,'?',size=8.,color=QUERY)
text(48,324,'node',size=8.)
for j,value in enumerate(['τ','0','0']):
 rect(81+j*48,312,42,24,'#edf2f5');text(102+j*48,324,value,size=8.)
# Native ARG identity continuities, same edges and age scale as all marginals.
plane_coords=[]
for i,t in enumerate(trees):
 x0=315+i*91;y0=314;depth=80;shear=31;ageh=.272
 pts=[(x0,y0),(x0+depth,y0-shear),(x0+depth,y0-shear-210),(x0,y0-210)]
 ax.add_patch(Polygon(pts,fc=PLANE[i],ec=PLANE[i],alpha=.32,lw=.6,zorder=0))
 pp={u:(x0+depth*sx(t,u),y0-shear*sx(t,u)-ageh*ts.node(u).time)for u in t.nodes()};plane_coords.append(pp)
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
 text(354+i*91,77,'T'+str(i+1),size=8.)
arrow([(293,314),(293,93)],GRAY,.6);text(292,69,'t',color=GRAY,size=8.)
arrow([(315,351),(590,351)],GRAY,.65);text(451,372,'genome',color=GRAY,size=8.)
arrow([(601,213),(649,213)],INK,1.)
flat_coords=[]
for i,t in enumerate(trees):
 cx=735+i*210;w=160;base=303;scale=.272
 pp={u:(cx-w/2+w*sx(t,u),base-scale*ts.node(u).time)for u in t.nodes()};flat_coords.append(pp)
 text(cx,77,'T'+str(i+1),size=8.)
 for u in t.nodes():
  p=t.parent(u)
  if p!=tskit.NULL:line([pp[u],pp[p]],INK,.9)
 for u,pt in pp.items():dot(*pt,sample=u if u<5 else None,r=4 if u<5 else 3)
 for u in range(5):text(pp[u][0],326,NAMES[u],size=8.,color=QUERY if u==0 else GRAY)
 interval_strip(cx-w/2,346,w,9,i);text(cx,373,'I'+str(i+1),size=8.)
# Trees feed both graph applications; the quiet grey line denotes topology.
line([(655,355),(655,388),(1235,388),(1235,355)],GRAY,.6)
arrow([(655,388),(345,388),(345,455)],GRAY,.65,head=4)
arrow([(935,388),(865,388),(865,455)],GRAY,.65,head=4)

# b: named evidence streams, then two applications of one learned GNN.
panel(18,385,'b')
for y,label,col in [(428,'site MLP',PST[0]),(520,'leaf MLP',PST[1]),(583,'node MLP',PST[2])]:
 box(105,y,120,33,col);text(165,y+16.5,label,size=8.)
 vector(35,y+9,w=35,h=15);arrow([(70,y+16.5),(105,y+16.5)],GRAY,.6,head=4)
for y,label in [(414,'alleles'),(506,'ancestry'),(569,'node')]:text(55,y,label,size=8.,color=GRAY)
box(105,472,120,31,'#f3f6f8');text(165,487.5,'pool',size=8.)
arrow([(165,461),(165,472)],GRAY,.6,head=4)
arrow([(165,503),(165,520)],GRAY,.6,head=4)
arrow([(225,536.5),(235,536.5),(235,521)],GRAY,.65,head=4)
arrow([(225,599.5),(245,599.5),(245,531)],GRAY,.65,head=4)
ax.add_patch(Circle((245,521),10,fc='white',ec=GRAY,lw=.65,zorder=6));text(245,521,'+',size=8.)
arrow([(255,521),(280,521)],INK,.85)
# A base-state skip runs above the trainable applications; no hidden direct first-pass skip.
arrow([(266,521),(266,409),(735,409),(735,503)],GRAY,.7,head=4);text(485,403,'b',size=8.,color=GRAY)
def gnn(x,number):
 box(x,455,130,132,'#f8fafc',edge=LIGHT)
 box(x+8,464,114,49,PST[0]);text(x+65,488.5,r'↑ GNN$_{\theta_u}$',size=8.1)
 box(x+8,528,114,49,PST[2]);text(x+65,552.5,r'↓ GNN$_{\theta_d}$',size=8.1)
 arrow([(x+65,513),(x+65,528)],GRAY,.65,head=4)
 text(x+65,608,'('+str(number)+')',size=8.,color=GRAY)
gnn(280,1)
text(592,432,r'shared $\theta_u,\theta_d$',size=8.,color=GRAY)
line([(410,432),(513,432)],GRAY,.5,ls='--')
line([(672,432),(800,432)],GRAY,.5,ls='--')
line([(410,432),(410,455)],GRAY,.5,ls='--')
line([(800,432),(800,455)],GRAY,.5,ls='--')
line([(410,521),(424,521)],INK,.8)
arrow([(424,521),(424,472),(455,472)],INK,.8)
arrow([(424,521),(424,573),(455,573)],INK,.8)
box(455,448,165,49,PST[1]);text(537.5,472,'local residual',size=8.)
box(455,548,165,49,PST[3]);text(531,572,'memory',size=8.3)
ax.add_patch(Circle((602,573),10,fc='white',ec=GRAY,lw=.65,zorder=6));text(602,573,'c',size=8.,color=GRAY)
arrow([(620,472),(680,472),(680,511)],INK,.8)
arrow([(620,573),(680,573),(680,531)],INK,.8)
text(650,590,'δ',size=8.,color=GRAY)
ax.add_patch(Circle((680,521),10,fc='white',ec=INK,lw=.7,zorder=6));text(680,521,'+',size=8.)
arrow([(690,521),(717,521)],INK,.75,head=4)
ax.add_patch(Circle((735,521),18,fc='white',ec=INK,lw=.85,zorder=6));text(735,521,'+',size=13)
arrow([(753,521),(800,521)],INK,.85)
gnn(800,2)
arrow([(930,521),(950,521)],INK,.85)
box(950,477,92,89,PST[1]);text(996,499,'Linear',size=8.)
line([(957,521),(1035,521)],LIGHT,.5);text(996,542,'tanh',size=8.)
arrow([(1042,521),(1054,521)],INK,.75,head=3)
# Every column and output strip now shares the same genomic direction.
for i in range(3):
 interval_strip(1090+i*59,421,44,6,i);text(1112+i*59,440,'T'+str(i+1),size=8.)
for j,name in enumerate(NAMES):
 text(1075,468+j*28,name,size=8.,color=QUERY if j==0 else GRAY)
 for i in range(3):vector(1090+i*59,460+j*28,w=44,h=16,query=j==0)
box(1056,452,219,146,'none',edge=LIGHT,r=2,lw=.5,z=1)
text(1165,619,'e',size=8.,color=GRAY)
# Dashed callout is explicitly a detail linkage, not a feature edge.
line([(537.5,597),(537.5,658),(537.5,690)],GRAY,.65,ls='--')
ax.add_patch(Circle((537.5,674),10,fc='white',ec=GRAY,lw=.65,zorder=6));text(537.5,674,'c',size=8.,color=GRAY)

# c: the actual T1 cut, explicit context broadcast, identity-aligned EMA and ancestor return.
panel(18,668,'c')
box(10,690,1020,304,'none',edge=LIGHT,r=7,lw=.5,z=1)
text(88,703,'T1',size=8.)
inset=trees[0];ip={u:(40+110*sx(inset,u),779-.10*ts.node(u).time)for u in inset.nodes()}
for u in inset.nodes():
 parent=inset.parent(u)
 if parent!=tskit.NULL:line([ip[u],ip[parent]],GRAY,.65)
line([(36,744),(154,744)],GRAY,.6,ls='--')
for u,pt in ip.items():dot(*pt,sample=u if u<5 else None,r=2.4)
ax.add_patch(Circle(ip[5],5,fc='#fff5e5',ec=REP,lw=1.,zorder=8))
arrow([ip[5],(ip[5][0],789),(92,789),(92,800)],REP,.65,head=4)
box(36,800,112,39,PST[3]);text(92,819.5,'Linear U',size=8.)
arrow([(148,819.5),(196,819.5)],REP,.75)
vector(160,815.5,w=31,h=8,outline=REP);text(176,803,'f',size=8.,color=REP)
# Copy f into both descendants q,r1 of the highlighted native ancestor (node 5).
line([(196,819.5),(202,819.5)],REP,.65)
arrow([(202,819.5),(202,775),(278,775),(278,789)],REP,.7,head=4)
arrow([(202,819.5),(202,853),(278,853),(278,839)],REP,.7,head=4)
cols=[243,421,599];ys=[796,832,868,904,940]
partitions=[[[0,1],[2],[3,4]],[[0],[1,2],[3,4]],[[0],[1,2],[3,4]]]
box(230,769,460,188,'none',edge=LIGHT,r=7,lw=.5,z=1)
text(462,711,'↔ EMA',size=8.5)
for i,x in enumerate(cols):
 text(x+35,744,'T'+str(i+1),size=8.)
 for group in partitions[i]:
  top=ys[min(group)]-15;bottom=ys[max(group)]+15
  box(x-5,top,83,bottom-top,'#eef2f6',edge='#aebdc9',r=6,lw=.55)
 for j in range(5):
  vector(x+7,ys[j]-7,w=59,h=14,query=j==0)
  if i==0:text(x-23,ys[j],NAMES[j],size=8.,color=QUERY if j==0 else GRAY)
box(cols[0]-5,ys[0]-15,83,ys[1]-ys[0]+30,'none',edge=REP,r=6,lw=.7,z=4)
for i in range(2):
 for j in range(5):arrow([(cols[i]+82,ys[j]),(cols[i+1]-12,ys[j])],QUERY if j==0 else '#9aabb9',.65,style='<->',head=4)
# Genomic widths remain exactly proportioned 3:4:3.
x0=299
for i,span in enumerate([90,120,90]):interval_strip(x0,966,span,9,i);x0+=span
text(652,971,'Δx',size=8.,color=GRAY);arrow([(449,966),(449,957)],GRAY,.6,head=4)
# This mean is over EMA directions and the current ancestor's sampled group.
arrow([(321,783),(335,783),(335,763),(719,763),(719,868),(749,868)],INK,.75)
box(749,838,114,60,PST[3]);text(806,854,'mean',size=8.);text(806,882,'− f',size=8.)
arrow([(863,868),(883,868)],INK,.75)
box(883,846,108,45,PST[3]);text(937,868,'Linear O',size=8.)
arrow([(991,868),(1012,868)],REP,.75,head=4);text(1006,842,'δ',size=8.,color=REP)
# A single outside return route ends at the same highlighted native ancestor.
arrow([(1012,868),(1012,986),(23,986),(23,ip[5][1]),(ip[5][0]-5,ip[5][1])],REP,.7,head=4)

# d: all-sample readout is top to bottom; q selects the displayed output.
panel(1055,630,'d')
arrow([(1165,598),(1165,682)],INK,.8)
for j,span in enumerate([27,36,27]):interval_strip(1060+sum([27,36,27][:j]),650,span,9,j)
text(1105,636,'Δx',size=8.,color=GRAY)
arrow([(1150,654.5),(1185,654.5),(1185,682)],GRAY,.65,head=4)
box(1085,682,190,55,PST[0]);text(1180,699,'BiGRU',size=8.5);text(1180,722,'1 layer',size=8.)
arrow([(1180,737),(1180,756)],INK,.8)
box(1085,756,190,34,'#f1f4f7');text(1180,773,'concat',size=8.)
arrow([(1275,598),(1290,598),(1290,773),(1275,773)],GRAY,.75,head=4);text(1282,744,'e',size=8.,color=GRAY)
arrow([(1180,790),(1180,808)],INK,.8)
box(1085,808,190,46,PST[1]);text(1180,831,'MLP',size=8.5)
arrow([(1180,854),(1180,861)],INK,.7,head=3)
ax.add_patch(Circle((1180,873),11,fc='white',ec=QUERY,lw=.75,zorder=6));text(1180,873,'q',size=8.,color=QUERY)
arrow([(1180,884),(1180,898)],INK,.7,head=3)
px=1085;pw=190;py=898;ph=36
for row,span in zip(record['illustrative_probabilities'],[.3,.4,.3]):
 h=0
 for prob,col in zip(row,ANC):rect(px,py+ph*(1-h-prob),pw*span,ph*prob,col,edge='white',lw=.3);h+=prob
 px+=pw*span
text(1080,916,'p(q)',size=8.,color=QUERY,ha='right')
for i,(name,col)in enumerate(zip(AN,ANC)):
 x=1085+(i%2)*106;y=939+(i//2)*24;rect(x,y,11,11,col,edge=col);text(x+15,y+6,name,ha='left',size=8.)
px=1085
for span,call in zip([.3,.4,.3],[0,2,3]):rect(px,984,pw*span,10,ANC[call],edge='white',lw=.4);px+=pw*span
text(1066,986,'q',size=8.,color=QUERY)
# Exports and science-preservation checks.
fig.canvas.draw();renderer=fig.canvas.get_renderer()
bounds=[]
for t in texts:
 bb=t.get_window_extent(renderer);bounds.append(dict(text=t.get_text(),font_points=t.get_fontsize(),pixel_bounds=[bb.x0,bb.y0,bb.x1,bb.y1]))
assert min(t.get_fontsize()for t in texts)>=8
for ext in ('svg','pdf','png'):fig.savefig(OUT/('Figure1_v15.'+ext),dpi=200)
fig.savefig(OUT/'Figure1_v15_2x.png',dpi=400)
fig.set_size_inches(7,7*H/W)
fig.savefig(OUT/'Figure1_v15_print100.png',dpi=100)
fig.set_size_inches(6.5,5.0)
(OUT/'text_bounds.v1.json').write_text(json.dumps(dict(width_inches=6.5,height_inches=5.0,min_font_points=8.,texts=bounds),indent=2)+'\n')
checks=dict(native_tskit_tree_sequence_valid=True,all3_marginals_from_same_recorded_ARG=True,leaf_order_identical_all_views=NAMES,parents_identical_to_recorded_native_edges=True,all_plane_and_marginal_branch_ages_same=True,only_same_native_node_IDs_have_continuity_lines=True,probabilities_illustrative_not_measured=True,editable_vector_text=True,printed_inches=[6.5,5.0],minimum_font_points=8.,no_panel_titles=True,model_executed=False,all_genomic_views_T1_T2_T3_left_to_right=True,gnn_applications=2,shared_gnn_parameters=True,memory_context_broadcast_descendants=[0,1],memory_delta_returns_to_native_ancestor=5,independent_image_review_pending=True)
(OUT/'geometry_checks.v1.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
