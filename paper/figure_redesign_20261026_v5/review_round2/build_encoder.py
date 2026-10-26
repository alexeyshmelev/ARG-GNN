"""Draw the frozen TreeLoom tensor contract. No model or simulation execution."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
OUT=Path(__file__).resolve().parent
INK='#26363b'
GOLD='#f8e6b8'
GOLD_EDGE='#8d6826'
FIX='#f5f6f6'
TEAL=LinearSegmentedColormap.from_list('embeddings',['#e7f6f8','#63b9c6','#145965'])
GRAY=LinearSegmentedColormap.from_list('observed',['#fafafa','#a8b0b6','#34434a'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7,8))
ax=fig.add_axes([0,0,1,1])
ax.set(xlim=(0,504),ylim=(576,0))
ax.axis('off')
texts=[]
def text(x,y,s,size=7.2,bold=False,color=INK):
    z=ax.text(x,y,s,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal',color=color,zorder=10)
    texts.append(z)
def box(x,y,w,h,s='',learn=False,size=7.2):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=GOLD if learn else FIX,edgecolor=GOLD_EDGE if learn else INK,lw=.65,zorder=4))
    if s:text(x+w/2,y+h/2,s,size)
def panel(x,y,w,h,label):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=INK,lw=.75,zorder=0))
    box(x,y,18,14,label,False,7.5)
def arrow(points,lw=.7):
    if len(points)>2:ax.plot(*zip(*points[:-1]),color=INK,lw=lw,zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=5,color=INK,lw=lw,shrinkA=0,shrinkB=0,zorder=3))
def mat(x,y,w,h,label,rows=1,cols=4,kind='emb',values=None,body=None):
    if body is not None and h<=16:
        shown=label if body=='1×1' else label+'='+body
        box(x,y,w,h,shown,size=7)
        return
    header=min(12,h*.55)
    text(x+w/2,y+header/2,label,7)
    if body is not None:
        ax.add_patch(Rectangle((x,y+header),w,h-header,facecolor='white',edgecolor=INK,lw=.5,zorder=4))
        text(x+w/2,y+header+(h-header)/2,body,7)
        return
    if values is None:
        values=.28+.6*(.5+.5*np.sin(np.arange(rows*cols).reshape(rows,cols)*1.73+.7))
    cmap=TEAL if kind=='emb' else GRAY
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x+w*j/cols,y+header+(h-header)*i/rows),w/cols,(h-header)/rows,facecolor=cmap(float(values[i,j])),edgecolor='white',lw=.3,zorder=5))
def op_arrow(x1,y,x2):arrow([(x1,y),(x2,y)])
def named_rows(x,y,w,h,label,names,zero_rows=()):
    if h < 6*len(names)+7:
        mat(x,y,w,h,label,len(names),4)
        return
    head=min(12,h-6*len(names))
    text(x+w/2,y+head/2,label,7)
    rowh=(h-head)/len(names)
    labelw=min(28,w*.36)
    for i,name in enumerate(names):
        box(x,y+head+i*rowh,labelw,rowh,name,size=7)
        for j in range(4):
            color='white' if name in zero_rows else TEAL(.28+.16*j)
            ax.add_patch(Rectangle((x+labelw+(w-labelw)*j/4,y+head+i*rowh),(w-labelw)/4,rowh,facecolor=color,edgecolor='#c9d6d9',lw=.3,zorder=5))
        if name in zero_rows:text(x+labelw+(w-labelw)/2,y+head+(i+.5)*rowh,'0',7)
def row_token(x,y,w,label):box(x,y,w,14,label,size=7)
nodes=['q','r1','r2','r3','r4','a','b','c','root']
# ENC is the exact parent block. U and O have explicit, separate routes.
panel(6,5,492,56,'')
box(6,5,18,13,'a',size=7.5)
box(24,5,474,13,'ENC — expansion of Figure 1c',size=7.5)
mat(12,32,31,25,'B:9×32',9,4)
box(50,32,30,25,'M(b)',True,7)
box(12,19,68,11,'G↑,G↓:8×32',size=7)
arrow([(65,30),(65,32)])
arrow([(43,44),(50,44)])
# One fork produces two matrices, enclosed as a pair of outputs.
ax.add_patch(Rectangle((91,19),44,42,fill=False,edgecolor=INK,lw=.5,zorder=1))
mat(94,23,38,17,'U¹:9×32',9,4)
mat(94,43,38,17,'O¹:9×32',9,4)
arrow([(80,44),(86,44),(86,32),(94,32)])
arrow([(86,44),(86,52),(94,52)])
# Only U enters structure. Both U and O enter memory.
mat(173,23,47,17,'S(d):9×32',9,4)
mat(173,43,47,17,'Δ(c):9×32',9,4)
arrow([(132,32),(173,32)])
box(148,44,18,15,'pair',size=7)
arrow([(140,32),(140,47),(148,47)])
arrow([(132,52),(148,52)])
arrow([(166,52),(173,52)])
# Add corrections to the original base, then repeat with exactly the same maps.
mat(238,19,52,12,'B:9×32',9,4)
box(238,39,52,18,'B+S+Δ',size=7)
arrow([(264,31),(264,39)])
arrow([(220,32),(229,32),(229,43),(238,43)])
arrow([(220,52),(238,52)])
box(299,32,49,25,'M(b)\nsame weights',True,7)
box(299,19,49,11,'G↑,G↓',size=7)
arrow([(323,30),(323,32)])
arrow([(290,47),(299,47)])
ax.add_patch(Rectangle((359,19),44,42,fill=False,edgecolor=INK,lw=.5,zorder=1))
mat(362,23,38,17,'U²:9×32',9,4)
mat(362,43,38,17,'O²:9×32',9,4)
arrow([(348,44),(354,44),(354,32),(362,32)])
arrow([(354,44),(354,52),(362,52)])
box(407,24,13,34,'')
z=ax.text(413.5,41,'pair',ha='center',va='center',fontsize=7,rotation=90,color=INK,zorder=10)
texts.append(z)
arrow([(400,32),(407,32)])
arrow([(400,52),(407,52)])
box(428,32,60,25,'U²,O² leaves\n→ Figure 1',size=7)
arrow([(420,41),(428,41)])
# Complete message operator. Child order and node placement are explicit.
panel(6,67,492,187,'b')
box(26,67,472,15,'M: all nodes and trees, distinct ↑ / ↓ maps, same maps in both rounds',size=7.3)
# Normalized saved messages from r1,r2 into their parent b.
mat(14,87,43,23,'U(r1,r2)',2,4)
box(64,87,44,23,'Linear\n32 → 32',True,7)
mat(115,87,37,23,'Z:2×32',2,4)
mat(160,84,48,14,'g↑:2×32',2,4)
box(160,101,48,13,'g↑ ⊙ Z',size=7)
arrow([(184,98),(184,101)])
arrow([(152,99),(156,99),(156,108),(160,108)])
mat(216,84,23,14,'d',kind='obs',body='2')
box(216,101,39,13,'÷ d',size=7)
arrow([(228,98),(228,101)])
op_arrow(208,108,216)
named_rows(263,84,61,30,'m:2×32',['r1','r2'])
box(332,87,58,23,'Sum rows',size=7)
mat(398,87,90,23,'A(b):1×32',1,5)
for xa,xb in [(57,64),(108,115),(255,263),(324,332),(390,398)]:op_arrow(xa,99 if xa not in (255,) else 108,xb)
# Upward base residual with each actual operand tensor.
mat(14,121,37,23,'B(b):32')
mat(58,121,37,23,'A(b):32')
box(102,121,43,23,'Concat\n32 + 32',size=7)
arrow([(51,132),(54,132),(54,147),(123,147),(123,144)])
op_arrow(95,132,102)
mat(152,121,33,23,'x:64')
box(192,121,68,23,'Linear 64 → 32\ntanh',True,7)
mat(267,121,33,23,'z:32')
mat(307,118,39,14,'B(b):32')
box(307,135,39,13,'z+B(b)',size=7)
arrow([(326,132),(326,135)])
arrow([(300,132),(303,132),(303,141),(307,141)])
box(354,121,58,23,'LayerNorm',True,7)
mat(420,121,68,23,'U(b):32')
for xa,xb in [(145,152),(185,192),(260,267),(346,354),(412,420)]:op_arrow(xa,141 if xa==346 else 132,xb)
# Four visible column vectors enter the recipient-exclusion formula.
for j,lab in enumerate(['O(b)','B(b)','A(b)','m(r1)']):mat(14+j*26,153,24,23,lab,1,4)
box(126,153,81,23,'O(b)+B(b)+A(b)\n−m(r1)',size=7)
mat(214,153,30,23,'C:32')
box(251,153,44,23,'Linear\n32 → 32',True,7)
mat(302,153,30,23,'z:32')
mat(339,150,39,14,'g↓:32')
box(339,167,39,13,'g↓ ⊙ z',size=7)
arrow([(358,164),(358,167)])
arrow([(332,164),(335,164),(335,174),(339,174)])
box(385,153,64,23,'Linear + tanh\nLayerNorm',True,7)
mat(456,153,32,23,'O(r1)')
for xa,xb in [(118,126),(207,214),(244,251),(295,302),(378,385),(449,456)]:op_arrow(xa,174 if xa==378 else 164,xb)
# Iteration order is a directed list, and the collected result has every node row.
for j,lab in enumerate(['leaves: A','a, b','c','root']):
    row_token(14,187+j*16,45,lab)
    if j:arrow([(36,185+j*16),(36,187+j*16)])
text(70.5,190,r'$0_{32}$',7)
for j in range(4):box(62+17*j/4,195,17/4,6,'',size=7)
arrow([(62,198),(59,198)])
named_rows(80,178,161,72,'U: stack all node results × 32',nodes)
arrow([(59,220),(80,220)])
for j,lab in enumerate(['root=0','a, c','q,b,r3,r4','r1,r2']):
    row_token(252,187+j*16,62,lab)
    if j:arrow([(283,185+j*16),(283,187+j*16)])
named_rows(334,178,154,72,'O: same nine node rows × 32',nodes,zero_rows=['root'])
arrow([(314,220),(334,220)])
# Stable-leaf memory. F is group context, f_t is one persistent leaf's copied row.
panel(6,260,492,190,'c')
box(26,260,472,15,'Memory: group context → leaf recurrence → group correction',size=7.3)
# Three observed group assignments. The selected r1 column changes d -> b -> b.
box(14,280,108,56,'')
box(14,280,108,10,'Groups at 350 generations',size=7)
for j,lab in enumerate(['','q','r1','r2','r3','r4']):
    box(14+18*j,290,18,9,lab,size=7)
assignments=[['T1','d','d','r2','a','a'],['T2','q','b','b','a','a'],['T3','q','b','b','a','a']]
for i,row in enumerate(assignments):
    for j,lab in enumerate(row):box(14+18*j,299+8*i,18,8,lab,size=7)
ax.add_patch(Rectangle((50,290),18,33,fill=False,edgecolor=INK,lw=1.4,zorder=12))
box(14,323,108,13,'d:200   a:175   b:212.5',size=7)
# Named first-round source ports feed a gather operation. The table supplies indices.
mat(131,277,24,12,'U¹',9,4)
mat(160,277,24,12,'O¹',9,4)
box(131,292,53,19,'T₂: gather rows\n[q, b, a]',size=7)
arrow([(143,289),(143,292)])
arrow([(172,289),(172,292)])
arrow([(122,303),(131,303)])
mat(191,282,27,29,'U:3×32',3,4)
mat(225,282,27,29,'O:3×32',3,4)
arrow([(184,301),(191,301)])
arrow([(184,296),(187,296),(187,278),(238,278),(238,282)])
box(260,282,14,29,'+')
arrow([(218,296),(221,296),(221,313),(267,313),(267,311)])
op_arrow(252,296,260)
box(281,282,45,29,'Linear\n32 → 8',True)
named_rows(333,282,37,29,'F₂:3×8',['q','b','a'])
box(377,282,55,29,'Copy rows\nq b b a a',size=7)
mat(440,282,48,29,'V₂:5×8',5,4)
for xa,xb in [(274,281),(326,333),(370,377),(432,440)]:op_arrow(xa,296,xb)
# V ports, not group identifiers, feed the context stack.
for j,lab in enumerate(['V₁','V₂','V₃']):
    mat(131+20*j,315,18,11,lab,5,4)
box(131,329,59,12,'stack, select r1',size=7)
for xx in [140,160,180]:arrow([(xx,326),(xx,329)])
named_rows(197,315,51,25,'f(r1):3×8',['T1:d','T2:b','T3:b'])
arrow([(190,333),(197,333)])
# Learned scalar, bounded map, length scalar, observed interval vector, decay.
box(255,317,20,21,'θ',True,7.5)
box(282,317,72,21,'100 + 99,900 ×\nsigmoid(θ)',size=7)
mat(361,317,29,21,'ℓ (eg)',body='5000')
mat(397,317,31,21,'L:bp',1,3,kind='obs',values=np.array([[.3,.4,.3]]))
for j,v in enumerate(['3k','4k','3k']):text(397+(j+.5)*31/3,333,v,7)
box(435,317,53,21,'exp(−L/ℓ)',size=7)
for xa,xb in [(275,282),(354,361),(428,435)]:op_arrow(xa,327,xb)
arrow([(390,327),(393,327),(393,340),(461,340),(461,338)])
# Recurrence operands sit physically above each update formula.
def ema_cell(x,y,w,alpha,previous,forcing):
    mat(x,y,w*.36,11,alpha,body='1×1')
    mat(x+w*.44,y,w*.56,11,forcing+'='+{'f₁':'F₁[d]','f₂':'F₂[b]','f₃':'F₃[b]'}[forcing],1,4)
    box(x,y+14,w,15,alpha+previous+'+(1−'+alpha+')'+forcing,size=7)
    arrow([(x+w*.18,y+11),(x+w*.18,y+14)])
    arrow([(x+w*.72,y+11),(x+w*.72,y+14)])
# r1: independent left-to-right and right-to-left recurrences.
mat(14,344,33,29,'f₁=F₁[d]')
box(54,344,31,29,'Copy')
mat(92,344,34,29,'m₁→:8')
ema_cell(137,344,98,'α₂','m₁→','f₂')
mat(242,344,34,29,'m₂→:8')
ema_cell(287,344,98,'α₃','m₂→','f₃')
mat(392,344,40,29,'m₃→:8')
text(464,349,'α₁ α₂ α₃',7)
for j,v in enumerate(['.55','.45','.55']):box(440+16*j,355,16,12,v,size=7)
arrow([(461,338),(461,344)])
for xa,xb in [(47,54),(85,92),(126,137),(235,242),(276,287),(385,392)]:op_arrow(xa,365,xb)
mat(14,376,34,29,'m₁←:8')
ema_cell(55,376,98,'α₁','m₂←','f₁')
mat(164,376,34,29,'m₂←:8')
ema_cell(205,376,98,'α₂','m₃←','f₂')
mat(314,376,34,29,'m₃←:8')
box(355,376,31,29,'Copy')
mat(393,376,39,29,'f₃=F₃[b]')
for xa,xb in [(55,48),(164,153),(205,198),(314,303),(355,348),(393,386)]:arrow([(xa,397),(xb,397)])
# Current tree T2: direction mean, group mean, centering, learned projection, scatter.
box(14,407,54,16,'Repeat for\nq r1 r2 r3 r4',size=7)
mat(14,424,24,18,'m₂→',5,4)
mat(44,424,24,18,'m₂←',5,4)
box(75,413,53,29,'Mean\ndirections',size=7)
named_rows(135,413,41,29,'M₂:5×8',['q','r1','r2','r3','r4'])
box(183,413,56,29,'Group mean\nq / b / a',size=7)
named_rows(246,413,41,29,'J₂:3×8',['q','b','a'])
named_rows(294,413,36,29,'F₂:3×8',['q','b','a'])
box(337,413,28,29,'J₂−F₂')
arrow([(287,428),(289,428),(289,447),(351,447),(351,442)])
op_arrow(330,428,337)
box(372,413,47,29,'Linear\n8 → 32',True,7)
named_rows(440,372,48,74,'Δ:scatter',nodes,zero_rows=['r1','r2','r3','r4','c','root'])
arrow([(419,428),(440,428)])
arrow([(38,438),(41,438),(41,446),(101,446),(101,442)])
arrow([(68,438),(75,438)])
for xa,xb in [(128,135),(176,183),(239,246),(365,372)]:op_arrow(xa,428,xb)
# Structural adapters are per-node, with explicit local scatter before summing nodes.
panel(6,456,492,93,'d')
box(26,456,472,15,'Shared correction maps: repeat proper clades and intervals, sum scattered rows → S',size=7.3)
# Flags are computed for the present clade at T2, using explicit adjacent bits.
box(14,478,84,65,'')
box(14,478,84,12,'C={r1,r2}, at T₂',size=7)
box(14,490,84,9,'2 ≤ |C| < 5',size=7)
for j,(t,role,value) in enumerate([('T₁','prev','0'),('T₂','now','1'),('T₃','next','1')]):
    box(14+28*j,499,28,9,t,size=7)
    box(14+28*j,508,28,9,role,size=7)
    box(14+28*j,517,28,9,value,size=7)
ax.add_patch(Rectangle((42,499),28,27,fill=False,edgecolor=INK,lw=1.1,zorder=12))
box(14,526,84,17,'consecutive run: 2\ne=1 when run ≥ 2',size=7)
mat(106,479,35,26,'u=U¹(b)')
box(148,479,44,26,'3 Linear\n32 → 8',True,7)
mat(199,479,44,26,'x,y,z:8',3,4)
box(250,476,36,11,'g=1−prev',size=7)
box(289,476,36,11,'l=1−next',size=7)
box(250,490,75,19,'g(x⊙y)−l(x⊙z)',size=7)
arrow([(268,487),(268,490)])
arrow([(307,487),(307,490)])
arrow([(243,492),(250,500)])
box(332,479,43,26,'Linear\n8 → 32',True,7)
mat(382,479,28,26,'s_c:32')
for xa,xb in [(141,148),(192,199),(325,332),(375,382)]:op_arrow(xa,492 if xa!=325 else 500,xb)
mat(106,516,35,26,'u=U¹(b)')
box(148,516,44,26,'3 Linear\n32 → 8',True,7)
mat(199,516,44,26,"x′,y′,z′:8",3,4)
box(250,516,75,26,"x′⊙y′ + z′",size=7)
box(332,516,43,26,'Linear\n8 → 32',True,7)
mat(382,513,28,11,'e',kind='obs',body='1')
box(382,527,28,15,'× e',size=7)
arrow([(396,524),(396,527)])
arrow([(375,529),(382,535)])
for xa,xb in [(141,148),(192,199),(243,250),(325,332)]:op_arrow(xa,529,xb)
box(416,506,16,18,'+')
arrow([(410,492),(424,492),(424,506)])
arrow([(410,535),(424,535),(424,524)])
named_rows(440,473,48,72,'scatter b',nodes,zero_rows=['q','r1','r2','r3','r4','a','c','root'])
arrow([(432,515),(440,515)])
# The two concrete T2 operations are repeated separately for every interval.
ax.add_patch(Rectangle((129,276),361,38,fill=False,edgecolor=INK,lw=.45,zorder=1))
z=ax.text(494.2,295,'t=1,2,3',ha='center',va='center',fontsize=7,rotation=90,color=INK,zorder=10)
texts.append(z)
ax.plot([489.5,491,491,489.5],[371,371,447,447],color=INK,lw=.5,zorder=1)
z=ax.text(495,409,'Δₜ for t=1,2,3',ha='center',va='center',fontsize=7,rotation=90,color=INK,zorder=10)
texts.append(z)
# Every color is explained. Matrices are schematic, scalar examples use initial ell.
box(6,557,113,14,'Gold: learned layer',True,7)
box(125,557,112,14,'Gray box: fixed step',size=7)
mat(243,557,120,14,'Teal: embedding',1,4)
mat(369,557,129,14,'Gray cells: observations',1,4,kind='obs')
fig.canvas.draw()
ren=fig.canvas.get_renderer()
for t in texts:
    b=t.get_window_extent(ren).transformed(fig.transFigure.inverted())
    assert b.x0>=0 and b.y0>=0 and b.x1<=1 and b.y1<=1,t.get_text()
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/('TreeLoom_encoder.'+ext),dpi=260,facecolor='white')
plt.close(fig)
print('Wrote TreeLoom_encoder PDF/SVG/PNG')
