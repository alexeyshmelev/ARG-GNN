"""Draw the compact TreeLoom encoder explanation. No model execution."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

OUT = Path(__file__).resolve().parent
INK = '#26363b'
GOLD = '#f8e6b8'
GOLD_EDGE = '#8d6826'
FIX = '#f3f5f5'
TEALS = ['#e2f0f2', '#a2d2d8', '#56a8b6', '#247786']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
    'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(7,4))
ax = fig.add_axes([0,0,1,1])
ax.set(xlim=(0,504), ylim=(288,0))
ax.axis('off')

def text(x,y,s,size=8,bold=False,ha='center'):
    return ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=INK,
        fontweight='bold' if bold else 'normal',zorder=9)

def rect(x,y,w,h,label='',learn=False,lw=.7):
    ax.add_patch(Rectangle((x,y),w,h,fc=GOLD if learn else FIX,
        ec=GOLD_EDGE if learn else INK,lw=lw,zorder=3))
    if label: text(x+w/2,y+h/2,label)

def line(points,lw=.7,color=INK):
    ax.plot(*zip(*points),lw=lw,c=color,zorder=6)

def arrow(points,lw=.8,style='-|>'):
    if len(points)>2: line(points[:-1],lw)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle=style,
        mutation_scale=6,lw=lw,color=INK,shrinkA=0,shrinkB=0,zorder=6))

def vector(x,y,w=28,h=8,offset=0):
    for j in range(4):
        ax.add_patch(Rectangle((x+j*w/4,y),w/4,h,
            fc=TEALS[(j+offset)%4],ec='white',lw=.3,zorder=4))
    ax.add_patch(Rectangle((x,y),w,h,fc='none',ec=INK,lw=.5,zorder=5))

def matrix(x,y,w,h):
    for r in range(6): vector(x,y+r*h/6,w,h/6,offset=r)

def badge(x,y,s):
    rect(x,y,13,13,s)

def two_round(x,y,index):
    rect(x,y,76,36,learn=True)
    text(x+38,y+7,f'Round {index}',bold=True)
    text(x+38,y+19,'↑ then ↓')
    text(x+38,y+30,'tree messages')

# a: The whole encoder, with the original-base path visibly separate.
text(9,9,'a',bold=True)
text(40,23,'each tree')
matrix(23,32,32,32)
arrow([(55,48),(71,48)])
two_round(71,30,1)
arrow([(147,48),(156,48),(156,37),(170,37)])
arrow([(156,48),(156,61),(170,61)])
rect(170,27,91,20,'Clade pattern',learn=True)
badge(170,27,'b')
rect(170,51,91,20,'Leaf memory',learn=True)
badge(170,51,'c')
# A junction receives both corrections, but not the first-round states.
arrow([(261,37),(272,37),(272,48),(284,48)])
arrow([(261,61),(272,61),(272,48),(284,48)])
ax.add_patch(Circle((291,48),7,fc=FIX,ec=INK,lw=.7,zorder=3))
text(291,48,'+',size=10)
line([(39,64),(39,81),(291,81)])
arrow([(291,81),(291,55)])
text(161,81,'original node states',size=8)
# White backing keeps the label attached to its long residual route.
for t in ax.texts[-1:]: t.set_bbox({'facecolor':'white','edgecolor':'none','pad':1})
arrow([(298,48),(317,48)])
two_round(317,30,2)
arrow([(393,48),(411,48)])
matrix(411,32,32,32)
text(427,23,'All node states')
# Same weights only connects the two tree rounds, not the different modules.
line([(109,27),(109,12),(355,12),(355,27)])
text(232,12,'weights shared across rounds and trees')
ax.texts[-1].set_bbox({'facecolor':'white','edgecolor':'none','pad':2})
# The output is the all-node matrix itself.

# b: One clade is compared by its member leaves, not ancestor identity.
# Whitespace separates the main encoder from its two expansions.
badge(8,101,'b')
rect(28,99,133,65)
text(94,108,'same leaf set {r1, r2}')
for x,t,present in [(42,'T₁',False),(82,'T₂',True),(122,'T₃',True)]:
    text(x+12,122,t)
    if present:
        line([(x+3,146),(x+3,137),(x+21,137),(x+21,146)])
        ax.add_patch(Circle((x+12,137),1.6,fc=INK,ec=INK,lw=.4,zorder=4))
    else:
        line([(x+3,146),(x+3,139)])
        line([(x+21,146),(x+21,139)])
    ax.add_patch(Circle((x+3,146),1.3,fc='white',ec=INK,lw=.6,zorder=4))
    ax.add_patch(Circle((x+21,146),1.3,fc='white',ec=INK,lw=.6,zorder=4))
    text(x+12,157,'yes' if present else 'no')
arrow([(161,138),(181,138)])
rect(181,121,88,35,'Clade change\n+ persistence',learn=True)
# Current node embedding is a separate input attached to its operation.
vector(210,107,29,8)
arrow([(224.5,115),(224.5,121)])
text(224,99,'current node in T₂')
arrow([(269,138),(291,138)])
vector(291,132,32,12,offset=1)
arrow([(323,138),(412,138)])
text(414,122,'T₂')
line([(414,138),(414,147)])
line([(389,160),(389,147),(439,147),(439,160)])
ax.add_patch(Circle((414,138),2,fc=INK,ec=INK,lw=.4,zorder=7))
text(426,132,'b')
for x,label in [(389,'r1'),(439,'r2')]:
    ax.add_patch(Circle((x,160),1.5,fc='white',ec=INK,lw=.6,zorder=7))
    text(x,169,label)
text(307,158,'correction')

# c: Representative context is copied to leaves and follows persistent leaf IDs.
line([(7,174),(497,174)],lw=.45,color='#bbc6c8')
badge(8,180,'c')
rect(28,181,125,73)
text(90,190,'age-cut groups → leaves')
# Three source groups, displaying their member labels explicitly.
for x,t,rep,members in [(37,'T₁','d',('q','r1')),(77,'T₂','b',('r1','r2')),(117,'T₃','b',('r1','r2'))]:
    text(x+10,204,t)
    vector(x+2,211,16,6,offset={'T₁':1,'T₂':2,'T₃':3}[t])
    text(x+10,224,rep)
    line([(x+10,228),(x+10,232),(x,232),(x,237)])
    line([(x+10,232),(x+20,232),(x+20,237)])
    text(x,244,members[0])
    text(x+20,244,members[1])
arrow([(153,217),(165,217)])
rect(165,181,125,73,learn=True)
text(227.5,191,'each leaf: r1 shown')
# Endpoint positions and paired arrows convey one forward and one reverse memory.
for x,label,off in [(173,'T₁',1),(214,'T₂',2),(255,'T₃',3)]:
    text(x+11,205,label)
    vector(x,213,22,9,off)
arrow([(184,209),(266,209)])
arrow([(266,228),(184,228)])
text(227.5,237,'genomic distance decay')
text(227.5,249,'mean directions')
arrow([(290,217),(302,217)])
rect(302,190,81,55,'Mean within\ncurrent group\n{r1, r2}')
arrow([(383,217),(396,217)])
rect(396,190,61,55,'Group\ncorrection',learn=True)
arrow([(457,217),(479,217)])
text(481,199,'T₂')
line([(481,217),(481,226)])
line([(467,237),(467,226),(495,226),(495,237)])
ax.add_patch(Circle((481,217),2,fc=INK,ec=INK,lw=.4,zorder=7))
text(490,210,'b')
for x,label in [(467,'r1'),(495,'r2')]:
    ax.add_patch(Circle((x,237),1.5,fc='white',ec=INK,lw=.6,zorder=7))
    text(x,248,label)

# Compact semantic legend, once for this standalone figure.
rect(51,277,12,6,learn=True)
text(68,280,'learned',ha='left')
rect(134,277,12,6)
text(151,280,'fixed / observed',ha='left')
vector(275,277,15,6)
text(295,280,'embedding',ha='left')
text(411,280,'schematic vectors')

for ext in ('png','pdf','svg'):
    fig.savefig(OUT/f'TreeLoom_encoder.{ext}',dpi=240,facecolor='white')
print(OUT/'TreeLoom_encoder.png')
