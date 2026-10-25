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
W, H = 7.0, 6.2
fig = plt.figure(figsize=(W, H))
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

# (b) The encoder is a single horizontal computation with one clean base bypass.
panel(.12, 2.23, 'b')
# Common observed-input collector, with a separate topology note in the caption.
line([(.44, 2.04), (.44, 2.11), (6.58, 2.11), (6.58, 2.04)], .65)
arrow([(.65, 2.11), (.65, 2.29)])
arrow([(.65, 2.65), (.65, 2.75)])
box(.20, 2.75, .90, .68, 'MLPs +\npooling')
text(.67, 2.47, 'Alleles, ages,\nref. labels', 7.5)
arrow([(1.10, 3.09), (1.43, 3.09)])
vector(1.17, 3.04, .19, .10, 3)
text(1.265, 3.45, '$b$', 8)
# The original feature state b is the input to the corrected second traversal.
line([(1.265, 3.09), (1.265, 2.45), (4.22, 2.45)], .75)
arrow([(4.22, 2.45), (4.22, 2.98)], lw=.75)
text(2.86, 2.32, 'Input features', 7.5)

def gnn(x, idx):
    box(x, 2.75, .88, .68)
    text(x+.44, 2.91, r'$\uparrow$ GNN$(T_t)$', 8)
    arrow([(x+.44, 3.01), (x+.44, 3.14)], lw=.65)
    text(x+.44, 3.24, r'$\downarrow$ GNN$(T_t)$', 8)
    text(x+.44, 3.58, '('+str(idx)+')', 8)

gnn(1.43, 1)
line([(2.31, 3.09), (2.49, 3.09)], .8)
# Both corrections depend on the first pass. No inter-tree episode-state pooling.
arrow([(2.49, 3.09), (2.49, 2.84), (2.78, 2.84)])
arrow([(2.49, 3.09), (2.49, 3.36), (2.78, 3.36)])
box(2.78, 2.64, 1.15, .40, 'Clade adapters', 7.7)
box(2.78, 3.16, 1.15, .40, 'Memory (c)', 8)
arrow([(3.93, 2.84), (4.22, 2.84), (4.22, 2.98)])
arrow([(3.93, 3.36), (4.22, 3.36), (4.22, 3.20)])
text(4.13, 3.58, r'$\delta$', 8)
ax.add_patch(Circle((4.22, 3.09), .11, fc=WHITE, ec=INK, lw=.8, zorder=5))
text(4.22, 3.09, '+', 12)
arrow([(4.33, 3.09), (4.55, 3.09)])
gnn(4.55, 2)
text(3.45, 3.76, 'GNN weights shared between (1) and (2)', 7.7)
arrow([(5.43, 3.09), (5.72, 3.09)])
# Query row shown for readout, all samples get their own sequence in the model.
for i, x in enumerate([5.76, 6.12, 6.48]):
    text(x+.13, 2.75, '$T_'+str(i+1)+'$', 8)
    vector(x, 2.94, .27, .29, 3)
text(6.26, 3.47, 'Query embeddings', 7.7)

# (c) Memory follows identity even when age-cut group membership changes.
# Dashed group boundaries use exact membership at the implementation's age cut.
panel(.12, 4.05, 'c')
cols = [1.82, 2.43, 3.04]
ys = [4.54+.22*j for j in range(5)]
# Group representatives are projected and broadcast before identity-wise memory.
vector(.47, 4.35, .36, .13)
text(.65, 4.16, 'Group state', 7.5)
arrow([(.65, 4.48), (.65, 4.65)])
box(.22, 4.65, .86, .45, 'Linear +\ncopy', 7.7)
arrow([(1.08, 4.88), (1.39, 4.88)])
box(1.39, 3.99, 2.13, 1.69)
text(2.455, 4.14, 'Bidirectional EMA', 8)
for i, x in enumerate(cols):
    text(x+.12, 4.34, '$T_'+str(i+1)+'$', 8)
for j, y in enumerate(ys):
    text(1.66, y+.055, NAMES[j], 7.5, ha='right')
    for x in cols:
        vector(x, y, .24, .11, 3)
    for x, xn in zip(cols[:-1], cols[1:]):
        arrow([(x+.30, y+.055), (xn-.06, y+.055)], both=True, lw=.6)
for i, tree in enumerate(trees):
    groups = {}
    for u in range(5):
        rep=u
        while tree.parent(rep)!=-1 and ts.node(tree.parent(rep)).time<=350:
            rep=tree.parent(rep)
        groups.setdefault(rep, []).append(u)
    for members in groups.values():
        lo, hi=min(members),max(members)
        assert members==list(range(lo,hi+1))
        box(cols[i]-.04, ys[lo]-.05, .32, ys[hi]-ys[lo]+.21, dashed=True)
text(2.455, 5.81, 'Genomic-distance decay', 7.7)
arrow([(3.52, 4.79), (3.67, 4.79), (3.67, 4.58), (3.79, 4.58)])
box(3.79, 4.39, .84, .38, 'Group mean', 7.5)
arrow([(4.21, 4.77), (4.21, 4.98)])
box(3.79, 4.98, .84, .47, 'Center +\nLinear', 7.5)
arrow([(4.21, 5.45), (4.21, 5.64)])
vector(4.03, 5.64, .36, .12)
text(4.21, 5.94, r'$\delta$', 8)

# (d) Standard recurrent readout, separate from memory inside the graph encoder.
panel(4.70, 4.05, 'd')
arrow([(6.26, 3.63), (6.26, 4.16)])
box(5.61, 4.16, 1.30, .42, 'BiGRU (1 layer)', 8)
text(6.61, 3.87, 'Interval\ngeometry', 7.5)
arrow([(6.61, 4.03), (6.61, 4.16)], lw=.6)
# Direct embedding skip has its own lane, clear of all text and panel boundaries.
line([(5.77, 3.89), (5.13, 3.89), (5.13, 4.84)], .75)
line([(6.26, 3.89), (5.77, 3.89)], .75)
arrow([(6.26, 4.58), (6.26, 4.72)])
box(5.61, 4.72, 1.30, .30, 'Concat + MLP', 8)
arrow([(5.13, 4.84), (5.61, 4.84)])
arrow([(6.26, 5.02), (6.26, 5.11)])
box(5.80, 5.11, .92, .23, 'Softmax', 7.7)
arrow([(6.26, 5.34), (6.26, 5.45)])
# Unfitted probabilities are copied from the preserved illustrative record.
probs = [[.62,.12,.14,.12], [.17,.16,.53,.14], [.11,.16,.21,.52]]
breaks = list(ts.breakpoints())
x0, x1 = 5.48, 6.91
for i, row in enumerate(probs):
    left=x0+(x1-x0)*breaks[i]/ts.sequence_length
    right=x0+(x1-x0)*breaks[i+1]/ts.sequence_length
    bottom=5.84
    for k,p in enumerate(row):
        ax.add_patch(Rectangle((left, bottom-.36*p),right-left,.36*p,fc=ANC[k],ec=WHITE,lw=.4))
        bottom-=.36*p
    ax.add_patch(Rectangle((left, 6.04),right-left,.11,fc=ANC[int(np.argmax(row))],ec=WHITE,lw=.4))
text(5.27, 5.64, '$p(q)$', 8, ha='right')
text(5.27, 6.095, 'argmax', 7.5, ha='right')
for i in range(3):
    text(x0+(x1-x0)*(breaks[i]+breaks[i+1])/(2*ts.sequence_length), 5.94, '$T_'+str(i+1)+'$', 7.5)
arrow([(6.86, 5.84), (6.86, 6.02)], lw=.6)

# Every color has an explicit key. Neutral color never distinguishes GNN layers.
# Legend will be placed in a dedicated appended band in the final canvas.
fig.set_size_inches(W, 6.25)
ax.set(ylim=(7.12, 0))
line([(.18, 6.30), (6.91, 6.30)], .55)
legend = [('Anatolian farmer (Ana)', .24, 6.46),
          ('Caucasus hunter-gatherer (CHG)', 3.55, 6.46),
          ('Western hunter-gatherer (WHG)', .24, 6.71),
          ('Eastern hunter-gatherer (EHG)', 3.55, 6.71)]
for color, (name,x,y) in zip(ANC, legend):
    ax.add_patch(Rectangle((x,y-.055),.14,.11,fc=color,ec='none'))
    text(x+.21,y,name,7.7,ha='left')
ax.add_patch(Rectangle((.24,6.905),.14,.11,fc=FEATURE,ec=INK,lw=.4))
text(.45,6.96,'Features / memory',7.7,ha='left')
line([(2.35,6.96),(2.49,6.96)],1.0)
text(2.56,6.96,'Structure / operations',7.7,ha='left')
dot(5.03,6.96,0,.045)
text(5.15,6.96,'Query: ancestry unknown',7.7,ha='left')
box(.20,5.90,.20,.12,dashed=True)
text(.48,5.96,'Group below $t_c$',7.7,ha='left')

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
bounds=[]
for t in texts:
    b=t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert b.x0>=0 and b.y0>=0 and b.x1<=1 and b.y1<=1, t.get_text()
    bounds.append({'text':t.get_text(),'bbox':list(b.bounds)})
for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'TreeLoom_architecture.{ext}',dpi=220)
svg = OUT/'TreeLoom_architecture.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
(OUT/'text_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
(OUT/'metadata.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)),
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'palette':{INK:'structure and operations', FEATURE:'learned features and memory',
               **dict(zip(ANC,['Anatolian farmer','CHG','WHG','EHG']))},
    'query':'hollow circle, unknown ancestry', 'predictions':'schematic, not fitted',
    'experiments_run':False,'figure_inches':list(fig.get_size_inches())},indent=2)+'\n')
print('Rendered', OUT)
