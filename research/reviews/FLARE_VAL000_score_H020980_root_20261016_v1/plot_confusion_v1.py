from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/FLARE_VAL000_score_H020980_root_20261016_v1'
terminal=json.loads((OUT/'actual_tool_terminal.root_v1.json').read_text())
assert terminal['actual_terminal_consumed'] and terminal['actual_tool_exit_code']==0
data=json.loads((ROOT/terminal['completed_ref']['path']).read_text())
counts=np.asarray(data['result']['confusion'])
pct=100*counts/counts.sum(axis=1,keepdims=True)
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.0,4.5),layout='constrained')
mesh=ax.imshow(pct,cmap='Blues',vmin=0,vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f'{counts[i,j]}\n{pct[i,j]:.2f}%',ha='center',va='center',
            color='white' if pct[i,j]>50 else '#18212d',fontsize=10)
ax.set_xticks(range(3),data['class_order']);ax.set_yticks(range(3),data['class_order'])
ax.set_xlabel('Predicted ancestry');ax.set_ylabel('True ancestry')
ax.set_title('Native FLARE · American VAL000\n10,000 fixed focal records',fontsize=11,pad=10)
fig.colorbar(mesh,ax=ax,label='Within true ancestry (%)',shrink=.87)
for suffix in ['png','svg','pdf']:
    fig.savefig(OUT/f'confusion_native_FLARE_VAL000.{suffix}',dpi=160)
plt.close(fig)
print('Saved observed native FLARE confusion figure in PNG, SVG and PDF')
