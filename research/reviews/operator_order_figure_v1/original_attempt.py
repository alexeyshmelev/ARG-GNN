#!/usr/bin/env python3
"""Trace a structural graph-order figure to the recorded synthetic fixture."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from arggnn.resources import ROOT,set_research_title
from arggnn.development_execution import configure_deterministic_development,deterministic_development_manifest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    set_research_title();torch.set_num_threads(1);configure_deterministic_development()
    helper=ROOT/'scripts/check_development_cuda_admission.py'
    spec=importlib.util.spec_from_file_location('operator_probe_fixture',helper)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    fixture=module.prepare_fixture()
    original=ROOT/'experiments/results/synthetic_cpu_full_admission_v1/fixture.json'
    if fixture.manifest!=json.loads(original.read_bytes()):raise RuntimeError('Recorded fixture changed')
    operators=fixture.anchors.operators['plain'].cqq[:,0].double()
    intervals=fixture.boundary.intervals.double();durations=intervals[:,1]-intervals[:,0]
    identity=torch.eye(64,dtype=torch.float64)
    rows=[]
    for rate in (5.,20.,80.):
        generators=rate*operators+identity
        holds=[torch.matrix_exp(-duration*generator) for duration,generator in zip(durations,generators)]
        ordered=identity.clone()
        for hold in holds:ordered=hold@ordered
        reversed_order=identity.clone()
        for hold in reversed(holds):reversed_order=hold@reversed_order
        divided=identity.clone()
        for duration,generator in zip(durations,generators):
            partial=torch.matrix_exp(-(duration/3)*generator)
            for _ in range(3):divided=partial@divided
        rows.append({'coupling_rate':rate,'damping':1.,
            'order_difference_frobenius':float(torch.linalg.norm(ordered-reversed_order)),
            'subdivision_difference_frobenius':float(torch.linalg.norm(ordered-divided)),
            'ordered_frobenius':float(torch.linalg.norm(ordered)),
            'ordered_minimum_entry':float(ordered.min())})
    if not all(r['order_difference_frobenius']>1e-8 and r['subdivision_difference_frobenius']<1e-11 for r in rows):
        raise RuntimeError('The synthetic order/subdivision probe did not meet its explicit contract')
    result=ROOT/'experiments/results/operator_order_probe_v1'
    if result.exists():raise RuntimeError('Preserve previous figure attempts')
    result.mkdir()
    figures=ROOT/'paper/figures';figures.mkdir(exist_ok=True)
    pdf=figures/'operator_order_probe_v1.pdf';png=result/'preview.png'
    if pdf.exists():raise RuntimeError('Preserve previous figure PDF')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,
        'ps.fonttype':42,'mathtext.fontset':'dejavuserif','axes.linewidth':.65})
    fig=plt.figure(figsize=(7.0,2.95),layout='constrained')
    grid=fig.add_gridspec(2,4,height_ratios=[1.55,1.0],width_ratios=[1.,1.,1.,1.1])
    low,high=float(operators.min()),float(operators.max())
    norm=TwoSlopeNorm(vmin=low,vcenter=0.,vmax=high)
    for i in range(3):
        ax=fig.add_subplot(grid[0,i]);image=ax.imshow(operators[i].numpy(),cmap='RdBu_r',norm=norm,
            interpolation='nearest',rasterized=True)
        ax.set_title(r'$C_{QQ}$'+f' on hold {i+1}',fontsize=8)
        ax.set_xticks([0,31,63]);ax.set_yticks([0,31,63]);ax.set_xlabel('Query index',fontsize=7)
        if i==0:ax.set_ylabel('Query index',fontsize=7)
        ax.tick_params(labelsize=6,length=2)
    color=fig.colorbar(image,ax=fig.axes[:3],shrink=.62,pad=.015)
    color.ax.tick_params(labelsize=6,length=2);color.set_label('Normalized response',fontsize=7)
    measure=fig.add_subplot(grid[1,:3])
    for i,(left,right) in enumerate(intervals.tolist()):
        measure.axvspan(left,right,facecolor=('.91' if i%2==0 else '.78'),edgecolor='none',zorder=0)
        measure.text((left+right)/2,1.65,f'Hold {i+1}',ha='center',va='center',fontsize=7)
    anchors=fixture.anchors.layout.anchors.numpy()
    readouts=fixture.boundary.positions.numpy()
    measure.scatter(anchors,[1.0]*len(anchors),marker='|',s=48,c='black',linewidths=1.)
    measure.scatter(readouts,[.32]*len(readouts),marker='x',s=18,c='black',linewidths=.85)
    measure.set_yticks([.32,1.]);measure.set_yticklabels(['Readouts','Fixed sources'],fontsize=7)
    measure.set_xticks([0,.31,.72,1.]);measure.tick_params(labelsize=7,length=2)
    measure.set_ylim(-.03,1.95);measure.set_xlim(0,1.);measure.set_xlabel('Physical coordinate x',fontsize=8)
    measure.spines[['top','right']].set_visible(False)
    probe=fig.add_subplot(grid[:,3])
    x=list(range(3));delta=.17
    for offset,key,label,color,hatch in [(-delta,'order_difference_frobenius','Reversed order','.20',''),
        (delta,'subdivision_difference_frobenius','Subdivided holds','.65','///')]:
        values=[max(r[key],1e-16) for r in rows]
        probe.bar([i+offset for i in x],values,width=.30,color=color,edgecolor='black',linewidth=.45,hatch=hatch,label=label)
    probe.set_yscale('log');probe.set_ylim(1e-16,1.)
    probe.set_xticks(x);probe.set_xticklabels(['5','20','80'])
    probe.set_xlabel(r'Coupling $\lambda$',fontsize=8)
    probe.set_ylabel('Propagator difference (Frobenius norm)',fontsize=7)
    probe.set_title('Untrained operator probe',fontsize=8)
    probe.legend(loc='lower left',bbox_to_anchor=(-.04,1.04),fontsize=6,frameon=False)
    probe.tick_params(labelsize=6,length=2);probe.spines[['top','right']].set_visible(False)
    fig.savefig(pdf,metadata={'Title':'Synthetic graph-order and subdivision probe','Author':'',
        'Subject':'Structural fixture only, no predictive result'},dpi=600)
    fig.savefig(png,dpi=250);plt.close(fig)
    deterministic_development_manifest()
    receipt={'kind':'synthetic_operator_order_and_subdivision_probe','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_fixture':str(original.relative_to(ROOT)),'source_fixture_sha256':sha(original),
        'generator_source':str(helper.relative_to(ROOT)),'generator_source_sha256':sha(helper),
        'probe_source':str(Path(__file__).resolve().relative_to(ROOT)),'probe_source_sha256':sha(__file__),
        'operator_sha256':module.tensor_digest(operators),'intervals':intervals.tolist(),
        'fixed_anchor_count':len(anchors),'observed_query_count':64,'observed_reference_count':24,
        'rows':rows,'numerical_dtype':'float64','matrix_exponential':'torch.matrix_exp',
        'reversal':'Reverse the same three (operator,duration) pairs, preserving their multiset',
        'subdivision':'Replace every hold by three identical-generator subholds of one-third duration',
        'plotting_floor_only':1e-16,'neural_fitting':False,'cohort_or_test_access':False,'cuda_context_initialized':torch.cuda.is_initialized(),
        'scientific_scope':'Numerical structural illustration of established ordered propagation/semigroup behavior. No ancestry accuracy, empirical superiority or new mathematical theorem.',
        'figure':str(pdf.relative_to(ROOT)),'figure_sha256':sha(pdf),'preview_sha256':sha(png)}
    (result/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'figure':str(pdf),'probe':rows}),flush=True)


if __name__=='__main__':main()
