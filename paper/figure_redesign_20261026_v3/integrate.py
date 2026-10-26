"""Integrate the figure redesign while preserving every table exactly."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib, json, re, shutil, subprocess
root=Path.cwd()
paper=root/'paper'
out=paper/'figure_redesign_20261026_v3'
p=paper/'current_full_panel_working.tex'
before=p.read_text()
head=subprocess.run(['git','show','HEAD:paper/current_full_panel_working.tex'],capture_output=True,text=True,check=True).stdout
captions={
'architecture':r'''\textbf{a,} Three marginal trees with the same sampled haplotypes. The allele matrix shows one illustrative site per interval. Observed allele frequencies and callability feed the site MLP, then pooling. Reference labels, sampling ages and node features also enter the feature MLPs. Query ancestry is unknown.
\textbf{b,} Vectors at every node show upward states $u$ and outside states $o$, including internal ancestors. Root outside state is zero. Two complete rounds reuse the same node operators, expanded in Figure~\ref{fig:operators}. Memory and structure modify input $b$ before round 2. Concatenating the final states gives one local embedding per haplotype per tree.
\textbf{c,} Interval length $L_t$ and adjacent center distances $d_-,d_+$ form three geometry features, each transformed by $\log(1+x)/\log(2000001)$. Missing neighbors give zero. Geometry joins each local query embedding before the BiGRU and classifier. Feature colors and ancestry probabilities are illustrative, not fitted results. Colors and shapes follow the legend.''',
'memory':r'''\textbf{a,} The age cut produces groups $G_1,G_2,G_3$. The marked ancestor of r1 and r2 supplies upward and outside states. Their sum projects to $f_2$, copied to both group members. The complete tree remains in the GNN.
\textbf{b,} Each haplotype retains separate forward and reverse exponential memories. Both labeled $m_2$ outputs are states at $T_2$. Direction means are computed per haplotype, then averaged within the current group. Subtracting $f_2$ and projecting gives correction $\delta$, added at the group ancestor before round 2. The connected expansion of M shows both vector and scalar inputs. Interval length $L_t$ and learned scale $\ell$ determine retention $\alpha_t$. The first state in each direction equals its current context. The same update applies at other intervals, whose context inputs are omitted. This exponential memory is distinct from the final BiGRU. Feature values are illustrative.''',
'operators':r'''\textbf{a,} Upward update at one node. Each child state is transformed and gated before the arithmetic mean over children. This mean joins input $b$, followed by a linear map, tanh, residual addition and learned layer normalization. The operator repeats from leaves to root.
\textbf{b,} Downward update. Parent outside state $o_p$, input $b_p$ and mean child message $a_p$ are combined, excluding the recipient's normalized contribution $m_{q\to p}$. The displayed maps and edge gate give outside state $o_q$. The root starts at zero. Nine observed edge features produce separate directional gates.
\textbf{c,} Descendant pairs illustrate loss, gain and persistence across trees. Binary flags select factored transformations of the current upward state $u_v$, with separate weights for the two corrections. Neighboring hidden states do not enter these circuits. The symbol $\odot$ denotes elementwise product, Scale $\times$ multiplies a vector by a scalar, and $\parallel$ concatenates. Input signs define subtraction order. Diagram state $u$ is $h$ in the equations. Colors have the meanings shown in the legend.'''
}
blocks=[]
for name,caption in captions.items():
    block='\\begin{figure*}[p]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/TreeLoom_'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+name+'}\n\\end{figure*}'
    (out/('caption_'+name+'.tex')).write_text(block+'\n')
    blocks.append(block)
pattern=r'\\begin\{figure\*\}\[[^]]*\].*?\\end\{figure\*\}'
found=re.findall(pattern,before,re.S)
assert len(found)==2 and 'fig:architecture' in found[0] and 'fig:operators' in found[1]
s=before.replace(found[0],'\n\n'.join(blocks)).replace(found[1],'')
s=s.replace('Sample memory supplies a different kind of context.', 'Sample memory supplies a different kind of context (Figure~\\ref{fig:memory}).')
tabpattern=r'\\begin\{table\*?\}.*?\\end\{table\*?\}'
oldtables=re.findall(tabpattern,head,re.S)
newtables=re.findall(tabpattern,s,re.S)
assert oldtables==newtables and len(newtables)==8
p.write_text(s)
for name in captions:
    for ext in ['pdf','svg','png']:
        shutil.copy2(out/('TreeLoom_'+name+'.'+ext),paper/'figures'/('TreeLoom_'+name+'.'+ext))
receipt={'source_before_sha256':hashlib.sha256(before.encode()).hexdigest(),'source_after_sha256':hashlib.sha256(s.encode()).hexdigest(),'table_source_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in newtables],'all_eight_result_tables_unchanged':True,'experiments_remain_stopped':True,'new_training_or_simulation':False}
(out/'integration.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'build').mkdir(exist_ok=True)
(out/'overleaf_build').mkdir(exist_ok=True)
with ZipFile(paper/'current_full_panel_overleaf_working.zip','w',ZIP_DEFLATED) as z:
    z.writestr('main.tex',s)
    for f in ['current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst']:
        z.write(paper/f,f)
    for name in captions:
        f='figures/TreeLoom_'+name+'.pdf'
        z.write(paper/f,f)
    z.writestr('README.txt','Compile main.tex with pdfLaTeX and BibTeX. The three figures describe the architecture, haplotype memory and node operators. This is a development manuscript, not a completed matched benchmark.\n')
with ZipFile(paper/'current_full_panel_overleaf_working.zip') as z:
    z.extractall(out/'overleaf_build')
print('Integrated three figures. All eight table source blocks match HEAD.')
