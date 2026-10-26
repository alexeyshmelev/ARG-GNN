"""Integrate two reviewed figures and preserve all experimental table source."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib, json, re, shutil, subprocess
root=Path.cwd()
paper=root/'paper'
out=paper/'figure_redesign_20261026_v4'
p=paper/'current_full_panel_working.tex'
before=p.read_text()
head=subprocess.run(['git','show','HEAD:paper/current_full_panel_working.tex'],capture_output=True,text=True,check=True).stdout
captions={
'architecture':r'''TreeLoom predicts local ancestry for each query leaf and interval. Each sampled leaf represents one haplotype.
\textbf{a,} Node features initialize all nodes, including ancestors. Observed site features are encoded and pooled over called sites within each interval separately for each leaf. Reference labels and sampling ages join this pooled evidence, and the resulting features are added only to leaf rows. Query ancestry labels are masked. Branch geometry and observed allele--branch agreement supply learned message gates.
\textbf{b,} The two rounds of the tree encoder are expanded in Figure~\ref{fig:encoder}. Its output has upward and outside embeddings for every node. Selecting leaf rows, concatenating the two states and projecting gives a local leaf embedding of width 32.
\textbf{c,} Rows for the same query leaf are stacked in tree order. Interval length and distances to adjacent interval centers add three columns before the BiGRU. Its output joins the original local embedding for classification. Four ancestry classes illustrate the European condition. All matrix values and probabilities are schematic, not fitted results.''',
'encoder':r'''Expansion of the tree encoder in Figure~\ref{fig:architecture}b.
\textbf{a,} A complete upward traversal produces $U^1$, followed by a downward traversal producing $O^1$. Rows retain node identities. Upward state $U$ corresponds to $h$ in the equations.
\textbf{b,} Each leaf joins the group represented by its highest ancestor no older than 350 generations. Older samples remain singleton groups. A projection of the representative's summed states gives the group context $f$, copied to its leaves. Separate histories follow each leaf across trees, even when its group changes. Forward and reverse exponential memories are averaged per leaf and then over the current group's leaves. Subtracting the current context and projecting gives a correction $\Delta$, added only at group representatives. The $T_2$ example is repeated at every interval. Here $m,\alpha,L,\ell$ correspond to $z,\rho,w,\lambda$ in the text.
\textbf{c,} Exact descendant sets determine gain, loss and persistence flags for proper clades. These flags may overlap. Learned factored maps $G,L,P$ transform the current upward state and place correction $S$ at its clade representative. Their equations appear in the supplement.
\textbf{d,} Both corrections enter the original base before the second complete upward and downward round, which reuses the first round's weights. Both all-node outputs return to leaf selection in Figure~\ref{fig:architecture}b.'''
}
blocks=[]
for name,caption in captions.items():
    block='\\begin{figure*}[!t]\n\\centering\n\\includegraphics[width=\\textwidth]{figures/TreeLoom_'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+name+'}\n\\end{figure*}'
    (out/('caption_'+name+'.tex')).write_text(block+'\n')
    blocks.append(block)
pattern=r'\\begin\{figure\*\}\[[^]]*\].*?\\end\{figure\*\}'
found=re.findall(pattern,before,re.S)
assert len(found)==3 and 'fig:architecture' in found[0]
s=before.replace(found[0],'\n\n'.join(blocks))
for block in found[1:]:s=s.replace(block,'')
s=s.replace('Figure~\\ref{fig:memory}', 'Figure~\\ref{fig:encoder}b')
s=s.replace('agreement between inferred mutations and observed alleles', 'agreement between observed allele partitions and inferred branches')
s=s.replace('gives a local embedding $e_{tq}', 'gives a local leaf embedding $e_{tq}')
s=s.replace('Aligning embeddings by haplotype identity produces one ordered sequence per haplotype.', 'Stacking leaf embeddings in tree order while preserving leaf identity produces one sequence per sampled chromosome.')
tabpattern=r'\\begin\{table\*?\}.*?\\end\{table\*?\}'
oldtables=re.findall(tabpattern,head,re.S)
newtables=re.findall(tabpattern,s,re.S)
assert oldtables==newtables and len(newtables)==8
assert len(re.findall(pattern,s,re.S))==2
assert 'fig:memory' not in s and 'fig:operators' not in s
p.write_text(s)
for name in captions:
    for ext in ['pdf','svg','png']:
        shutil.copy2(out/('TreeLoom_'+name+'.'+ext),paper/'figures'/('TreeLoom_'+name+'.'+ext))
receipt={'source_before_sha256':hashlib.sha256(before.encode()).hexdigest(),'source_after_sha256':hashlib.sha256(s.encode()).hexdigest(),'table_source_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in newtables],'all_eight_result_tables_unchanged':True,'experiments_remain_stopped':True,'new_training_or_simulation':False,'main_figures':2}
(out/'integration.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'build').mkdir(exist_ok=True)
(out/'overleaf_build').mkdir(exist_ok=True)
with ZipFile(paper/'current_full_panel_overleaf_working.zip','w',ZIP_DEFLATED) as z:
    z.writestr('main.tex',s)
    for f in ['current_full_panel_refs.bib','aaai2027.sty','aaai2027.bst']:z.write(paper/f,f)
    for name in captions:
        f='figures/TreeLoom_'+name+'.pdf'
        z.write(paper/f,f)
    z.writestr('README.txt','Compile main.tex with pdfLaTeX and BibTeX. Two figures explain the prediction pipeline and its expanded tree encoder. This is a development manuscript, not a completed matched benchmark.\n')
with ZipFile(paper/'current_full_panel_overleaf_working.zip') as z:z.extractall(out/'overleaf_build')
print('Integrated two figures. All eight table source blocks match HEAD.')
