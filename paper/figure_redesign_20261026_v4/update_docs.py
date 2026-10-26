from pathlib import Path
import re,hashlib,json
root=Path.cwd()
out=root/'paper/figure_redesign_20261026_v4'
figures='Figures 1 and 2 show the complete prediction path and an expansion of its two-round tree encoder. The diagram explicitly separates all-node embeddings from selected leaf embeddings, follows one leaf across tree rows, appends the three geometry columns, and distinguishes group memory from the final BiGRU. Learned blocks are gold, embeddings teal, observed features gray and probabilities purple, with legends. [Architecture](figures/TreeLoom_architecture.png) · [Tree encoder](figures/TreeLoom_encoder.png) · [Generator and independent review](figure_redesign_20261026_v4/README.md).'
checks='The manuscript and a freshly extracted Overleaf bundle compile to ten pages, with no errors, undefined references, overfull boxes or oversized floats. The two figures occupy pages 3 and 4. Five underfull diagnostics remain and the layouts were visually inspected. All eight result tables are unchanged. [Build and review record](figure_redesign_20261026_v4/integration.json).'
for name in ['README.md','CURRENT_FULL_PANEL_README.md']:
 p=root/'paper'/name
 s=p.read_text()
 s=re.sub(r'Figures 1–3 were redesigned.*?(?=\n\n)',figures,s,flags=re.S)
 s=re.sub(r'The manuscript and extracted Overleaf bundle compile.*?(?=\n\n)',checks,s,flags=re.S)
 s=s.replace('twelve-page','ten-page').replace('pages 1–8','pages 1–6').replace('pages 9–12','pages 7–10')
 p.write_text(s)
p=root/'README.md'
s=p.read_text()
pattern=r'<!-- figure-redesign-current:start -->.*?<!-- figure-redesign-current:end -->'
old=re.search(pattern,s,re.S).group()
new='''<!-- figure-redesign-current:start -->
**TreeLoom figure revision, 26 October:** the main paper now has two diagrams. [Prediction flow](paper/figures/TreeLoom_architecture.png) and [expanded tree encoder](paper/figures/TreeLoom_encoder.png) show node and leaf representations, sequence stacking, structural corrections, sample memory and genomic classification. A fresh agent reconstructed every key conceptual stage from the images alone after two failed rounds and revisions. [Updated ten-page paper](paper/current_full_panel_working.pdf) · [Overleaf bundle](paper/current_full_panel_overleaf_working.zip) · [Review record](paper/figure_redesign_20261026_v4/review.md). All eight result tables are unchanged. Experiments remain stopped.
<!-- figure-redesign-current:end -->'''
after=s.replace(old,new)
assert s.replace(old,'')==after.replace(new,'')
p.write_text(after)
d=json.loads((out/'integration.json').read_text())
d['root_readme_outside_figure_marker_unchanged']=True
d['protected_readme_sections']={}
for key in ['owner-stop-current','paper-benchmark-rebuild-current']:
 block=re.search(r'<!-- '+key+r':start -->.*?<!-- '+key+r':end -->',s,re.S).group()
 assert block in after
 d['protected_readme_sections'][key]=hashlib.sha256(block.encode()).hexdigest()
(out/'integration.json').write_text(json.dumps(d,indent=2)+'\n')
p=root/'paper/figure_redesign_20261026_v3/README.md'
s=p.read_text()
s+='''
## Superseded by the two-figure redesign

The owner-reported open issues above remain preserved as the record of this version. They were addressed in the [v4 redesign](../figure_redesign_20261026_v4/README.md), which replaces the three figures with a prediction diagram and an expanded encoder. New independent readers had no source, paper text or prior conversation. Two failed readings led to further changes, and a subsequent fresh reader correctly explained the complete conceptual flow. This is not owner approval. The detailed review and remaining implementation-level scope are recorded in v4. Experiments remain stopped and the numerical tables are unchanged.
'''
p.write_text(s)
print('Updated figure documentation only. Stop and benchmark snapshot blocks preserved.')
