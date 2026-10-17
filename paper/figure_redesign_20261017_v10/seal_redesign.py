from pathlib import Path
import json,hashlib,re,subprocess
from PIL import Image, ImageChops
out=Path("/workspace-SR003.nfs2/ARG-GNN/paper/figure_redesign_20261017_v10")
old=out.parent/"figure_redesign_20261017_v9"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
oldsha=json.loads((out/"v9_predecessor_sha256.json").read_text())
for n,h in oldsha.items():assert sha(old/n)==h,(n,"sealed v9 changed")
meta=json.loads((out/"figure_metadata.json").read_text())
prevmeta=json.loads((old/"figure_metadata.json").read_text())
for k,v in prevmeta.items():
    if k!="revision":assert meta[k]==v,(k,"unintended metadata change")
# All labels/fonts/positions remain identical except moving panel letter b.
bounds=json.loads((out/"text_layout_bounds.json").read_text())
prevbounds=json.loads((old/"text_layout_bounds.json").read_text())
assert len(bounds)==len(prevbounds)
for a,b in zip(bounds,prevbounds):
    assert a["text"]==b["text"] and a["fontsize"]==b["fontsize"]
    if a["text"]!="b":assert a["bbox_figure"]==b["bbox_figure"]
over=[]
for i,a in enumerate(bounds):
    x,y,w,h=a["bbox_figure"]
    for b in bounds[:i]:
        bx,by,bw,bh=b["bbox_figure"]
        if min(x+w,bx+bw)>max(x,bx)+1e-6 and min(y+h,by+bh)>max(y,by)+1e-6:
            over.append([a["text"],b["text"]])
assert not over,over
assert min(b["fontsize"] for b in bounds)>=7
# Pixel-level guarantee that the lower architecture and ARG are unchanged.
a=Image.open(old/"TreeLoom_architecture.png").convert("RGB")
b=Image.open(out/"TreeLoom_architecture.png").convert("RGB")
assert a.size==b.size==(1540,1452)
diff=ImageChops.difference(a,b);bbox=diff.getbbox()
outside=diff.copy()
# Allowed only: b frame, moved letter, and new single output route.
allowed=(int(2.47*220),int((6.60-6.51)*220),int(6.85*220),int((6.60-4.13)*220))
outside.paste((0,0,0),allowed)
assert outside.getbbox() is None,(bbox,allowed)
caption=(out/"caption.tex").read_text()
assert ";" not in caption
assert ". diamonds" not in caption and ". query" not in caption and ". concatenation" not in caption
plain=re.sub(r"\\[A-Za-z]+","",caption);plain=re.sub(r"[{}$]","",plain)
mapping=json.loads((old/"source_mapping.json").read_text())
mapping["status"]="v10 narrow input-routing successor; ROOT integration pending"
mapping["immutable_predecessor"]="paper/figure_redesign_20261017_v9"
mapping["predecessor_seal_sha256"]=sha(old/"handoff_seal.json")
mapping["narrow_change"]={
 "purpose":"Make explicit that all observed feature glyphs feed initialization, not only ancestry vectors.",
 "boundary_inches":{"x":2.68,"y":4.15,"width":4.15,"height":2.34},
 "included":"three local trees carrying observed node features, all allele cells, known and masked ancestry vectors, internal age/root/sample callout",
 "single_exit":"left boundary at (2.68,5.10), orthogonal route to MLP at (0.52,3.82)",
 "removed":"old ancestry-only bottom bracket/connector",
 "panel_letter":"b moved x2.67 to x2.54 to clear source boundary",
 "caption":"Tree edges define both GNN traversals. Numeric initialization still follows the actual feature encoder. The boundary is input grouping, not a claim that topology itself is numerically encoded by the MLP.",
 "unchanged":"ARG/topologies, all alleles, every other label location, layers, memory, skips, outputs, probabilities, dimensions, typography and all lower architecture pixels"
}
mapping["b"]["all_observed_input_collector"]=meta["observed_input_collector"]
mapping["scope"]="Only fresh paper/figure_redesign_20261017_v10 written. Sealed v9/v7, manuscript, canonical assets, scientific sources and blind review unchanged."
mapping["source_of_truth"]["paper/figure_redesign_20261017_v9/build_figure.py"]={"sha256":sha(old/"build_figure.py"),"role":"sealed source preserved; successor differs only by documented input collector, panel letter and provenance fields"}
(out/"source_mapping.json").write_text(json.dumps(mapping,indent=2)+"\n")
qa={
 "revision":"v10 whole-panel observed-input collector",
 "actual_PDF_render":"Poppler pdftoppm -singlefile -r220 -png",
 "visual_review":"Latest actual PDF-rendered PNG inspected. Whole b input group is bounded; one left exit connects to MLP. No line crosses text, no overlap/clipping, all model modules unchanged.",
 "minimum_font_pt":min(v["fontsize"] for v in bounds),"text_overlap_pairs":over,
 "label_positions":"all identical to sealed v9 except panel letter b",
 "pixel_difference_bbox":bbox,"allowed_difference_bbox":allowed,
 "pixels_unchanged_outside_input_collector":True,
 "sealed_v9_hashes_unchanged":True,
 "native_toy_copies_unchanged":all(sha(out/n)==sha(old/n) for n in ["illustrative_full_arg.trees","illustrative_tree_sequence.trees"]),
 "caption_words_rough":len(plain.split()),"caption_semicolons":0,
 "caption_postperiod_capitalization":True,
 "PDF_info":subprocess.check_output(["pdfinfo",str(out/"TreeLoom_architecture.pdf")],text=True),
 "model_execution":False,"thread_limits":"all plotting BLAS/OMP limits1",
 "operation_marker":"same authorized PDF edit continuation; ROOT marker reused, not repeated"
}
(out/"redesign_QA.json").write_text(json.dumps(qa,indent=2)+"\n")
(out/"HANDOFF.txt").write_text("""TreeLoom v10 narrow successor: all panel-b observed inputs visibly grouped
-----------------------------------------------------------------------
Ready: TreeLoom_architecture.pdf / .svg / .png.
Actual PDF render: TreeLoom_architecture_render.png.
Editable: build_figure.py. Exact narrow source diff: v9_to_v10.patch.
Caption: caption.tex. Provenance: source_mapping.json. Checks: redesign_QA.json.
Single subtle b frame collects trees carrying age/status features, allele cells
and known/masked ancestry vectors, with one connected output to MLP. Caption
states tree edges define the GNN traversals separately.
Only panel letter b moved, to clear the boundary. All other labels, modules,
biology and lower architecture pixels are unchanged. Text >=7pt, no overlaps.
Sealed v9 and v7 remain byte-for-byte unchanged. Only fresh v10 written.
No scientific model, Git/resource operation or job.
ROOT may integrate after independent image-only routing verification.
""")
files=[p for p in out.iterdir() if p.is_file() and p.name not in ("SHA256SUMS","handoff_seal.json")]
sums={p.name:sha(p) for p in sorted(files)}
(out/"SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k,v in sums.items()))
seal={"status":"sealed; independent routing verification and ROOT integration pending",
 "directory":str(out),"predecessor":"sealed v9","predecessor_SHA256":oldsha,
 "sha256":sums,"SHA256SUMS_sha256":sha(out/"SHA256SUMS"),
 "image_SHA256":sums["TreeLoom_architecture.png"],"pdf_SHA256":sums["TreeLoom_architecture.pdf"],
 "source_SHA256":sums["build_figure.py"],"only_fresh_v10_written":True}
(out/"handoff_seal.json").write_text(json.dumps(seal,indent=2)+"\n")
print(json.dumps({"png_sha256":seal["image_SHA256"],"pdf_sha256":seal["pdf_SHA256"],"caption_words":qa["caption_words_rough"],"text_overlap_pairs":len(over),"unchanged_outside_collector":True},indent=2))
