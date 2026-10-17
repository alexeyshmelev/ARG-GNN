from pathlib import Path
import json,hashlib,re,ast,difflib,subprocess
from PIL import Image, ImageChops
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
out=ROOT/"paper/figure_redesign_20261017_v11"
old=out.parent/"figure_redesign_20261017_v10"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
snapshot=json.loads((out/"v10_predecessor_sha256.json").read_text())
for n,h in snapshot.items():assert sha(old/n)==h,(n,"sealed v10 changed")
assert {str(p.relative_to(old)) for p in old.rglob("*") if p.is_file()}==set(snapshot),"v10 file inventory changed"
meta=json.loads((out/"figure_metadata.json").read_text())
prevmeta=json.loads((old/"figure_metadata.json").read_text())
for k in ["figure_inches","minimum_font_pt","panel_titles","panels","trees","samples","genomic_breakpoints_bp","ARG_native_edge_surfaces","ARG_edge_surface_count","ARG_layout","marginal_trees","query_identity","feature_values","memory_display_order","sample_readout_display_order","residual","readout","geometry_input","no_graph_pooling","output_probs","observed_input_collector"]:
    assert meta[k]==prevmeta[k],(k,"unintended science/geometry change")
for n in ["illustrative_full_arg.trees","illustrative_tree_sequence.trees"]:
    assert sha(out/n)==sha(old/n)
before=(old/"build_figure.py").read_text()
after=(out/"build_figure.py").read_text()
oldtree=ast.parse(before);newtree=ast.parse(after)
def assigned(tree,name):
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.dump(node.value,include_attributes=False)
    raise KeyError(name)
for name in ["probs","ANAMES","ANC","ROW","NAMES"]:
    assert assigned(oldtree,name)==assigned(newtree,name),(name,"numerical example/identity altered")
for name in ["display_tree","message_tree","small_vector","proj"]:
    a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),(name,"science icon changed")
(out/"v10_to_v11.patch").write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="v10/build_figure.py",tofile="v11/build_figure.py")))
bounds=json.loads((out/"text_layout_bounds.json").read_text())
over=[]
for i,a in enumerate(bounds):
    x,y,w,h=a["bbox_figure"]
    for b in bounds[:i]:
        bx,by,bw,bh=b["bbox_figure"]
        if min(x+w,bx+bw)>max(x,bx)+1e-6 and min(y+h,by+bh)>max(y,by)+1e-6:
            over.append([a["text"],b["text"]])
assert not over,over
assert min(b["fontsize"] for b in bounds)>=7
# Only orthogonal main dataflow routes are tested here. Icons and all arrowheads
# were additionally inspected in the actual PDF render at 220 and 100 dpi.
routes=[ast.literal_eval(n.args[0]) for n in ast.walk(newtree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="route"]
inter=[]
for ridx,points in enumerate(routes):
    for a,b in zip(points[:-1],points[1:]):
        for t in bounds:
            tx,ty,tw,th=t["bbox_figure"];x,y,w,h=tx*7,ty*6.6,tw*7,th*6.6
            if a[0]==b[0]:
                hit=x<a[0]<x+w and min(a[1],b[1])<y+h and max(a[1],b[1])>y
            else:
                assert a[1]==b[1]
                hit=y<a[1]<y+h and min(a[0],b[0])<x+w and max(a[0],b[0])>x
            if hit:inter.append({"route":ridx,"segment":[a,b],"text":t["text"]})
assert not inter,inter
svg=(out/"TreeLoom_architecture.svg").read_text()
assert "<image" not in svg,"raster element in SVG"
pdfimages=subprocess.check_output(["pdfimages","-list",str(out/"TreeLoom_architecture.pdf")],text=True)
assert len(pdfimages.strip().splitlines())==2,pdfimages
# The ARG panel is byte-identical in the PNG, independent of changed labels.
a=Image.open(old/"TreeLoom_architecture.png").convert("RGB")
b=Image.open(out/"TreeLoom_architecture.png").convert("RGB")
assert a.size==b.size==(1540,1452)
crop=(0,0,int(2.44*220),int((6.6-4.19)*220))
assert ImageChops.difference(a.crop(crop),b.crop(crop)).getbbox() is None
caption=(out/"caption.tex").read_text()
assert ";" not in caption
plain=re.sub(r"\\[A-Za-z]+","",caption);plain=re.sub(r"[{}$]","",plain)
assert len(plain.split())<170,len(plain.split())
mapping=json.loads((out/"source_mapping.json").read_text())
source_audit={}
for n,e in mapping["source_of_truth"].items():
    p=ROOT/n
    actual=sha(p)
    source_audit[n]={"expected_sha256":e["sha256"],"actual_sha256":actual,"match":actual==e["sha256"]}
qa={"status":"mechanical preservation and author visual QA passed; independent image-only review pending",
 "revision":"v11 operation and interval clarity","figure_inches":[7,6.6],
 "minimum_font_pt":min(v["fontsize"] for v in bounds),"text_overlap_pairs":over,
 "main_orthogonal_route_text_intersections":inter,"tested_main_routes":len(routes),
 "vector_outputs":{"SVG_image_elements":0,"PDF_embedded_images":0},
 "actual_PDF_render":"Poppler pdftoppm -singlefile -r220 -png and separate -r100 print-size view",
 "visual_review":"Both actual PDF renderings inspected. No clipping, textual collision or new arrow/text overlap seen. Small schematic branches remain an operator abstraction; Up/Down labels and caption state traversal order.",
 "exact_genuine_trees":True,"illustrative_probability_array_unchanged":True,
 "ARG_panel_pixels_unchanged":True,"sample_identity_and_ancestry_classes_unchanged":True,
 "native_edge_geometry_and_marginal_times_unchanged":True,
 "complete_architecture":"base + both current-node structural residuals + centered sample-memory correction -> second shared GNN -> all-sample local projection -> one-layer BiGRU + local concatenation -> classifier",
 "memory_operation":"actual bidirectional span-decay EMA, followed by group averaging, centering and learned Linear",
 "caption_words_rough":len(plain.split()),"caption_semicolons":0,
 "full_equations_inferable_from_image":False,
 "remaining_review":"ROOT will obtain fresh image-only interpretation. No reader verdict is asserted by this worker.",
 "sealed_v10_hashes_unchanged":True,"v10_files_checked":len(snapshot),
 "source_sha_audit":source_audit,
 "PDF_info":subprocess.check_output(["pdfinfo",str(out/"TreeLoom_architecture.pdf")],text=True),
 "model_execution":False,"GPU_or_admissions":False,"canonical_edits_or_build":False,"Git_operations":False,
 "runtime":"existing .runtime/env/bin/python -B -s; plotting threads1; no installs",
 "scope":"all new files in fresh paper/figure_redesign_20261017_v11, no project Mac writes",
 "operation_marker":"ROOT executed once earlier in continuous authorized request; worker did not duplicate"}
(out/"redesign_QA.json").write_text(json.dumps(qa,indent=2)+"\n")
(out/"HANDOFF.txt").write_text("""TreeLoom v11 inactive candidate: operation, node-feature and interval clarity
---------------------------------------------------------------------------
Ready: TreeLoom_architecture.pdf / .svg / .png / _2x.png.
Actual PDF renders: TreeLoom_architecture_render.png (220dpi),
TreeLoom_architecture_print100.png (100dpi / 7inch printed-size inspection).
Editable source: build_figure.py. Source diff: v10_to_v11.patch.
Caption: caption.tex (under170words). Mapping: source_mapping.json.
Checks: redesign_QA.json. Predecessor snapshot: v10_predecessor_sha256.json.

Changes: observed allele labels replace opaque g tuples; node-age/status
caption identifies tau/0/0; numbered GNN_theta(T_t) headers state per-tree
operation; shared theta label; Bidirectional EMA names actual sample memory;
matching I1/I2/I3 labels appear under q1 output track. Return lane shortened.

Preserved: exact genuine ARG/tree bytes, native edges and ages, all sample
identities and biological classes, illustrative probabilities, base residual,
both current-node structural residuals, memory correction, second complete
shared Up/Down pass, final one-layer BiGRU, local concatenation and classifier.
No performance metric appears or changes. No model/training/GPU/admission,
canonical manuscript/build, source seal, commit/push or Mac project write.

Mechanical checks and author PDF-render inspection pass. Full encoder,
structural and memory equations remain in Methods; the image alone does not
specify them completely. ROOT still needs a fresh image-only reader before
judging clarity or integrating. This candidate is inactive.
""")
files=[p for p in out.iterdir() if p.is_file() and p.name not in ("SHA256SUMS","handoff_seal.json")]
sums={p.name:sha(p) for p in sorted(files)}
(out/"SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k,v in sums.items()))
seal={"status":"inactive candidate sealed; independent image-only review and ROOT integration pending",
 "directory":str(out),"predecessor":"sealed v10","predecessor_seal_sha256":sha(old/"handoff_seal.json"),
 "sha256":sums,"SHA256SUMS_sha256":sha(out/"SHA256SUMS"),"image_sha256":sums["TreeLoom_architecture_render.png"],
 "pdf_sha256":sums["TreeLoom_architecture.pdf"],"source_sha256":sums["build_figure.py"],
 "only_fresh_v11_written":True}
(out/"handoff_seal.json").write_text(json.dumps(seal,indent=2)+"\n")
print(json.dumps({"directory":str(out),"caption_words":qa["caption_words_rough"],"overlap_pairs":len(over),"main_route_text_intersections":len(inter),"SVG_images":0,"PDF_images":0,"unchanged_v10_files":len(snapshot),"source_sha_mismatches":[n for n,e in source_audit.items() if not e["match"]],"PDF_sha256":seal["pdf_sha256"],"render_sha256":seal["image_sha256"]},indent=2))
