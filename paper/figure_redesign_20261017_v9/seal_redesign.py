from pathlib import Path
import json, hashlib, re, subprocess, os
out=Path("/workspace-SR003.nfs2/ARG-GNN/paper/figure_redesign_20261017_v9")
root=out.parent.parent
prev=root/"paper/figure_redesign_20261017_v7"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={
 "paper/current_full_panel_working.tex":{"lines":"68-142","role":"current model description: observed features, full traversals, current-node structure, same-sample memory, final GRU/readout"},
 "src/arggnn/anonymous_clade_gnn_v1.py":{"lines":"90-166","role":"prepare_tree_base_v1, sweep_tree_state_v1, leaf_context_v1, inherited forward_holds"},
 "src/arggnn/mutation_routed_whole_tree_gnn_v1.py":{"lines":"255-307","role":"site/leaf/node projections, gate layers, GRU and classifier declarations"},
 "src/arggnn/anonymous_clade_flag_control_v1.py":{"lines":"222-246","role":"rank-eight current-node gained/lost structural residual"},
 "src/arggnn/ordered_clade_episode_gnn_v1.py":{"lines":"346-390","role":"own-state residual eligibility and second traversal"},
 "research/reviews/order2_cut_memory_implementation_20261015_v1/order2_cut_memory_v1.py":{"lines":"372-434","role":"same-sample span-decay forward/reverse scans; current-group average/centering; output projection"},
 "research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/direct_readout_cut_memory_v1.py":{"lines":"123-152","role":"actual final encoder splice: structural sum + memory injected at cut representatives before shared sweep2"},
 "paper/figure_redesign_20261017_v7/build_figure.py":{"lines":"full source","role":"immutable preceding toy input and exact marginal topology source"},
 "paper/figure_redesign_20261017_v7/source_mapping.json":{"lines":"full mapping","role":"immutable provenance and scientific operation mapping"}
}
for f,v in sources.items():v["sha256"]=sha(root/f)
meta=json.loads((out/"figure_metadata.json").read_text())
mapping={
 "status":"v9 isolated redraw; ROOT integration pending",
 "source_of_truth":sources,
 "immutable_predecessor":"paper/figure_redesign_20261017_v7",
 "output_geometry":{"width_inches":7.0,"height_inches":6.6,"minimum_label_font_pt":7.0,"panels":"a, b, c; no figure/panel titles"},
 "a":{
   "native_source":"illustrative_full_arg.trees, copied byte-for-byte from v7",
   "geometry":"Every ribbon is one actual native tskit edge from its child to parent over exactly edge.left..edge.right. Three exact full-ARG marginal slices are at their interval midpoints. Global node horizontal coordinate is a constant display coordinate derived from the union of true descendants. Ribbon opacity encodes no probability. No invented lineage connectors.",
   "recombination":"native node flags & 131072, diamond at original recorded time",
   "intervals_bp":meta["genomic_breakpoints_bp"]
 },
 "b":{
   "native_source":"illustrative_tree_sequence.trees, copied byte-for-byte from v7",
   "topology":"exact three unary-collapsed marginals, original node ages and planar sample order",
   "alleles":"one actual hypothetical observed site per toy interval, full.genotype_matrix()[interval_index,sample_id]; all five genotypes observed",
   "references":"r1 Ana, r2 CHG, r3 WHG, r4 EHG. Known one-hot order Ana/CHG/WHG/EHG. q1 is displayed as ?, abbreviating the real zero ancestry-vector plus unknown indicator; alleles stay observed.",
   "internal_feature":"genuine internal nonroot node with age tau, root=0, sampled=0. No ancestral ancestry label.",
   "abstraction":"The main figure abbreviates full site/frequency/callability/age feature encoding, defined in Methods. It does not substitute one scalar allele for the actual encoder."
 },
 "c":{
   "input":"MLP icon summarizes site/leaf/node encoders and additive initialization at every node b_v",
   "GNN":"Both GNN_theta modules complete an upward and a downward traversal of every native node, with identical learned weights and observed gates. Tree glyphs preserve genuine T1 topology but are representative operators applied to all three trees, not selection of T1. Every icon branch displays its message direction.",
   "structural":"Delta_str is the sum of gained/lost and eligible own-state local residuals. Its input is only the current node state plus observed topology conditioning. No arrow imports another tree's learned internal-node state.",
   "base_residual":"The b_v junction supplies the upper lane into plus. The first-GNN output junction supplies separate structure and sample-memory branches. The plus returns b_v + Delta_str + Delta_memory to traversal2.",
   "sample_memory":"Sample memory box abstracts cut representative projection to forcing, assignment to each sampled identity, independent forward/reverse physical-span-decay scans, direction average, current-group average and current-context subtraction. Its explicit Linear is the real output projection to width32 at the current cut-group representative. Grouping, centering and decay internals are intentionally in Methods; no new operator is introduced.",
   "identity":"q1,r1,r2,r3,r4 is one fixed visual row permutation in memory and readout across all3trees. Display order is independent of changing planar tip order; sample identities never merge.",
   "readout":"A learned per-sample projection of second-pass upward/outside states gives e_tq; the matrix glyphs preserve all samples, not whole-graph pooling. One-layer BiGRU receives these embeddings and observed geometry for every sample. Concat joins local32 with recurrent64, then the inherited classifier makes four probabilities.",
   "geometry":"I_t strip abbreviates log-scaled interval width and left/right neighboring-center distances. No geometry feature is omitted from actual model.",
   "output":"Only displayed output selects q1. The illustrative probabilities/calls are copied from v7; they are not fitted predictions. Exact interval widths are preserved."
 },
 "simplifications":"Removed the tiny clade-presence matrix, cut-group chip sets, forcing/history/centered-vector algebra, span subscripts, and geometry tuple. No supplementary inset is necessary; Methods hold the details.",
 "scope":"All writes within fresh paper/figure_redesign_20261017_v9. Canonical assets, manuscript, source modules, v7 assets and blind review remain unchanged. No scientific model, data evaluation, Git operation, resource admission or job.",
 "operation_marker":"ROOT executed PDF edit marker successfully exactly once before this redraw. Worker did not duplicate it."
}
(out/"source_mapping.json").write_text(json.dumps(mapping,indent=2)+"\n")
bounds=json.loads((out/"text_layout_bounds.json").read_text())
over=[]
for i,a in enumerate(bounds):
    ax,ay,aw,ah=a["bbox_figure"]
    for j in range(i):
        b=bounds[j];bx,by,bw,bh=b["bbox_figure"]
        ix=min(ax+aw,bx+bw)-max(ax,bx);iy=min(ay+ah,by+bh)-max(ay,by)
        if ix>1e-6 and iy>1e-6:
            over.append({"a":a["text"],"b":b["text"],"intersection_figure":[ix,iy]})
# These are text bounding rectangles, not paths; all are checked visually too.
assert not over,over
assert all(v["fontsize"]>=7 for v in bounds)
assert sha(out/"illustrative_full_arg.trees")==sha(prev/"illustrative_full_arg.trees")
assert sha(out/"illustrative_tree_sequence.trees")==sha(prev/"illustrative_tree_sequence.trees")
caption=(out/"caption.tex").read_text()
plain=re.sub(r"\\[A-Za-z]+","",caption)
plain=re.sub(r"[{}$]","",plain)
qa={
 "actual_PDF_render":"pdftoppm -singlefile -r 220 -png TreeLoom_architecture.pdf TreeLoom_architecture_render",
 "visual_review":"Latest PDF-rendered PNG inspected after final routing correction. No visible overlap or clipping; residual lane clears GNN header, single source dots, arrows terminate on boxes. All meaningful labels >=7pt at exact7-inch output.",
 "text_count":len(bounds),"minimum_font_pt":min(v["fontsize"] for v in bounds),
 "text_bbox_overlap_pairs":over,
 "outside_text":False,
 "pdf_page_info":subprocess.check_output(["pdfinfo",str(out/"TreeLoom_architecture.pdf")],text=True),
 "caption_word_count_rough":len(plain.split()),
 "native_toy_copy_SHA_match":True,
 "native_ARG_edge_surfaces":len(meta["ARG_native_edge_surfaces"]),
 "no_whole_graph_pooling":True,
 "same_sample_row_names":meta["memory_display_order"],
 "model_execution":False,
 "thread_limits":"OMP/OpenBLAS/MKL/NumExpr/VECLIB each set1 before plotting imports",
 "fit_estimate":"Figure6.6in + about1.2in caption at9pt fits a full7in-wide page within common journal text heights. ROOT to verify final manuscript integration."
}
(out/"redesign_QA.json").write_text(json.dumps(qa,indent=2)+"\n")
handoff="""TreeLoom v9 isolated architecture redraw
--------------------------------------
Editable geometry/source: build_figure.py; SVG keeps text editable.
Primary outputs: TreeLoom_architecture.pdf / .svg / .png.
Actual PDF QA render: TreeLoom_architecture_render.png.
Caption: caption.tex (no figure title). Scientific correspondence: source_mapping.json.
Geometry: 7 x 6.6 inches, every text label at least 7 pt.
Substantial changes: native recorded ARG edge-span ribbons; explicit 0/1 allele cells
and shared ancestry vectors; larger two-tier pipeline; two complete shared Up/Down
passes; fixed named sample rows; separate simplified memory and final BiGRU;
explicit original-base residual and final local skip; retained probability colors.
All outputs are schematic; no model was executed and no efficacy claim was made.
Only this fresh v9 directory was written. v7, canonical figure, manuscript and the
earlier image-only review are untouched. ROOT artifact marker reused, not repeated.
ROOT should inspect and integrate only after the separate fresh blind image read.
"""
(out/"HANDOFF.txt").write_text(handoff)
files=[p for p in out.iterdir() if p.is_file() and p.name not in ("SHA256SUMS","handoff_seal.json")]
sums={p.name:sha(p) for p in sorted(files)}
(out/"SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k,v in sums.items()))
seal={"status":"sealed after final PDF-render visual QA; ROOT review/integration pending",
 "directory":str(out),"sha256":sums,"SHA256SUMS_sha256":sha(out/"SHA256SUMS"),
 "image_SHA256":sums["TreeLoom_architecture.png"],"pdf_SHA256":sums["TreeLoom_architecture.pdf"],
 "source_SHA256":sums["build_figure.py"],"predecessor_image_SHA256":sha(prev/"TreeLoom_architecture.png"),
 "no_edits_outside_directory":True,"blind_review_unchanged":True}
(out/"handoff_seal.json").write_text(json.dumps(seal,indent=2)+"\n")
print(json.dumps({"png_sha256":seal["image_SHA256"],"pdf_sha256":seal["pdf_SHA256"],"caption_words_rough":qa["caption_word_count_rough"],"text_overlap_pairs":len(over),"min_font":qa["minimum_font_pt"]},indent=2))
