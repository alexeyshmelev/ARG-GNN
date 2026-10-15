"""Deterministic editable SVG from frozen source facts only; imports no model/data."""
from pathlib import Path
from html import escape
import hashlib,json,xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
W,H=1800,1450
C={"ink":"#183047","muted":"#52677B","line":"#7990A3","reuse":"#EDF3FA","reuse_stroke":"#698BB2","candidate":"#E5F5F1","candidate_stroke":"#218A79","control":"#FFF3DE","control_stroke":"#BA8B3D","panel":"#F9FBFD","white":"#FFFFFF"}
parts=[]
def add(s):parts.append(s)
def txt(x,y,s,size=20,weight=400,fill=None,anchor="start",italic=False):
 add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill or C["ink"]}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+f'>{escape(s)}</text>')
def lines(x,y,ss,size=20,step=27,weight=400,fill=None,anchor="start"):
 for i,s in enumerate(ss):txt(x,y+i*step,s,size,weight,fill,anchor)
def box(id,x,y,w,h,title,body,kind="reuse",size=20):
 add(f'<g id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{C[kind]}" stroke="{C[kind+"_stroke"]}" stroke-width="1.8"/>')
 txt(x+17,y+31,title,21,700)
 lines(x+17,y+61,body,size,25)
 add("</g>")
def panel(id,x,y,w,h,title):
 add(f'<g id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{C["panel"]}" stroke="#CED9E3" stroke-width="1.4"/>')
 txt(x+18,y+32,title,24,700);add("</g>")
def path(d,color=None,dash=False,width=2.6,arrow=True):
 add(f'<path d="{d}" fill="none" stroke="{color or C["line"]}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(' stroke-dasharray="7 6"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
def node(x,y,r=4,fill="#FFFFFF",stroke=None,ring=False):
 if ring:add(f'<circle cx="{x}" cy="{y}" r="{r+4}" fill="none" stroke="{C["candidate_stroke"]}" stroke-width="2.4"/>')
 add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or C["ink"]}" stroke-width="1.4"/>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="figtitle figdesc">')
add('<title id="figtitle">H049 candidate architecture: sample-witness order-2 cut memory</title>')
add('<desc id="figdesc">Source-aligned editable vector architecture. Complete dated observed trees and persistent sampled-leaf identities feed reused two-sweep PyG computation. A candidate bidirectional rank-eight memory remaps pair-cell states using actual three-partition sample intersections between sweeps. Two matched controls replace this remapping. The inherited full-parent bidirectional GRU and H048 SNP plus 100-base-pair grid decoder produce four-class probabilities. This is a candidate, not an accepted or validated method.</desc>')
add('<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7.8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="#7990A3"/></marker></defs>')
add('<style>text{font-family:Arial,Helvetica,sans-serif}path,rect,circle{vector-effect:non-scaling-stroke}</style>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
txt(30,43,"H049 candidate: sample-witness order-2 cut memory",32,700)
txt(31,75,"Source-aligned architecture  •  candidate, not an accepted or validated method",19,400,C["muted"])
for x,label,k in [(1140,"Reused backbone / decoder","reuse"),(1455,"Candidate memory","candidate"),(1645,"Matched controls","control")]:
 add(f'<rect x="{x}" y="62" width="15" height="15" rx="3" fill="{C[k]}" stroke="{C[k+"_stroke"]}"/>');txt(x+23,75,label,16)

panel("A",28,105,815,306,"A  Complete observed tree sequence and cut relation")
panel("B",865,105,907,306,"B  One memory arm per matched model")
# Toy complete trees: five persistent leaf identities, local node IDs vary.
add('<g id="A1-complete-trees">')
palette={"a":"#6D91C8","b":"#CF775F","c":"#6AA57D","d":"#A385B9","e":"#D6A646"}
for j,label in enumerate(("A","B","C")):
 ox=49+j*260
 # y positions preserve monotone observed dates; older sample e is at 450.
 pts={"root":(ox+109,170),"top":(ox+83,205),"a":(ox+18,318),"b":(ox+61,318),"c":(ox+105,318),"d":(ox+149,318),"e":(ox+213,218)}
 if label=="A":
  pts.update(g=(ox+61,263),ab=(ox+39,295))
  edges=[("root","top"),("root","e"),("top","g"),("top","d"),("g","ab"),("g","c"),("ab","a"),("ab","b")]
  reps={"g","d","e"}
 elif label=="B":
  pts.update(ab=(ox+39,269),cd=(ox+127,264))
  edges=[("root","top"),("root","e"),("top","ab"),("top","cd"),("ab","a"),("ab","b"),("cd","c"),("cd","d")]
  reps={"ab","cd","e"}
 else:
  pts.update(g=(ox+105,263),bc=(ox+83,295))
  edges=[("root","top"),("root","e"),("top","a"),("top","g"),("g","bc"),("g","d"),("bc","b"),("bc","c")]
  reps={"a","g","e"}
 for u,v in edges:
  x1,y1=pts[u];x2,y2=pts[v];path(f"M{x1} {y1} L{x2} {y2}",C["ink"],width=1.5,arrow=False)
 path(f"M{ox+1} 242 H{ox+230}",C["candidate_stroke"],dash=True,width=1.5,arrow=False)
 for n,(x,y) in pts.items():node(x,y,5 if n in palette else 4,palette.get(n,"#FFFFFF"),ring=n in reps)
 for n in palette:
  x,y=pts[n];txt(x,y+23,n,17,700,palette[n],"middle")
 txt(ox+118,153,f"partition {label}",18,700,anchor="middle")
 txt(ox+9,232,"350 gen.",14,400,C["candidate_stroke"])
txt(51,365,"Schematic identities a–e; e is sampled above 350 generations.",17,400,C["muted"])
txt(51,389,"All 952 sampled leaves + all internal nodes remain in every whole-tree computation.",17,700)
add("</g>")
# Matched arms and correct triple-supported remapping.
for id,y,title,body,kind in [
 ("B1",153,"Candidate: triple-supported pair memory","m(A,B) → m(B,C), weighted by n(A∩B∩C) / n(B∩C)","candidate"),
 ("B2",221,"Control 1: adjacent-marginal null","Pool within B using n(A∩B) / n(B); discard dependence on C","control"),
 ("B3",289,"Control 2: stable-leaf memory","Keep a rank-8 state for each persistent sampled-leaf identity","control")]:
 add(f'<g id="{id}"><rect x="886" y="{y}" width="864" height="59" rx="9" fill="{C[kind]}" stroke="{C[kind+"_stroke"]}" stroke-width="1.4"/>')
 txt(902,y+24,title,19,700);txt(902,y+47,body,18);add("</g>")
txt(889,380,"All arms: same U, O, ℓ parameterization; 513 memory scalars + 31,933 common scalars.",17,700)
txt(889,400,"Identical warm start / fresh initialization; separate models and independent AdamW.",17,400,C["muted"])

panel("C",28,433,1744,595,"C  Actual computation order: candidate memory enters between the two whole-tree sweeps")
box("C1-inputs",49,495,232,139,"Observed inputs",["Dated topology + spans","SNPs / callability","Proxy labels + ages","Edge geometry / support"],size=18)
box("C2-first-sweep",321,495,239,139,"First PyG tree sweep",["Site / node base: 32-d","Upward aggregation","Downward cavity messages","All native nodes retained"],size=18)
box("C3-clade",620,495,266,111,"Reused clade injection",["Local anonymous-clade flags","Applied to upward states"],size=18)
box("C5-sum",945,495,159,139,"Add at nodes",["base + clade"," + candidate Δ","at real cut reps"],size=18)
box("C6-second-sweep",1150,495,240,139,"Second PyG sweep",["Same full dated trees","Upward + cavity states","Leaf context: 32-d"],size=18)
box("C7-genomic-gru",1434,495,316,139,"Reused genomic history",["Same 952 leaf columns","Context + span geometry","Bidirectional genomic GRU","State [T, 952, 96]"],size=18)
# Forward paths and explicit base bypass.
path("M281 550 H321")
path("M560 532 H620")
txt(575,519,"up",16,400,C["muted"])
path("M886 550 H945")
path("M1104 550 H1150")
path("M1390 550 H1434")
path("M165 634 V661 H1010 V634",dash=True)
txt(603,654,"unchanged base",17,400,C["muted"],"middle")
# Candidate memory branch with exact internal order.
add(f'<g id="C4-memory"><rect x="321" y="695" width="783" height="298" rx="15" fill="{C["candidate"]}" stroke="{C["candidate_stroke"]}" stroke-width="2"/>')
txt(340,727,"Candidate cut memory: actual two-direction, order-2 path",22,700)
txt(340,758,"At real representative r:  f = U(upward[r] + outside[r])  ∈ ℝ⁸",20)
txt(340,787,"Persistent sample identities witness A ∩ B ∩ C; pair cells carry m(A,B).",18)
add('<g id="C4a-three-partitions">')
for x,label in [(389,"A"),(535,"B"),(681,"C")]:
 add(f'<rect x="{x}" y="804" width="78" height="41" rx="7" fill="#FFFFFF" stroke="#218A79" stroke-width="1.5"/>')
 txt(x+39,833,label,23,700,anchor="middle")
path("M467 825 H535",C["candidate_stroke"])
path("M613 825 H681",C["candidate_stroke"])
txt(902,822,"L→R: (A,B) → (B,C)",17,700,C["candidate_stroke"],"middle")
txt(902,847,"R→L: (C,B) → (B,A)",17,700,C["candidate_stroke"],"middle")
txt(340,874,"1  At a changed partition: remap m(A,B) → m(B,C) using the selected arm.",18)
txt(340,901,"2  Each native hold: m ← ρm + (1−ρ)f,  ρ = exp(−span / ℓ).",19)
txt(340,928,"3  Mean the two directions on leaves → mean within the current cut group.",18)
txt(340,955,"4  Δ = O(read − f) ∈ ℝ³²; scatter only to actual representative nodes.",19)
txt(340,980,"First state = f; unchanged partitions keep their layout.  ℓ ∈ (100, 100,000) bp.",17,400,C["muted"])
add("</g></g>")
path("M442 634 V695",C["candidate_stroke"])
txt(457,686,"up + outside",17,400,C["candidate_stroke"])
path("M1104 855 H1124 V664 H1060 V634",C["candidate_stroke"])
txt(1150,722,"Cut plan is a grouping relation,",20,700)
txt(1150,750,"not a pruned tree or cross-tree",20)
txt(1150,778,"matching of internal node IDs.",20)
txt(1150,821,"At age ≤ 350: highest actual",18,400,C["muted"])
txt(1150,847,"ancestor at or below 350.",18,400,C["muted"])
txt(1150,879,"At age > 350: actual sampled",18,400,C["muted"])
txt(1150,905,"node remains a singleton.",18,400,C["muted"])
txt(1434,724,"T = original native tree holds",18,400,C["muted"])
txt(1434,754,"No genomic-history reset",18,700)
txt(1434,781,"or sample / node subsampling",18)
txt(1434,830,"Each arm: 32,446 parameters",18,700)
txt(1434,859,"All common and memory",18)
txt(1434,886,"parameters remain trainable.",18)

panel("D",28,1051,1744,309,"D  Reused SNP evidence + original 100-bp grid decoder (H048 graph_to_transition)")
box("D1-tree-logits",49,1114,291,115,"Tree states → local logits",["Classifier; select 686 queries","Gather each scoring position's hold"],size=18)
box("D2-emission",390,1114,321,115,"Grid emissions",["Local tree logits"," + observed SNP-bin residual"],size=18)
box("D4-chain",1115,1114,310,115,"Exact reset-chain marginals",["Physical 100-bp center gaps","Full chromosome per query block"],size=18)
box("D5-probabilities",1470,1114,280,115,"Softmax probabilities",["[position, query, 4]","P original centers × 686 × 4"],size=18)
box("D2a-SNP-bins",49,1263,662,70,"Observed SNP evidence",["Disjoint ≤100-bp bins → 5 features → marker head"],size=18)
box("D3-transition-rates",758,1221,313,112,"Graph-conditioned rates",["Neighbor 32-d tree contexts","λ = λ₀ exp(g); four-class π"],size=18)
path("M340 1172 H390")
path("M711 1172 H1115")
path("M1425 1172 H1470")
path("M550 1263 V1229")
path("M1071 1272 H1092 V1203 H1115")
# GRU state feeds both classifier and rate head, no decoder state reinterpretation.
path("M1592 634 H1760 V1097 H194 V1114")
txt(1630,1088,"96-d state",16,400,C["muted"])
add('<path d="M915 1160 V1184" stroke="#FFFFFF" stroke-width="10"/>')
path("M915 1097 V1221")
txt(928,1140,"first 32 channels",16,400,C["muted"])
txt(32,1400,"Cut intersections use persistent sample identities only. Shown trees are schematic; no data, performance or novelty claim is encoded.",18,400,C["muted"])
txt(32,1428,"Editable vector artwork; exact block-to-function and line mapping, source hashes, and implementation notes accompany this SVG.",17,400,C["muted"])
add("</svg>")
svg="\n".join(parts)+"\n"
out=HERE/"architecture.svg"
out.write_text(svg)
tree=ET.fromstring(svg)
ids=[e.attrib["id"] for e in tree.iter() if "id" in e.attrib]
assert len(ids)==len(set(ids))
assert not any(e.tag.rsplit("}",1)[-1] in {"image","foreignObject","script"} for e in tree.iter())
assert all(e.attrib.get("font-size") is None or float(e.attrib["font-size"])>=14 for e in tree.iter())
print(json.dumps({"svg":str(out),"sha256":hashlib.sha256(svg.encode()).hexdigest(),"editable_text_elements":sum(e.tag.endswith("}text") for e in tree.iter()),"unique_ids":len(ids),"raster_images":0,"model_imports":False,"scientific_payload_reads":False}))
