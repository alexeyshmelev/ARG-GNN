# Render saved European development diagnostics without fitting or rescoring.
from pathlib import Path
import hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/reviews/development_confusion_20261016_v1"
FILES={"H045":"experiments/results/H045_GPU5_episode_evaluate_seed101_v1/foreign_episode.selected_full_diagnostics.json","native":"research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json","scope":"research/reviews/native_common_cohort_diagnostics_20261015_v1/same_support_comparison.json","selection":"research/reviews/H045_completed_comparison_root_20261015_v1/summary.json"}
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 assert not OUT.exists(),"Preserve prior rendering"
 raw={k:json.loads((ROOT/f).read_text())for k,f in FILES.items()}
 assert raw["scope"]["class_order"]==["Ana","CHG","WHG","EHG"]
 assert raw["selection"]["arms"]["foreign_episode"]["selected_pass"]==30 and not raw["H045"]["selection_revised"]
 series={"H045":raw["H045"]["cells"],"ARGMix":raw["native"]["native_ARGMix"],"AncestralPaths C4":raw["native"]["native_AP"]}
 parents=["ap_expanded_val_%03d"%i for i in range(8)]
 rows={name:{v["parent_id"]:v["groups"]["primary_present_day_182"]for v in cells}for name,cells in series.items()}
 norm={}
 for name,cells in rows.items():
  assert sorted(cells)==parents
  arrays=[]
  for parent in parents:
   g=cells[parent];base=rows["H045"][parent]
   for field in ("query_count","denominator_query_bp","coverage_original_bp0","represented_span_bp"):assert g[field]==base[field],(name,parent,field)
   c=np.asarray(g["confusion_truth_rows_prediction_columns_query_bp"],dtype=np.float64)
   b=np.asarray(base["confusion_truth_rows_prediction_columns_query_bp"])
   assert c.shape==(4,4) and np.isfinite(c).all() and (c>=0).all() and (c.sum(1)>0).all()
   assert np.array_equal(c.sum(1),b.sum(1)),(name,parent,"identical truth denominators")
   assert math.isclose(float(c.sum()),g["denominator_query_bp"],rel_tol=1e-12,abs_tol=1e-6)
   assert math.isclose(float(np.trace(c)/c.sum()),g["accuracy"],rel_tol=1e-12,abs_tol=1e-12)
   for i,cl in enumerate(g["per_class"]):assert cl["class_0based"]==i and math.isclose(float(c[i,i]/c[i].sum()),cl["recall"],rel_tol=1e-12,abs_tol=1e-12)
   arrays.append(c/c.sum(1,keepdims=True)*100)
  norm[name]=np.stack(arrays)
 mean={k:v.mean(0)for k,v in norm.items()};sd={k:v.std(0,ddof=1)for k,v in norm.items()}
 paired=np.diagonal(norm["H045"]-norm["ARGMix"],axis1=1,axis2=2)
 mu=paired.mean(0);spread=paired.std(0,ddof=1);classes=["ANA","CHG","WHG","EHG"]
 result=dict(schema="existing_European_development_confusion_v1",scope="Present182, eight reused VAL simulations, 2Mb, one training seed",class_order=classes,normalization="true-class span within each parent, then equal mean over eight parents",SD_unit="eight validation simulations, not training seeds or a confidence interval",truth_rows_prediction_columns=True,new_predictions_or_fits=False,new_truth_access=False,original_paper_reproduction=False,architecture_only_attribution=False,parents=parents,mean_percent={k:v.tolist()for k,v in mean.items()},sample_SD_percent={k:v.tolist()for k,v in sd.items()},per_parent_normalized_percent={k:v.tolist()for k,v in norm.items()},paired_H045_minus_ARGMix_recall_pp=dict(mean=mu.tolist(),sample_SD=spread.tolist(),per_parent=paired.tolist()),source_refs={k:ref(ROOT/f)for k,f in FILES.items()},render_source_ref=ref(Path(__file__).resolve()))
 OUT.mkdir();(OUT/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
 plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","svg.fonttype":"none","pdf.fonttype":42})
 fig,axes=plt.subplots(1,3,figsize=(11.2,4.25),layout="constrained")
 for ax,name in zip(axes,series):
  a=mean[name];im=ax.imshow(a,cmap="Blues",vmin=0,vmax=100)
  ax.set_xticks(range(4),classes);ax.set_yticks(range(4),classes);ax.set_xlabel("Predicted ancestry");ax.set_ylabel("True ancestry");ax.set_title(name)
  for i in range(4):
   for j in range(4):ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",color="white"if a[i,j]>55 else"black",fontsize=9)
 fig.colorbar(im,ax=axes,shrink=.78,label="Mean true-class span (%)");fig.suptitle("European development panel: present-day queries",fontsize=13)
 for ext in ("png","svg","pdf"):fig.savefig(OUT/("present_day_confusion."+ext),dpi=180)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.6,3.65),layout="constrained")
 ax.errorbar(range(4),mu,yerr=spread,fmt="o",color="#174c79",capsize=5,markersize=7);ax.axhline(0,color="0.4",lw=1);ax.set_xticks(range(4),classes)
 ax.set_ylabel("H045 minus ARGMix recall (pp)");ax.set_title("Paired class recall differences on eight VAL simulations",fontsize=11)
 fig.supxlabel("Bars: sample SD of paired differences. One training seed per model.",fontsize=9)
 for ext in ("png","svg","pdf"):fig.savefig(OUT/("paired_class_recall."+ext),dpi=180)
 plt.close(fig)
 lines=["# Existing European development confusion diagnostic","","Rendered from saved selected metrics. No model fitting, prediction generation, target joining or checkpoint selection was performed.","","Present182, eight reused validation simulations, 2 Mb. Normalize each true-class row within each simulation, then equally average eight simulations. All methods have identical per-parent scored support and true-class denominators.","","[Confusion matrices](present_day_confusion.png) · [Paired recall differences](paired_class_recall.png) · [Values and source hashes](metrics.json)","","| True ancestry | H045 recall % | ARGMix recall % | AP C4 recall % | Paired gain, pp ± SD |","|---|---:|---:|---:|---:|"]
 for i,cl in enumerate(classes):
  h=mean["H045"][i,i];a=mean["ARGMix"][i,i];b=mean["AncestralPaths C4"][i,i]
  lines.append(f"| {cl} | {h:.3f} | {a:.3f} | {b:.3f} | {mu[i]:+.3f} ± {spread[i]:.3f} |")
 lines += ["","SD describes validation simulations, not training seeds. This does not reproduce original-paper tables or isolate architecture. ARGMix inputs, training budgets and model selection differ. AncestralPaths uses the released Dense model with six-path probabilities collapsed to four classes before argmax. H045 is the selected foreign-episode seed101/pass30 development checkpoint.","","This diagnostic does not establish independent generalization, useful cross-tree novelty or performance against ARGformer/FLARE."]
 (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
 products={p.name:ref(p)for p in OUT.iterdir()if p.is_file()}
 (OUT/"render_receipt.json").write_text(json.dumps(dict(schema="existing_scientific_diagnostic_render_v1",source_ref=ref(Path(__file__).resolve()),products=products,model_fits=0,new_predictions=0),indent=2)+"\n")
 print(json.dumps(dict(output=str(OUT),paired_recall_gain_pp=mu.tolist(),paired_SD_pp=spread.tolist(),new_model_fit=False)))
if __name__=="__main__":main()
