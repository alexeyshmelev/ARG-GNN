"""Prospectively frozen selected-checkpoint tract diagnostics; original scores unchanged."""
from __future__ import annotations
import numpy as np
from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1, FullPanelExternalTargetV1, means_v1, PRIMARY

STRATA=((0.,1000.),(1000.,2000.),(2000.,5000.),(5000.,10000.))
DEFINITION=dict(length_strata_bp=[list(x) for x in STRATA],short_max_bp=10000.,
    true_recovery_rule="at_least_half_of_original_class_overlap_on_represented_support",
    false_short_rule="less_than_half_predicted_tract_span_overlaps_matching_true_class",
    endpoint_censor="exclude_support_endpoint_censored_and_original_domain_endpoint_tracts",
    typed_boundaries="directed_ancestry_transition_same_original_1kb_matching",
    wrong_bp_normalization="original2Mb_times_group_query_count")


def compact_diagnostics_v1(reports,helpers,tolerance):
    typed={str(a)+"->"+str(b):dict(true_boundaries=0,predicted_boundaries=0,matched=0,missed=0,extra=0)
        for a in range(4) for b in range(4) if a!=b}
    strata=[dict(left_exclusive_bp=lo,right_inclusive_bp=hi,true_eligible=0,true_recovered=0,
        predicted_short=0,false_predicted_short=0,false_short_wrong_bp=0.) for lo,hi in STRATA]
    censored_true=censored_predicted=0
    for report in reports:
        true,pred=report["true_tracts"],report["predicted_tracts"]
        for key,counts in typed.items():
            a,b=map(int,key.split("->"))
            tb=[left["right_bp"] for left,right in zip(true,true[1:])
                if left["class_0based"]==a and right["class_0based"]==b]
            pb=[left["right_bp"] for left,right in zip(pred,pred[1:])
                if left["class_0based"]==a and right["class_0based"]==b]
            found=helpers.boundary_match(tb,pb,tolerance)
            for name in counts:counts[name]+=found[name]
        for tract in report["short_true_tracts"]:
            if tract["support_endpoint_censored"] or tract["original_domain_endpoint"]:
                censored_true+=1;continue
            length=tract["original_span_bp"]
            for row,(lo,hi) in zip(strata,STRATA):
                if lo<length<=hi:
                    row["true_eligible"]+=1
                    row["true_recovered"]+=int(tract["correct_overlap_fraction"]>=.5)
        support_left=true[0]["left_bp"];support_right=true[-1]["right_bp"]
        for tract in pred:
            length=tract["right_bp"]-tract["left_bp"]
            if length>10000.:continue
            if tract["left_bp"]==support_left or tract["right_bp"]==support_right:
                censored_predicted+=1;continue
            correct=sum(max(0.,min(tract["right_bp"],t["right_bp"])-max(tract["left_bp"],t["left_bp"]))
                for t in true if t["class_0based"]==tract["class_0based"])
            for row,(lo,hi) in zip(strata,STRATA):
                if lo<length<=hi:
                    row["predicted_short"]+=1
                    row["false_predicted_short"]+=int(correct/length<.5)
                    row["false_short_wrong_bp"]+=max(0.,length-correct)
    for counts in typed.values():
        nt,npred,matched=counts["true_boundaries"],counts["predicted_boundaries"],counts["matched"]
        counts.update(precision=matched/npred if npred else None,recall=matched/nt if nt else None,
            f1=2*matched/(nt+npred) if nt else None)
    for row in strata:
        row["true_short_recall"]=row["true_recovered"]/row["true_eligible"] if row["true_eligible"] else None
        row["false_short_wrong_bp_per_Mb"]=row["false_short_wrong_bp"]/(2.*len(reports))
    eligible=sum(r["true_eligible"] for r in strata);recovered=sum(r["true_recovered"] for r in strata)
    return dict(definition=DEFINITION,typed_transitions=typed,short_tract_strata=strata,
        true_short_eligible=eligible,true_short_recovered=recovered,
        uncensored_true_short_recall=recovered/eligible if eligible else None,
        censored_true_short_excluded=censored_true,censored_predicted_short_excluded=censored_predicted,
        false_short_wrong_bp_per_Mb=sum(r["false_short_wrong_bp"] for r in strata)/(2.*len(reports)))


class H058ValidationV1(FullPanelPrimaryValidationV1):
    def full_final(self,seals):
        # Exact original primary/full-score calls and cohort/support definitions.
        # Only compact diagnostics are added before the original per_query discard.
        if self.full_calls:raise ValueError("Full selected diagnostics already computed")
        records,probabilities=self._verify(seals)
        cells=[]
        for p,parent in self.parents.items():
            target=FullPanelExternalTargetV1(parent,self.bindings[p],"VAL_after_sealed_predictions",
                self.reader,sealed_records=list(records.values()))
            cohorts=parent["cohorts"]
            groups={PRIMARY:np.flatnonzero(cohorts=="present_day"),
                "secondary_ancient_504":np.flatnonzero(cohorts!="present_day"),"all_queries_686":np.arange(686)}
            groups.update({"age_stratum_"+c:np.flatnonzero(cohorts==c) for c in ("Bronze_Age","BAA","Neo","Yam")})
            scores={}
            for name,columns in groups.items():
                score=self.helpers.full_span_score(probabilities[p],parent,target,columns,self.cfg["metrics"])
                score["H058_tract_diagnostics"]=compact_diagnostics_v1(score["per_query"],self.helpers,
                    self.cfg["metrics"]["boundary_tolerance_bp"])
                score.pop("per_query",None);score.pop("retained_original_grid_indexes",None)
                scores[name]=score
            cells.append(dict(parent_id=p,split="val",prediction_seal=records[p],groups=scores))
        self.full_calls+=1
        return cells,means_v1(cells,full=True)
