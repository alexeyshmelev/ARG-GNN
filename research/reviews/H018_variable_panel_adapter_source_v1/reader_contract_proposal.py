"""Pure metadata/coordinate proposal; contains no file reader or execution boundary."""
from decimal import Decimal,ROUND_HALF_EVEN
import math,struct
CLASSES=("ANA","CHG","WHG","EHG")
def need(ok,message):
    if not ok:raise ValueError(message)
def panel_metadata_v1(rows):
    need(isinstance(rows,list) and len(rows)==40,"Exactly40 original roster rows")
    labels=[];ages=[];subjects=[]
    for i,row in enumerate(rows):
        need(row["sample_index"]==i and row["haplotype"]==i%2,"Exact persistent observed order")
        need(row["role"]==("reference" if i<16 else "query"),"OriginalR16/Q24 roles")
        cls=row["observed_reference_class"]
        if i<16:
            need(type(cls) is int and 0<=cls<4 and row["observed_cohort"]==CLASSES[cls],"Observed source class order")
            labels.append(cls)
        else:
            need(cls is None and row["query_ancestry_truth"] is None,"Unknown query labels and no query truth")
            labels.append(-1)
        need(row["date_zero"]=="calBP1950" and row["years_per_generation"]==28,"Original date convention")
        age=row["age_generations"]
        need(type(age) in (int,float) and math.isfinite(age) and age>=0
          and age==row["age_mean_cal_BP"]/28.,"Original source fractional sampling ages")
        ages.append(age);subjects.append(row["subject_id"])
        if i%2:
            need(subjects[i]==subjects[i-1] and labels[i]==labels[i-1]
              and ages[i]==ages[i-1],"Adjacent original diploid pair")
    need(len(set(subjects))==20 and tuple(labels.count(c) for c in range(4))==(4,4,4,4),"Original20subjects/four reference haps per source")
    counts=[sum(label==c and (age<350 if cohort==0 else age>=350)
      for label,age in zip(labels,ages)) for cohort in range(2) for c in range(4)]
    return dict(haplotypes=40,reference_class=labels,query_column=list(range(16,40)),
      source_sample_age_generations=ages,donor_group_counts=counts,
      classes=list(CLASSES),class_counts=[4,4,4,4],source_roster_order_preserved=True)
def verify_native_tip_dates_v1(source_ages,native_times):
    need(len(source_ages)==len(native_times)==40,"Complete40 tip clocks")
    f32=lambda x:struct.unpack("f",struct.pack("f",x))[0]
    parsed=[float(format(Decimal.from_float(float(a)).quantize(Decimal("0.000001"),rounding=ROUND_HALF_EVEN),"f")) for a in source_ages]
    need(list(native_times)==parsed,"Exact author six-decimal tip-clock semantics")
    need(all(f32(a)==f32(b) for a,b in zip(source_ages,native_times)),
      "Original sample-age and untouched native-clock FP32 bank equality must hold; no time edits")
    return dict(native_times_edited=False,source_ages_edited=False,
      original_source_age_and_native_sixdecimal_clock_kept_separate=True,model_FP32_tip_equal=True)
def translated_interval_view_v1(native_intervals,original_local_POS_bp1):
    need(bool(native_intervals) and bool(original_local_POS_bp1),"Nonempty original graph/site support")
    need(all(type(p) is int and p>=1 for p in original_local_POS_bp1)
      and all(a<b for a,b in zip(original_local_POS_bp1,original_local_POS_bp1[1:])),"Exact ordered unique original local SNP positions")
    first,last=original_local_POS_bp1[0],original_local_POS_bp1[-1]
    left,right=first-1,last
    translated=[]
    for i,pair in enumerate(native_intervals):
        need(len(pair)==2 and all(math.isfinite(v) for v in pair) and pair[0]<pair[1],"Exact finite native intervals")
        if i:need(native_intervals[i-1][1]==pair[0],"All native marginal holds contiguous")
        a,b=pair[0]-1,pair[1]-1
        a,b=max(a,left),min(b,right)
        need(a<b,"Every original hold must have genuine support; no dropped holds")
        translated.append([a,b])
    need(native_intervals[0][0]==0 and native_intervals[-1][1]==last+1,"Untouched converter native extent")
    need(translated[0][0]==left and translated[-1][1]==right
      and all(a[1]==b[0] for a,b in zip(translated,translated[1:])),"Complete genuine translated view")
    positions=[p-1 for p in original_local_POS_bp1]
    need(all(left<=p<right for p in positions),"Every original SNP retained in genuine support")
    return dict(intervals_regional_bp0=translated,positions_regional_bp0=positions,
      genuine_regional_bp0=[left,right],interval_count_preserved=True,
      sites_dropped=0,native_tables_modified=False,coordinate_transform="native_minus1")
