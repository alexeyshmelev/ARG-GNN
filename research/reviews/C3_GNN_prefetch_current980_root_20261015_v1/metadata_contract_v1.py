"""Exact original registry validation relocated for Torch-free root metadata checks."""
from arggnn.american_c3_common_v1 import CLASSES,require,read
REGISTRY_REF={'path': 'research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json', 'sha256': 'c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37'}
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TEST=tuple("american_c3_test_%03d"%i for i in range(10))
def authoritative_registry_v1(record=REGISTRY_REF):
    require(record==REGISTRY_REF,"Authoritative20x2Mb TRAIN registry must remain exact")
    value=read(record);rows=value["parents"]
    require(value["schema"]=="american_c3_independent_2Mb_TRAIN_wave_registry_v1"
        and value["class_order"]==list(CLASSES) and value["sequence_length_bp"]==2000000
        and [r["parent_id"] for r in rows]==list(TRAIN) and len(rows)==20,"Fixed independent TRAIN roster")
    for i,row in enumerate(rows):
        require(row["split"]=="train" and row["sequence_length_bp"]==2000000
            and row["sample_unit"]=="haploid" and row["sample_counts"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and row["ancestry_seed"]==22001+i and row["mutation_seed"]==1022001+i
            and row["path_sampling_seed"]==92201+i,"Frozen independent seeds/counts/length")
    held=value["deferred_heldout_30Mb_original_records"]
    require(tuple(r["parent_id"] for r in held if r["split"]=="val")==VAL
        and tuple(r["parent_id"] for r in held if r["split"]=="test")==TEST,"Separate original held-out registry")
    return {r["parent_id"]:r for r in rows}
