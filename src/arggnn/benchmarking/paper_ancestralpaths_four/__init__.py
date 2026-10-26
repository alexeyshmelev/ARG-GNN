"""Separate BAA-excluded AncestralPaths experiment from supplement section 3.1.

The inferred genealogies stay unchanged. BAA contributes neither distribution
features nor target queries. This is a fresh four-output classifier, not a sum
of probabilities from the six-output experiment. Imports launch no work.
"""

CONDITION = "ancestralpaths_historical_cnn_four_no_baa"
PATH_NAMES = ("Ana", "CHG", "WHG", "EHG")
QUERY_COHORTS = ("present_day", "Bronze_Age", "Neo", "Yam")
COHORT_FEATURE_ORDER = ("Bronze_Age", "Yam", "Neo", "WHG", "EHG", "Ana", "CHG")
FEATURE_NAMES = COHORT_FEATURE_ORDER + ("ancestor_age_div_1500",)
