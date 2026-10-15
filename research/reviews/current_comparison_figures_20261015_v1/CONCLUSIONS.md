# Supported descriptive error patterns

These figures use the already completed European C4 original eight VAL parents and the Present182 query cohort. Every method has exactly the same per-parent true-class query-bp totals and denominators, and the receipts share original parent coordinates/coverage. The nominal interval is2Mb. Recorded represented support ranges from363,867,322 to363,990,900 query-bp per parent, compared with364,000,000 nominal query-bp. The class order is ANA, CHG, WHG, EHG; source label Ana is capitalized for display without reordering.

Each true-class row is normalized separately within each parent, then the eight normalized matrices receive equal weight. Cell SD is sample SD with ddof1 across eight independent simulation parents, expressed in percentage points. It is not optimizer-seed variation, SE or a confidence interval. Normalizing the pooled count matrix would change individual cells by up to0.153669percentage points and is not the plotted statistic.

Native ARGMix has its largest mean directed errors between CHG and ANA: CHG to ANA is 4.9315% +/- 2.7712 percentage points, and ANA to CHG is 4.8237% +/- 1.4919 percentage points. Its WHG to EHG error is 3.8004% +/- 2.0898 percentage points. These are conditional assignment errors within this simulated development panel, without a causal demographic interpretation.

Derived C4 AncestralPaths most often misassigns CHG to ANA: 5.3977% +/- 0.8745 percentage points. The figure uses the recorded C4 physical query-bp confusion counts. It does not collapse or reconstruct the separate six-path SNP-focal confusion matrix.

H045 foreign_episode has higher mean recall for all four classes than either native setup: ANA 97.1980% +/- 1.0535percentage points, CHG 95.9305% +/- 1.4606percentage points, WHG 96.0460% +/- 1.2921percentage points, EHG 97.6067% +/- 1.0954percentage points. This pattern describes these completed fits and does not establish an architecture effect.

The improvement is not uniform across individual error channels. WHG to EHG is 2.2415% +/- 0.9910 percentage points for H045 versus 1.5316% +/- 1.0118 percentage points for derived C4 AP. H045 CHG to EHG is 1.3993% +/- 0.9862 percentage points versus 1.2036% +/- 0.6432 percentage points for ARGMix. Higher total recall can coexist with larger individual off-diagonal entries.

The population panel and cohort table keep H047 original-risk and balanced-risk arms separate. Their already recorded Present182 balanced-minus-original accuracy difference is-0.020029percentage points and the registered benefit criterion remains false. No new model selection, prediction, significance test or scientific evaluation was performed.

Native AP has recorded infinite_nll_query_bp in some parents, including800 query-bp in Present182 VAL006. The table explicitly reports infinite NLL with undefined sample SD and exact affected-parent counts, while preserving the source aggregate's null mean/SD. Other absent aggregate values remain unavailable, including the native Yam boundary F1 aggregate. Source-defined n below8 stays visible.

Native ARGMix used11epochs with native early stopping. Native AP used5epochs with its native fixed schedule and recorded six-path probability collapse to C4. H045 and both H047 arms use H042 warmstarts plus30passes/960updates per arm, seed101. Inputs, objectives, representations and selection/training budgets differ. These are shared-support pipeline comparisons on internal development data, without architecture attribution, exact original-paper reproduction or independent TEST confirmation.
