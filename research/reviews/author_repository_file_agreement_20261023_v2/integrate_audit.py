"""Integrate static source review evidence without importing reviewed projects."""
import sys
sys.stderr = sys.stdout
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import tarfile

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
BASE = ROOT / 'research/reviews/author_repository_file_agreement_20261023_v2'
PRIOR = ROOT / 'research/reviews/author_repository_deep_audit_20261023_v1'

def read(path):
    return json.loads(path.read_text())

def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

am = read(BASE / 'argmix/file_coverage.json')
af = read(BASE / 'argformer/file_coverage.json')
ap = read(BASE / 'ancestralpaths/file_coverage.json')
gst = read(BASE / 'gst/file_coverage.json')
flare_other = read(BASE / 'argformer/flare_non_java_coverage.json')

current = {
    'ARGMix': [r for r in am['files'] if r['public_author_file']],
    'ARGformer': [r for r in af['argformer_current'] if r['official']],
    'AncestralPaths': [r for r in ap['files'] if r['scope'] == 'current_upstream'],
    'GST': gst['rows'],
    'FLARE_Java': af['flare_java'],
    'FLARE_non_Java': flare_other['files'],
}
expected_counts = {'ARGMix': 85, 'ARGformer': 60, 'AncestralPaths': 17,
                   'GST': 166, 'FLARE_Java': 96, 'FLARE_non_Java': 9}
for repo, rows in current.items():
    assert len(rows) == expected_counts[repo], (repo, len(rows))

expected_paths = {
    'ARGMix': {r['path'] for r in read(PRIOR / 'argmix/current_tree.json')['tree'] if r['type'] == 'blob'},
    'ARGformer': {r['path'] for r in read(PRIOR / 'argformer/current_tree_local_verification.json')['file_checks']},
    'AncestralPaths': {r['path'] for r in read(PRIOR / 'ancestralpaths/current_tree_local_verification.json')},
    'GST': {r['path'] for r in read(PRIOR / 'gst/tree_main.json')['tree'] if r['type'] == 'blob'},
}
observed_paths = {
    'ARGMix': {r['author_relative_path'] for r in current['ARGMix']},
    'ARGformer': {r['upstream_relative_path'] for r in current['ARGformer']},
    'AncestralPaths': {r['path'] for r in current['AncestralPaths']},
    'GST': {r['path'] for r in current['GST']},
}
for repo, paths in expected_paths.items():
    assert paths == observed_paths[repo], (repo, paths - observed_paths[repo], observed_paths[repo] - paths)
flare_tag = read(PRIOR / 'argformer/flare_tag_source_census.json')
assert {r['path'] for r in flare_tag} == {r['path'] for r in current['FLARE_Java'] + current['FLARE_non_Java']}

hash_checks = []
def verify(repo, row, path):
    actual = digest(path)
    assert actual == row['sha256'], (repo, str(path), actual, row['sha256'])
    hash_checks.append({'repository': repo, 'path': str(path.relative_to(ROOT)), 'sha256': actual})

for repo in ['ARGMix', 'ARGformer']:
    for row in current[repo]:
        verify(repo, row, ROOT / row['path'])
for row in current['AncestralPaths']:
    verify('AncestralPaths', row, ROOT / row['local_path'])
with tarfile.open(PRIOR / 'gst/history/6469d56aa131463325058b897b1e435c6d2edcd2/source.tar.gz', 'r:gz') as archive:
    gst_payloads = {member.name.split('/', 1)[1]: archive.extractfile(member).read()
                    for member in archive.getmembers() if member.isfile()}
for row in current['GST']:
    actual = hashlib.sha256(gst_payloads[row['path']]).hexdigest()
    assert actual == row['sha256'], ('GST', row['path'])
    hash_checks.append({'repository': 'GST', 'archive_member': row['path'], 'sha256': actual})
flare_root = ROOT / 'data/native_FLARE_build_existing_JDK_root_v1/flare'
for row in current['FLARE_Java']:
    verify('FLARE', row, flare_root / row['path'])
with tarfile.open(BASE / 'argformer/flare_official_tag.tar.gz', 'r:gz') as archive:
    payloads = {member.name.split('/', 1)[1]: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}
for row in current['FLARE_non_Java']:
    actual = hashlib.sha256(payloads[row['path']]).hexdigest()
    assert actual == row['sha256']
    hash_checks.append({'repository': 'FLARE', 'archive_member': row['path'], 'sha256': actual})

am_partial = [r['author_relative_path'] for r in current['ARGMix']
              if not r.get('review_complete', False) or r['content_review_mode'] not in
              ['full_text_semantic_review', 'complete_structured_dependency_lock_review']]
assert not am_partial, ('ARGMix reading remains unfinished', am_partial)
for repo in ['argmix', 'argformer', 'ancestralpaths', 'gst']:
    assert (BASE / repo / 'REPORT.md').is_file(), repo

category_map = {
    'resolved': 'resolved_definition_or_agreement',
    'resolved_with_disclosure': 'agreement_with_disclosure',
    'source-conflict': 'paper_release_conflict',
    'source_conflict': 'paper_release_conflict',
    'conflict': 'paper_release_conflict',
    'upstream-unavailable': 'missing_upstream_detail',
    'upstream_ambiguity': 'ambiguous_paper_definition',
    'upstream_version_conflict': 'incoherent_source_versions',
    'explicit-study-choice': 'explicit_new_study_choice',
    'explicit-choice': 'explicit_new_study_choice',
    'source_condition': 'disclosed_source_condition',
    'resolved_with_remaining_choices': 'explicit_new_study_choice',
    'resolved_with_remaining_work': 'unfinished_project_implementation',
}
gaps = []
for repo, key in [('argmix', 'requirements'), ('argformer', 'entries'), ('ancestralpaths', 'rows'), ('gst', 'rows')]:
    matrix = read(BASE / repo / 'agreement_matrix.json')
    for row in matrix[key]:
        status = row['status']
        assert status in category_map, (repo, status)
        resolution = row.get('bounded_resolution', row.get('resolution', ''))
        gaps.append({'repository': repo, 'id': row['id'], 'category': category_map[status],
                     'source_status': status, 'topic': row.get('topic', row.get('detail', '')),
                     'resolution_or_next_action': resolution,
                     'source_matrix': repo + '/agreement_matrix.json',
                     'source_matrix_sha256': digest(BASE / repo / 'agreement_matrix.json'),
                     'live_change_performed_by_audit': False,
                     'benchmark_performance_impact_measured_by_audit': False})
assert len({r['id'] for r in gaps}) == len(gaps)
counts = dict(Counter(row['category'] for row in gaps))
write(BASE / 'gap_register.json', {
    'schema': 'all_author_paper_agreement_register_v2',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'counts': counts, 'rows': gaps,
    'meaning': 'Resolved describes the stated definition or evidence question, not successful runtime, exact publication provenance or benchmark completion.',
})
write(BASE / 'coverage_validation.json', {
    'schema': 'all_author_file_coverage_validation_v2',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'current_file_counts': expected_counts, 'current_file_total': sum(expected_counts.values()),
    'all_current_public_paths_accounted_for': True,
    'current_payload_hash_checks': len(hash_checks),
    'hash_mismatches': [], 'missing_coverage_rows': [],
    'ARGMix_remaining_partial_text_files': am_partial,
    'review_depth_limit': 'File census and exact-byte verification do not imply runtime certification. See per-file review modes. FLARE Java is contract/data-flow review plus critical bodies, locks are structured record review, and thirteen current PNG images have metadata-only review.',
    'checks': hash_checks,
})
write(BASE / 'evidence.json', {
    'schema': 'four_paper_file_agreement_integration_v2',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'contributors': ['ROOT: GST and integration', 'ARGMix reviewer', 'ARGformer/FLARE reviewer', 'AncestralPaths reviewer'],
    'cross_assistance': 'ARGformer and AncestralPaths reviewers finished disjoint remaining ARGMix Rust subsets.',
    'public_current_files': expected_counts,
    'bounded_public_histories_commits': {'argmix': 9, 'argformer': 12, 'ancestralpaths': 27, 'gst': 18},
    'agreement_items': len(gaps), 'agreement_categories': counts,
    'artifact_refs': ['coverage_validation.json', 'gap_register.json', 'static_counterexamples.json'],
    'audit_actions': {'author_runtime_execution': False, 'model_construction': False, 'new_training_or_inference': False,
                     'new_benchmark_scores': False, 'source_pins_changed': False, 'live_jobs_changed': False,
                     'inactive_patch_applicability_checks': True, 'standard_library_arithmetic_checks': True},
    'explicit_remaining_limits': ['Original paper-run identities and some upstream methods/settings remain unknown.',
                                 'Thirteen current PNG assets have hash/header-only review.',
                                 'Historical AncestralPaths tree/VCF assets and checkpoint weights are not fully content-reviewed.',
                                 'The active queue does not implement all original four-paper benchmarks.',
                                 'No whole-repository audit of our own changing implementation is claimed by this author-repository packet.'],
})
report = f'''# File-by-file audit of the four author repositories

Completed 24 October 2026 on anogena. Three reviewers and the primary agent divided ARGMix, ARGformer/FLARE, AncestralPaths and GST. Reviewers subsequently shared the remaining ARGMix Rust files. This packet extends the earlier public-history audit and corrects incomplete or overly broad statements in it.

## Actual coverage

| Repository | Current official files | Review depth |
| --- | ---: | --- |
| ARGMix | 85 | Complete source/configuration reading, with structured dependency-lock review |
| ARGformer | 60 | Complete Python bodies, configuration/prose and structured lock review. One logo has hash/header review |
| AncestralPaths | 17 | Complete text review and visual review of its demographic schematic |
| GST | 166 | Complete text/record scans with scientific route review. Twelve PNGs have hash/header review |
| FLARE v0.6.0 | 105 | All 96 Java class contracts, declarations and option/data flow, plus critical method bodies. All nine other files scanned, including full structured scans of both VCF fixtures |

**All 433 current official files have a coverage record.** Integration compared exact path sets with the retained official trees and rechecked every payload hash. There are no omitted current paths or hash mismatches. This does not mean every Java utility was mathematically verified, every image was visually reviewed or every historical dataset was read.

Reachable public histories remain bounded at 9 ARGMix, 12 ARGformer, 27 AncestralPaths and 18 GST commits. Relevant historical source variants and differences were reviewed. Thirty AncestralPaths historical tree/VCF contents remain outside full review, including one header-only VCF. Its example checkpoint has metadata-only review, without numerical weights or optimizer values. Private and unreachable history are outside this claim.

## Material findings

| Method | Details recovered or verified | Conflicts or missing details |
| --- | --- | --- |
| ARGMix | Low-level 25-reference/three-ancestor components, manual export path, paper sampling budgets and distinct donor/output vocabularies | Current attention selects key i where the equation says key j. Fast raw-linear distance narrows before clipping. Supported full pipeline still blocks the paper ancestor geometry, restricts TRAIN replicate handling and rejects extra admixed donor tokens at the inference CLI. Original roster/map/error mechanism, exact architecture configuration and paper checkpoints remain missing |
| ARGformer | Full historical environment specification recovered. Native path, masking, pooling, objectives, checkpoint and retrieval routes traced through our wrapper | Release temperature, weighting fallback, corruption and extant pooling differ from paper descriptions. Original PCA10 ancestry classifier and several simulation/inference/evaluation details are not released |
| AncestralPaths | Historical CNN topology, exact Keras Flatten order, named 476-person roster and sample/age ordering recovered. Native feature bridge preserves ancient focal self-inclusion | Current Dense model is a different released condition. Native sampling and some train/test splits differ from the supplement. Broken painting interval output and incomplete downstream date aggregation remain. Exact empirical QC/genotype/map manifest and paper-run association are unavailable |
| GST | Synthetic per-model/cell/seed totals and every published BH training-set count explained by the correct source branch. Deleted GSTv2/CIFAR sources recovered | Gate architecture, learning rates, optimizer and uncertainty labels differ. Saved USPTO results are a 1,000-training-reaction pilot. Candidate adapters and full paired original-task experiments remain unfinished |
| FLARE | Official source identity, multiallelic input/MAC rules, ancestry order and native posterior/call handling verified. gen=10 initializes EM rather than fixing generations | Focal-site projection is an explicit study choice. Optional FLARE2 clustering is separate. Toy fixtures contain no ancestry truth and cannot stand in for the American benchmark |

The register contains {len(gaps)} source-backed dispositions. Its categories distinguish resolved definitions, conflicts, explicit study choices, missing upstream detail and unfinished project work. A resolved row does not establish reproduction of a printed paper cell.

## Repairs and decisions made concrete

Two exact inactive ARGMix patches are included: `argmix/attention_key_j.inactive.patch` and `argmix/fast_linear_clamp.inactive.patch`. Both pass `git apply --check` against their exact pinned bases. The attention patch preserves directed psi(i,j) while selecting key j. The linear patch clips before narrowing. Small independent standard-library examples verify those logical effects. No author model was executed, and no accuracy effect was measured.

The historical AncestralPaths CNN has a concrete inactive specification, including its channels-first Conv2D, Keras 2.3.1 permutation before Flatten, six outputs, native features and final-checkpoint training rule. A naive NCHW flatten would permute Dense inputs. Runtime and numerical qualification remain required before calling a new implementation faithful.

Every method has a remediation matrix separating unchanged released controls from proposed equation/configuration corrections, historical compositions and new evaluators. Missing original settings are recorded explicitly. They are not filled by arbitrary defaults. Deterministic fixes do not identify unknown publication-run choices.

The FLARE non-Java supplement closes the earlier four unread test fixtures: 2,000 markers in each VCF, 300 reference samples in three panels and two query samples, all phased/nonmissing, with identical marker order and an exact panel/sample join. The map extends to 300 Mb while observed toy variants stop near 0.613 Mb. The launcher downloads unpinned latest files and was read, never run. It is a smoke example, not an accuracy benchmark.

## Consequence for our comparisons

The existing shared-data queue can still yield useful paired pipeline comparisons under its declared conditions. This audit **does not certify that those conditions reproduce all original paper benchmarks**. In particular, released ARGMix on the AP panel differs from its paper's 1,611-person, chromosome-22, genotype-error and 25+3 scenario. Released Dense AncestralPaths differs from the historical CNN. Native ARGformer retrieval is not the unreleased original PCA10 classifier. GST candidate experiments remain absent from that queue.

Reports must name the released and reconstructed conditions separately, retain shared sample/site/truth joins, disclose information access and training budgets, and keep corrected variants separate from untouched author baselines. No all-method superiority claim or fresh accuracy metric follows from this static review.

No running source, job, policy, frozen data, model or scientific setting was changed. All new files, source acquisitions, checks and Git operations for this audit stay inside the project on anogena. The Mac is only the SSH transport.

## Inspect the evidence

- [ARGMix report](argmix/REPORT.md), [agreement matrix](argmix/agreement_matrix.json), [patch review](argmix/patch_review.md).
- [ARGformer and FLARE report](argformer/REPORT.md), [agreement matrix](argformer/agreement_matrix.json), [FLARE non-Java supplement](argformer/flare_non_java_coverage.json).
- [AncestralPaths report](ancestralpaths/REPORT.md), [agreement matrix](ancestralpaths/agreement_matrix.json), [historical CNN specification](ancestralpaths/historical_CNN_INACTIVE_spec.json).
- [GST report](gst/REPORT.md), [agreement matrix](gst/agreement_matrix.json), [sample-count derivations](gst/count_derivations.json).
- [Unified gap register](gap_register.json), [coverage validation](coverage_validation.json), [logical checks](static_counterexamples.json).

Full acquired sources and verbose source displays remain on NFS2. Compact reports, file coverage, receipts and hashes are versioned. The report deliberately exposes the limits of each review mode instead of equating a file inventory with complete semantic or runtime validation.
'''
(BASE / 'REPORT.md').write_text(report)
print(json.dumps({'current_files': sum(expected_counts.values()), 'hash_checks': len(hash_checks),
                  'agreement_items': len(gaps), 'categories': counts, 'ARGMix_partial': am_partial}, indent=2))
