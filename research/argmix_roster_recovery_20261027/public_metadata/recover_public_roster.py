"""Recover source-bound ARGMix roster candidates without inventing missing selections.

Only public_metadata/derived and its report files are written. Existing simulation,
benchmark, queue, and author-source files are read-only inputs. The output is not
an executable 1,611-person simulation roster.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import datetime
import hashlib
import json
import re

from read_xlsx import sheets

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD = ROOT / 'research/reviews/ARGMix_original_input_recovery_20261015_v1'
ALLENTOFT = ROOT / '.runtime/author_sources/public_realdata_metadata_join_v1/springer_MOESM6.xlsx'
QC_IDS = ROOT / '.runtime/author_sources/ancestralpaths_public_realdata_inventory_v1/postglacial_noflag.inds.txt'
WORKBOOK = BASE / 'sources/raw/ancestralpaths_supplementary_tables.xlsx'
OUT = BASE / 'derived'
TARGET_COUNTS = {'Ana': 22, 'CHG': 13, 'EHG': 42, 'WHG': 48, 'NF': 159, 'Yam': 13, 'PostNeo': 654, 'Present': 660}
REF_MAP = {'CHG': 'CHG', 'EHG': 'EHG', 'WHG': 'WHG', 'Yamnaya': 'Yam'}
FARMER_GROUPS = {'FarmerAnatolian', 'FarmerEarly', 'FarmerMiddle', 'FarmerLate'}


def normalized_id(value):
    # XLSX serializes some all-digit individual IDs as numeric values ending in .0.
    # This exact textual alias rule does not alter IDs containing letters or hyphens.
    return re.sub(r'^([0-9]+)\.0$', r'\1', str(value))


def read_tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_tsv(name, rows):
    assert rows, name
    fields = list(rows[0])
    assert all(set(row) == set(fields) for row in rows)
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def equal_number(left, right):
    missing = {'', 'NA', 'NaN', 'nan'}
    if str(left) in missing and str(right) in missing:
        return True
    try:
        return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal('0.00000001')
    except InvalidOperation:
        return left == right


def main():
    OUT.mkdir(exist_ok=True)
    tabs = dict(sheets(WORKBOOK, {'Table S1', 'Table S1.2.1'}))
    all_rows = [r for r in tabs['Table S1'] if r['row'] >= 12 and 'A' in r['cells']]
    allmeta = {normalized_id(r['cells']['A']): r for r in all_rows}
    ref_rows = [r for r in tabs['Table S1.2.1'] if r['row'] >= 3 and 'B' in r['cells']]
    refs = {normalized_id(r['cells']['B']): r for r in ref_rows}
    ancient_rows = read_tsv(OLD / 'upstream_ancient_candidate_lookup.v1.tsv')
    ancient = {normalized_id(r['source_sample_id']): r for r in ancient_rows}
    assert len(all_rows) == len(allmeta) == 1664
    assert len(ref_rows) == len(refs) == 318
    assert len(ancient_rows) == len(ancient) == 1015
    qc = {key for key, row in allmeta.items() if row['cells'].get('R') in {'0', '0.0'}}
    assert len(qc) == 1492
    public_qc = {normalized_id(v.strip()) for v in QC_IDS.read_text().splitlines() if v.strip()}
    assert public_qc == qc, 'Independent public QC ID list differs after documented normalization'
    allentoft_rows = next(sheets(ALLENTOFT))[1]
    allentoft = {normalized_id(r['cells']['A']): r for r in allentoft_rows if r['row'] >= 12 and 'A' in r['cells']}
    assert set(allentoft) == set(allmeta)
    allentoft_qc = {key for key, row in allentoft.items() if row['cells'].get('R') in {'0', '0.0'}}
    assert allentoft_qc == qc
    painted_qc = set(ancient) & qc
    assert len(painted_qc) == 992
    provenance = {'source_file': str(WORKBOOK.relative_to(ROOT)), 'source_sha256': digest(WORKBOOK)}
    full = []
    aliases = []
    date_mismatches = []
    allentoft_date_mismatches = []
    for key, row in allmeta.items():
        cells = row['cells']
        prior = allentoft[key]['cells']
        for column in ['I', 'J', 'K']:
            if not equal_number(cells.get(column, ''), prior.get(column, '')):
                allentoft_date_mismatches.append({'sample_id': key, 'column': column, 'ap_raw': cells.get(column, ''), 'allentoft_raw': prior.get(column, '')})
        if key != cells['A']:
            aliases.append({'source': 'AP Table S1', 'raw_sample_id': cells['A'], 'normalized_sample_id': key})
        if key != prior['A']:
            aliases.append({'source': 'Allentoft supplementary dataset', 'raw_sample_id': prior['A'], 'normalized_sample_id': key})
        full.append({'sample_id': key, 'source_sample_id_raw': cells['A'], 'source_excel_row': row['row'], 'country': cells.get('C', ''), 'region': cells.get('D', ''), 'group_label': cells.get('E', ''), 'date_upper_raw': cells.get('I', ''), 'date_lower_raw': cells.get('J', ''), 'date_average_raw': cells.get('K', ''), 'dating_source': cells.get('L', ''), 'flag_raw': cells.get('R', ''), 'qc_pass': str(key in qc).lower(), 'in_public_ap_painted_candidates': str(key in ancient).lower(), 'ap_reference_population': refs.get(key, {}).get('cells', {}).get('A', ''), 'source_sheet': 'Table S1', **provenance})
    write_tsv('all_1664_ancient_source_metadata.tsv', full)
    write_tsv('qc_1492_sample_ids.tsv', [{'sample_id': key} for key in sorted(qc)])
    write_tsv('numeric_id_aliases.tsv', aliases)
    references = []
    for key, row in refs.items():
        cells = row['cells']
        references.append({'sample_id': key, 'source_sample_id_raw': cells['B'], 'ap_reference_population': cells['A'], 'qc_pass': str(key in qc).lower(), 'in_public_ap_painted_candidates': str(key in ancient).lower(), 'country': cells.get('C', ''), 'group_label': cells.get('D', ''), 'date_average_raw': cells.get('G', ''), 'source_excel_row': row['row'], 'source_sheet': 'Table S1.2.1', **provenance})
    write_tsv('ap_reference_list_318.tsv', references)
    candidates = []
    for key in sorted(ancient):
        observed = ancient[key]
        source = allmeta[key]
        cells = source['cells']
        ref = refs.get(key)
        pop = ref['cells']['A'] if ref else ''
        donor = REF_MAP.get(pop, '') if key in painted_qc else ''
        if donor:
            selection = 'public_reference_QC_painted_intersection_matches_published_cohort_count'
            unresolved = 'author_final_1611_roster_not_released'
        elif key not in qc:
            selection = 'excluded_by_published_Flag0_QC_rule'
            unresolved = ''
        elif pop in FARMER_GROUPS:
            selection = 'farmer_candidate_requires_final_selection_and_ADMIXTURE_assignment'
            unresolved = '183_candidates_180_expected_before_Iceman_and_unreleased_K4_WHG_lt_0p03_split'
        else:
            selection = 'nonreference_ancient_candidate_requires_final_selection'
            unresolved = '693_candidates_654_published_PostNeo'
        for column, old_column in [('I', 'ageHigh_raw'), ('J', 'ageLow_raw'), ('K', 'ageAverage_raw')]:
            if not equal_number(cells.get(column, ''), observed.get(old_column, '')):
                date_mismatches.append({'sample_id': key, 'column': column, 'supplement_raw': cells.get(column, ''), 'painted_metadata_raw': observed.get(old_column, '')})
        candidates.append({'sample_id': key, 'ap_reference_population': pop, 'reconstructed_argmix_reference_cohort': donor, 'date_upper_raw': cells.get('I', ''), 'date_lower_raw': cells.get('J', ''), 'date_average_raw': cells.get('K', ''), 'reference_table_date_average_raw': ref['cells'].get('G', '') if ref else '', 'selected_simulation_age': '', 'painted_age_raw': observed['age_raw'], 'painted_date_upper_raw': observed['ageHigh_raw'], 'painted_date_lower_raw': observed['ageLow_raw'], 'mean_date_agrees_across_sources': str(equal_number(cells.get('K', ''), observed['ageAverage_raw'])).lower(), 'dating_source': cells.get('L', ''), 'country': cells.get('C', ''), 'region': cells.get('D', ''), 'group_label': cells.get('E', ''), 'flag_raw': cells.get('R', ''), 'qc_pass': str(key in qc).lower(), 'argmix_final_roster_confirmed': 'false', 'argmix_Ana_NF_assignment': '', 'age_generations': '', 'selection_status': selection, 'unresolved_reason': unresolved, 'metadata_source_sheet': 'Table S1', 'metadata_source_excel_row': source['row'], 'reference_source_sheet': 'Table S1.2.1' if ref else '', 'reference_source_excel_row': ref['row'] if ref else '', **provenance})
    write_tsv('ancient_painted_1015_candidates.tsv', candidates)
    write_tsv('ancient_QC_painted_992_candidates.tsv', [r for r in candidates if r['qc_pass'] == 'true'])
    resolved = [r for r in candidates if r['reconstructed_argmix_reference_cohort']]
    donor_counts = dict(Counter(r['reconstructed_argmix_reference_cohort'] for r in resolved))
    assert donor_counts == {'CHG': 13, 'EHG': 42, 'WHG': 48, 'Yam': 13}, donor_counts
    write_tsv('reconstructed_CHG_EHG_WHG_Yam_reference_116.tsv', resolved)
    farmers = [r for r in candidates if r['qc_pass'] == 'true' and r['ap_reference_population'] in FARMER_GROUPS]
    other = [r for r in candidates if r['qc_pass'] == 'true' and not r['ap_reference_population']]
    assert len(farmers) == 183 and len(other) == 693
    write_tsv('farmer_183_candidates_unsplit.tsv', farmers)
    write_tsv('nonreference_ancient_693_candidates.tsv', other)
    modern = read_tsv(OLD / 'upstream_modern_EUR_candidate_lookup.v1.tsv')
    original_modern = {r['s']: r for r in read_tsv(OLD / 'gnomad_meta_updated.tsv')}
    panel = BASE / 'sources/1000g_phase3_integrated_call_samples_v3.20130502.ALL.panel'
    phase3_eur = {r['sample'] for r in read_tsv(panel) if r.get('super_pop') == 'EUR'} if panel.exists() else set()
    modern_out = []
    for row in modern:
        raw = original_modern[row['source_s']]
        modern_out.append({**row, 'release_duplicate': raw.get('sample_filters.release_duplicate', ''), 'project_sample_id_raw': raw.get('project_meta.sample_id', ''), 'study_region_raw': raw.get('hgdp_tgp_meta.Study.region', ''), 'metadata_region_raw': raw.get('region', ''), 'in_original_1000G_phase3_EUR_panel': str(row['source_sample_id'] in phase3_eur).lower() if phase3_eur else '', 'sample_time_present_day': '0', 'membership_status': 'candidate_only_final_660_selection_unresolved'})
    assert len(modern_out) == 788
    assert len({r['source_sample_id'] for r in modern_out}) == 788
    write_tsv('modern_EUR_788_candidates.tsv', modern_out)
    modern_diagnostics = {}
    policies = {'all_EUR': modern_out, 'release_true': [r for r in modern_out if r['release'] == 'true'], 'release_true_in_chr22': [r for r in modern_out if r['release'] == 'true' and r['in_released_chr22_header'] == 'true'], 'not_related_in_chr22': [r for r in modern_out if r['release_related'] == 'false' and r['in_released_chr22_header'] == 'true']}
    for name, rows in policies.items():
        modern_diagnostics[name] = {'count': len(rows), 'by_project_population': dict(sorted(Counter(r['project'] + ':' + r['population_updated'] for r in rows).items()))}
    if phase3_eur:
        modern_diagnostics['original_phase3_EUR'] = {'count': len(phase3_eur), 'found_in_metadata': sum(r['source_sample_id'] in phase3_eur for r in modern_out), 'found_in_chr22': sum(r['source_sample_id'] in phase3_eur and r['in_released_chr22_header'] == 'true' for r in modern_out), 'missing_from_EUR_metadata': sorted(phase3_eur - {r['source_sample_id'] for r in modern_out})}
    modern_diagnostics['HGDP_EUR'] = {'all': sum(r['project'] == 'HGDP' for r in modern_out), 'in_chr22': sum(r['project'] == 'HGDP' and r['in_released_chr22_header'] == 'true' for r in modern_out)}
    modern_diagnostics['duplicate_sample_ids'] = []
    modern_diagnostics['policy_counts_are_diagnostics_not_author_selections'] = True
    (OUT / 'modern_selection_diagnostics.json').write_text(json.dumps(modern_diagnostics, indent=2) + '\n')
    iceman = json.loads((OLD / 'iceman_source_metadata.v1.json').read_text())
    iceman['published_argmix_cohort'] = 'NF'
    iceman['point_date_or_generation_age_invented'] = False
    (OUT / 'iceman_public_identity_date.json').write_text(json.dumps(iceman, indent=2) + '\n')
    input_paths = [WORKBOOK, ALLENTOFT, QC_IDS, OLD / 'upstream_ancient_candidate_lookup.v1.tsv', OLD / 'mesoneo__data__Ancestral_paths_new__ancestral_paths_merged_filtered_age.sampleInfo.tsv', OLD / 'upstream_modern_EUR_candidate_lookup.v1.tsv', OLD / 'gnomad_meta_updated.tsv', OLD / 'HGDP1KGP_chr22_released_sample_ids.txt', OLD / 'iceman_source_metadata.v1.json', OLD / 'AP_paper_fulltext.xml', OLD / 'MesoNeo_paper_fulltext.xml']
    if panel.exists(): input_paths.append(panel)
    manifest = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in input_paths]
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = {'schema': 'argmix_public_roster_recovery_20261027_v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'final_1611_roster_recovered': False, 'simulation_roster_generated': False, 'target_counts': TARGET_COUNTS, 'all_ancient_source_rows': len(allmeta), 'independently_verified_QC_rows': len(qc), 'painted_ancient_candidates': len(ancient), 'QC_painted_ancient_candidates': len(painted_qc), 'reconstructed_reference_counts': donor_counts, 'reconstructed_reference_count': len(resolved), 'farmer_candidates_before_Iceman': len(farmers), 'published_expected_farmers_before_Iceman': 180, 'remaining_farmer_exclusions_unknown': 3, 'nonreference_ancient_candidates': len(other), 'published_PostNeo': 654, 'remaining_nonreference_exclusions_unknown': 39, 'total_unresolved_ancient_exclusions': 42, 'farmer_Anatolian_Neolithic_split_recovered': False, 'modern_660_roster_recovered': False, 'age_generations_converted': False, 'numeric_ID_aliases': aliases, 'AP_Allentoft_date_differences': allentoft_date_mismatches, 'AP_painted_metadata_date_differences': date_mismatches, 'source_dates_description': 'Published radiocarbon/context date upper and lower bounds and reported mean, preserved verbatim. Main MesoNeo paper describes calibrated and reservoir-corrected date intervals and midpoints. The ARGMix generation interval and point-date rule remain unverified.', 'scientific_status': 'Partial original metadata recovery with exact public reference identities for four cohorts. No fabricated final membership, farmer ancestry split, ages, or claim to replicate the original real panel.'}
    summary['reference_table_vs_full_metadata_mean_date_differences'] = sum(bool(r['reference_table_date_average_raw']) and not equal_number(r['reference_table_date_average_raw'], r['date_average_raw']) for r in candidates)
    summary['painted_vs_full_metadata_mean_date_differences'] = sum(r['mean_date_agrees_across_sources'] == 'false' for r in candidates)
    summary['all_reference_table_dates_agree_with_painted_repository_dates'] = all(not r['reference_table_date_average_raw'] or equal_number(r['reference_table_date_average_raw'], r['painted_age_raw']) for r in candidates)
    (OUT / 'recovery_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['numeric_ID_aliases', 'AP_painted_metadata_date_differences', 'AP_Allentoft_date_differences']}, indent=2))
    print('Date differences:', len(allentoft_date_mismatches), len(date_mismatches))


if __name__ == '__main__':
    main()
