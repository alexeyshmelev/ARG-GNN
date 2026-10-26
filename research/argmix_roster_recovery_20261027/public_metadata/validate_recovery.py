"""Validate bounded exported metadata tables against their public source identities."""
from collections import Counter
from pathlib import Path
import csv
import datetime
import hashlib
import json
import re

from read_xlsx import sheets

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OUT = BASE / 'derived'


def read(name):
    with (OUT / name).open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert len(rows) == len({r['sample_id'] for r in rows}), name
    return {r['sample_id']: r for r in rows}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    full = read('all_1664_ancient_source_metadata.tsv')
    qc = read('qc_1492_sample_ids.tsv')
    painted = read('ancient_painted_1015_candidates.tsv')
    intersection = read('ancient_QC_painted_992_candidates.tsv')
    donor = read('reconstructed_CHG_EHG_WHG_Yam_reference_116.tsv')
    farmers = read('farmer_183_candidates_unsplit.tsv')
    others = read('nonreference_ancient_693_candidates.tsv')
    refs = read('ap_reference_list_318.tsv')
    assert [len(x) for x in [full, qc, painted, intersection, donor, farmers, others, refs]] == [1664, 1492, 1015, 992, 116, 183, 693, 318]
    assert set(intersection) == set(qc) & set(painted)
    assert set(intersection) == set(donor) | set(farmers) | set(others)
    assert not set(donor) & set(farmers) and not set(donor) & set(others) and not set(farmers) & set(others)
    assert Counter(r['reconstructed_argmix_reference_cohort'] for r in donor.values()) == {'CHG': 13, 'EHG': 42, 'WHG': 48, 'Yam': 13}
    assert all(r['argmix_final_roster_confirmed'] == 'false' and r['selected_simulation_age'] == '' and r['age_generations'] == '' for r in painted.values())
    assert all(k in refs and r['ap_reference_population'] == refs[k]['ap_reference_population'] for k, r in donor.items())
    source = BASE / 'sources/raw/ancestralpaths_supplementary_tables.xlsx'
    tab = next(sheets(source, {'Table S1.2.1'}))[1]
    raw_refs = {re.sub(r'^([0-9]+)\.0$', r'\1', r['cells']['B']): r for r in tab if r['row'] >= 3 and 'B' in r['cells']}
    assert set(raw_refs) == set(refs)
    assert all(r['ap_reference_population'] == raw_refs[k]['cells']['A'] and r['date_average_raw'] == raw_refs[k]['cells']['G'] for k, r in refs.items())
    modern_path = OUT / 'modern_EUR_788_candidates.tsv'
    with modern_path.open() as f:
        modern = list(csv.DictReader(f, delimiter='\t'))
    assert len(modern) == len({r['source_sample_id'] for r in modern}) == 788
    assert all(r['membership_status'] == 'candidate_only_final_660_selection_unresolved' for r in modern)
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    for entry in manifest:
        p = ROOT / entry['path']
        assert p.stat().st_size == entry['bytes'] and digest(p) == entry['sha256'], entry['path']
    summary = json.loads((OUT / 'recovery_summary.json').read_text())
    assert summary['painted_vs_full_metadata_mean_date_differences'] == 166
    assert summary['reference_table_vs_full_metadata_mean_date_differences'] == 107
    assert summary['all_reference_table_dates_agree_with_painted_repository_dates']
    assert not summary['final_1611_roster_recovered'] and not summary['simulation_roster_generated']
    receipt = {'schema': 'argmix_metadata_recovery_validation_v1', 'validated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True, 'unique_ID_counts_verified': [1664, 1492, 1015, 992, 116, 183, 693, 318, 788], 'source_manifest_files_rehashed': len(manifest), 'source_reference_list_and_dates_verified': True, 'cohort_partition_verified': True, 'unresolved_membership_and_ages_preserved': True, 'final_simulation_roster_is_not_emitted': True, 'no_queue_or_training_changes': True}
    (OUT / 'validation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
