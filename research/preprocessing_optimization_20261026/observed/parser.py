"""Vectorized ASCII genotype parsing with an exact scalar fallback."""
from __future__ import annotations
import numpy as np
from arggnn.benchmarking.european_treeloom import _need


def scalar_calls(sample_text, format_names, n_samples, allele_count):
    """The original loop, including int spellings, error order and exceptions."""
    gt_index = format_names.index('GT')
    calls = []
    for sample in sample_text.split('\t'):
        gt = sample.split(':')[gt_index]
        _need('|' in gt, 'Observed haplotypes must remain phased')
        pair = gt.split('|')
        _need(len(pair) == 2, 'Two actual haplotypes per VCF sample required')
        for call in pair:
            value = -1 if call == '.' else int(call)
            _need(-1 <= value < allele_count, 'Observed nucleotide genotype index required')
            calls.append(value)
    return np.asarray(calls, dtype=np.int8)


def fast_calls(sample_text, format_names, n_samples, allele_count):
    """Return exact int8 calls, delegating all uncommon or invalid forms.

    The fast path only accepts a one-byte nucleotide index (0..3) or missing dot
    on each side of a single phase separator. Python int representations such
    as +1, 01, -1, whitespace or Unicode digits retain the original fallback.
    FORMAT may put GT anywhere and may have variable-length other values.
    """
    gt_index = format_names.index('GT')
    try:
        raw_bytes = sample_text.encode('ascii')
    except UnicodeEncodeError:
        return scalar_calls(sample_text, format_names, n_samples, allele_count)
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    if format_names == ['GT'] and len(raw) == 4 * n_samples - 1:
        rectangle = np.frombuffer(raw_bytes + b'\t', dtype=np.uint8).reshape(n_samples, 4)
        if (rectangle[:, 1] == 124).all() and (rectangle[:, 3] == 9).all():
            calls = rectangle[:, (0, 2)].reshape(-1)
        else:
            return scalar_calls(sample_text, format_names, n_samples, allele_count)
    else:
        separators = np.flatnonzero((raw == 9) | (raw == 58))
        sample_tokens = np.r_[0, np.flatnonzero(raw[separators] == 9) + 1]
        if len(sample_tokens) != n_samples:
            return scalar_calls(sample_text, format_names, n_samples, allele_count)
        sample_ends = np.r_[sample_tokens[1:], len(separators) + 1]
        gt_tokens = sample_tokens + gt_index
        if (gt_tokens >= sample_ends).any():
            return scalar_calls(sample_text, format_names, n_samples, allele_count)
        starts = np.r_[0, separators + 1][gt_tokens]
        ends = np.r_[separators, len(raw)][gt_tokens]
        if not ((ends - starts == 3).all() and (raw[starts + 1] == 124).all()):
            return scalar_calls(sample_text, format_names, n_samples, allele_count)
        calls = np.stack((raw[starts], raw[starts + 2]), axis=1).reshape(-1)
    if not (((calls >= 48) & (calls < 48 + allele_count)) | (calls == 46)).all():
        return scalar_calls(sample_text, format_names, n_samples, allele_count)
    result = calls.astype(np.int8) - 48
    result[calls == 46] = -1
    return result


def read_observed_calls(vcf_path, rows, wanted, n_sites, *, resource_check):
    """Preserve original VCF/sample/site validation and callback boundaries."""
    h = len(rows)
    genotype = np.empty((n_sites, h), dtype=np.int8)
    allele_count = np.empty(n_sites, dtype=np.int8)
    seen, header, previous = set(), None, 0
    with vcf_path.open() as stream:
        for line in stream:
            if line.startswith('##'):
                continue
            fields = line.rstrip('\n').split('\t', 9)
            if line.startswith('#CHROM'):
                header = fields[9].split('\t') if len(fields) == 10 else []
                _need(h % 2 == 0 and header == [r['sample_id'] for r in rows[::2]]
                      and all(rows[i]['sample_id'] == rows[i+1]['sample_id']
                              and [rows[i]['haplotype'], rows[i+1]['haplotype']] == [0, 1]
                              for i in range(0, h, 2)), 'Original phased diploid sample columns required')
                continue
            # Count columns before parsing coordinates, exactly as the original.
            n_fields = len(fields) + (fields[9].count('\t') if len(fields) == 10 else 0)
            _need(header is not None and n_fields == len(header) + 9,
                  'Complete original VCF rows required')
            position = int(fields[1])
            _need(position > previous, 'Original observed VCF positions must remain ordered')
            previous = position
            index = wanted.get(position)
            if index is None:
                continue
            if len(seen) % 4096 == 0:
                resource_check('European_TreeLoom_complete_VCF_join')
            alleles = [fields[3]] + ([] if fields[4] == '.' else fields[4].split(','))
            _need(0 < len(alleles) <= 4 and all(a in {'A', 'C', 'G', 'T'} for a in alleles)
                  and len(set(alleles)) == len(alleles), 'Original nucleotide allele order required')
            format_names = fields[8].split(':')
            _need('GT' in format_names, 'Actual observed GT calls required')
            genotype[index] = (fast_calls(fields[9], format_names, len(header), len(alleles))
                               if header else np.empty(0, dtype=np.int8))
            allele_count[index] = len(alleles)
            seen.add(index)
    _need(header is not None and len(seen) == n_sites,
          'Every native SNP requires original observed calls')
    return genotype, allele_count
