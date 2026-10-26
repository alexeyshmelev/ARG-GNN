"""CPU-only differential qualification against the frozen original reader."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = ROOT / 'research/preprocessing_optimization_20261026/observed'
sys.path.insert(0, str(ROOT))
import numpy as np
import setproctitle
import tskit
from arggnn.benchmarking import european_treeloom as original
from research.preprocessing_optimization_20261026.observed import adapter, parser, reference_reader

setproctitle.setproctitle('AI Auto Research Pipeline Test')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
RESULT = {'cases': {}, 'failures': []}
FIXTURES = HERE / 'fixtures'
FIXTURES.mkdir(exist_ok=True)


def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def outcome(fn, *args, **kwargs):
    try:
        return 'ok', fn(*args, **kwargs)
    except Exception as error:
        return 'error', (type(error).__name__, str(error))


def compare(a, b, name):
    if a[0] != b[0]:
        raise AssertionError((name, a[0], b[0], a[1] if a[0] == 'error' else None, b[1] if b[0] == 'error' else None))
    if a[0] == 'error':
        assert a[1] == b[1], (name, a, b)
    else:
        pairs = zip(a[1], b[1]) if isinstance(a[1], tuple) else [(a[1], b[1])]
        for x, y in pairs:
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), name


def sample_rows(n):
    return [dict(sample_id='s' + str(i // 2), haplotype=i % 2) for i in range(2 * n)]


def header(n):
    return '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT' + ''.join('\ts' + str(i) for i in range(n)) + '\n'


def row(pos=1, ref='A', alt='C', fmt='GT', samples=('0|1',)):
    return '\t'.join(['1', str(pos), '.', ref, alt, '.', 'PASS', '.', fmt, *samples]) + '\n'


def source_contract_test():
    source = (ROOT / 'src/arggnn/benchmarking/european_treeloom.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == 'd06122d90c45e3fb013e5493f62ce87f6c5a54a904a2dea891111c136f718dd8'
    original_loader = source[source.index('def load_relate_observation('):source.index('\n\ndef _sampled_records(')]
    parse_start = original_loader.index('    h = len(rows)')
    parse_end = original_loader.index('    labels, queries, ages = [], [], []', parse_start)
    expected = (original_loader[:parse_start]
                + '    genotype, allele_count = read_observed_calls(\n'
                  '        vcf_path, rows, wanted, ts.num_sites, resource_check=resource_check)\n'
                + original_loader[parse_end:])
    candidate = (HERE / 'adapter.py').read_text()
    actual = candidate[candidate.index('def load_relate_observation('):].rstrip()
    assert actual == expected.rstrip(), 'All nonparser full-loader code must remain exactly original'
    RESULT['cases']['unchanged_loader_source_contract'] = 1


def parser_tests():
    count = 0
    formats = [('GT',), ('GT', 'DP'), ('DP', 'GT'), ('DP', 'GT', 'AD'), ('GT', 'GT'), ('A', 'B', 'GT', 'X')]
    for k in range(1, 5):
        representations = ['.', '-1', '0', '+0', '00', ' 0 ', '\u0660', '-0', '-00', '0_0', '\uff10'] + [str(i) for i in range(1, k)]
        for x, y, fmt in itertools.product(representations, representations, formats):
            gt = x + '|' + y
            values = [gt if f == 'GT' else ('17' if f == 'DP' else '') for f in fmt]
            text = '\t'.join([':'.join(values)] * 3)
            compare(outcome(parser.scalar_calls, text, list(fmt), 3, k),
                    outcome(parser.fast_calls, text, list(fmt), 3, k), f'accepted {gt} {fmt} {k}')
            count += 1
    # Exhaust every one-character ASCII call/phase separator: exact errors too.
    for a, b, separator in itertools.product(['.', '-1', '0', '1', '2', '3', '4', '-2', '', '+', 'x', ' 1', '1 ', '1.0', '99999999999999999999999'],
                                           ['.', '0', '1', '3', '4', '', 'x'], ['|', '/', '||', '']):
        for fmt, prefix, suffix in [(['GT'], '', ''), (['DP', 'GT'], '9:', ''), (['GT', 'DP'], '', ':9')]:
            text = prefix + a + separator + b + suffix
            compare(outcome(parser.scalar_calls, text, fmt, 1, 4),
                    outcome(parser.fast_calls, text, fmt, 1, 4), f'malformed {text} {fmt}')
            count += 1
    rng = random.Random(711)
    for trial in range(6000):
        n = rng.choice([1, 2, 5, 16, 257])
        k = rng.randrange(1, 5)
        fmt = rng.choice(formats)
        columns = []
        for _ in range(n):
            gt = rng.choice(['.', '-1', '+0', '00', *map(str, range(k))]) + '|' + rng.choice(['.', *map(str, range(k))])
            columns.append(':'.join(gt if key == 'GT' else str(rng.randrange(0, 100000)) for key in fmt))
        if trial % 7 == 0:
            columns[rng.randrange(n)] = rng.choice(['0', '', '0|1|2', '0/1', ':', '::', '0|9', 'x|0', '0|1:', '0|0::'])
        text = '\t'.join(columns)
        compare(outcome(parser.scalar_calls, text, list(fmt), n, k),
                outcome(parser.fast_calls, text, list(fmt), n, k), f'fuzz {trial}')
        count += 1
    for gt_index in range(64):
        for n in [1, 9]:
            fmt = ['A'] * gt_index + ['GT', 'B']
            values = ['100'] * gt_index + ['.|3', '']
            text = '\t'.join([':'.join(values)] * n)
            compare(outcome(parser.scalar_calls, text, fmt, n, 4),
                    outcome(parser.fast_calls, text, fmt, n, 4), f'GT index {gt_index} sample count {n}')
            count += 1
    for text in ['0' * 5000 + '|1', '0|1' + ':' * 256, '0|1\x00', '\uff10|\uff11', '-00|0_0']:
        compare(outcome(parser.scalar_calls, text, ['GT'], 1, 4),
                outcome(parser.fast_calls, text, ['GT'], 1, 4), f'extended lexical case {len(text)}')
        count += 1
    RESULT['cases']['genotype_parser_differential'] = count


def vcf_tests():
    path = FIXTURES / 'edge.vcf'
    cases = []
    base = header(2) + row(samples=('0|1', '.|0')) + row(2, alt='C,G,T', samples=('3|2', '1|0'))
    cases.append(('normal', base, sample_rows(2), {1: 0, 2: 1}, 2))
    cases += [(name, value, sample_rows(2), {1: 0, 2: 1}, 2) for name, value in [
        ('CRLF', base.replace('\n', '\r\n')), ('no final newline', base.rstrip('\n')),
        ('empty line', base + '\n'), ('missing header', base.split('\n', 1)[1]),
        ('duplicate position', base + row(2, samples=('0|1', '0|0'))),
        ('out of order unretained', base + row(1, samples=('0|1', '0|0'))),
        ('extra column', base.replace('0|1\t.|0', '0|1\t.|0\t0|0')),
        ('missing column', base.replace('0|1\t.|0', '0|1')),
        ('missing GT', base.replace('\tGT\t', '\tDP\t')),
        ('ambiguous nucleotide', base.replace('\tA\t', '\tN\t')),
        ('duplicate allele', base.replace('\tA\tC\t', '\tA\tA\t')),
        ('too many alleles', base.replace('\tC,G,T\t', '\tC,G,T,A\t')),
        ('invalid position', base.replace('1\t1\t', '1\tx\t')),
        ('zero position', base.replace('1\t1\t', '1\t0\t')),
        ('negative position', base.replace('1\t1\t', '1\t-1\t')),
        ('different samples', base.replace('\ts1\n', '\ts2\n')),
        ('extra header', base + header(2)), ('wrong extra header', base + header(3)),
        ('meta after body', base + '##extra=yes\n'), ('truncated header', '#CHROM\tPOS\n'),
        ('missing retained', header(2) + row(samples=('0|1', '0|0'))),
        ('header only', header(2)), ('empty file', ''),
    ]]
    cases.extend([
        ('reverse native index order', base, sample_rows(2), {1: 1, 2: 0}, 2),
        ('unretained invalid GT ignored', base + row(3, samples=('bad', 'bad')), sample_rows(2), {1: 0, 2: 1}, 2),
        ('unretained wrong column rejected', base + row(3, samples=('bad',)), sample_rows(2), {1: 0, 2: 1}, 2),
        ('monomorphic', header(1) + row(alt='.', samples=('0|.',)), sample_rows(1), {1: 0}, 1),
        ('multiallelic missing', header(1) + row(alt='C,G,T', samples=('3|.',)), sample_rows(1), {1: 0}, 1),
        ('negative one accepted', header(1) + row(samples=('-1|0',)), sample_rows(1), {1: 0}, 1),
        ('no diploid samples', header(0) + row(samples=()), sample_rows(0), {1: 0}, 1),
        ('bad hap order', base, [dict(r, haplotype=1-r['haplotype']) for r in sample_rows(2)], {1: 0, 2: 1}, 2),
        ('odd hap count', base, sample_rows(2)[:-1], {1: 0, 2: 1}, 2),
    ])
    for k in range(1, 5):
        for bases in itertools.permutations(['A', 'C', 'G', 'T'], k):
            for a, b in itertools.product(['.', *map(str, range(k))], repeat=2):
                text = header(1) + row(ref=bases[0], alt=','.join(bases[1:]) or '.', samples=(a+'|'+b,))
                cases.append((f'exhaustive alleles {bases} {a}|{b}', text, sample_rows(1), {1: 0}, 1))
    for n in [1, 2, 63, 64, 65, 4095, 4096, 4097, 8193]:
        text = header(1) + ''.join(row(i + 1) for i in range(n))
        cases.append((f'callback boundary {n}', text, sample_rows(1), {i + 1: i for i in range(n)}, n))
    rng = random.Random(732)
    for trial in range(600):
        n, sites = rng.randint(1, 16), rng.randint(1, 30)
        text = header(n)
        wanted = {}
        for site in range(sites):
            k = rng.randint(1, 4)
            bases = rng.sample(['A', 'C', 'G', 'T'], k)
            text += row(site + 1, bases[0], ','.join(bases[1:]) or '.', samples=[rng.choice(['.', *map(str, range(k))]) + '|' + rng.choice(['.', *map(str, range(k))]) for _ in range(n)])
            if rng.random() > .3:
                wanted[site + 1] = len(wanted)
        if not wanted:
            wanted[1] = 0
        cases.append((f'whole VCF fuzz {trial}', text, sample_rows(n), wanted, len(wanted)))
    for name, text, rows, wanted, sites in cases:
        path.write_bytes(text.encode())
        a_events, b_events = [], []
        a = outcome(reference_reader.read_observed_calls, path, rows, wanted, sites, resource_check=a_events.append)
        b = outcome(parser.read_observed_calls, path, rows, wanted, sites, resource_check=b_events.append)
        compare(a, b, name)
        assert a_events == b_events, name
    RESULT['cases']['whole_VCF_differential'] = len(cases)
    path.unlink()


def full_loader_tests():
    protocol = 'ancestralpaths_paper200mb_native6'
    condition = dict(length_bp=100, haplotypes=8, queries=4)
    fake_implementation = lambda _: SimpleNamespace(CONDITIONS={original.PROTOCOLS[protocol]: condition})
    tables = tskit.TableCollection(80)
    for _ in range(8):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    ancestor = tables.nodes.add_row(time=5)
    for leaf in range(8):
        tables.edges.add_row(0, 80, ancestor, leaf)
    for position in [0, 9, 50, 79]:
        tables.sites.add_row(position, 'A')
    tables.sort()
    tree_path = FIXTURES / 'observed.trees'
    tables.tree_sequence().dump(tree_path)
    rows = [dict(r, sample_index=i, observed_node_id=i,
                 observed_reference_class=None if i < 4 else (i - 4),
                 role='query' if i < 4 else 'pure_source_reference', age_generations=0)
            for i, r in enumerate(sample_rows(4))]
    vcf_path, manifest_path, receipt_path = [FIXTURES / n for n in ['observed.vcf', 'samples.json', 'inference.json']]
    text = header(4) + ''.join(row(p, alt='C,G,T', samples=('0|1', '.|-1', '2|3', '+1|00')) for p in [11, 20, 61, 90])
    vcf_path.write_text(text)
    def receipt_for(current_rows):
        manifest_path.write_text(json.dumps(current_rows))
        return dict(status='post_five_iterations_inferred', sequence_length_bp=100, truth_or_oracle_read=False,
                    input_observed_sha256={'genotypes.vcf': ref(vcf_path)['sha256']},
                    observed_sample_manifest=ref(manifest_path), inferred_tree=ref(tree_path),
                    post_five_iterations=dict(coverage_original_bp0=[10, 90], local_to_original_offset_bp=10,
                    site_join=[dict(native_site_id=i, local_position_0based=p-11, original_position_0based=p-1, VCF_position_1based=p) for i, p in enumerate([11,20,61,90])]))
    tweaks = [None, 'status', 'length', 'truth', 'vcf_hash', 'offset', 'coverage', 'join_order', 'join_coord',
              'node_id', 'sample_order', 'role', 'reference_label', 'query_label', 'age', 'sites_cap', 'metadata_cap']
    for tweak in tweaks:
        current_rows = deepcopy(rows)
        if tweak == 'node_id': current_rows[0]['observed_node_id'] = 1
        if tweak == 'sample_order': current_rows[0]['sample_index'] = 1
        if tweak == 'role': current_rows[0]['role'] = 'invalid'
        if tweak == 'reference_label': current_rows[4]['observed_reference_class'] = 4
        if tweak == 'query_label': current_rows[0]['observed_reference_class'] = 1
        if tweak == 'age': current_rows[0]['age_generations'] = 1
        receipt = receipt_for(current_rows)
        if tweak == 'status': receipt['status'] = 'not_ready'
        if tweak == 'length': receipt['sequence_length_bp'] = 101
        if tweak == 'truth': receipt['truth_or_oracle_read'] = True
        if tweak == 'vcf_hash': receipt['input_observed_sha256']['genotypes.vcf'] = '0' * 64
        if tweak == 'offset': receipt['post_five_iterations']['local_to_original_offset_bp'] = 11
        if tweak == 'coverage': receipt['post_five_iterations']['coverage_original_bp0'] = [10, 91]
        if tweak == 'join_order': receipt['post_five_iterations']['site_join'].reverse()
        if tweak == 'join_coord': receipt['post_five_iterations']['site_join'][0]['VCF_position_1based'] = 12
        receipt_path.write_text(json.dumps(receipt))
        kw = dict(protocol_id=protocol, max_sites=3 if tweak == 'sites_cap' else 100,
                  max_metadata_bytes=1 if tweak == 'metadata_cap' else 2**20)
        a_events, b_events = [], []
        with patch.object(original, '_implementation', fake_implementation), patch.object(adapter, '_implementation', fake_implementation):
            a = outcome(original.load_relate_observation, ref(receipt_path), ref(vcf_path), resource_check=a_events.append, **kw)
            b = outcome(adapter.load_relate_observation, ref(receipt_path), ref(vcf_path), resource_check=b_events.append, **kw)
        assert a[0] == b[0], tweak
        if a[0] == 'error':
            assert a == b, (tweak, a, b)
        else:
            assert a[1].keys() == b[1].keys()
            for key in a[1]:
                x, y = a[1][key], b[1][key]
                if isinstance(x, np.ndarray):
                    assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (tweak, key)
                elif key == 'ts':
                    assert x.tables.equals(y.tables)
                else:
                    assert x == y, (tweak, key)
        assert a_events == b_events, tweak
    RESULT['cases']['full_loader_contract_differential'] = len(tweaks)
    for path in [tree_path, vcf_path, manifest_path, receipt_path]: path.unlink()


def real_benchmark():
    tasks = json.loads((ROOT / 'research/operations_resume_20261026/european_gst/replacement_tasks.json').read_text())
    item = tasks[0]['call']['factory']['kwargs']['heldout_inputs']['ancestralpaths_european_test_000']
    source = ROOT / item['observed_vcf_ref']['path']
    receipt = json.loads((ROOT / item['inference_ref']['path']).read_text())
    rows = json.loads((ROOT / receipt['observed_sample_manifest']['path']).read_text())
    # Copy at most 2,048 complete observed rows, preserving actual sample width.
    path = FIXTURES / 'bounded_real_observed.vcf'
    positions, nbytes = [], 0
    with source.open() as incoming, path.open('w') as outgoing:
        for line in incoming:
            outgoing.write(line)
            nbytes += len(line.encode())
            if not line.startswith('#'):
                positions.append(int(line.split('\t', 2)[1]))
            if len(positions) >= 2048 or nbytes > 256 * 2**20:
                break
    results = {}
    for keep_every in [1, 3]:
        wanted = {pos: i for i, pos in enumerate(positions[::keep_every])}
        pair = []
        for name, fn in [('reference', reference_reader.read_observed_calls), ('vectorized', parser.read_observed_calls)]:
            timings = []
            value = None
            for repeat in range(3):
                start = time.perf_counter()
                value = fn(path, rows, wanted, len(wanted), resource_check=lambda _: None)
                timings.append(time.perf_counter() - start)
            pair.append(value)
            results[f'{name}_retain_every_{keep_every}'] = dict(seconds=timings, median_seconds=float(np.median(timings)))
        compare(('ok', pair[0]), ('ok', pair[1]), 'real observed data')
        results[f'speedup_retain_every_{keep_every}'] = results[f'reference_retain_every_{keep_every}']['median_seconds'] / results[f'vectorized_retain_every_{keep_every}']['median_seconds']
    RESULT['real_observed_benchmark'] = dict(source_ref=item['observed_vcf_ref'],
        sample_manifest_ref=receipt['observed_sample_manifest'], copied_rows=len(positions),
        haplotypes=len(rows), bytes=path.stat().st_size, results=results,
        genotypes_and_allele_counts_byte_identical=True, truth_opened=False,
        qualification_subset_not_scientific_benchmark=True)
    path.unlink()


def main():
    start = time.perf_counter()
    source_contract_test()
    parser_tests()
    vcf_tests()
    full_loader_tests()
    real_benchmark()
    RESULT.update(status='pass', elapsed_seconds=time.perf_counter()-start,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={str(p.relative_to(ROOT)): ref(p)['sha256'] for p in [HERE/'parser.py', HERE/'adapter.py', HERE/'reference_reader.py', ROOT/'src/arggnn/benchmarking/european_treeloom.py']},
                  no_training_started=True, no_native_competitor_modified=True)
    (HERE / 'qualification.json').write_text(json.dumps(RESULT, indent=2) + '\n')
    print(json.dumps(RESULT, indent=2), flush=True)

if __name__ == '__main__':
    main()
