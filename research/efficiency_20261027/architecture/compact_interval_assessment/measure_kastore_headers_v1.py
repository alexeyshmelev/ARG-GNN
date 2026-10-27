"""Bounded table row counts from Kastore metadata, without array payload reads."""
from pathlib import Path
import hashlib
import json
import resource
import struct
import sys
import time

BASE = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = BASE / 'research/efficiency_20261027/architecture/compact_interval_assessment'
TYPE_WIDTHS = (1, 1, 2, 2, 4, 4, 8, 8, 4, 8)
MAGIC = b'\x89KAS\r\n\x1a\n'
MAX_PREFIX = 65536


def inspect_header(relative_path, declared_sha256):
    path = BASE / relative_path
    stat = path.stat()
    with path.open('rb') as stream:
        header = stream.read(64)
        assert len(header) == 64 and header[:8] == MAGIC
        major, minor, count, file_size = struct.unpack_from('<HHIQ', header, 8)
        assert sys.byteorder == 'little' and major == 1
        assert file_size == stat.st_size and 0 < count <= 256
        descriptor_bytes = stream.read(64 * count)
        assert len(descriptor_bytes) == 64 * count
        descriptors = []
        expected_key_start = 64 + 64 * count
        for index in range(count):
            descriptor = descriptor_bytes[64 * index:64 * (index + 1)]
            item_type = descriptor[0]
            key_start, key_length, array_start, array_length = struct.unpack_from('<QQQQ', descriptor, 8)
            assert item_type < len(TYPE_WIDTHS)
            assert key_length > 0 and key_start == expected_key_start
            expected_key_start += key_length
            assert expected_key_start <= MAX_PREFIX
            assert array_start + array_length * TYPE_WIDTHS[item_type] <= file_size
            descriptors.append((item_type, key_start, key_length, array_start, array_length))
        first_array = descriptors[0][3]
        assert expected_key_start <= first_array <= MAX_PREFIX
        key_bytes = stream.read(first_array - (64 + 64 * count))
        prefix = header + descriptor_bytes + key_bytes
        assert len(prefix) == first_array
        items = {}
        expected_array_start = first_array
        for item_type, key_start, key_length, array_start, array_length in descriptors:
            expected_array_start += (-expected_array_start) % 8
            assert array_start == expected_array_start
            expected_array_start = array_start + array_length * TYPE_WIDTHS[item_type]
            key = prefix[key_start:key_start + key_length].decode('ascii')
            assert key not in items
            items[key] = {'type': item_type, 'length': array_length, 'payload_bytes': array_length * TYPE_WIDTHS[item_type]}
        assert expected_array_start == file_size
    row_keys = {'nodes': 'nodes/time', 'edges': 'edges/left', 'sites': 'sites/position', 'mutations': 'mutations/site'}
    return {
        'tree_ref': {'path': relative_path, 'declared_sha256': declared_sha256},
        'full_file_rehashed_in_this_header_check': False,
        'file_size_bytes': file_size,
        'mtime_ns': stat.st_mtime_ns,
        'kastore_version': [major, minor],
        'descriptor_count': count,
        'prefix_bytes_read': len(prefix),
        'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
        'array_payload_read': False,
        'table_rows': {name: items[key]['length'] for name, key in row_keys.items()},
        'descriptors': items,
    }


def main():
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    started = time.monotonic()
    american = inspect_header(
        'experiments/results/benchmark_campaign_20261018/fresh_american/argformer_american_train_000/inference/observed/tsinfer_tsdate_sanitized.trees',
        '2ba88d5f02523c33736611283cae445fbed435593b3ef1768599105c801d3a9b')
    counts_path = OUT / 'TRAIN000_compact_counts_v2.json'
    previous = json.loads(counts_path.read_text())
    for name, count in american['table_rows'].items():
        assert count == previous['counts'][name]
    european = inspect_header(
        'experiments/results/benchmark_campaign_20261018_workflow/release_v8/inherited/AP__ancestralpaths_european_train_000__ap_infer/native/post_size.observed.trees',
        '95b55b36b8f498a0fc0645eb2abc978486c171b41c003b1d6efd176d55f4dee6')
    converter_ref = OUT / 'references/relate_ConvertToTreeSequence.cpp'
    converter_hash = hashlib.sha256(converter_ref.read_bytes()).hexdigest()
    assert converter_hash == '452298c9a5ad60159d9f86dd402f71798e8a88108f2bb217fcb941d28778bfc4'
    sample_count, tree_count = 952, 95625
    expected = {'nodes': sample_count + (sample_count - 1) * tree_count,
                'edges': (2 * sample_count - 2) * tree_count}
    receipt = {
        'schema': 'bounded_kastore_header_comparison_v1',
        'passed': True,
        'american': american,
        'american_count_validation_ref': {'path': str(counts_path.relative_to(BASE)), 'sha256': hashlib.sha256(counts_path.read_bytes()).hexdigest()},
        'european_declared_samples': sample_count,
        'european_declared_trees': tree_count,
        'european': european,
        'european_converter_expected_counts': expected,
        'european_converter_count_equality': {key: european['table_rows'][key] == value for key, value in expected.items()},
        'converter_source_ref': {'path': str(converter_ref.relative_to(BASE)), 'sha256': converter_hash},
        'caveat': 'European tree content was not rehashed; current table row counts are header-only checks against a previously declared artifact. Samples and tree counts come from existing observation evidence, not descriptor inference.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_seconds': time.monotonic() - started,
        'maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'GPU_initialized': False,
        'genotype_payload_or_private_truth_read': False,
    }
    result_path = OUT / 'KASTORE_headers_v1.json'
    with result_path.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: receipt[key] for key in ['passed', 'european_converter_expected_counts', 'european_converter_count_equality', 'wall_seconds', 'maxrss_bytes']}))
    for name, item in [('american', american), ('european', european)]:
        print(json.dumps({'name': name, 'table_rows': item['table_rows'], 'prefix_bytes_read': item['prefix_bytes_read'], 'file_size_bytes': item['file_size_bytes']}))
    print(json.dumps({'path': str(result_path), 'sha256': hashlib.sha256(result_path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
