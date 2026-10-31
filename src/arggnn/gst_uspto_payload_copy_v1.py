"""H022 bounded byte copy of the frozen GST USPTO ZIP, source-only by default.

No row/text parser, chemistry library, author model, scoring or GPU entry exists.
Actual copying requires a fresh exact root authority and reviewed v4 reservation.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import zipfile
import zlib
from . import project_budget_v4 as budget
from . import gst_uspto_archive_inventory_v2 as inventory_core

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE = ROOT / 'src/arggnn/gst_uspto_payload_copy_v1.py'
CONFIG = ROOT / 'configs/gst_uspto_payload_copy_v1.json'
CONFIG_SHA256 = '9c89b25bf07bb4ed049a0884e318215c1c4cea93cd4dc76df9f5ec1a448afd97'
BLOCK = 65536
RESERVATION_BYTES = 16000000
RECEIPT_CAP = 1000000
PROGRESS_CAP = 256000
ENTRY_HEADROOM = 8192


def _require(value, message):
    if not value:
        raise ValueError(message)


def _same(value, expected, name):
    _require(budget.encoded(value) == budget.encoded(expected), 'Exact typed ' + name + ' required')


def _active(value):
    inventory_core._active(value)


def policy_v1():
    cfg = budget.bound(ROOT, {'path': str(CONFIG.relative_to(ROOT)), 'sha256': CONFIG_SHA256}, True)
    _same(cfg['reservation_bytes'], RESERVATION_BYTES, '16MB scope cap')
    _same(cfg['stream_block_bytes'], BLOCK, 'bounded stream block')
    _same(cfg['receipt_max_bytes'], RECEIPT_CAP, 'final receipt cap')
    _same(cfg['progress_max_bytes'], PROGRESS_CAP, 'progress log cap')
    _same(cfg['raw'], inventory_core.policy()['raw'], 'unchanged opaque ZIP identity')
    for name in ('budget_source', 'budget_config', 'inventory_source', 'inventory_config', 'native_protocol'):
        budget.bound(ROOT, cfg[name])
    frozen = budget.bound(ROOT, cfg['inventory'], True)
    _require(frozen['status'] == 'metadata_only_complete' and frozen['archive_sha256'] == cfg['raw']['sha256'] and frozen['archive_bytes'] == cfg['raw']['bytes'], 'Completed pinned raw inventory required')
    _same(frozen['declared_total_uncompressed_bytes'], cfg['payload_bytes'], 'complete payload cap')
    _same(frozen['member_count'], 5, 'all five actual members')
    _require(cfg['payload_bytes'] + RECEIPT_CAP + PROGRESS_CAP + 16 * ENTRY_HEADROOM < RESERVATION_BYTES, 'Payload/receipt/directory headroom must fit cap')
    return cfg, frozen


def source_report_v1():
    cfg, frozen = policy_v1()
    return {'kind': 'gst_uspto_payload_copy_v1_source_report', 'status': 'source_only',
            'implementation': budget.reference(ROOT, SOURCE), 'config': budget.reference(ROOT, CONFIG),
            'raw': cfg['raw'], 'inventory': cfg['inventory'], 'members': frozen['member_count'],
            'payload_bytes': cfg['payload_bytes'], 'reservation_bytes': RESERVATION_BYTES,
            'archive_opened': False, 'member_payload_opened': False, 'dataset_rows_parsed': False,
            'native_model_run': False, 'live_budget_or_lock_access': False}


def admit_v1(record, *, api=budget):
    cfg, frozen = policy_v1(); source_ref = budget.reference(ROOT, SOURCE)
    config_ref = {'path': str(CONFIG.relative_to(ROOT)), 'sha256': CONFIG_SHA256}
    value = budget.bound(ROOT, record, True)
    fixed = {'kind': 'gst_uspto_payload_copy_v1_root_admission', 'authorizer': 'root_agent',
             'active': True, 'payload_copy_only': True, 'dataset_parse_authorized': False,
             'native_model_authorized': False, 'GPU_authorized': False,
             'implementation': source_ref, 'config': config_ref, 'raw': cfg['raw'], 'inventory': cfg['inventory']}
    _require(isinstance(value, dict) and set(value) == set(fixed) | {'created_utc', 'expires_utc', 'registration', 'independent_review', 'output_directory', 'reservation', 'initial_ledger_pins'}, 'Exact narrow payload-copy root authority required')
    for key, expected in fixed.items():
        _same(value[key], expected, 'root authority ' + key)
    _active(value)
    review = budget.bound(ROOT, value['independent_review'], True)
    review_fields = {'kind': 'gst_uspto_payload_copy_v1_independent_review', 'review_role': 'independent_reviewer',
                     'verdict': 'PASS', 'implementation': source_ref, 'config': config_ref,
                     'source_handmade_payload_copy_verified': True, 'actual_archive_opened': False,
                     'actual_member_payload_opened': False, 'fixture_mode': False, 'actual_admission': False}
    for key, expected in review_fields.items():
        _same(review.get(key), expected, 'independent review ' + key)
    registration = budget.bound(ROOT, value['registration'], True)
    registration_fields = {'kind': 'gst_uspto_payload_copy_v1_root_registration', 'authorizer': 'root_agent',
                           'hypothesis_id': 'H022', 'payload_copy_run_registered': True,
                           'implementation': source_ref, 'config': config_ref, 'raw': cfg['raw'],
                           'inventory': cfg['inventory'], 'dataset_parse_authorized': False,
                           'native_model_authorized': False}
    for key, expected in registration_fields.items():
        _same(registration.get(key), expected, 'root registration ' + key)
    _require(isinstance(registration.get('experiment_id'), str) and registration['experiment_id'], 'Registered unfinished H022 execution ID required')
    output = budget.canonical(ROOT, value['output_directory'])
    _require(not os.path.lexists(output) and output.parent.is_dir(), 'Exclusive new destination with existing admitted parent required')
    reservation = budget.bound(ROOT, value['reservation'], True)
    _same(reservation['output'], str(output), 'exact owned destination')
    _same(reservation['reserved_bytes'], RESERVATION_BYTES, 'small exact scope cap')
    _same(reservation['auxiliary_scopes'], [], 'one output scope')
    for ref in (source_ref, config_ref, value['independent_review'], cfg['inventory']):
        _require(any(budget.encoded(ref) == budget.encoded(item) for item in reservation['consumer_bindings']), 'Reservation omitted pinned copy consumer or inventory')
    pins = value['initial_ledger_pins']
    _require(isinstance(pins, dict) and set(pins) == {'legacy_ledger_sha256', 'supplemental_ledger_sha256'} and all(isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x) for x in pins.values()), 'Exact first-read ledger pins required')
    view = api.check_reservation(ROOT, reservation, expected_source_sha256=cfg['budget_source']['sha256'], expected_config_sha256=cfg['budget_config']['sha256'], initial_ledger_pins=pins, boundary=True, required_write_bytes=RESERVATION_BYTES)
    return value, cfg, frozen, reservation, output, view


def _inventory_matches(raw, frozen):
    actual = inventory_core.inventory(raw)
    for key in ('schema', 'status', 'archive_bytes', 'archive_sha256', 'archive_comment_bytes', 'members', 'member_count', 'declared_total_uncompressed_bytes', 'logical_split_member_paths'):
        _same(actual[key], frozen[key], 'frozen inventory ' + key)
    return actual


def _output_fd(output):
    output = budget.canonical(ROOT, output)
    parent_fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in output.parent.relative_to(ROOT).parts:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd); parent_fd = next_fd
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    return output, fd


class CopyScopeV1:
    """One exclusive copy destination; member opens stay relative to no-follow FDs."""
    def __init__(self, output, guard, max_bytes):
        self.output, self.fd = _output_fd(output)
        self.identity = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        self.guard = guard; self.max_bytes = max_bytes; self.used = ENTRY_HEADROOM
        self.progress_bytes = 0
        try:
            self.progress = self.file('progress.jsonl', guarded=False)
        except BaseException:
            os.close(self.fd)
            raise
    def stable(self):
        current = self.output.lstat()
        _require(stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == self.identity and self.output.resolve() == self.output, 'Owned destination changed identity')
    def check(self, planned):
        self.stable(); _require(type(planned) is int and planned >= 0 and self.used + planned <= self.max_bytes, 'Bounded output cap exceeded')
        self.guard(planned)
    def directories(self, parts):
        fd = os.dup(self.fd)
        try:
            for component in parts:
                _require(component not in ('', '.', '..') and '/' not in component and '\\' not in component, 'Validated unchanged member component required')
                try:
                    nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError:
                    self.check(ENTRY_HEADROOM)
                    os.mkdir(component, mode=0o700, dir_fd=fd); self.used += ENTRY_HEADROOM
                    nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = nxt
            return fd
        except BaseException:
            os.close(fd); raise
    def file(self, name, *, guarded=True):
        parts = name.split('/'); parent = self.directories(parts[:-1])
        try:
            if guarded: self.check(ENTRY_HEADROOM)
            fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            self.used += ENTRY_HEADROOM
            return fd
        finally:
            os.close(parent)
    def write(self, fd, payload, *, writer=os.write, update=None):
        self.stable(); _require(self.used + len(payload) <= self.max_bytes, 'Stream output cap exceeded')
        pending = memoryview(payload)
        while pending:
            _require(self.used + len(pending) <= self.max_bytes, 'Remaining stream output cap exceeded')
            wrote = writer(fd, pending)
            _require(type(wrote) is int and 0 < wrote <= len(pending), 'Positive bounded actual write required')
            part = pending[:wrote]; self.used += wrote
            if update is not None: update(part)
            pending = pending[wrote:]
    def event(self, kind, **fields):
        raw = budget.encoded({'event': kind, **fields}) + b'\n'
        _require(self.progress_bytes + len(raw) <= PROGRESS_CAP, 'Bounded progress cap exceeded')
        self.write(self.progress, raw); self.progress_bytes += len(raw)
        os.fsync(self.progress)
    def finish(self, result):
        raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        _require(len(raw) <= RECEIPT_CAP, 'Final payload receipt cap exceeded')
        self.check(len(raw) + ENTRY_HEADROOM)
        fd = self.file('payload_manifest.json', guarded=False)
        try: self.write(fd, raw); os.fsync(fd)
        finally: os.close(fd)
        return hashlib.sha256(raw).hexdigest()
    def close(self):
        os.close(self.progress); os.close(self.fd)


def _member_chunks(raw, archive, item, row):
    """Independently count the full stored/deflated stream, including overrun.

    ZipFile.open validates the local name and overlapping member boundaries but
    its payload reader is not used: declared-size truncation cannot hide bytes.
    """
    with archive.open(item, 'r'):
        pass
    header = struct.unpack_from('<IHHHHHIIIHH', raw, item.header_offset)
    signature, version, flags, method, _, _, crc, compressed, uncompressed, name_len, extra_len = header
    _require(signature == 0x04034B50 and flags == item.flag_bits and method == item.compress_type, 'Local header flags/compression differ from frozen member')
    start = item.header_offset + 30
    name = raw[start:start + name_len].decode('utf8' if flags & 0x800 else 'cp437', errors='strict')
    _same(name, row['name'], 'local member name')
    if not flags & 8:
        _require((crc, compressed, uncompressed) == (item.CRC, item.compress_size, item.file_size), 'Local CRC/size differs from central directory')
    start += name_len + extra_len; end = start + item.compress_size
    _require(0 <= start <= end <= archive.start_dir and end <= len(raw), 'Compressed payload bounds invalid')
    if method == zipfile.ZIP_STORED:
        _require(item.compress_size == item.file_size, 'Stored member sizes differ')
        for offset in range(start, end, BLOCK):
            yield raw[offset:min(offset + BLOCK, end)]
    elif method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        for offset in range(start, end, BLOCK):
            pending = raw[offset:min(offset + BLOCK, end)]
            while pending:
                previous = pending; out = decoder.decompress(pending, BLOCK)
                pending = decoder.unconsumed_tail
                _require(not decoder.unused_data, 'Trailing compressed bytes after deflate end')
                _require(out or pending != previous, 'Deflate decoder made no progress')
                if out: yield out
        while not decoder.eof:
            out = decoder.decompress(b'', BLOCK)
            if not out: break
            yield out
        _require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, 'Incomplete or trailing deflate member')
    else:
        raise ValueError('Only frozen stored/deflated methods are supported')


def copy_payload_v1(raw, frozen, output, *, guard=lambda _: None, max_bytes=RESERVATION_BYTES, payload_writer=os.write, receipt_context=None):
    """Finite bytes-only copy primitive; handmade fixtures supply their own inventory.

    The production CLI binds the actual fixed inventory/admission before calling.
    Failure never removes files, renames members, parses rows or accepts a prefix.
    """
    _require(isinstance(raw, bytes) and len(raw) <= inventory_core.RAW_BYTES, 'Bounded opaque input bytes required')
    actual = _inventory_matches(raw, frozen)
    _require(not any(row['name'].split('/')[0].casefold() in ('progress.jsonl', 'payload_manifest.json') for row in actual['members']), 'Member collides with copy receipt namespace')
    payload_cap = actual['declared_total_uncompressed_bytes']
    _require(payload_cap + RECEIPT_CAP + PROGRESS_CAP + 16 * ENTRY_HEADROOM < max_bytes <= RESERVATION_BYTES, 'Explicit complete payload/write cap required')
    guard(max_bytes)
    scope = CopyScopeV1(output, guard, max_bytes)
    rows = [{**row, 'status': 'unreached', 'payload_bytes_written': 0, 'CRC_payload_verified': False} for row in actual['members']]
    result = {'schema': 'gst_uspto_payload_manifest_v1', 'status': 'running', 'archive_bytes': len(raw),
              'archive_sha256': hashlib.sha256(raw).hexdigest(), 'frozen_inventory_sha256': hashlib.sha256(budget.encoded(frozen)).hexdigest(),
              'member_count': len(rows), 'members': rows, 'logical_split_member_paths': actual['logical_split_member_paths'],
              'declared_total_uncompressed_bytes': payload_cap, 'total_payload_bytes_written': 0,
              'member_payload_opened': False, 'data_extracted': False, 'dataset_rows_parsed': False,
              'native_model_run': False, 'scoring_performed': False, 'original_author_used_snapshot_proven': False,
              'output_directory': str(scope.output), 'maximum_output_bytes': max_bytes, 'stream_block_bytes': BLOCK,
              'execution_bindings': receipt_context, 'complete_payload_copy': False, 'partial_payload_preserved': False, 'dataset_accepted_for_parsing': False}
    current = None
    try:
        scope.event('copy_started', archive_sha256=result['archive_sha256'], member_count=len(rows), payload_bytes=payload_cap)
        with zipfile.ZipFile(io.BytesIO(raw), 'r') as archive:
            for item, current in zip(archive.infolist(), rows):
                expected = current['uncompressed_bytes']; digest = hashlib.sha256(); crc = 0
                scope.check(expected + ENTRY_HEADROOM + RECEIPT_CAP + PROGRESS_CAP)
                current.update(status='opened', payload_prefix_sha256=digest.hexdigest(), payload_prefix_CRC32='00000000'); scope.event('member_opened', archive_order=current['archive_order'], name=current['name'])
                fd = None
                if current['directory']:
                    dfd = scope.directories(current['name'][:-1].split('/')); os.close(dfd)
                else:
                    fd = scope.file(current['name'])
                def written(part):
                    nonlocal crc
                    digest.update(part); crc = zlib.crc32(part, crc)
                    current['payload_bytes_written'] += len(part); result['total_payload_bytes_written'] += len(part)
                    current['payload_prefix_sha256'] = digest.hexdigest(); current['payload_prefix_CRC32'] = f'{crc & 0xffffffff:08x}'
                    scope.event('member_prefix_written', archive_order=current['archive_order'], payload_bytes_written=current['payload_bytes_written'], payload_prefix_sha256=current['payload_prefix_sha256'])
                try:
                    result['member_payload_opened'] = True
                    for chunk in _member_chunks(raw, archive, item, current):
                        _require(current['payload_bytes_written'] + len(chunk) <= expected and result['total_payload_bytes_written'] + len(chunk) <= payload_cap, 'Actual decompressed size exceeds frozen member/total')
                        _require(fd is not None, 'Directory emitted forbidden payload bytes')
                        scope.write(fd, chunk, writer=payload_writer, update=written)
                    _require(current['payload_bytes_written'] == expected, 'Actual decompressed size differs from frozen member')
                    _require(f'{crc & 0xffffffff:08x}' == current['CRC32'], 'Actual member CRC32 differs from frozen inventory')
                    if fd is not None: os.fsync(fd)
                    current.update(status='complete', content_sha256=digest.hexdigest(), CRC32_actual=f'{crc & 0xffffffff:08x}', CRC_payload_verified=True)
                    scope.event('member_complete', archive_order=current['archive_order'], name=current['name'], content_sha256=digest.hexdigest(), payload_bytes_written=current['payload_bytes_written'])
                finally:
                    if fd is not None: os.close(fd)
        _require(result['total_payload_bytes_written'] == payload_cap and all(row['status'] == 'complete' for row in rows), 'Complete byte and member coverage required')
        result.update(status='complete_payload_bytes_verified_no_rows_parsed', data_extracted=True, complete_payload_copy=True, CRC_payload_verified=True, member_content_SHA256_known=True)
        scope.event('copy_complete', total_payload_bytes_written=result['total_payload_bytes_written'])
    except Exception as error:
        if current is not None and current['status'] != 'complete': current['status'] = 'failed_partial'
        result.update(status='failed_partial_preserved', error_type=type(error).__name__, error=str(error)[:2000], error_message_truncated=len(str(error))>2000, CRC_payload_verified=False, member_content_SHA256_known=False, data_extracted=result['total_payload_bytes_written'] > 0, partial_payload_preserved=result['total_payload_bytes_written'] > 0)
        try: scope.event('copy_failed', error_type=type(error).__name__, error=str(error)[:2000], total_payload_bytes_written=result['total_payload_bytes_written'])
        except Exception as log_error: result['failure_event_error'] = type(log_error).__name__ + ': ' + str(log_error)[:1000]
    try:
        receipt_sha = scope.finish(result)
    except Exception as receipt_error:
        result['final_receipt_written'] = False; result['final_receipt_error'] = type(receipt_error).__name__ + ': ' + str(receipt_error)[:1000]
        receipt_sha = None
        if result['status'].startswith('complete'): result['status'] = 'failed_receipt_preserved_payload_not_admitted'
    finally:
        scope.close()
    return result, receipt_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--admission'); parser.add_argument('--expected-admission-sha256')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(source_report_v1(), sort_keys=True)); return 0
    _require(Path.cwd().resolve() == ROOT and args.admission and args.expected_admission_sha256, 'Explicit exact root payload-copy admission required')
    ref = {'path': args.admission, 'sha256': args.expected_admission_sha256}
    admission, cfg, frozen, reservation, output, view = admit_v1(ref)
    def guard(planned):
        budget.bound(ROOT, ref); _active(admission)
        budget.check_reservation(ROOT, reservation, expected_source_sha256=cfg['budget_source']['sha256'], expected_config_sha256=cfg['budget_config']['sha256'], boundary=True, required_write_bytes=planned)
    # Registration, independent source review and fresh root cap precede raw read.
    context = {'admission_ref': ref, 'implementation': admission['implementation'], 'config': admission['config'], 'registration': admission['registration'], 'independent_review': admission['independent_review'], 'inventory_ref': cfg['inventory'], 'initial_storage_admission': view}
    try:
        raw = inventory_core._opaque_bytes()
        result, receipt_sha = copy_payload_v1(raw, frozen, output, guard=guard, receipt_context=context)
    except Exception as error:
        if os.path.lexists(output):
            raise  # Existing partial output is never overwritten or relabeled.
        guard(RESERVATION_BYTES)
        scope = CopyScopeV1(output, guard, RESERVATION_BYTES)
        result = {'schema': 'gst_uspto_payload_manifest_v1', 'status': 'failed_preflight_preserved', 'error_type': type(error).__name__, 'error': str(error)[:2000], 'total_payload_bytes_written': 0, 'member_payload_opened': False, 'complete_payload_copy': False, 'data_extracted': False, 'dataset_rows_parsed': False, 'native_model_run': False, 'execution_bindings': context}
        try:
            scope.event('preflight_failed', error_type=type(error).__name__, error=str(error)[:2000])
            receipt_sha = scope.finish(result)
        finally:
            scope.close()
    print(json.dumps({'status': result['status'], 'output_directory': str(output), 'receipt_sha256': receipt_sha, 'admission_ref': ref, 'initial_storage_admission': view, 'dataset_rows_parsed': False, 'native_model_run': False}, sort_keys=True))
    return 0 if result['status'] == 'complete_payload_bytes_verified_no_rows_parsed' and receipt_sha else 1


if __name__ == '__main__':
    raise SystemExit(main())
