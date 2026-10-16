"""Small streaming BGZF container boundary; no IO, network, execution or authority."""
import hashlib, struct, zlib
BLOCK_RAW = 65280
EOF_BLOCK = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def block_v1(raw):
    require(isinstance(raw, bytes) and len(raw) <= BLOCK_RAW, "Bounded raw BGZF block")
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    body = compressor.compress(raw) + compressor.flush()
    size = 18 + len(body) + 8
    require(size <= 65536, "BGZF total block at most65536 bytes")
    return (b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00"
            + struct.pack("<H", size - 1) + body
            + struct.pack("<II", zlib.crc32(raw) & 0xffffffff, len(raw)))

class BGZFEncoderV1:
    """emit(bytes) must guard and persist every compressed block before returning."""
    def __init__(self, emit):
        require(callable(emit), "Root-owned guarded compressed-byte emitter")
        self.emit, self.pending, self.closed = emit, bytearray(), False
        self.logical_bytes = self.compressed_bytes = 0
        self.logical_sha, self.compressed_sha = hashlib.sha256(), hashlib.sha256()
    def _emit(self, raw):
        self.emit(raw)
        self.compressed_bytes += len(raw)
        self.compressed_sha.update(raw)
    def append(self, raw):
        require(not self.closed and isinstance(raw, bytes), "Open byte-only BGZF stream")
        self.logical_bytes += len(raw)
        self.logical_sha.update(raw)
        self.pending.extend(raw)
        while len(self.pending) >= BLOCK_RAW:
            self._emit(block_v1(bytes(self.pending[:BLOCK_RAW])))
            del self.pending[:BLOCK_RAW]
    def close(self):
        require(not self.closed, "BGZF stream closes exactly once")
        if self.pending:
            self._emit(block_v1(bytes(self.pending)))
            self.pending.clear()
        self._emit(EOF_BLOCK)
        self.closed = True
        return dict(encoding="BGZF", compressed_bytes=self.compressed_bytes,
                    compressed_sha256=self.compressed_sha.hexdigest(),
                    logical_bytes=self.logical_bytes,
                    logical_sha256=self.logical_sha.hexdigest())
