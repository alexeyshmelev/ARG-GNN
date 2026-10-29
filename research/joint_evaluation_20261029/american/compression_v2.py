"""Lossless data compression through installed public C APIs; no executable payloads."""
import ctypes
import ctypes.util

_zstd = ctypes.CDLL(ctypes.util.find_library('zstd'))
_zstd.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
_zstd.ZSTD_compressBound.restype = ctypes.c_size_t
_zstd.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_zstd.ZSTD_compress.restype = ctypes.c_size_t
_zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
_zstd.ZSTD_decompress.restype = ctypes.c_size_t
_zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
_zstd.ZSTD_isError.restype = ctypes.c_uint
_lz4 = ctypes.CDLL(ctypes.util.find_library('lz4'))
_lz4.LZ4_compressBound.argtypes = [ctypes.c_int]
_lz4.LZ4_compressBound.restype = ctypes.c_int
_lz4.LZ4_compress_default.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lz4.LZ4_compress_default.restype = ctypes.c_int
_lz4.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lz4.LZ4_decompress_safe.restype = ctypes.c_int


def compress(data, codec='zstd', level=3):
    if codec == 'zstd':
        capacity = _zstd.ZSTD_compressBound(len(data))
        output = ctypes.create_string_buffer(capacity)
        size = _zstd.ZSTD_compress(output, capacity, data, len(data), level)
        if _zstd.ZSTD_isError(size):
            raise ValueError('zstd compression failure')
    elif codec == 'lz4':
        capacity = _lz4.LZ4_compressBound(len(data))
        output = ctypes.create_string_buffer(capacity)
        size = _lz4.LZ4_compress_default(data, output, len(data), capacity)
        if size <= 0:
            raise ValueError('lz4 compression failure')
    else:
        raise ValueError('Unregistered compression codec')
    return output.raw[:size]


def decompress(data, raw_size, codec='zstd'):
    if not 0 < raw_size <= 128 * 2**20:
        raise ValueError('Bounded prepared-state size')
    output = ctypes.create_string_buffer(raw_size)
    if codec == 'zstd':
        size = _zstd.ZSTD_decompress(output, raw_size, data, len(data))
        if _zstd.ZSTD_isError(size):
            raise ValueError('zstd decompression failure')
    elif codec == 'lz4':
        size = _lz4.LZ4_decompress_safe(data, output, len(data), raw_size)
    else:
        raise ValueError('Unregistered compression codec')
    if size != raw_size:
        raise ValueError('Exact decoded data size required')
    return output.raw[:size]
