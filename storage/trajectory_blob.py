"""HEKB vNext: `TrajectoryBlob` v2 file I/O.

SPEC-HEKB-REFACTOR-2026-v2.0 section 2.1 ("Step 1" of the refactor roadmap).
No prior trajectory-persistence code exists in `hekb-vnext` (verified: `git
log`/`store_service.py` only ever write `objects/`, `relations/`,
`lineages/`, `meta/` JSON records; no `.bin`/`.zst` writer exists anywhere in
this repo before this file). New, disclosed infrastructure.

Binary layout (128-byte header + raw float32 payload, zstd-compressed on
disk, + 32-byte tail hash):

    [ Header, 128 bytes, little-endian, UNCOMPRESSED on disk ]
      magic_bytes    6s   b"TRJ_V2"
      format_version H    0x0002
      trajectory_id  32s  SHA-256(raw uncompressed payload bytes)
      dimensions     H    256
      steps          H    128
      dtype          B    0x02 (float32) or 0x03 (int64, HEKBV3-BUILD002, disclosed:
                            already-quantized BoardRecord cell values, element size 8B)
      endianness     B    0x01 (little-endian)
      created_at     Q    unix ms
      reserved       74s  zero padding
    [ Payload, zstd level 3 compressed on disk; 131,072 raw bytes
      (128 steps x 256 dims x 4-byte float32) once decompressed ]
    [ Tail, 32 bytes, UNCOMPRESSED on disk ]
      payload_sha256 32s  SHA-256(header_bytes + raw_uncompressed_payload)

[OBSERVED DIFFERENCE, disclosed]: SPEC-HEKB-REFACTOR-2026-v2.0 section 2.1's
own field list sums to 54 bytes (6+2+32+2+2+1+1+8) and then states "reserved:
69 bytes", which totals 123 bytes -- five short of the header's own stated
128-byte total. This is an internal inconsistency in the spec, not something
this implementation invented. Since "Header Block (128 Bytes)" is the
stronger, structurally load-bearing constraint (matching the fixed-offset
read path implied by the rest of section 2), this implementation uses
reserved=74 bytes to make the header exactly 128 bytes, and flags the
discrepancy here rather than silently picking one number.

[OBSERVED DIFFERENCE, disclosed]: zstd is applied only to the payload block,
not the header/tail, so the on-disk file is `header || zstd(payload) ||
tail` -- not a single zstd stream over the whole file. This is deliberate,
not an oversight: it keeps the 128-byte header (id, dims, steps, timestamp)
readable via a plain byte-offset seek, without decompressing anything, for
faster metadata scans -- a real design tradeoff the spec's own layout diagram
implies (it draws Header / Payload / Tail as separate blocks) but does not
say explicitly. Flagging the choice rather than asserting the spec mandates
it.

zstd support: prefers the Python 3.14+ stdlib `compression.zstd` module
(PEP 784) when available (this repo's own `python3` is 3.14.4). BUT the
experiment harness scripts that call this module (e.g.
`EXP-SensOS001/edsr_task1_cumr_contrastive_geometry.py`) run under
`sensos/.venv`, which is Python 3.12 -- verified directly by attempting the
import there and hitting `ModuleNotFoundError: No module named
'compression'` before adding this fallback. So this module falls back to
the third-party `zstandard` package (installed into `sensos/.venv` via
`pip install zstandard` -- a new, disclosed dependency; see this repo's
`requirements.txt`) when the stdlib module isn't present, through a tiny
`compress(data, level)`/`decompress(data)` shim matching the stdlib API
this module actually uses.
"""
from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from compression import zstd  # Python 3.14+ stdlib (PEP 784)
except ImportError:
    import zstandard as _zstandard

    class _ZstdShim:
        @staticmethod
        def compress(data: bytes, level: int) -> bytes:
            return _zstandard.ZstdCompressor(level=level).compress(data)

        @staticmethod
        def decompress(data: bytes) -> bytes:
            return _zstandard.ZstdDecompressor().decompress(data)

    zstd = _ZstdShim()

MAGIC = b"TRJ_V2"
FORMAT_VERSION = 0x0002
DTYPE_FLOAT32 = 0x02
# [HEKBV3-BUILD002, disclosed]: new dtype code for storing already-quantized BoardRecord
# cell values (nvs_edge/record.py's fixed-point int64 grid, step 2^QUANT_STEP_LOG2) without
# dequantizing first -- ingestion keeps the Edge's own int64 values byte-for-byte; the index
# build step is the one place that calls dequantize(). Element size is 8 bytes (int64), vs
# 4 for float32 -- `_ELEMENT_FORMAT`/`_ELEMENT_SIZE` below branch on this field.
DTYPE_INT64 = 0x03
ENDIANNESS_LITTLE = 0x01
_ELEMENT_FORMAT = {DTYPE_FLOAT32: "f", DTYPE_INT64: "q"}
_ELEMENT_SIZE = {DTYPE_FLOAT32: 4, DTYPE_INT64: 8}
DIMENSIONS = 256          # default/legacy: every MAC001/002/004 dataset inspected is 256-dim.
                          # [HEKBV3-BUILD001, disclosed]: the on-disk header already carries
                          # `dimensions` as a variable uint16 field (not a fixed magic constant),
                          # so this format is dimension-agnostic by construction; only the
                          # encode-side validation below used to hardcode 256. Generalized so
                          # the identical header/zstd/tail format can be reused for NVS's 64-D
                          # Trajectory positions (HEKBv3 改修設計書 v0.1 §3.3 Layer 4) without a
                          # new blob format. `DIMENSIONS` is kept only as the legacy default
                          # for callers that don't pass `dimensions` explicitly.
DEFAULT_STEPS = 128       # the spec's own worked example (128 steps x 256 dims = 32,768 elements / 128KB).
HEADER_SIZE = 128
TAIL_SIZE = 32
RESERVED_SIZE = 74  # see module docstring: spec's own arithmetic is short by 5 bytes; this fills the header to 128B exactly.
ZSTD_LEVEL = 3

# [OBSERVED DIFFERENCE, disclosed]: the spec's own worked example fixes
# "steps = 128" as a constant, matching MAC001/MAC002 (always 128 steps) and
# EXP-BABY-MAC005's Check B acceptance criterion. But the header format
# itself carries a `steps: uint16` FIELD (not a fixed magic constant like
# `dimensions`), and real MAC004 trajectories are NOT uniformly 128 steps
# (observed lengths {128,102,75,108,78} -- see EXP-BABY-MAC005's own data
# inventory). Treating `steps` as fixed would make this format unable to
# store real MAC004 data, which Step 5 ("全軌跡" -- all trajectories) needs
# to cover. So this implementation reads `steps` from the header (and sizes
# the payload accordingly: dimensions * steps * 4 bytes) rather than
# hardcoding it to 128 -- honoring the header's own variable field instead
# of the prose example that assumes every trajectory has the same length.

_HEADER_STRUCT = struct.Struct(f"<6sH32sHHBBQ{RESERVED_SIZE}s")
assert _HEADER_STRUCT.size == HEADER_SIZE, _HEADER_STRUCT.size


class TrajectoryBlobError(ValueError):
    pass


@dataclass
class TrajectoryBlobHeader:
    trajectory_id: bytes       # 32 raw bytes (not hex)
    dimensions: int
    steps: int
    created_at_ms: int
    format_version: int = FORMAT_VERSION
    dtype: int = DTYPE_FLOAT32
    endianness: int = ENDIANNESS_LITTLE

    @property
    def trajectory_id_hex(self) -> str:
        return self.trajectory_id.hex()


def _flatten(points: list[list[float]] | list[list[int]], dimensions: int, dtype: int) -> tuple[bytes, int]:
    """points: steps x dimensions (row-major) -> (raw little-endian bytes, steps).
    `steps` is whatever len(points) is -- not assumed to be 128 (see module docstring)."""
    steps = len(points)
    if steps == 0:
        raise TrajectoryBlobError("trajectory has zero steps")
    flat = []
    for row in points:
        if len(row) != dimensions:
            raise TrajectoryBlobError(f"expected {dimensions} dims per step, got {len(row)}")
        flat.extend(row)
    fmt = _ELEMENT_FORMAT[dtype]
    return struct.pack(f"<{steps * dimensions}{fmt}", *flat), steps


def _unflatten(raw: bytes, steps: int, dimensions: int, dtype: int) -> list[list[float]]:
    element_count = steps * dimensions
    fmt = _ELEMENT_FORMAT[dtype]
    flat = struct.unpack(f"<{element_count}{fmt}", raw)
    return [list(flat[i * dimensions:(i + 1) * dimensions]) for i in range(steps)]


def encode_trajectory_blob(
    points: list[list[float]], created_at_ms: int | None = None, dimensions: int = DIMENSIONS,
    dtype: int = DTYPE_FLOAT32,
) -> tuple[bytes, TrajectoryBlobHeader]:
    """points: N x `dimensions` real trajectory -> (file_bytes, header). Defaults to 256-D
    float32 (legacy MAC00x callers); pass `dimensions=64` for NVS Trajectory positions, etc.
    -- see the generalization note above `DIMENSIONS`. Pass `dtype=DTYPE_INT64` to persist
    already-quantized int64 values (e.g. BoardRecord cells) byte-for-byte, with no cast.
    trajectory_id is derived from the raw on-disk-dtype payload bytes (post-cast for
    float32, as-is for int64, since that IS what's persisted) -- not from the caller's
    original float64 values, matching the spec's own formula 'SHA256(Raw Payload Bytes)'."""
    if dtype not in _ELEMENT_FORMAT:
        raise TrajectoryBlobError(f"unsupported dtype: {dtype}")
    raw_payload, steps = _flatten(points, dimensions, dtype)
    expected_size = steps * dimensions * _ELEMENT_SIZE[dtype]
    if len(raw_payload) != expected_size:
        raise TrajectoryBlobError(f"internal error: payload size {len(raw_payload)} != {expected_size}")
    if steps > 0xFFFF:
        raise TrajectoryBlobError(f"steps={steps} exceeds uint16 range for the header field")
    if dimensions > 0xFFFF:
        raise TrajectoryBlobError(f"dimensions={dimensions} exceeds uint16 range for the header field")

    trajectory_id = hashlib.sha256(raw_payload).digest()
    header = TrajectoryBlobHeader(
        trajectory_id=trajectory_id,
        dimensions=dimensions,
        steps=steps,
        created_at_ms=created_at_ms if created_at_ms is not None else int(time.time() * 1000),
        dtype=dtype,
    )
    header_bytes = _HEADER_STRUCT.pack(
        MAGIC, header.format_version, header.trajectory_id,
        header.dimensions, header.steps, header.dtype, header.endianness,
        header.created_at_ms, b"\x00" * RESERVED_SIZE,
    )
    tail = hashlib.sha256(header_bytes + raw_payload).digest()
    if len(tail) != TAIL_SIZE:
        raise TrajectoryBlobError("internal error: tail hash size mismatch")

    compressed_payload = zstd.compress(raw_payload, level=ZSTD_LEVEL)
    file_bytes = header_bytes + compressed_payload + tail
    return file_bytes, header


def decode_trajectory_blob(file_bytes: bytes) -> tuple[list[list[float]], TrajectoryBlobHeader]:
    if len(file_bytes) < HEADER_SIZE + TAIL_SIZE:
        raise TrajectoryBlobError("file too short to contain header + tail")

    header_bytes = file_bytes[:HEADER_SIZE]
    tail = file_bytes[-TAIL_SIZE:]
    compressed_payload = file_bytes[HEADER_SIZE:-TAIL_SIZE]

    magic, format_version, trajectory_id, dimensions, steps, dtype, endianness, created_at_ms, _reserved = (
        _HEADER_STRUCT.unpack(header_bytes)
    )
    if magic != MAGIC:
        raise TrajectoryBlobError(f"bad magic bytes: {magic!r}")
    if format_version != FORMAT_VERSION:
        raise TrajectoryBlobError(f"unsupported format_version: {format_version}")
    if dtype not in _ELEMENT_FORMAT or endianness != ENDIANNESS_LITTLE:
        raise TrajectoryBlobError(f"unsupported dtype/endianness: {dtype}/{endianness}")
    # dimensions is read from the header's own variable field, not asserted against a global
    # (see the generalization note above `DIMENSIONS`) -- any dimensionality this format's
    # uint16 field can hold decodes correctly.

    expected_size = dimensions * steps * _ELEMENT_SIZE[dtype]
    raw_payload = zstd.decompress(compressed_payload)
    if len(raw_payload) != expected_size:
        raise TrajectoryBlobError(f"decompressed payload size {len(raw_payload)} != expected {expected_size} (steps={steps} from header)")

    recomputed_trajectory_id = hashlib.sha256(raw_payload).digest()
    if recomputed_trajectory_id != trajectory_id:
        raise TrajectoryBlobError("trajectory_id mismatch: payload does not match header (corruption)")

    recomputed_tail = hashlib.sha256(header_bytes + raw_payload).digest()
    if recomputed_tail != tail:
        raise TrajectoryBlobError("tail checksum mismatch: header+payload does not match tail (corruption)")

    header = TrajectoryBlobHeader(
        trajectory_id=trajectory_id, dimensions=dimensions, steps=steps,
        created_at_ms=created_at_ms, format_version=format_version,
        dtype=dtype, endianness=endianness,
    )
    return _unflatten(raw_payload, steps, dimensions, dtype), header


def trajectory_path(trajectories_root: Path, trajectory_id_hex: str) -> Path:
    """Content-addressed 2-level subtree, per SPEC-HEKB-REFACTOR-2026-v2.0
    section 2.2: trajectories/<first 2 hex>/<next 2 hex>/trj_<full hex>.bin.zst"""
    return trajectories_root / trajectory_id_hex[:2] / trajectory_id_hex[2:4] / f"trj_{trajectory_id_hex}.bin.zst"


def write_trajectory(
    trajectories_root: Path, points: list[list[float]], created_at_ms: int | None = None,
    dimensions: int = DIMENSIONS,
) -> TrajectoryBlobHeader:
    file_bytes, header = encode_trajectory_blob(points, created_at_ms=created_at_ms, dimensions=dimensions)
    path = trajectory_path(trajectories_root, header.trajectory_id_hex)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(file_bytes)
    tmp_path.replace(path)
    return header


def read_trajectory(trajectories_root: Path, trajectory_id_hex: str) -> list[list[float]]:
    path = trajectory_path(trajectories_root, trajectory_id_hex)
    if not path.exists():
        raise TrajectoryBlobError(f"no TrajectoryBlob found for trajectory_id={trajectory_id_hex} at {path}")
    points, _header = decode_trajectory_blob(path.read_bytes())
    return points
