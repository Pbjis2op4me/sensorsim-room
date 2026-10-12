"""Audio reading/writing for the real-data comparison (no extra Python packages).

WAV (PCM 16/24/32-bit integer, IEEE float 32/64, WAVE_FORMAT_EXTENSIBLE) is parsed here directly so
that a channel/duration subset can be taken from a *stream* (a remote zip member) without holding the
whole file. Lossy formats (MP3 previews, Ogg Vorbis) are decoded with the host's `ffmpeg` binary if
present (it is an existing host tool, not a Python dependency); without ffmpeg those clips are skipped.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TAG_PCM, _TAG_FLOAT, _TAG_EXT = 1, 3, 0xFFFE


@dataclass(frozen=True)
class WavFormat:
    tag: int              # 1 PCM integer, 3 IEEE float
    channels: int
    fs: int
    bits: int
    data_bytes: int       # size of the data chunk (may be 0/huge for streamed writers)
    endian: str = "<"
    valid_bits: int | None = None
    channel_mask: int = 0  # descriptive only: preserve file channel order

    @property
    def block_align(self) -> int:
        return self.channels * self.bits // 8

    @property
    def n_frames(self) -> int:
        return self.data_bytes // self.block_align

    @property
    def sample_format(self) -> str:
        return f"{'float' if self.tag == _TAG_FLOAT else 'int'}{self.bits}"


def _read_exact(fh, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = fh.read(min(size - len(data), 1 << 20))
        if not part:
            raise ValueError("truncated WAV stream")
        data.extend(part)
    return bytes(data)


def _discard(fh, size: int) -> None:
    while size:
        n = min(size, 1 << 20)
        _read_exact(fh, n)
        size -= n


def read_wav_header(fh) -> WavFormat:
    """Parse RIFF/RIFX/RF64, leaving fh at data. Unsupported encodings fail closed.

    PCM container scaling follows Microsoft's WAVEFORMATEXTENSIBLE (valid bits
    are left aligned); RF64 data sizes follow EBU Tech 3306 section 3.4.
    """
    riff = _read_exact(fh, 12)
    if riff[:4] not in (b"RIFF", b"RIFX", b"RF64") or riff[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    endian = ">" if riff[:4] == b"RIFX" else "<"
    data64 = None
    sizes64 = {}
    fmt = None
    while True:
        hdr = _read_exact(fh, 8)
        cid, size = hdr[:4], struct.unpack(endian + "I", hdr[4:])[0]
        if size == 0xffffffff:
            if cid == b"data" and data64 is not None:
                size = data64
            elif sizes64.get(cid):
                size = sizes64[cid].pop(0)
            else:
                raise ValueError("WAV chunk has unknown size (RF64 requires ds64)")
        if cid == b"ds64" and riff[:4] == b"RF64":
            if size < 28:
                raise ValueError("invalid RF64 ds64 chunk")
            _, data64, _, count = struct.unpack("<QQQI", _read_exact(fh, 28))
            if 28 + count * 12 > size:
                raise ValueError("invalid RF64 size table")
            for _ in range(count):
                name, length = struct.unpack("<4sQ", _read_exact(fh, 12))
                sizes64.setdefault(name, []).append(length)
            _discard(fh, size - 28 - count * 12 + (size & 1))
        elif cid == b"fmt ":
            if size < 16:
                raise ValueError("invalid WAV fmt chunk")
            body = _read_exact(fh, min(size, 40))
            _discard(fh, max(0, size - 40) + (size & 1))
            tag, ch, fs, br, ba, bits = struct.unpack(endian + "HHIIHH", body[:16])
            valid, mask = bits, 0
            if tag == _TAG_EXT:
                if size < 40 or struct.unpack(endian + "H", body[16:18])[0] < 22:
                    raise ValueError("invalid WAVEFORMATEXTENSIBLE fmt chunk")
                valid, mask = struct.unpack(endian + "HI", body[18:24])
                tag, guid2, guid3 = struct.unpack(endian + "IHH", body[24:32])
                if (guid2, guid3, body[32:40]) != (0, 0x10, bytes.fromhex("800000aa00389b71")):
                    raise ValueError("unsupported WAV SubFormat GUID")
                valid = valid or bits
            supported = (tag == _TAG_PCM and bits in (8, 16, 24, 32)) or (tag == _TAG_FLOAT and bits in (32, 64))
            if not supported or ch < 1 or fs < 1 or ba != ch * bits // 8 or br != fs * ba:
                raise ValueError("unsupported or inconsistent WAV format")
            if not 0 < valid <= bits or (tag == _TAG_FLOAT and valid != bits):
                raise ValueError("invalid WAV valid bits")
            fmt = (tag, ch, fs, bits, valid, mask)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            if size % (fmt[1] * fmt[3] // 8):
                raise ValueError("WAV data is not a whole number of frames")
            return WavFormat(*fmt[:4], size, endian, *fmt[4:])
        else:
            _discard(fh, size + (size & 1))


def decode_frames(raw: bytes, f: WavFormat) -> np.ndarray:
    """Interleaved sample bytes -> (n, channels) array in the file's own numeric type
    (int16/int32/float32/float64; 24-bit PCM is returned as int32 scaled by 2^8)."""
    if len(raw) % f.block_align:
        raise ValueError("truncated WAV frame")
    n = len(raw) // f.block_align
    if f.tag == _TAG_FLOAT:
        dt = f.endian + {32: "f4", 64: "f8"}[f.bits]
        a = np.frombuffer(raw, dtype=dt).reshape(n, f.channels)
        if not np.isfinite(a).all():
            raise ValueError("nonfinite WAV samples")
        return a
    if f.tag != _TAG_PCM:
        raise ValueError("unsupported WAV encoding")
    if f.bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        if f.endian == ">":
            b = b[:, ::-1]
        v = (b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)
        return v.reshape(n, f.channels)
    dt = f.endian + {8: "u1", 16: "i2", 32: "i4"}[f.bits]
    a = np.frombuffer(raw, dtype=dt).reshape(n, f.channels)
    return (a.astype(np.int16) - 128) * 256 if f.bits == 8 else a


def to_float(a: np.ndarray) -> np.ndarray:
    """Integer PCM -> float64 in [-1, 1) (full scale); float data unchanged (as float64)."""
    if a.dtype == np.uint8:
        return (a.astype(np.float64) - 128) / 128
    if np.issubdtype(a.dtype, np.signedinteger):
        return a.astype(np.float64) / float(-np.iinfo(a.dtype).min)
    return a.astype(np.float64)


def _channels(channels, count):
    if channels is not None and (not len(channels) or any(
            not isinstance(c, (int, np.integer)) or not 1 <= c <= count for c in channels)):
        raise ValueError(f"channels must be nonempty 1-based indices in 1..{count}")
    return slice(None) if channels is None else [c - 1 for c in channels]


def _window(t0, dur):
    if not math.isfinite(t0) or t0 < 0 or (dur is not None and (not math.isfinite(dur) or dur < 0)):
        raise ValueError("audio window start/duration must be finite and nonnegative")


def _frames(fh, f, n, channels):
    selection = _channels(channels, f.channels)
    parts = []
    # Discard unrequested channels per block; a short subset never buffers the full file.
    block = max(1, (1 << 20) // f.block_align)
    for start in range(0, n, block):
        a = decode_frames(_read_exact(fh, min(block, n-start) * f.block_align), f)
        parts.append(np.array(a[:, selection], copy=True))
    if not parts:
        return decode_frames(b"", f)[:, selection]
    return np.ascontiguousarray(np.concatenate(parts))


def read_wav_subset(fh, channels: tuple[int, ...] | None = None, seconds: float | None = None
                    ) -> tuple[np.ndarray, WavFormat]:
    """Read the first `seconds` of `channels` (1-based) from a WAV stream; returns native-type samples."""
    _window(0, seconds)
    f = read_wav_header(fh)
    n = f.n_frames if seconds is None else min(f.n_frames, int(round(seconds * f.fs)))
    return _frames(fh, f, n, channels), f


def write_wav(path: Path, fs: int, a: np.ndarray) -> None:
    """Write int16/int32/float32 samples (n, ch) as PCM/float WAV (scipy writer)."""
    from scipy.io import wavfile
    wavfile.write(str(path), int(fs), np.ascontiguousarray(a))


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: Path) -> dict:
    """fs, channels, duration_s, sample_format of an audio file (WAV natively, others via ffprobe)."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with open(path, "rb") as fh:
            f = read_wav_header(fh)
            start = fh.tell()
            fh.seek(0, 2)
            if fh.tell() - start < f.data_bytes:
                raise ValueError("truncated WAV data chunk")
        return dict(fs=f.fs, channels=f.channels, duration_s=f.n_frames / f.fs, sample_format=f.sample_format,
                    codec="wav")
    if not have_ffmpeg():
        return {}
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels,codec_name,bit_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60, check=True)
    d = json.loads(out.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    return dict(fs=int(s.get("sample_rate", 0)), channels=int(s.get("channels", 0)),
                duration_s=float(d.get("format", {}).get("duration", 0.0)), sample_format="lossy",
                codec=s.get("codec_name"), bit_rate=int(s["bit_rate"]) if s.get("bit_rate") else None)


def read_audio(path: Path, channels: tuple[int, ...] | None = None, t0: float = 0.0,
               dur: float | None = None) -> tuple[np.ndarray, int]:
    """(n, ch) float64 samples and fs. WAV: full-scale-normalised ints / raw floats. Lossy: ffmpeg f32."""
    _window(t0, dur)
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with open(path, "rb") as fh:
            f = read_wav_header(fh)
            i0 = min(f.n_frames, int(round(t0 * f.fs)))
            n = f.n_frames - i0 if dur is None else min(f.n_frames - i0, int(round(dur * f.fs)))
            fh.seek(i0 * f.block_align, 1)
            a = _frames(fh, f, n, channels)
        a = to_float(a)
        return a, f.fs
    if not have_ffmpeg():
        raise RuntimeError(f"ffmpeg needed to decode {path.name}")
    info = probe(path)
    selection = _channels(channels, info["channels"])
    if info["fs"] <= 0 or info["channels"] <= 0:
        raise ValueError("no decodable audio stream")
    if dur == 0:
        return np.empty((0, info["channels"]))[:, selection], info["fs"]
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{t0}", "-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur}"]
    cmd += ["-map", "0:a:0", "-threads", "2", "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=300, check=True).stdout
    a = np.frombuffer(raw, dtype="<f4").reshape(-1, info["channels"]).astype(np.float64)
    return a[:, selection], info["fs"]
