"""BeatLeader BSOR v1 replay reader (and a writer, used to build test fixtures).

Layout verified against a real replay; see docs/phases/phase-1-data.md §6.1.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x442D3D69

INFO_STRINGS_1 = ("version", "gameVersion", "timestamp", "playerID", "playerName", "platform",
                  "trackingSystem", "hmd", "controller", "hash", "songName", "mapper", "difficulty")
INFO_STRINGS_2 = ("mode", "environment", "modifiers")
INFO_TAIL = (("jumpDistance", "f"), ("leftHanded", "?"), ("height", "f"), ("startTime", "f"),
             ("failTime", "f"), ("speed", "f"))

POSE = ("px", "py", "pz", "qx", "qy", "qz", "qw")
FRAME_DTYPE = np.dtype([("t", "<f4"), ("fps", "<i4")]
                       + [(f"{part}_{c}", "<f4") for part in ("head", "left", "right") for c in POSE])

EVENT_GOOD, EVENT_BAD, EVENT_MISS, EVENT_BOMB = 0, 1, 2, 3
CUT_FORMAT = "<????f3fiff3f3fffff"
CUT_FIELDS = ("speed_ok", "direction_ok", "saber_type_ok", "cut_too_soon", "saber_speed",
              "saber_dir_x", "saber_dir_y", "saber_dir_z", "saber_type", "time_deviation",
              "cut_dir_deviation", "cut_point_x", "cut_point_y", "cut_point_z", "cut_normal_x",
              "cut_normal_y", "cut_normal_z", "cut_distance_to_center", "cut_angle",
              "before_cut_rating", "after_cut_rating")


class BsorError(ValueError):
    pass


@dataclass
class Replay:
    info: dict
    frames: np.ndarray  # structured, FRAME_DTYPE
    notes: list[dict]
    walls: list[tuple[int, float, float, float]]  # wall hits: id, energy, time, spawn time
    heights: list[tuple[float, float]]  # (height, time)
    pauses: list[tuple[int, float]]  # (duration, time)
    saber_offsets: tuple[float, ...] | None = None  # left pos+rot, right pos+rot (14 floats)
    custom_data: dict[str, bytes] = field(default_factory=dict)


def decode_note_id(note_id: int) -> tuple[int, int, int, int, int]:
    """noteID = scoringType*10000 + lineIndex*1000 + lineLayer*100 + colorType*10 + cutDirection."""
    return (note_id // 10000, note_id // 1000 % 10, note_id // 100 % 10, note_id // 10 % 10,
            note_id % 10)


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def unpack(self, fmt: str):
        fmt = fmt if fmt[0] in "<>=!@" else "<" + fmt
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.buf):
            raise BsorError(f"unexpected end of file at byte {self.pos}")
        values = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return values if len(values) > 1 else values[0]

    def string(self) -> str:
        n = self.unpack("i")
        if not 0 <= n <= len(self.buf) - self.pos or n > 10_000:
            raise BsorError(f"bad string length {n} at byte {self.pos - 4}")
        raw = self.buf[self.pos:self.pos + n]
        self.pos += n
        return raw.decode("utf-8", "replace")

    def info_string(self, followed_by_string: bool) -> str:
        """Info-section string, tolerating the old recorder bug.

        Recorders before ~0.6 wrote the length as the UTF-16 unit count (C# `string.Length`)
        but the bytes as UTF-8, so non-ASCII names/titles under-report their size. If the
        byte-length reading leaves an implausible next field, re-read by UTF-16 units.
        """
        start = self.pos
        n = self.unpack("i")
        if 0 <= n <= len(self.buf) - self.pos:
            end = self.pos + n
            if not followed_by_string or self._plausible_string_at(end):
                try:
                    value = self.buf[self.pos:end].decode("utf-8")
                    self.pos = end
                    return value
                except UnicodeDecodeError:
                    pass
        if n < 0:
            raise BsorError(f"bad string length {n} at byte {start}")
        chars, units = [], 0
        text_start = self.pos
        while units < n:
            if self.pos >= len(self.buf):
                raise BsorError(f"string at byte {start} runs past end of file")
            lead = self.buf[self.pos]
            width = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
            ch = self.buf[self.pos:self.pos + width].decode("utf-8", "replace")
            chars.append(ch)
            units += 2 if width == 4 else 1
            self.pos += width
        if followed_by_string and not self._plausible_string_at(self.pos):
            raise BsorError(f"cannot resync string at byte {start} (bytes {text_start}..{self.pos})")
        return "".join(chars)

    def _plausible_string_at(self, pos: int) -> bool:
        if pos + 4 > len(self.buf):
            return False
        n = struct.unpack_from("<i", self.buf, pos)[0]
        if not 0 <= n <= 1000 or pos + 4 + n > len(self.buf):
            return False
        try:
            self.buf[pos + 4:pos + 4 + n].decode("utf-8")
        except UnicodeDecodeError as e:
            # The next string may itself be mis-sized; then only its last character is cut off.
            return e.start >= n - 3
        return True

    def count(self, max_count: int = 10_000_000) -> int:
        n = self.unpack("i")
        if not 0 <= n <= max_count:
            raise BsorError(f"bad count {n} at byte {self.pos - 4}")
        return n

    def section(self, expected: int) -> None:
        got = self.unpack("B")
        if got != expected:
            raise BsorError(f"expected section {expected}, got {got} at byte {self.pos - 1}")


def parse(buf: bytes) -> Replay:
    r = _Reader(buf)
    if r.unpack("I") != MAGIC:
        raise BsorError("not a BSOR file (bad magic)")
    if (version := r.unpack("B")) != 1:
        raise BsorError(f"unsupported BSOR version {version}")

    r.section(0)
    # `difficulty` is followed by the score int and `modifiers` by a float, not a string.
    info: dict = {k: r.info_string(followed_by_string=k != "difficulty") for k in INFO_STRINGS_1}
    info["score"] = r.unpack("i")
    info.update({k: r.info_string(followed_by_string=k != "modifiers") for k in INFO_STRINGS_2})
    info.update({k: r.unpack(fmt) for k, fmt in INFO_TAIL})

    r.section(1)
    n = r.count()
    end = r.pos + n * FRAME_DTYPE.itemsize
    if end > len(buf):
        raise BsorError("truncated frames section")
    frames = np.frombuffer(buf, dtype=FRAME_DTYPE, count=n, offset=r.pos).copy()
    r.pos = end

    r.section(2)
    notes = []
    for _ in range(r.count()):
        note_id, event_time, spawn_time, event_type = r.unpack("iffi")
        note = {"note_id": note_id, "event_time": event_time, "spawn_time": spawn_time,
                "event_type": event_type}
        if event_type in (EVENT_GOOD, EVENT_BAD):
            note.update(zip(CUT_FIELDS, r.unpack(CUT_FORMAT)))
        elif event_type not in (EVENT_MISS, EVENT_BOMB):
            raise BsorError(f"unknown note event type {event_type}")
        notes.append(note)

    r.section(3)
    walls = [r.unpack("ifff") for _ in range(r.count())]
    r.section(4)
    heights = [r.unpack("ff") for _ in range(r.count())]
    r.section(5)
    pauses = [r.unpack("qf") for _ in range(r.count())]

    replay = Replay(info, frames, notes, walls, heights, pauses)
    while r.pos < len(buf):
        sec = r.unpack("B")
        if sec == 6:
            replay.saber_offsets = r.unpack("14f")
        elif sec == 7:
            for _ in range(r.count()):
                key = r.string()
                size = r.count(len(buf))
                replay.custom_data[key] = r.buf[r.pos:r.pos + size]
                r.pos += size
        else:
            break  # unknown trailing section from a newer recorder; ignore the rest
    return replay


def write(replay: Replay, legacy_lengths: bool = False) -> bytes:
    """Serialise a Replay; exists so tests can build small, exact fixtures.

    `legacy_lengths` reproduces the old recorder bug (length = UTF-16 units, bytes = UTF-8).
    """
    out = bytearray(struct.pack("<IB", MAGIC, 1))

    def s(v: str) -> None:
        b = v.encode("utf-8")
        n = len(v.encode("utf-16-le")) // 2 if legacy_lengths else len(b)
        out.extend(struct.pack("<i", n) + b)

    out.append(0)
    for k in INFO_STRINGS_1:
        s(str(replay.info.get(k, "")))
    out.extend(struct.pack("<i", int(replay.info.get("score", 0))))
    for k in INFO_STRINGS_2:
        s(str(replay.info.get(k, "")))
    for k, fmt in INFO_TAIL:
        out.extend(struct.pack("<" + fmt, replay.info.get(k, 0)))

    out.append(1)
    frames = np.asarray(replay.frames, dtype=FRAME_DTYPE)
    out.extend(struct.pack("<i", len(frames)) + frames.tobytes())

    out.append(2)
    out.extend(struct.pack("<i", len(replay.notes)))
    for n in replay.notes:
        out.extend(struct.pack("<iffi", n["note_id"], n["event_time"], n["spawn_time"], n["event_type"]))
        if n["event_type"] in (EVENT_GOOD, EVENT_BAD):
            out.extend(struct.pack(CUT_FORMAT, *(n.get(f, 0) for f in CUT_FIELDS)))

    for sec, rows, fmt in ((3, replay.walls, "<ifff"), (4, replay.heights, "<ff"),
                           (5, replay.pauses, "<qf")):
        out.append(sec)
        out.extend(struct.pack("<i", len(rows)))
        for row in rows:
            out.extend(struct.pack(fmt, *row))
    if replay.saber_offsets is not None:
        out.append(6)
        out.extend(struct.pack("<14f", *replay.saber_offsets))
    out.append(7)
    out.extend(struct.pack("<i", len(replay.custom_data)))
    for k, v in replay.custom_data.items():
        s(k)
        out.extend(struct.pack("<i", len(v)) + v)
    return bytes(out)
