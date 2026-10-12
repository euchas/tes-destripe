"""Minimal reader for attached-label PDS3 / ISIS2 QUBE files.

Covers the single-band global map products distributed as PDS3 QUBE objects
(for example the TES global maps): the label is parsed for the record size, the
QUBE pointer, the core dimensions, axis order, item type and scaling, and the
core is read with numpy. Detached labels, suffix planes and VAX floating-point
cores are not supported.
"""
from __future__ import annotations

import os
import re

import numpy as np

LABEL_MAX_BYTES = 65536
DEFAULT_AXES = ("SAMPLE", "LINE", "BAND")   # ISIS2 convention: first axis varies fastest


def _np_dtype(item_type: str, nbytes: int) -> np.dtype:
    """Map a PDS CORE_ITEM_TYPE and CORE_ITEM_BYTES to a numpy dtype."""
    t = item_type.upper().strip().strip('"')
    if "VAX" in t:
        raise NotImplementedError(f"VAX core item type {t!r} is not supported")
    endian = "<" if ("PC" in t or "LSB" in t) else ">"
    if "REAL" in t or "FLOAT" in t:
        return np.dtype(f"{endian}f{nbytes}")
    if "UNSIGNED" in t:
        return np.dtype(f"{endian}u{nbytes}")
    return np.dtype(f"{endian}i{nbytes}")


def read_label_text(path: str) -> str:
    """Return the attached label (up to the first END statement) as text."""
    with open(path, "rb") as f:
        head = f.read(LABEL_MAX_BYTES)
    m = re.search(rb"(?m)^\s*END\s*$", head)
    label = head[: m.start()] if m else head
    return label.decode("latin-1", "ignore")


def label_value(text: str, key: str, default=None):
    """Value of `key = value` in a label (first occurrence), or `default`."""
    m = re.search(r"(?m)^\s*" + re.escape(key) + r"\s*=\s*([^\r\n]+)", text)
    return m.group(1).strip() if m else default


def _core_offset(text: str, rec_bytes: int) -> int:
    ptr = label_value(text, "^QUBE")
    if ptr is not None:
        p = ptr.strip().strip("()").strip()
        if p.isdigit():                       # record number, 1-based
            return (int(p) - 1) * rec_bytes
        if "<BYTES>" in p.upper():            # byte offset
            return int(re.findall(r"\d+", p)[0])
        raise ValueError(f"unsupported ^QUBE pointer {ptr!r} (detached labels are not supported)")
    lbl_recs = label_value(text, "LABEL_RECORDS")
    if lbl_recs is None:
        raise ValueError("label has neither ^QUBE nor LABEL_RECORDS")
    return int(lbl_recs) * rec_bytes


def read_pds_qube(path: str, band: int = 0):
    """Read one band of an attached-label PDS3 QUBE.

    Returns ``(data, meta)`` where ``data`` is a float64 array indexed
    ``[line, sample]`` (CORE_BASE and CORE_MULTIPLIER applied) and ``meta`` holds
    ``samples``, ``lines``, ``bands``, ``ppd`` (samples / 360), ``null`` (the
    CORE_NULL value, or None), ``dtype`` and ``offset``.
    """
    text = read_label_text(path)
    rec_bytes = int(label_value(text, "RECORD_BYTES", "512"))
    offset = _core_offset(text, rec_bytes)

    items_txt = label_value(text, "CORE_ITEMS")
    if items_txt is None:
        raise ValueError("label has no CORE_ITEMS")
    items = [int(x) for x in re.findall(r"-?\d+", items_txt)]
    axes_txt = label_value(text, "AXIS_NAME") or label_value(text, "AXES_NAME")
    axes = (tuple(a.strip().strip('"').upper() for a in axes_txt.strip("()").split(","))
            if axes_txt else DEFAULT_AXES[: len(items)])
    if len(axes) != len(items) or "SAMPLE" not in axes or "LINE" not in axes:
        raise ValueError(f"cannot interpret AXIS_NAME {axes} with CORE_ITEMS {items}")
    dim = dict(zip(axes, items))
    suffix = label_value(text, "SUFFIX_ITEMS")
    if suffix and any(int(x) for x in re.findall(r"-?\d+", suffix)):
        raise NotImplementedError("QUBEs with suffix planes are not supported")

    nbytes = int(label_value(text, "CORE_ITEM_BYTES", "4"))
    dtype = _np_dtype(label_value(text, "CORE_ITEM_TYPE", "PC_REAL"), nbytes)
    base = float(label_value(text, "CORE_BASE", "0") or 0)
    mult = float(label_value(text, "CORE_MULTIPLIER", "1") or 1)
    null_txt = label_value(text, "CORE_NULL")
    null = float(null_txt) if null_txt not in (None, "", "N/A") else None

    shape_c = tuple(dim[a] for a in reversed(axes))       # C order: last label axis slowest
    count = int(np.prod(shape_c))
    if offset + count * nbytes > os.path.getsize(path):
        raise ValueError("file is shorter than the core described by its label")
    cube = np.fromfile(path, dtype=dtype, count=count, offset=offset).reshape(shape_c)
    names_c = list(reversed(axes))
    want = [n for n in ("BAND", "LINE", "SAMPLE") if n in names_c]
    cube = np.transpose(cube, [names_c.index(n) for n in want])
    if "BAND" not in names_c:
        cube = cube[None]
    nbands = cube.shape[0]
    if not 0 <= band < nbands:
        raise IndexError(f"band {band} out of range for {nbands} band(s)")
    data = cube[band].astype(np.float64)
    if base != 0.0 or mult != 1.0:
        data = base + mult * data
    meta = dict(samples=dim["SAMPLE"], lines=dim["LINE"], bands=nbands,
                ppd=dim["SAMPLE"] / 360.0, null=null, dtype=str(dtype), offset=offset)
    return data, meta


def detect_fill(data: np.ndarray, min_fraction: float = 0.9):
    """Guess the constant fill (nodata) value of a global map.

    Global TES maps carry constant-fill polar caps. If more than `min_fraction`
    of the first or the last row equals the corner value, that value is taken as
    the fill; otherwise None is returned.
    """
    cand = data[0, 0]
    if np.mean(data[0, :] == cand) > min_fraction or np.mean(data[-1, :] == cand) > min_fraction:
        return float(cand)
    return None
