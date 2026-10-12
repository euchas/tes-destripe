"""Synthetic data helpers: striped maps and tiny PDS3 QUBE files."""
import numpy as np
import pytest
from scipy import ndimage

RECORD_BYTES = 512
LABEL_RECORDS = 2

_DTYPES = {
    "PC_REAL": "<f4", "IEEE_REAL": ">f4", "SUN_REAL": ">f4",
    "SUN_INTEGER": ">i2", "LSB_INTEGER": "<i2", "LSB_UNSIGNED_INTEGER": "<u2",
}


def make_terrain(H, W, seed=1):
    """Smooth isotropic random field with a few sharper isotropic features (albedo-like values)."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal((H, W)), 8.0)
    field /= field.std()
    y, x = np.mgrid[0:H, 0:W]
    for _ in range(6):
        cy, cx, r = rng.uniform(0, H), rng.uniform(0, W), rng.uniform(3, 8)
        field -= 0.8 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))
    return 0.2 + 0.05 * field


def make_stripes(H, W, tilt_deg, amp=0.02, seed=0):
    """Broadband stripes tilted `tilt_deg` from vertical (positive: upper-left to lower-right)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(float)
    t = np.deg2rad(tilt_deg)
    u = x * np.cos(t) - y * np.sin(t)          # constant along the stripe direction (sin t, cos t)
    s = np.zeros((H, W))
    for f in (0.08, 0.11, 0.15, 0.20, 0.27, 0.33):
        s += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * f * u + rng.uniform(0, 2 * np.pi))
    return amp * s / s.std()


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def write_qube(path, data, item_type="PC_REAL", axis_key="AXIS_NAME",
               axes=("SAMPLE", "LINE", "BAND"), base=0.0, mult=1.0, null=None,
               pointer="record"):
    """Write `data` [line, sample] (or [band, line, sample]) as an attached-label PDS3 QUBE."""
    data = np.asarray(data)
    cube = data[None] if data.ndim == 2 else data
    nb, H, W = cube.shape
    dtype = np.dtype(_DTYPES[item_type])
    stored = (cube - base) / mult
    if dtype.kind in "iu":
        stored = np.round(stored)
    names = ["BAND", "LINE", "SAMPLE"]
    c_order = list(reversed(axes))
    cube_c = np.transpose(stored, [names.index(n) for n in c_order]).astype(dtype)
    items = {"SAMPLE": W, "LINE": H, "BAND": nb}
    core_items = "(" + ",".join(str(items[a]) for a in axes) + ")"
    offset = LABEL_RECORDS * RECORD_BYTES
    lines = [
        "PDS_VERSION_ID = PDS3",
        "RECORD_TYPE = FIXED_LENGTH",
        f"RECORD_BYTES = {RECORD_BYTES}",
    ]
    if pointer == "record":
        lines += [f"LABEL_RECORDS = {LABEL_RECORDS}", f"^QUBE = {LABEL_RECORDS + 1}"]
    elif pointer == "bytes":
        lines += [f"^QUBE = {offset} <BYTES>"]
    elif pointer == "label_records":
        lines += [f"LABEL_RECORDS = {LABEL_RECORDS}"]
    lines += [
        "OBJECT = QUBE",
        "  AXES = 3",
        f"  {axis_key} = (" + ",".join(axes) + ")",
        f"  CORE_ITEMS = {core_items}",
        f"  CORE_ITEM_BYTES = {dtype.itemsize}",
        f"  CORE_ITEM_TYPE = {item_type}",
        f"  CORE_BASE = {base}",
        f"  CORE_MULTIPLIER = {mult}",
    ]
    if null is not None:
        lines.append(f"  CORE_NULL = {null}")
    lines += ["  SUFFIX_ITEMS = (0,0,0)", "END_OBJECT = QUBE", "END"]
    label = ("\r\n".join(lines) + "\r\n").encode("ascii")
    assert len(label) <= offset
    with open(path, "wb") as f:
        f.write(label.ljust(offset, b" "))
        f.write(cube_c.tobytes())
    return path


@pytest.fixture
def striped_scene():
    """(image, terrain, stripes) for a 96 x 192 synthetic map."""
    H, W = 96, 192
    terrain = make_terrain(H, W)
    stripes = make_stripes(H, W, tilt_deg=12.0)
    return terrain + stripes, terrain, stripes
