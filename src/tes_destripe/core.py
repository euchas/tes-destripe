"""Latitude-adaptive directional Fourier ("wedge") destriping.

The image is processed in overlapping latitude strips. In each strip the
dominant stripe orientation is estimated from the high-pass power spectrum, and
only a narrow angular wedge of spatial frequencies around that orientation is
attenuated; the lowest frequencies are protected by a guard radius so that
large-scale terrain is untouched. Strips are blended with Hanning weights in
latitude. Columns are treated as periodic, which is seam-free for a global map
in longitude. Nodata pixels are filled by nearest neighbour before the FFT.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULTS = {
    "wedge_sigma_deg": 9.0,   # angular half-width (Gaussian sigma) of the attenuated wedge, degrees
    "guard": 0.05,            # low-frequency guard radius, cycles per pixel
    "atten": 0.95,            # maximum attenuation inside the wedge (0 = none, 1 = remove)
    "strip": 512,             # latitude strip height, pixels
    "step": 256,              # strip step; smaller than `strip` gives overlap
    "pad": 128,               # context rows added above and below each strip
}
ORIENT_R_MIN = 0.04      # annulus (cycles per pixel) used for the orientation estimate
ORIENT_R_MAX = 0.45
ORIENT_HP_SIGMA = 6.0    # Gaussian sigma (pixels) of the high-pass used for the estimate
FLAT_POWER_PER_PIXEL = 1e-9


def fill_nodata_nearest(data, valid=None):
    """Replace invalid pixels by the value of the nearest valid pixel."""
    data = np.asarray(data, dtype=np.float64)
    if valid is None or np.all(valid):
        return data
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        raise ValueError("no valid pixels to fill from")
    ind = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def _frequency_grids(shape):
    H, W = shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    FY = np.broadcast_to(fy, (H, W))
    FX = np.broadcast_to(fx, (H, W))
    R = np.sqrt(FX ** 2 + FY ** 2)
    ANG = np.mod(np.arctan2(FY, FX), np.pi)
    return R, ANG


def estimate_orientation(block, r_min=ORIENT_R_MIN, r_max=ORIENT_R_MAX, hp_sigma=ORIENT_HP_SIGMA):
    """Orientation of the dominant oriented texture in `block`.

    Returns the direction (radians, in [0, pi), measured from the sample axis
    towards the line axis) along which the texture *varies*, i.e. the normal to
    the stripes, at 1 degree resolution. Returns None when the block has no
    high-pass power (for example a constant-fill strip).
    """
    block = np.asarray(block, dtype=np.float64)
    R, ANG = _frequency_grids(block.shape)
    hp = block - ndimage.gaussian_filter(block, hp_sigma)
    power = np.abs(np.fft.fft2(hp)) ** 2
    ann = (R > r_min) & (R < r_max)
    if power[ann].sum() < FLAT_POWER_PER_PIXEL * block.size:
        return None
    by_deg = np.bincount(np.degrees(ANG[ann]).astype(int), weights=power[ann], minlength=180)[:180]
    return float(np.deg2rad(np.argmax(by_deg)))


def tilt_from_normal(normal_rad):
    """Stripe tilt in degrees from vertical, in (-90, 90].

    Positive tilt: the stripe runs from upper-left to lower-right of the array
    (sample index increasing with line index); negative: upper-right to
    lower-left.
    """
    return 90.0 - ((np.degrees(normal_rad) - 90.0) % 180.0)


def wedge_filter(block, wedge_sigma_deg=9.0, guard=0.05, atten=0.95):
    """Attenuate the dominant oriented frequency wedge of a 2-D block.

    Returns ``(filtered, tilt_deg)``; ``tilt_deg`` is None (and the block is
    returned unchanged) when no orientation could be estimated.
    """
    block = np.asarray(block, dtype=np.float64)
    normal = estimate_orientation(block)
    if normal is None:
        return block.copy(), None
    R, ANG = _frequency_grids(block.shape)
    sig = np.deg2rad(wedge_sigma_deg)
    ad = np.abs(np.mod(ANG - normal + np.pi / 2, np.pi) - np.pi / 2)      # angular distance to the normal
    soft = np.exp(-(ad / sig) ** 2) * (1.0 - np.exp(-(R / guard) ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(block) * (1.0 - atten * soft)))
    return out, float(tilt_from_normal(normal))


def destripe(data, valid=None, wedge_sigma_deg=9.0, guard=0.05, atten=0.95,
             strip=512, step=256, pad=128, lat_top=90.0, lat_span=180.0):
    """Latitude-adaptive directional-wedge destripe of a whole map.

    Parameters
    ----------
    data : 2-D array indexed [line, sample] (north at the top for a global map).
    valid : optional boolean mask of valid pixels; invalid pixels are filled by
        nearest neighbour before filtering (the caller restores them afterwards).
    wedge_sigma_deg, guard, atten : see `wedge_filter`.
    strip, step, pad : strip height, strip step (overlap when step < strip) and
        context rows added above and below each strip, all in pixels.
    lat_top, lat_span : latitude of the first line and the latitude span of the
        map, used only to label the per-strip tilts.

    Returns ``(out, tilts)``: the filtered map (float64, same shape) and a list of
    ``(latitude_of_strip_top, tilt_deg_or_None)``.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array")
    if not 0 < step <= strip:
        raise ValueError("step must be between 1 and strip")
    H, W = data.shape
    filled = fill_nodata_nearest(data, valid)
    acc = np.zeros((H, W))
    wacc = np.zeros((H, W))
    tilts = []
    a = 0
    while a < H:
        b = min(a + strip, H)
        keep = b - a
        w0, w1 = max(a - pad, 0), min(b + pad, H)
        out, tilt = wedge_filter(filled[w0:w1], wedge_sigma_deg, guard, atten)
        tilts.append((lat_top - a * lat_span / H, tilt))
        centre = out[a - w0: a - w0 + keep]
        wy = np.maximum(np.hanning(max(keep, 3))[:keep], 1e-3)[:, None]
        acc[a:b] += centre * wy
        wacc[a:b] += wy
        if b == H:
            break
        a += step
    return acc / wacc, tilts


def destripe_map(data, fill=None, clip=True, **params):
    """Destripe a map in one call: fill handling, filtering, clipping, fill restore.

    `fill` is the nodata value (None: all pixels valid). With `clip`, FFT
    overshoot is clipped back to the range of the valid input pixels. Other
    keyword arguments go to `destripe`. Returns ``(out, tilts)`` with ``out`` as
    float32 and nodata pixels set back to `fill`.
    """
    data = np.asarray(data, dtype=np.float64)
    valid = None if fill is None else data != fill
    out, tilts = destripe(data, valid, **params)
    out = out.astype(np.float32)
    if clip:
        src = data if valid is None else data[valid]
        np.clip(out, float(src.min()), float(src.max()), out=out)
    if valid is not None:
        out[~valid] = np.float32(fill)
    return out, tilts
