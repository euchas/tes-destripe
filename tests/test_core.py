import numpy as np
import pytest

from tes_destripe import core
from conftest import make_stripes, make_terrain, rms


def test_flat_block_is_returned_unchanged():
    block = np.full((64, 128), 3.0)
    out, tilt = core.wedge_filter(block)
    assert tilt is None
    assert np.array_equal(out, block)
    assert core.estimate_orientation(block) is None


@pytest.mark.parametrize("tilt_deg", [0.0, 12.0, -20.0, 35.0])
def test_tilt_estimate(tilt_deg):
    H, W = 128, 256
    block = make_terrain(H, W) + make_stripes(H, W, tilt_deg, amp=0.03)
    _, tilt = core.wedge_filter(block)
    assert tilt is not None
    assert abs(tilt - tilt_deg) <= 3.0


def test_wedge_filter_removes_stripe_energy():
    H, W = 128, 256
    terrain = make_terrain(H, W)
    stripes = make_stripes(H, W, 12.0)
    out, _ = core.wedge_filter(terrain + stripes)
    # the small synthetic block leaves some leakage outside the wedge and the
    # guarded low frequencies; real maps are much larger and do better
    assert rms(out - terrain) < 0.4 * rms(stripes)


def test_destripe_reduces_stripes_and_blends_strips(striped_scene):
    image, terrain, stripes = striped_scene
    out, tilts = core.destripe(image, None, strip=32, step=16, pad=8)
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))
    assert rms(out - terrain) < 0.35 * rms(stripes)
    assert len(tilts) >= 5
    lats = [lat for lat, _ in tilts]
    assert lats[0] == 90.0 and lats == sorted(lats, reverse=True)
    assert all(t is not None and abs(t - 12.0) <= 4.0 for _, t in tilts)


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_isotropic_terrain_is_mostly_preserved(seed):
    H, W = 96, 192
    terrain = make_terrain(H, W, seed=seed)
    out_iso, _ = core.destripe(terrain, None, strip=48, step=24, pad=12)
    removed_iso = rms(out_iso - terrain)
    assert removed_iso < 0.1 * np.std(terrain)
    striped = terrain + make_stripes(H, W, 12.0)
    out_str, _ = core.destripe(striped, None, strip=48, step=24, pad=12)
    removed_str = rms(out_str - striped)
    assert removed_iso < 0.25 * removed_str


def test_nodata_is_filled_and_restored():
    H, W = 96, 192
    terrain = make_terrain(H, W)
    image = terrain + make_stripes(H, W, 12.0)
    fill = -999.0
    image[:10] = fill
    image[-10:] = fill
    valid = image != fill
    filled = core.fill_nodata_nearest(image, valid)
    assert np.all(np.isfinite(filled)) and not np.any(filled == fill)
    assert np.array_equal(filled[10], image[10]) and np.array_equal(filled[0], image[10])
    out, _ = core.destripe_map(image, fill=fill, strip=32, step=16, pad=8)
    assert out.dtype == np.float32
    assert np.all(out[:10] == fill) and np.all(out[-10:] == fill)
    inner = slice(10, H - 10)
    assert rms(out[inner] - terrain[inner]) < 0.5 * rms(image[inner] - terrain[inner])
    assert out[valid].min() >= image[valid].min() - 1e-6
    assert out[valid].max() <= image[valid].max() + 1e-6


def test_destripe_rejects_bad_arguments():
    with pytest.raises(ValueError):
        core.destripe(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        core.destripe(np.zeros((16, 16)), step=0)
    with pytest.raises(ValueError):
        core.fill_nodata_nearest(np.zeros((4, 4)), np.zeros((4, 4), bool))
