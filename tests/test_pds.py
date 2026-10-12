import numpy as np
import pytest

from tes_destripe import pds
from conftest import write_qube


@pytest.fixture
def small_map():
    rng = np.random.default_rng(3)
    data = rng.uniform(0.1, 0.4, size=(24, 48)).astype(np.float32).astype(np.float64)
    data[:3] = -1.0
    data[-3:] = -1.0
    return data


def test_pc_real_round_trip(tmp_path, small_map):
    path = write_qube(tmp_path / "map.pds", small_map)
    data, meta = pds.read_pds_qube(str(path))
    assert data.shape == (24, 48) and data.dtype == np.float64
    np.testing.assert_array_equal(data, small_map)
    assert (meta["samples"], meta["lines"], meta["bands"]) == (48, 24, 1)
    assert meta["ppd"] == pytest.approx(48 / 360)
    assert meta["null"] is None
    assert meta["dtype"] == "float32"


def test_big_endian_integer_with_scaling_and_byte_pointer(tmp_path, small_map):
    base, mult = 0.1, 0.001
    path = write_qube(tmp_path / "map.cub", small_map, item_type="SUN_INTEGER",
                      base=base, mult=mult, null=-32768, pointer="bytes", axis_key="AXES_NAME")
    data, meta = pds.read_pds_qube(str(path))
    np.testing.assert_allclose(data, small_map, atol=mult / 2 + 1e-9)
    assert meta["null"] == -32768.0
    assert meta["dtype"] == ">i2"


def test_label_records_pointer_and_bip_axis_order(tmp_path, small_map):
    path = write_qube(tmp_path / "map.pds", small_map, pointer="label_records",
                      axes=("BAND", "SAMPLE", "LINE"))
    data, _ = pds.read_pds_qube(str(path))
    np.testing.assert_array_equal(data, small_map)


def test_multiband_selects_band(tmp_path, small_map):
    cube = np.stack([small_map, small_map * 2.0, small_map * 3.0])
    path = write_qube(tmp_path / "cube.pds", cube)
    d0, meta = pds.read_pds_qube(str(path))
    d2, _ = pds.read_pds_qube(str(path), band=2)
    assert meta["bands"] == 3
    np.testing.assert_array_equal(d0, small_map)
    np.testing.assert_allclose(d2, small_map * 3.0, rtol=1e-6)
    with pytest.raises(IndexError):
        pds.read_pds_qube(str(path), band=3)


def test_truncated_file_is_rejected(tmp_path, small_map):
    path = write_qube(tmp_path / "map.pds", small_map)
    with open(path, "r+b") as f:
        f.truncate(path.stat().st_size - 100)
    with pytest.raises(ValueError):
        pds.read_pds_qube(str(path))


def test_detect_fill(small_map):
    assert pds.detect_fill(small_map) == -1.0
    assert pds.detect_fill(small_map[3:-3]) is None


def test_dtype_mapping():
    assert pds._np_dtype("PC_REAL", 4) == np.dtype("<f4")
    assert pds._np_dtype("SUN_REAL", 8) == np.dtype(">f8")
    assert pds._np_dtype("LSB_UNSIGNED_INTEGER", 2) == np.dtype("<u2")
    assert pds._np_dtype("MSB_INTEGER", 4) == np.dtype(">i4")
    with pytest.raises(NotImplementedError):
        pds._np_dtype("VAX_REAL", 4)
