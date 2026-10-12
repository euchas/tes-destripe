import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from tes_destripe import cli, geotiff
from conftest import make_stripes, make_terrain, rms, write_qube


def test_default_mars_grid_matches_usgs_mosaic():
    crs, transform = geotiff.default_mars_georef(2880, 1440)
    assert transform.a == pytest.approx(7410.0) and transform.e == pytest.approx(-7410.0)
    assert (transform.c, transform.f) == (-geotiff.MARS_X_HALF, geotiff.MARS_Y_HALF)
    wkt = crs.to_wkt()
    assert "3396000" in wkt and "Equirectangular" in wkt
    crs180, _ = geotiff.default_mars_georef(720, 360, central_meridian=180.0)
    assert "180" in crs180.to_wkt()


def test_write_and_read_geotiff_round_trip(tmp_path):
    arr = make_terrain(40, 80).astype(np.float32)
    arr[:4] = -1.0
    crs, transform = geotiff.default_mars_georef(80, 40)
    path = tmp_path / "out.tif"
    geotiff.write_geotiff(path, arr, crs, transform, nodata=-1.0, tags={"A": 1.5}, overviews=True)
    data, crs2, transform2, nodata = geotiff.read_raster(path)
    np.testing.assert_array_equal(data, arr)
    assert nodata == -1.0 and transform2 == transform and crs2 is not None
    with rasterio.open(path) as ds:
        assert ds.tags(1)["A"] == "1.5"
    crs3, transform3 = geotiff.georef_from_reference(path, 160, 80)
    assert transform3.a == pytest.approx(transform.a / 2)
    assert crs3 == crs2


def test_cli_end_to_end(tmp_path):
    H, W = 96, 192
    terrain = make_terrain(H, W)
    image = terrain + make_stripes(H, W, 12.0)
    fill = -1.0
    image[:8] = fill
    image[-8:] = fill
    qube = write_qube(tmp_path / "dci.pds", image.astype(np.float32), null=-3.0e38)
    out = tmp_path / "dci_destriped.tif"
    rc = cli.main([str(qube), "-o", str(out), "--strip", "32", "--step", "16", "--pad", "8",
                   "--no-overviews", "--also-raw"])
    assert rc == 0 and out.exists() and (tmp_path / "dci.tif").exists()
    data, crs, transform, nodata = geotiff.read_raster(out)
    assert nodata == fill and data.shape == (H, W)
    assert np.all(data[:8] == fill) and np.all(data[-8:] == fill)
    valid = image != fill
    assert data[valid].min() >= image[valid].min() - 1e-6
    assert data[valid].max() <= image[valid].max() + 1e-6
    inner = slice(8, H - 8)
    assert rms(data[inner] - terrain[inner]) < 0.5 * rms(image[inner] - terrain[inner])
    assert transform.a == pytest.approx(2 * geotiff.MARS_X_HALF / W)
    with rasterio.open(out) as ds:
        tags = ds.tags(1)
    assert tags["SOURCE"] == "dci.pds" and tags["STRIP"] == "32"


def test_cli_raster_input_keeps_georef_and_writes_qa(tmp_path):
    pytest.importorskip("matplotlib")
    H, W = 64, 128
    image = make_terrain(H, W) + make_stripes(H, W, -15.0)
    crs, transform = geotiff.default_mars_georef(W, H, central_meridian=180.0)
    src = tmp_path / "in.tif"
    geotiff.write_geotiff(src, image, crs, transform, nodata=None, overviews=False)
    out = tmp_path / "in_destriped.tif"
    rc = cli.main([str(src), "--strip", "32", "--step", "16", "--pad", "8", "--no-overviews", "--qa"])
    assert rc == 0 and out.exists() and (tmp_path / "in_destriped_qa.png").exists()
    data, crs2, transform2, nodata = geotiff.read_raster(out)
    assert transform2 == transform and nodata is None and crs2 == crs
    assert np.all(np.isfinite(data))
