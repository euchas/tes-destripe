"""Georeferencing and GeoTIFF output (rasterio is imported on use)."""
from __future__ import annotations

import numpy as np

MARS_RADIUS = 3396000.0
# Metres per degree used for the default global extent. 59,280 m is the value
# implied by the USGS TES global mosaics (7,410 m/pixel at 8 pixels/degree), so
# the default grid co-registers with them pixel for pixel. The exact value for a
# 3,396,000 m sphere would be 59,274.7 m; pass a reference raster for exact
# co-registration with another basemap.
MARS_M_PER_DEG = 59280.0
MARS_X_HALF = 180.0 * MARS_M_PER_DEG     # 10,670,400 m
MARS_Y_HALF = 90.0 * MARS_M_PER_DEG      # 5,335,200 m


def mars_equirect_wkt(radius=MARS_RADIUS, central_meridian=0.0):
    """WKT of the Mars equirectangular (simple cylindrical) sphere projection."""
    return (f'PROJCS["SimpleCylindrical Mars",GEOGCS["GCS_Mars",DATUM["D_Mars",'
            f'SPHEROID["Mars",{radius},0]],PRIMEM["Reference_Meridian",0],'
            f'UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],'
            f'PARAMETER["standard_parallel_1",0],PARAMETER["central_meridian",{central_meridian}],'
            f'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
            f'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')


def default_mars_georef(width, height, central_meridian=0.0):
    """CRS and affine transform of a global map on the default Mars grid.

    The map is assumed to span 360 degrees of longitude and 180 degrees of
    latitude, with longitude `central_meridian` at the centre (0 for a map
    stored -180..180, 180 for a map stored 0..360).
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    crs = CRS.from_wkt(mars_equirect_wkt(MARS_RADIUS, central_meridian))
    return crs, from_bounds(-MARS_X_HALF, -MARS_Y_HALF, MARS_X_HALF, MARS_Y_HALF, width, height)


def georef_from_reference(path, width, height):
    """CRS and transform copied from a reference raster, re-gridded to width x height."""
    import rasterio
    from rasterio.transform import from_bounds

    with rasterio.open(path) as r:
        return r.crs, from_bounds(*r.bounds, width, height)


def read_raster(path, band=1):
    """Read one band of a rasterio-readable file: (data float64, crs, transform, nodata)."""
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(band).astype(np.float64), ds.crs, ds.transform, ds.nodata


def write_geotiff(path, arr, crs, transform, nodata=None, tags=None, overviews=True):
    """Write a float32 LZW-compressed tiled GeoTIFF, optionally with overviews."""
    import rasterio
    from rasterio.enums import Resampling

    arr = np.asarray(arr, dtype=np.float32)
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", crs=crs, transform=transform, compress="LZW", tiled=True)
    if nodata is not None:
        profile["nodata"] = float(nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(1, **{k: str(v) for k, v in tags.items()})
    if overviews:
        factors = [f for f in (2, 4, 8, 16) if min(arr.shape) // f >= 16]
        if factors:
            with rasterio.open(path, "r+") as dst:
                dst.build_overviews(factors, Resampling.average)
