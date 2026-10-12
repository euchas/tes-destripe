"""tes-destripe: remove orbit-track striping from MGS TES global maps.

Reads an attached-label PDS3/ISIS2 QUBE (or any raster rasterio can open),
removes the oriented orbit-track stripes with a latitude-adaptive directional
Fourier filter, and writes a georeferenced GeoTIFF.
"""
from ._version import __version__
from .core import DEFAULTS, destripe, destripe_map, estimate_orientation, fill_nodata_nearest, wedge_filter
from .geotiff import (
    MARS_M_PER_DEG,
    MARS_RADIUS,
    MARS_X_HALF,
    MARS_Y_HALF,
    default_mars_georef,
    georef_from_reference,
    mars_equirect_wkt,
    read_raster,
    write_geotiff,
)
from .pds import detect_fill, read_pds_qube

__all__ = [
    "__version__",
    "DEFAULTS",
    "destripe",
    "destripe_map",
    "estimate_orientation",
    "fill_nodata_nearest",
    "wedge_filter",
    "MARS_M_PER_DEG",
    "MARS_RADIUS",
    "MARS_X_HALF",
    "MARS_Y_HALF",
    "default_mars_georef",
    "georef_from_reference",
    "mars_equirect_wkt",
    "read_raster",
    "write_geotiff",
    "detect_fill",
    "read_pds_qube",
]
