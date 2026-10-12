# tes-destripe

Removes the orbit-track striping from Mars Global Surveyor Thermal Emission
Spectrometer (MGS TES) global maps and writes a georeferenced GeoTIFF. Input is
an attached-label PDS3/ISIS2 QUBE, as distributed for the TES global maps
(Dust Cover Index, albedo, thermal inertia), or any raster that rasterio can
open.

TES global mosaics carry near-vertical streaks along the MGS ground tracks.
The streaks are tilted some degrees off the meridians, are broadband (no
single repeat period) and run coherently along track, so they bias any
per-pixel threshold applied to the map, for example when selecting low-dust
terrain. `tes-destripe` removes that oriented component with a
latitude-adaptive directional Fourier filter and leaves isotropic terrain and
large-scale patterns in place.

## Install

```bash
pip install git+https://github.com/euchas/tes-destripe
# optional: matplotlib for the --qa quick-look figure
pip install "tes-destripe[plot] @ git+https://github.com/euchas/tes-destripe"
```

Requires Python 3.10 or newer, numpy, scipy and rasterio. For development:
`git clone`, then `pip install -e .[test,plot]` and `pytest -q`.

## Quickstart

```bash
tes-destripe dci_lo_ice_dust_16ppd.pds                   # writes dci_lo_ice_dust_16ppd_destriped.tif
tes-destripe dci_lo_ice_dust_16ppd.pds --also-raw --qa   # also the unfiltered GeoTIFF and a before/after PNG
tes-destripe map.pds --ref Mars_MGS_TES_Albedo_mosaic_global_7410m.tif   # copy CRS and extent from a reference
tes-destripe map.pds --cm 180                            # product stored 0..360 E instead of -180..180
tes-destripe already_georeferenced.tif                   # keeps the input CRS and extent
tes-destripe -h                                          # all filter and output options
```

```python
from tes_destripe import read_pds_qube, detect_fill, destripe_map, default_mars_georef, write_geotiff

data, meta = read_pds_qube("dci_lo_ice_dust_16ppd.pds")   # float64 [line, sample]
fill = detect_fill(data)                                 # constant polar fill, or None
out, tilts = destripe_map(data, fill=fill)               # float32, clipped, nodata restored
crs, transform = default_mars_georef(meta["samples"], meta["lines"])
write_geotiff("dci_destriped.tif", out, crs, transform, nodata=fill)
```

`tilts` lists the stripe tilt found in each latitude strip (degrees from
vertical), which is a quick check that the filter locked on to the tracks.

## Inputs and outputs

* **PDS3 QUBE** (`.pds`, `.cub`, `.qub`, `.img` with an attached label): the
  reader uses `RECORD_BYTES`, `^QUBE` or `LABEL_RECORDS`, `CORE_ITEMS` with
  `AXIS_NAME`, `CORE_ITEM_TYPE`/`CORE_ITEM_BYTES`, `CORE_BASE`/`CORE_MULTIPLIER`
  and `CORE_NULL`. Multi-band cubes: `--band`.
* **Any other raster** readable by rasterio: its CRS and extent are kept.
* **Nodata**: detected automatically when the first or last row is constant
  fill (the polar caps of the TES maps), otherwise `CORE_NULL`; override with
  `--fill` or `--no-fill`.
* **Output**: float32, LZW-compressed, tiled GeoTIFF with overviews, the nodata
  value and tags recording the parameters, source file and version. Default
  grid for a QUBE input: Mars equirectangular (simple cylindrical) on a
  3,396,000 m sphere, central meridian 0 (`--cm 180` for a 0..360 map), global
  extent of 59,280 m per degree, which co-registers pixel for pixel with the
  USGS TES global mosaics (7,410 m per pixel at 8 pixels per degree).

## How it works

1. Nodata pixels are filled by nearest neighbour so the FFT sees no edges.
2. The map is cut into overlapping latitude strips (512 rows, step 256, plus
   128 rows of context above and below).
3. In each strip the dominant stripe orientation is read from the angular
   distribution of the high-pass power spectrum (0.04 to 0.45 cycles per pixel).
4. A Gaussian wedge of 9 degrees (sigma) around that orientation is attenuated
   by up to 95 %. Frequencies below 0.05 cycles per pixel are guarded, so
   large-scale terrain is untouched. Columns are periodic, which keeps the
   longitude seam of a global map clean.
5. The strips are blended with Hanning weights, overshoot is clipped to the
   input range and nodata is restored.

All of these are command line options (`--wedge-sigma-deg`, `--guard`,
`--atten`, `--strip`, `--step`, `--pad`).

## Known limitations and assumptions

| Assumption | Where |
|---|---|
| The default grid and the latitude labels of the tilt table assume a global map (360 degrees of longitude, 90 N to 90 S) | `geotiff.default_mars_georef`, `core.destripe` |
| One dominant stripe direction per strip; a second family of tracks is only removed if it dominates a strip (widen `--wedge-sigma-deg` or run twice) | `core.estimate_orientation` |
| Isotropic content that falls inside the wedge is attenuated as well, roughly a tenth of the high-frequency power, so sharp isotropic features lose a little contrast (the tests quantify this on synthetic scenes) | `core.wedge_filter` |
| Stripe components below the guard radius remain by design | `core.DEFAULTS["guard"]` |
| 59,280 m per degree (the USGS TES mosaic grid) rather than pi R / 180 = 59,274.7 m; use `--ref` for exact co-registration with another basemap | `geotiff.MARS_M_PER_DEG` |
| Nodata detection assumes constant-fill polar rows; otherwise `CORE_NULL` or `--fill` | `pds.detect_fill` |
| Reader: attached labels only, no suffix planes, no VAX floating point, axis order from `AXIS_NAME` (default SAMPLE, LINE, BAND) | `pds.read_pds_qube` |

## Tests

```bash
pip install -e .[test,plot]
pytest -q
```

The tests generate their own data: synthetic terrain with tilted broadband
stripes, and tiny PDS3 QUBE files in several item types and axis orders. No
downloads and no mission data are needed.

## Citing

If this software is useful in your work, please cite it (see `CITATION.cff`):

> Valantinas, A. (2026). tes-destripe: orbit-track destriping of MGS TES global
> maps (version 0.1.0). https://github.com/euchas/tes-destripe

and the instrument and data it is built for:

* TES: Christensen, P. R. et al. (1992), Journal of Geophysical Research 97(E5),
  7719-7734, https://doi.org/10.1029/92JE00453; Christensen, P. R. et al.
  (2001), Journal of Geophysical Research 106(E10), 23823-23871,
  https://doi.org/10.1029/2000JE001370.
* Dust Cover Index: Ruff, S. W. and Christensen, P. R. (2002), Journal of
  Geophysical Research 107(E12), https://doi.org/10.1029/2001JE001580.
* Data: TES global maps from the Mars Global Data Sets of Arizona State
  University, https://www.mars.asu.edu/data/ (Dust Cover Index at 16 pixels
  per degree: https://www.mars.asu.edu/data/tes_ruffdust); TES albedo mosaic
  from USGS Astrogeology,
  https://astrogeology.usgs.gov/search/map/mars_mgs_tes_global_bolometric_albedo_map_7410m.

## Acknowledgement

Much of the code was written with an AI coding assistant (Claude, Anthropic)
under the direction of the author, who reviewed the code and results.

## License

MIT, see `LICENSE`.
