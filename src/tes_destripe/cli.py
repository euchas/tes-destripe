"""Command line interface: PDS3 QUBE or raster in, destriped GeoTIFF out."""
from __future__ import annotations

import argparse
import os

import numpy as np

from ._version import __version__
from .core import DEFAULTS, destripe
from .geotiff import MARS_RADIUS, default_mars_georef, georef_from_reference, read_raster, write_geotiff
from .pds import detect_fill, read_pds_qube


def build_parser():
    p = argparse.ArgumentParser(
        prog="tes-destripe",
        description="Remove orbit-track striping from a global TES map (PDS3 QUBE or raster) "
                    "and write a georeferenced GeoTIFF.")
    p.add_argument("input", help="input PDS3 QUBE (.pds/.cub/.qub/.img with attached label) or any raster rasterio can open")
    p.add_argument("-o", "--output", help="output GeoTIFF (default: <input>_destriped.tif)")
    p.add_argument("--band", type=int, default=1, help="1-based band to read (default 1)")
    p.add_argument("--fill", type=float, help="nodata value (default: auto-detect from the polar rows, then CORE_NULL)")
    p.add_argument("--no-fill", action="store_true", help="treat all pixels as valid")
    p.add_argument("--also-raw", action="store_true",
                   help="also write the georeferenced but not destriped map (<input>.tif)")
    p.add_argument("--atten", type=float, default=DEFAULTS["atten"],
                   help="maximum wedge attenuation 0..1 (default %(default)s)")
    p.add_argument("--wedge-sigma-deg", type=float, default=DEFAULTS["wedge_sigma_deg"],
                   help="wedge angular half-width, degrees (default %(default)s)")
    p.add_argument("--guard", type=float, default=DEFAULTS["guard"],
                   help="low-frequency guard radius, cycles/pixel (default %(default)s)")
    p.add_argument("--strip", type=int, default=DEFAULTS["strip"], help="latitude strip height, pixels (default %(default)s)")
    p.add_argument("--step", type=int, default=DEFAULTS["step"],
                   help="strip step, pixels; smaller than --strip for overlap (default %(default)s)")
    p.add_argument("--pad", type=int, default=DEFAULTS["pad"], help="context rows above and below each strip (default %(default)s)")
    p.add_argument("--ref", help="reference raster to copy CRS and extent from (default: standard global Mars grid)")
    p.add_argument("--cm", type=float, default=0.0,
                   help="central meridian, degrees, when building the CRS: 0 for a -180..180 map, 180 for 0..360 (default 0)")
    p.add_argument("--no-clip", action="store_true", help="do not clip FFT overshoot to the input range")
    p.add_argument("--no-overviews", action="store_true", help="skip building GeoTIFF overviews")
    p.add_argument("--qa", action="store_true", help="also write <output>_qa.png (raw / destriped / removed); needs matplotlib")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _is_pds(path):
    return path.lower().endswith((".pds", ".cub", ".qub", ".img"))


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_path = args.output or (os.path.splitext(args.input)[0] + "_destriped.tif")
    src_crs = src_transform = None
    null = None

    print(f"reading {args.input}")
    if _is_pds(args.input):
        data, meta = read_pds_qube(args.input, band=args.band - 1)
        null = meta["null"]
        print(f"  {meta['samples']} x {meta['lines']} px ({meta['ppd']:.3g} ppd), {meta['bands']} band(s)")
    else:
        data, src_crs, src_transform, null = read_raster(args.input, band=args.band)
    H, W = data.shape

    if args.no_fill:
        fill = None
    elif args.fill is not None:
        fill = args.fill
    else:
        fill = detect_fill(data)
        if fill is None and null is not None and np.any(data == null):
            fill = float(null)
    valid = np.ones((H, W), bool) if fill is None else (data != fill)
    print(f"  fill/nodata = {fill}   valid fraction = {valid.mean():.3f}")

    if src_crs is not None and src_transform is not None:
        crs, transform = src_crs, src_transform
        print("  georef: kept from the input raster")
    elif args.ref:
        crs, transform = georef_from_reference(args.ref, W, H)
        print(f"  georef: CRS and extent copied from {os.path.basename(args.ref)}")
    else:
        crs, transform = default_mars_georef(W, H, args.cm)
        print(f"  georef: Mars equirectangular (R = {MARS_RADIUS:.0f} m, central meridian {args.cm:g}), standard global extent")

    if args.also_raw and _is_pds(args.input):
        raw_path = os.path.splitext(args.input)[0] + ".tif"
        print(f"writing georeferenced raw map to {raw_path}")
        write_geotiff(raw_path, data, crs, transform, fill,
                      tags=dict(SOURCE=os.path.basename(args.input)), overviews=not args.no_overviews)

    print("destriping (latitude-adaptive directional wedge)...")
    out, tilts = destripe(data, valid, args.wedge_sigma_deg, args.guard, args.atten,
                          args.strip, args.step, args.pad)
    out = out.astype(np.float32)
    if not args.no_clip:
        lo, hi = float(data[valid].min()), float(data[valid].max())
        np.clip(out, lo, hi, out=out)
        print(f"  clipped to input range [{lo:.4f}, {hi:.4f}]")
    if fill is not None:
        out[~valid] = np.float32(fill)
    print("  per-strip tilt (latitude: degrees from vertical): "
          + "  ".join(f"{lat:+.0f}:{'--' if t is None else f'{t:+.0f}'}" for lat, t in tilts))

    print(f"writing {out_path}")
    write_geotiff(out_path, out, crs, transform, fill,
                  tags=dict(PROCESSING="tes-destripe directional-wedge destripe (latitude-adaptive)",
                            TES_DESTRIPE_VERSION=__version__, ATTEN=args.atten,
                            WEDGE_SIGMA_DEG=args.wedge_sigma_deg, GUARD=args.guard,
                            STRIP=args.strip, STEP=args.step, PAD=args.pad,
                            SOURCE=os.path.basename(args.input)),
                  overviews=not args.no_overviews)
    if args.qa:
        from .qa import write_qa_figure
        qa_path = os.path.splitext(out_path)[0] + "_qa.png"
        write_qa_figure(data, out, valid, qa_path)
        print(f"  wrote {qa_path}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
