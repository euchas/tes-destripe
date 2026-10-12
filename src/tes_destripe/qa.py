"""Before/after/difference quick-look figure (matplotlib is imported on use)."""
from __future__ import annotations

import numpy as np


def write_qa_figure(raw, destriped, valid, png_path, dpi=110):
    """Write a 2 x 3 panel figure: raw, destriped and removed component, full and zoomed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(raw, dtype=np.float64)
    destriped = np.asarray(destriped, dtype=np.float64)
    rawm = np.where(valid, raw, np.nan)
    dem = np.where(valid, destriped, np.nan)
    diff = dem - rawm
    H, W = raw.shape
    dec = max(1, W // 1920)
    v = np.nanpercentile(rawm, [2, 98])
    m = np.nanpercentile(np.abs(diff), 99) or 1e-6
    zr = slice(int(H * 0.46), max(int(H * 0.54), int(H * 0.46) + 1))
    zc = slice(int(W * 0.38), max(int(W * 0.45), int(W * 0.38) + 1))
    panels = [
        (rawm[::dec, ::dec], "raw", {}),
        (dem[::dec, ::dec], "destriped", {}),
        (diff[::dec, ::dec], "removed (stripes)", dict(cmap="RdBu", vmin=-m, vmax=m)),
        (rawm[zr, zc], "raw, zoom", {}),
        (dem[zr, zc], "destriped, zoom", {}),
        (diff[zr, zc], "removed, zoom", dict(cmap="RdBu", vmin=-m, vmax=m)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, (img, title, kw) in zip(axes.ravel(), panels):
        opts = dict(cmap="gray", vmin=v[0], vmax=v[1])
        opts.update(kw)
        ax.imshow(img, aspect="auto", **opts)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return png_path
