import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────────
IN_DIR              = Path(
    r"G:\My Drive\1. LC Avengers\3. Haptic touch display\Raw data\Dektak"
    r"\5.5 embossing depth to thickness ratio\5 may S53P11D37\Graphs\S53p11.2 1.5mm"
)
OUT_DIR             = IN_DIR / "plots_analysis"

SHOW_QC             = 3          # how many random QC plots to pop up

BASELINE_WIDTH_CM   = 0.07       # ± region around each ridge-foot for baseline fit
BASELINE_POLY_DEG   = 1          # degree of polynomial for baseline (1=straight)

PEAK_PROMINENCE     = 10         # µm
PEAK_MIN_WIDTH_CM   = 0.01       # discard peaks narrower than this (cm)

# Hybrid smoothing params
MEDIAN_KERNEL       = 3          # small median filter to remove spikes
GAUSSIAN_SIGMA      = 4.0        # gaussian blur radius (tune as needed)

X_MAJOR_TICK        = 0.2
X_MINOR_TICK        = 0.02
Y_MAJOR_TICK        = 100
Y_MINOR_TICK        = 25

REJECT_CRATERS      = []         # crater IDs to drop manually
# ────────────────────────────────────────────────────────────────────────────────

def read_profile_csv(fp: Path) -> pd.DataFrame:
    """
    Reads a Dektak CSV file, locates the 'Lateral(µm)' and 'Total Profile(Å)' columns,
    and converts them to cm and µm, respectively.
    """
    lines = fp.read_text('utf-8-sig').splitlines()
    for i, line in enumerate(lines):
        if "Lateral" in line:
            hdr  = [c.strip() for c in line.split(",") if c.strip()]
            data = [l.split(",")[:len(hdr)] for l in lines[i+1:] if l.strip()]
            break
    else:
        raise ValueError(f"No header in {fp}")

    df = pd.DataFrame(data, columns=hdr).apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=["Lateral(µm)","Total Profile(Å)"], how='all', inplace=True)

    df["Lateral_cm"] = df["Lateral(µm)"] / 1e4
    df["Profile_um"] = df["Total Profile(Å)"] / 1e4
    return df.reset_index(drop=True)

def smooth(y):
    """
    1) Small median filter to remove spikes
    2) Gaussian filter for final smoothing
    """
    # 1) Median filter
    y_med = medfilt(y, kernel_size=MEDIAN_KERNEL)

    # 2) Gaussian smoothing
    y_gauss = gaussian_filter1d(y_med, sigma=GAUSSIAN_SIGMA)

    return y_gauss

def detect_peaks(x, y_s):
    """
    Use scipy.signal.find_peaks with a minimum prominence, distance, and width.
    """
    dx   = x[1] - x[0]
    dist = max(1, int(PEAK_MIN_WIDTH_CM / dx))
    width_samples = PEAK_MIN_WIDTH_CM / dx

    peaks, _ = find_peaks(
        y_s,
        prominence=PEAK_PROMINENCE,  # in µm
        distance=dist,
        width=width_samples
    )
    return peaks

def find_foot(y_s, peak_idx, direction):
    """
    Walk from peak_idx in 'direction' (-1=left, +1=right) until slope changes sign.
    """
    i = peak_idx
    if direction < 0:
        while i > 0 and y_s[i-1] < y_s[i]:
            i -= 1
    else:
        while i < len(y_s)-1 and y_s[i+1] < y_s[i]:
            i += 1
    return i

def analyze_pair(nem_df, iso_df):
    """
    For a single sample's cold (nematic) and hot (isotropic) profiles:
      1) Smooth data
      2) Detect peaks (ridges)
      3) Pair ridges in sets of two => define crater region
      4) Local baseline => measure crater valley
    """
    x_n = nem_df["Lateral_cm"].values
    y_n = nem_df["Profile_um"].values
    x_i = iso_df["Lateral_cm"].values
    y_i = iso_df["Profile_um"].values

    # Smooth each
    y_n_s = smooth(y_n)
    y_i_s = smooth(y_i)

    # Detect ridges in nematic
    peaks = detect_peaks(x_n, y_n_s)

    craters = []
    crater_id = 1

    # Pair peaks: (0,1), (2,3), ...
    for idx in range(0, len(peaks) - 1, 2):
        if crater_id in REJECT_CRATERS:
            crater_id += 1
            continue

        p0, p1 = peaks[idx], peaks[idx+1]
        l, r = p0, p1  # crater from left to right ridge

        # Feet for local baseline
        f0 = find_foot(y_n_s, p0, -1)
        f1 = find_foot(y_n_s, p1,  1)

        # sample around each foot ± BASELINE_WIDTH_CM
        xl0, xr0 = x_n[f0] - BASELINE_WIDTH_CM, x_n[f0] + BASELINE_WIDTH_CM
        xl1, xr1 = x_n[f1] - BASELINE_WIDTH_CM, x_n[f1] + BASELINE_WIDTH_CM

        mask0 = (x_n >= xl0) & (x_n <= xr0)
        mask1 = (x_n >= xl1) & (x_n <= xr1)

        Xb = np.concatenate([x_n[mask0], x_n[mask1]])
        Yb = np.concatenate([y_n_s[mask0], y_n_s[mask1]])

        coeff = np.polyfit(Xb, Yb, deg=BASELINE_POLY_DEG)
        baseline_fn = np.poly1d(coeff)

        # crater segment
        xs = x_n[l : r+1]
        yn_segment = y_n_s[l : r+1]
        yi_segment = np.interp(xs, x_i, y_i_s)
        bl_segment = baseline_fn(xs)

        # find valley in nematic segment
        valley_idx_local = np.argmin(yn_segment)
        valley_n = yn_segment[valley_idx_local]
        valley_i = yi_segment[valley_idx_local]
        thick    = bl_segment[valley_idx_local]

        d_n = thick - valley_n
        d_i = thick - valley_i
        if thick == 0:
            edt = np.nan
            eff_pct = np.nan
        else:
            edt = d_n / thick
            eff_pct = (d_n - d_i) / thick * 100

        craters.append({
            "crater"       : crater_id,
            "x_l_cm"       : x_n[l],
            "x_r_cm"       : x_n[r],
            "foot0_cm"     : x_n[f0],
            "foot1_cm"     : x_n[f1],
            "thickness_um" : thick,
            "nematic_um"   : d_n,
            "isotropic_um" : d_i,
            "EDT"          : edt,
            "efficiency_%" : eff_pct
        })

        crater_id += 1

    return peaks, craters, y_n_s, y_i_s

def plot_pair(name, nem_df, iso_df, peaks, craters, y_n_s, y_i_s, show, outp):
    """
    Creates a QC plot showing raw vs. smoothed (nematic/isotropic), ridge peaks, crater shading, etc.
    """
    x_n = nem_df["Lateral_cm"].values
    y_n = nem_df["Profile_um"].values
    x_i = iso_df["Lateral_cm"].values

    fig, ax = plt.subplots(figsize=(10,5.625), dpi=100)
    ax.plot(x_n, y_n, c="lightgray", label="raw nematic")
    ax.plot(x_n, y_n_s, c="purple", label="smooth nematic", lw=2)
    ax.plot(x_i, y_i_s, c="goldenrod", label="smooth isotropic", lw=2)

    ax.plot(x_n[peaks], y_n_s[peaks], 'r^', ms=6, label='ridges')

    for c in craters:
        ax.axvspan(c["x_l_cm"], c["x_r_cm"], color="red", alpha=0.12)
        xs_foot = np.array([c["foot0_cm"], c["foot1_cm"]])
        mask0 = (x_n >= xs_foot[0] - BASELINE_WIDTH_CM) & (x_n <= xs_foot[0] + BASELINE_WIDTH_CM)
        mask1 = (x_n >= xs_foot[1] - BASELINE_WIDTH_CM) & (x_n <= xs_foot[1] + BASELINE_WIDTH_CM)

        Xb = np.concatenate([x_n[mask0], x_n[mask1]])
        Yb = np.concatenate([y_n_s[mask0], y_n_s[mask1]])
        bl_fn = np.poly1d(np.polyfit(Xb, Yb, deg=BASELINE_POLY_DEG))

        bx = np.linspace(c["x_l_cm"], c["x_r_cm"], 2)
        ax.plot(bx, bl_fn(bx), c="green", ls="--", lw=1)

        # Mark the valley
        lidx = np.searchsorted(x_n, c["x_l_cm"])
        seg  = y_n_s[lidx : np.searchsorted(x_n, c["x_r_cm"]) + 1]
        mloc = lidx + int(np.argmin(seg))
        ax.plot(x_n[mloc], y_n_s[mloc], 'v', c="black", ms=6)

    ax.set_title(name, fontsize=18)
    ax.set_xlabel("Lateral (cm)", fontsize=16)
    ax.set_ylabel("Profile (µm)", fontsize=16)
    ax.tick_params(labelsize=14)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":",  alpha=0.3)
    ax.legend(fontsize=12, loc="best")

    fig.tight_layout()
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    files = sorted(IN_DIR.glob("*.csv"))

    # 1) Define groups before the loop
    groups = {}

    # 2) Populate groups by matching CSV filenames
    for f in files:
        m = re.match(r"(.+?)\s*(\d+\.?\d*)[Cc]", f.stem)
        if not m:
            continue
        base, T = m.group(1).strip(), float(m.group(2))
        groups.setdefault(base, []).append((T, f))

    results = []
    # If there are fewer than SHOW_QC groups, random.sample won't raise an error
    # but let's just do this safely:
    qc_bases = random.sample(list(groups), min(SHOW_QC, len(groups)))

    # 3) Now iterate over the items in groups
    for base, lst in groups.items():
        lst.sort(key=lambda x: x[0])  # sort by temperature
        nem_fp, iso_fp = lst[0][1], lst[-1][1]  # first = coldest, last = hottest

        nem_df = read_profile_csv(nem_fp)
        iso_df = read_profile_csv(iso_fp)

        peaks, craters, y_n_s, y_i_s = analyze_pair(nem_df, iso_df)

        for c in craters:
            c["sample"] = base
            results.append(c)

        show_plot = (base in qc_bases)
        plot_pair(base, nem_df, iso_df, peaks, craters, y_n_s, y_i_s, show_plot,
                  OUT_DIR / f"{base}.png")

    if results:
        df = pd.DataFrame(results)[[
            "sample","crater","thickness_um",
            "nematic_um","isotropic_um","EDT","efficiency_%"
        ]]
        df.to_csv(OUT_DIR / "results.csv", index=False)
        print("✅ Results:", OUT_DIR / "results.csv")
        print("✅ Plots:  ", OUT_DIR)
    else:
        print("❌ No valid CSV pairs found in", IN_DIR)
if __name__ == "__main__":
    main()
