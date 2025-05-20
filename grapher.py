import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import medfilt, savgol_filter, find_peaks
from pathlib import Path

# ─── USER CONFIGURATION ─────────────────────────────────────────────────────────
IN_DIR              = Path(
    r"G:\My Drive\1. LC Avengers\3. Haptic touch display\Raw data\Dektak"
    r"\5.5 embossing depth to thickness ratio\5 may S53P11D37\Graphs\S53p11.2 1.5mm"
)
OUT_DIR             = IN_DIR / "plots_analysis"
SHOW_QC             = 3          # how many random QC plots to pop up
BASELINE_WIDTH_CM   = 0.02       # ± region around each ridge-foot for baseline fit
BASELINE_POLY_DEG   = 1          # degree of polynomial for baseline (1=straight)
PEAK_PROMINENCE     = 20      # µm
PEAK_MIN_WIDTH_CM   = 0.04      # discard peaks narrower than this (cm)
MEDIAN_KERNEL       = 1
SAV_GOLAY_WINDOW    = 31
SAV_GOLAY_POLY      = 11
X_MAJOR_TICK        = 0.2
X_MINOR_TICK        = 0.02
Y_MAJOR_TICK        = 100
Y_MINOR_TICK        = 25
REJECT_CRATERS      = []         # crater IDs to drop manually
# ────────────────────────────────────────────────────────────────────────────────

def read_profile_csv(fp: Path) -> pd.DataFrame:
    lines = fp.read_text('utf-8-sig').splitlines()
    for i,line in enumerate(lines):
        if "Lateral" in line:
            hdr  = [c.strip() for c in line.split(",") if c.strip()]
            data = [line.split(",")[:len(hdr)] for line in lines[i+1:] if line.strip()]
            break
    else:
        raise ValueError(f"No header in {fp}")
    df = pd.DataFrame(data, columns=hdr).apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=["Lateral(µm)","Total Profile(Å)"], how='all', inplace=True)
    df["Lateral_cm"] = df["Lateral(µm)"] / 1e4
    df["Profile_um"] = df["Total Profile(Å)"] / 1e4
    return df.reset_index(drop=True)

def smooth(y):
    y_med = medfilt(y, kernel_size=MEDIAN_KERNEL)
    wl = min(SAV_GOLAY_WINDOW, len(y_med) - (1 - len(y_med) % 2))
    if wl % 2 == 0:
        wl -= 1
    if wl < SAV_GOLAY_POLY + 2 or wl < 3:
        return y_med
    try:
        return savgol_filter(y_med, window_length=wl, polyorder=SAV_GOLAY_POLY)
    except Exception:
        return y_med

def detect_peaks(x,y_s):
    dx   = x[1]-x[0]
    dist = max(1,int(PEAK_MIN_WIDTH_CM/dx))
    width_samples = PEAK_MIN_WIDTH_CM/dx
    peaks, _ = find_peaks(
        y_s,
        prominence=PEAK_PROMINENCE,
        distance=dist,
        width=width_samples
    )
    return peaks

def find_foot(y_s, peak_idx, direction):
    i = peak_idx
    if direction<0:
        while i>0 and y_s[i-1]<y_s[i]:
            i-=1
    else:
        while i<len(y_s)-1 and y_s[i+1]<y_s[i]:
            i+=1
    return i

def analyze_pair(nem_df, iso_df):
    x_n, y_n = nem_df["Lateral_cm"].values, nem_df["Profile_um"].values
    x_i, y_i = iso_df["Lateral_cm"].values, iso_df["Profile_um"].values

    y_n_s = smooth(y_n)
    y_i_s = smooth(y_i)

    peaks = detect_peaks(x_n, y_n_s)
    craters = []
    crater_id = 1

    # build crater intervals from peak-top to next peak-top
    for idx in range(0, len(peaks)-1, 2):
        if crater_id in REJECT_CRATERS:
            crater_id += 1
            continue

        p0, p1 = peaks[idx], peaks[idx + 1]
        left_idx, right_idx = p0, p1  # crater window = peak-tops

        # determine feet only for baseline fitting
        f0 = find_foot(y_n_s, p0, -1)
        f1 = find_foot(y_n_s, p1,  1)

        # sample around feet
        xl0, xr0 = x_n[f0]-BASELINE_WIDTH_CM, x_n[f0]+BASELINE_WIDTH_CM
        xl1, xr1 = x_n[f1]-BASELINE_WIDTH_CM, x_n[f1]+BASELINE_WIDTH_CM
        mask0 = (x_n>=xl0)&(x_n<=xr0)
        mask1 = (x_n>=xl1)&(x_n<=xr1)
        Xb = np.concatenate([x_n[mask0], x_n[mask1]])
        Yb = np.concatenate([y_n_s[mask0], y_n_s[mask1]])
        coeff = np.polyfit(Xb, Yb, deg=BASELINE_POLY_DEG)
        baseline_fn = np.poly1d(coeff)

        xs = x_n[left_idx:right_idx + 1]
        yn = y_n_s[left_idx:right_idx + 1]
        yi = np.interp(xs, x_i, y_i_s)
        bl = baseline_fn(xs)
        valley_idx  = int(np.argmin(yn))
        valley_n    = yn[valley_idx]
        valley_i    = yi[valley_idx]
        thick       = bl[valley_idx]
        d_n         = thick - valley_n
        d_i         = thick - valley_i
        edt         = d_n/thick if thick else np.nan
        eff_pct     = (d_n-d_i)/thick*100 if thick else np.nan

        craters.append({
            "crater"       : crater_id,
            "x_l_cm"       : x_n[left_idx],
            "x_r_cm"       : x_n[right_idx],
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

def plot_pair(name,nem_df,iso_df,peaks,craters,y_n_s,y_i_s,show,outp):
    x_n, y_n = nem_df["Lateral_cm"].values, nem_df["Profile_um"].values
    x_i       = iso_df["Lateral_cm"].values

    fig,ax = plt.subplots(figsize=(10,5.625), dpi=100)
    ax.plot(x_n, y_n,   c="lightgray", label="raw nematic")
    ax.plot(x_n, y_n_s, c="purple",    label="smooth nematic", lw=2)
    ax.plot(x_i, y_i_s, c="goldenrod", label="smooth isotropic", lw=2)
    ax.plot(x_n[peaks], y_n_s[peaks], 'r^', ms=6, label='ridges')

    for c in craters:
        ax.axvspan(c["x_l_cm"], c["x_r_cm"], color="red", alpha=0.12)
        # baseline
        xs = np.array([c["foot0_cm"], c["foot1_cm"]])
        mask0 = (x_n>=xs[0]-BASELINE_WIDTH_CM)&(x_n<=xs[0]+BASELINE_WIDTH_CM)
        mask1 = (x_n>=xs[1]-BASELINE_WIDTH_CM)&(x_n<=xs[1]+BASELINE_WIDTH_CM)
        Xb = np.concatenate([x_n[mask0], x_n[mask1]])
        Yb = np.concatenate([y_n_s[mask0], y_n_s[mask1]])
        bl_fn = np.poly1d(np.polyfit(Xb, Yb, deg=BASELINE_POLY_DEG))
        bx = np.linspace(c["x_l_cm"], c["x_r_cm"], 2)
        ax.plot(bx, bl_fn(bx), c="green", ls="--", lw=1)
        # valley
        lidx = np.searchsorted(x_n, c["x_l_cm"])
        seg  = y_n_s[lidx: np.searchsorted(x_n, c["x_r_cm"])+1]
        mloc = lidx + int(np.argmin(seg))
        ax.plot(x_n[mloc], y_n_s[mloc], 'v', c="black", ms=6)

    ax.set_title(name,fontsize=18)
    ax.set_xlabel("Lateral (cm)",fontsize=16)
    ax.set_ylabel("Profile (µm)",fontsize=16)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_TICK))
    ax.xaxis.set_minor_locator(MultipleLocator(X_MINOR_TICK))
    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_TICK))
    ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR_TICK))
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
    groups = {}
    for f in files:
        m = re.match(r"(.+?)\s*(\d+\.?\d*)[Cc]", f.stem)
        if not m:
            continue
        base, T = m.group(1).strip(), float(m.group(2))
        groups.setdefault(base,[]).append((T,f))

    results = []
    qc_bases = random.sample(list(groups), min(SHOW_QC,len(groups)))
    for base,lst in groups.items():
        lst.sort(key=lambda x:x[0])
        nem_fp, iso_fp = lst[0][1], lst[-1][1]
        nem_df,iso_df  = read_profile_csv(nem_fp), read_profile_csv(iso_fp)

        peaks,craters,y_n_s,y_i_s = analyze_pair(nem_df,iso_df)
        for c in craters:
            c["sample"] = base
            results.append(c)

        show = base in qc_bases
        plot_pair(base,nem_df,iso_df,peaks,craters,y_n_s,y_i_s,show, OUT_DIR/f"{base}.png")

    if results:
        df = pd.DataFrame(results)[[
            "sample","crater","thickness_um",
            "nematic_um","isotropic_um","EDT","efficiency_%"
        ]]
        df.to_csv(OUT_DIR/"results.csv",index=False)
        print("✅ Results:", OUT_DIR/"results.csv")
        print("✅ Plots:  ", OUT_DIR)
    else:
        print("❌ No valid CSV pairs found in", IN_DIR)

if __name__=="__main__":
    main()
