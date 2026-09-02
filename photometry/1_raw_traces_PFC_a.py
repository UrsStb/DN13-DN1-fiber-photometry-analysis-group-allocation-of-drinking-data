# =============================================================================
# inspect_pfc_raw_traces.py
#
# PURPOSE
#   Load a Doric-style fiber photometry export and INSPECT the RAW PFC traces
#   ONLY. This script does NOT correct anything (no bleaching correction, no
#   motion/isosbestic regression, no dF/F). Its only job is to let you SEE the
#   raw data honestly before any preprocessing, so you can spot problems early:
#   photobleaching decay, LED-onset transients, saturation/clipping, dropped
#   frames, irregular sampling, or dead channels.
#
# WHAT IT ASSUMES ABOUT THE FILE (edit CONFIG below if any of this is wrong)
#   - There are TWO header rows:
#       row 1 = group label   (---, CAM #1, ..., ROI 0, ROI 0, ROI 1, ROI 1, ...)
#       row 2 = channel label (Time(s), -, ..., CAM 1 EXC 1, CAM 1 EXC 2, ...)
#     Data starts on row 3.
#   - PFC = ROI 1. Isosbestic (405 nm) = CAM 1 EXC 1. Signal (465/470 nm) =
#     CAM 1 EXC 2. In the screenshot those are columns H and I (0-indexed 7 & 8).
#   - The two excitations are TIME-MULTIPLEXED (interleaved): each photometry
#     column only holds a value on the frames where that LED was on, so most
#     cells in the ROI columns are blank. The script de-interleaves by dropping
#     the blank rows per channel. IMPORTANT CONSEQUENCE: the isosbestic and the
#     signal are sampled at DIFFERENT time points (offset by ~half a frame), so
#     they are NOT simultaneous. That matters for later regression, but for raw
#     inspection we just plot each on its own real timestamps.
#
# HOW TO USE
#   1. Set FILE_PATH to your .csv or .xlsx.
#   2. Run it. It first PRINTS every detected column with its index and the
#      combined header text, so you can VERIFY the mapping before trusting plots.
#   3. If the printed names at TIME_COL / ISOS_COL / SIGNAL_COL are wrong, change
#      those indices and re-run. Nothing is hard-coded to header strings, so a
#      truncated/renamed header won't silently break it.
# =============================================================================

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- CONFIG (edit me) ------------------------------
FILE_PATH   = "example_photometry.csv"  # <-- your data file (.csv or .xlsx)
N_HEADER_ROWS = 2      # number of header rows before the data
TIME_COL    = 0        # column A  -> Time(s)
ISOS_COL    = 7        # column H  -> ROI 1 / CAM 1 EXC 1  (PFC isosbestic, 405 nm)
SIGNAL_COL  = 8        # column I  -> ROI 1 / CAM 1 EXC 2  (PFC signal, 465/470 nm)
INSPECT_FIRST_SECONDS = 30   # zoom window (s) to inspect the LED-onset transient
SAVE_FIGS   = False    # display only; set True to also write PNGs next to the input file
SAVE_DIR    = ""       # "" = save next to the input file. Or set an absolute path.
# -----------------------------------------------------------------------------


def load_two_header_file(path, n_header):
    """Read the file, build combined column names from the header rows, return
    (data_frame, combined_names). Works for both .csv and .xlsx.

    IMPORTANT: the file is read in a SINGLE pass. Doric-style exports are often
    "ragged" — the trailing event columns (F1..F4 Event) are only written on
    rows where an event fires, so header rows can have more fields than data
    rows. Reading header and data separately makes pandas infer two different
    column counts and crashes with a length mismatch. To avoid that, we first
    find the maximum field count anywhere in the file and read every row padded
    to that fixed width, so header and data always agree."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        full = pd.read_excel(path, header=None, dtype=str)
    else:  # treat anything else as CSV
        # Pass 1: find the widest row (csv.reader respects quoted fields).
        with open(path, "r", newline="") as fh:
            max_cols = max((len(row) for row in csv.reader(fh)), default=0)
        # Pass 2: read every row padded to max_cols (short rows -> trailing NaN).
        full = pd.read_csv(path, header=None, dtype=str,
                           names=range(max_cols), engine="python")

    # Split the combined frame back into header rows and data rows. Because it
    # is one frame, both have identical column counts.
    header_block = full.iloc[:n_header].reset_index(drop=True)
    df = full.iloc[n_header:].reset_index(drop=True)

    # Combine the header rows into one readable label per column, e.g.
    # "ROI 1 | CAM 1 EXC 1". Missing pieces are skipped.
    combined = []
    for col in range(header_block.shape[1]):
        parts = [str(header_block.iloc[r, col]).strip()
                 for r in range(n_header)
                 if str(header_block.iloc[r, col]).strip() not in ("", "nan", "-")]
        combined.append(" | ".join(parts) if parts else f"col_{col}")
    df.columns = combined
    return df, combined


def extract_channel(df, time_idx, value_idx):
    """De-interleave one photometry channel: take (time, value), coerce to
    numeric, and drop rows where the value is blank (the frames when this LED
    was off). Returns a clean 2-column DataFrame on the channel's own real
    timestamps."""
    sub = df.iloc[:, [time_idx, value_idx]].copy()
    sub.columns = ["time", "value"]
    sub["time"]  = pd.to_numeric(sub["time"], errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"]).reset_index(drop=True)
    return sub


def describe_channel(name, ch):
    """Print basic sanity diagnostics for one channel."""
    if len(ch) < 2:
        print(f"  [{name}] WARNING: only {len(ch)} samples found — check the column index.")
        return
    dt = np.diff(ch["time"].values)
    fs = 1.0 / np.median(dt)            # effective per-channel sampling rate
    print(f"  [{name}]")
    print(f"    samples            : {len(ch)}")
    print(f"    duration           : {ch['time'].iloc[0]:.3f}  ->  {ch['time'].iloc[-1]:.3f} s")
    print(f"    effective rate     : {fs:.3f} Hz  (median frame period {np.median(dt)*1000:.2f} ms)")
    print(f"    frame-period spread: min {dt.min()*1000:.2f} / max {dt.max()*1000:.2f} ms "
          f"(large max = dropped frames or gaps)")
    print(f"    value range        : {ch['value'].min():.3f} .. {ch['value'].max():.3f} "
          f"(mean {ch['value'].mean():.3f})")


def main():
    print("=" * 70)
    print("Loading:", FILE_PATH)
    df, names = load_two_header_file(FILE_PATH, N_HEADER_ROWS)

    # ---- STEP 1: show the column map so you can verify before trusting plots.
    print("\nDetected columns (index : combined header):")
    for i, n in enumerate(names):
        print(f"  {i:>2} : {n}")
    print("\nUsing ->")
    print(f"  TIME   = col {TIME_COL}  : {names[TIME_COL]}")
    print(f"  ISOS   = col {ISOS_COL}  : {names[ISOS_COL]}")
    print(f"  SIGNAL = col {SIGNAL_COL}  : {names[SIGNAL_COL]}")
    print("If any of those three are wrong, fix the indices in CONFIG and re-run.")

    # ---- STEP 2: de-interleave the two PFC channels onto their real timestamps.
    iso = extract_channel(df, TIME_COL, ISOS_COL)
    sig = extract_channel(df, TIME_COL, SIGNAL_COL)

    print("\nDiagnostics:")
    describe_channel("isosbestic (405)", iso)
    describe_channel("signal (465/470)", sig)

    # Median offset between the two interleaved channels (a half-frame is normal).
    if len(iso) and len(sig):
        n = min(len(iso), len(sig))
        offset = np.median(sig["time"].values[:n] - iso["time"].values[:n]) * 1000
        print(f"  median iso->signal timing offset: {offset:.2f} ms "
              f"(expected ~half a frame; confirms interleaving)")

    # Resolve an absolute, existing output directory. Default ("") = the folder
    # the input file lives in, which we know exists and just read from. This
    # avoids depending on the current working directory (a common source of
    # FileNotFoundError, e.g. in OneDrive-synced Documents folders).
    out_dir = SAVE_DIR or os.path.dirname(os.path.abspath(FILE_PATH))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"\n[warn] could not create output dir '{out_dir}': {e}")

    def save(fig, fname):
        """Save a figure; never crash the run if the write fails."""
        if not SAVE_FIGS:
            return
        path = os.path.join(out_dir, fname)
        try:
            fig.savefig(path, dpi=150)
            print(f"  saved: {path}")
        except OSError as e:
            print(f"  [warn] could not save {path}: {e}")
            print("         (plots still shown; set SAVE_FIGS=False or change SAVE_DIR)")

    # ---- STEP 3: plot raw traces (no correction of any kind).
    # Fig 1: full session, stacked so the two channels' native scales are honest.
    fig1, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(iso["time"], iso["value"], lw=0.6, color="tab:purple")
    ax[0].set_ylabel("Isosbestic (a.u.)")
    ax[0].set_title("Raw PFC isosbestic (405 nm) — uncorrected")
    ax[1].plot(sig["time"], sig["value"], lw=0.6, color="tab:green")
    ax[1].set_ylabel("Signal (a.u.)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_title("Raw PFC signal (465/470 nm) — uncorrected")
    fig1.tight_layout()

    # Fig 2: zoom on the first INSPECT_FIRST_SECONDS to inspect the onset transient.
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    for a, ch, c, lab in [(ax2[0], iso, "tab:purple", "isosbestic"),
                          (ax2[1], sig, "tab:green",  "signal")]:
        m = ch["time"] <= INSPECT_FIRST_SECONDS
        a.plot(ch["time"][m], ch["value"][m], lw=0.8, color=c)
        a.set_ylabel(f"{lab} (a.u.)")
    ax2[0].set_title(f"First {INSPECT_FIRST_SECONDS} s — inspect LED-onset transient / early bleach")
    ax2[1].set_xlabel("Time (s)")
    fig2.tight_layout()

    print("\nFigures:")
    save(fig1, "pfc_raw_full.png")
    save(fig2, "pfc_raw_onset.png")
    print("=" * 70)
    plt.show()   # opens interactive windows in VS Code; no-op if headless


if __name__ == "__main__":
    main()