# =============================================================================
# fp_session_qc.py  --  ONE-FILE SANITY CHECK for a Doric photometry session
# -----------------------------------------------------------------------------
# PURPOSE
#   Point it at one raw photometry file (+ optionally the behavioural video),
#   run it, read the printout, and decide whether the session is worth
#   analysing at all. It writes nothing by default and changes nothing.
#
#   It consolidates the checks that were previously spread over
#   inspect_pfc_raw_traces / sync-test / isosbestic-diagnostic / htr_sync, and
#   adds the video-integrity checks that were missing (mid-recording cuts,
#   frozen/duplicated frames).
#
# WHAT IT CHECKS
#   A. File structure & column map      -- headers actually say what we assume
#   B. Master timebase integrity        -- monotonic, no duplicates, no gaps
#   C. De-interleaving                  -- iso/sig counts, rates, half-frame offset
#   D. Per-channel signal quality       -- clipping, dead segments, bleaching, steps
#   E. Isosbestic usability             -- r (raw AND drift-free), OLS vs robust
#   F. dF/F preview                     -- does the correction flatten the steps?
#   G. Digital lines                    -- CAM/EXC/DIO edge trains, rate, gaps
#   H. F1-F4 event columns              -- present? how many? when?
#   I. Video integrity                  -- decoded frame count, CUTS, frozen frames
#   J. Verdict                          -- PASS / WARN / FAIL list
#
# WHAT IT DELIBERATELY DOES NOT DO
#   * It does not prove the behavioural camera is frame-locked to DI/O #2.
#     DI/O #2 and EXC #1 are indistinguishable at this file's time resolution;
#     that is consistent with a triggered camera AND with DI/O #2 being an
#     internal mirror of the LED gate. Section G reports the comparison and
#     says so; it cannot settle it.
#   * It does not decide whether surplus video frames LEAD or TRAIL. It reports
#     the count difference; the lead/trail question needs a landmark visible in
#     both streams.
#   * The thresholds in the CONFIG block are pragmatic heuristics chosen to make
#     obvious failures loud. They are not literature-derived cut-offs. Treat
#     WARN as "go look at it", not as an automated verdict.
#
# DEPENDENCIES
#   required : numpy, pandas
#   optional : scipy (better filtering), statsmodels (RLM), matplotlib (plots),
#              opencv-python (video section). Everything degrades gracefully and
#              prints which engine it used.
# =============================================================================

import os
import csv
import sys
import time as _time

import numpy as np
import pandas as pd

# ----------------------------- CONFIG (edit me) ------------------------------
PHOTOMETRY_PATH = r"path/to/photometry.csv"   # .csv or .xlsx
VIDEO_PATH      = None                 # "" or None -> skip video section

# --- file layout (Doric export; INVARIANT for this setup, but verified in A) --
N_HEADER_ROWS = 2
COL_TIME = 0     # A  Time(s)
COL_CAM1 = 1     # B  CAM #1   -- 40 Hz photometry exposure strobe (NOT the behav. cam)
COL_EXC1 = 2     # C  EXC #1   -- isosbestic LED gate (~20 Hz)
COL_EXC2 = 3     # D  EXC #2   -- signal LED gate (~20 Hz)
COL_DIO2 = 4     # E  DI/O #2  -- assumed behavioural-camera frame TTL (~20 Hz)
COL_ISO  = 7     # H  ROI 1 | CAM 1 EXC 1 -- PFC isosbestic (405)
COL_SIG  = 8     # I  ROI 1 | CAM 1 EXC 2 -- PFC signal (465/470)
COL_FEVENTS = [9, 10, 11, 12]   # J-M  F1-F4 Event (ragged: only on event rows)

SYNC_COL     = COL_DIO2   # which digital line is treated as the camera frame TTL
COMPARE_COL  = COL_EXC1   # compared against SYNC_COL for the identity check
NOMINAL_FPS  = 20.0       # what the AVI container claims / what scoring assumes

# --- processing parameters (keep IDENTICAL across sessions you compare) ------
LOWPASS_HZ    = 3.0       # pre-fit low-pass, as in the analysis pipeline
ROBUST_METHOD = "tukey"   # "tukey" or "huber"
SLOW_HZ       = 0.01      # cutoff used ONLY to split "slow drift" vs "fast" for QC

# --- QC thresholds (heuristics, not standards) -------------------------------
STEP_BIN_S      = 1.0     # bin size for step detection
STEP_Z          = 8.0     # robust z on binned differences
STEP_MIN_FRAC   = 0.010   # and at least 1% of the local median
DEAD_WIN_S      = 5.0     # window for flat/dead-segment detection
R_WARN          = 0.30    # Pearson r below this -> isosbestic questionable
SLOPE_GAP_WARN  = 5.0     # % OLS-vs-robust slope gap above which robust matters
CLIP_WARN_PCT   = 0.10    # % of samples pinned at the extreme -> saturation

# --- video parameters --------------------------------------------------------
VIDEO_STRIDE     = 1      # 1 = every frame (needed for reliable cut detection)
VIDEO_THUMB      = 64     # frames downscaled to THUMB x THUMB grey for diffing
VIDEO_MAX_FRAMES = None   # e.g. 20000 for a quick first pass; None = whole file
FROZEN_RUN       = 2      # >=N consecutive bit-identical frames -> padding
# Cut detection. A raw frame-difference threshold does NOT work: a rat rearing
# or grooming produces differences as large as a splice does, so any threshold
# tuned to catch cuts flags hundreds of movement frames. What separates them is
# WHERE the change happened. In continuous movement the pixels the animal
# vacates and the pixels it arrives at are adjacent; in a splice the animal is
# somewhere else entirely, so those two regions are far apart. CUT_SEP_PX is
# that distance, measured on the THUMB x THUMB grid.
CUT_DIFF_PCT     = 99.5   # only consider the largest frame differences
# Heuristic treshold for detecting abrupt scene changes. This value was 
# empirically selected for the present recording setup and should be validated
# when applied to other acquisition systems.
CUT_SEP_PX       = 12.0   # vacated-vs-arrived centroid distance (of THUMB px)
CUT_FRAC_GLOBAL  = 0.50   # or: this fraction of ALL pixels changed -> scene change
EDGE_SKIP        = 5      # ignore the first/last N frames (start-up settling)

SHOW_PLOTS  = True
SAVE_REPORT = False       # try to write <session>_qc.txt next to the script
# -----------------------------------------------------------------------------


# =============================================================================
# report plumbing
# =============================================================================
class Report:
    """Collects PASS/INFO/WARN/FAIL lines so the end of the run is readable
    even if the middle scrolled past."""

    LEVELS = ("PASS", "INFO", "WARN", "FAIL")

    def __init__(self):
        self.items = []
        self.lines = []

    def say(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def head(self, title):
        self.say("")
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)

    def add(self, level, section, msg):
        assert level in self.LEVELS
        self.items.append((level, section, msg))
        self.say(f"  [{level}] {msg}")

    def verdict(self):
        self.head("J. VERDICT")
        fails = [i for i in self.items if i[0] == "FAIL"]
        warns = [i for i in self.items if i[0] == "WARN"]
        for lev, sec, msg in fails + warns:
            self.say(f"  [{lev}] ({sec}) {msg}")
        if not fails and not warns:
            self.say("  No FAIL or WARN raised. Nothing here blocks analysis.")
        self.say("")
        self.say(f"  FAIL: {len(fails)}   WARN: {len(warns)}   "
                 f"PASS: {sum(1 for i in self.items if i[0] == 'PASS')}")
        self.say("")
        if fails:
            self.say("  >> At least one FAIL. Do not analyse this session until")
            self.say("     the listed problem is understood or excluded.")
        elif warns:
            self.say("  >> No hard failure, but the WARNs above are things a")
            self.say("     reviewer would ask about. Inspect before proceeding.")
        else:
            self.say("  >> Session looks internally consistent. Note that this")
            self.say("     script cannot verify camera<->photometry frame-lock;")
            self.say("     see section G.")


R = Report()


# =============================================================================
# small numeric helpers
# =============================================================================
try:
    from scipy.signal import butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import statsmodels.api as sm
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False


def robust_sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def lowpass(x, fs, cutoff_hz):
    """Zero-phase low-pass; falls back to a moving average without scipy."""
    if cutoff_hz is None or cutoff_hz <= 0 or x.size < 10:
        return np.asarray(x, dtype=float).copy()
    if HAVE_SCIPY and cutoff_hz < fs / 2:
        b, a = butter(2, cutoff_hz / (fs / 2), btype="low")
        return filtfilt(b, a, x)
    win = max(1, int(round(fs / cutoff_hz)))
    win = min(win, max(1, x.size // 2))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def irls(x, y, method="tukey", n_iter=50, tol=1e-9):
    """Robust straight-line fit y ~ a*x + b (Tukey biweight or Huber), pure
    NumPy. Same routine as the analysis script so the QC dF/F matches it."""
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = 4.685 if method == "tukey" else 1.345
    for _ in range(n_iter):
        resid = y - X @ beta
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if s <= 0:
            break
        u = resid / (s * c)
        if method == "tukey":
            w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        else:
            w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X + 1e-12 * np.eye(2), XtW @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta[0], beta[1]


def robust_fit(x, y, method="tukey"):
    if HAVE_STATSMODELS:
        norm = (sm.robust.norms.TukeyBiweight() if method == "tukey"
                else sm.robust.norms.HuberT())
        res = sm.RLM(y, sm.add_constant(x), M=norm).fit()
        return float(res.params[1]), float(res.params[0]), "statsmodels RLM"
    a, b = irls(x, y, method=method)
    return a, b, "numpy IRLS"


# =============================================================================
# A. LOAD + STRUCTURE
# =============================================================================
def _max_fields(path, skip_lines=0, block=1 << 22):
    """Widest row, in comma-separated fields, at or after line `skip_lines`.
    Byte-level scan: ~1 s on a 180 MB / 5M-row export.

    Header and data widths must be measured SEPARATELY. In real exports the
    two header rows are often WIDER than the data rows (trailing separators for
    F-event columns that are never written unless an event occurs). Passing a
    header-derived width to read_csv as `names` while also passing `usecols`
    makes the C parser raise 'Too many columns specified'."""
    mx = 0
    leftover = b""
    seen = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            lines = (leftover + buf).split(b"\n")
            leftover = lines.pop()
            for ln in lines:
                seen += 1
                if seen <= skip_lines:
                    continue
                c = ln.count(b",") + 1
                if c > mx:
                    mx = c
    if leftover.strip() and seen + 1 > skip_lines:
        mx = max(mx, leftover.count(b",") + 1)
    return mx


def _header_block_csv(path, n_header):
    """First n_header lines, as a DataFrame of strings (quoting-safe)."""
    rows = []
    with open(path, "r", newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if i >= n_header:
                break
            rows.append(row)
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, dtype=str), width


def load_session(path, n_header=N_HEADER_ROWS):
    """Returns (labels, df, header_ncol, data_ncol). df carries ONLY the needed
    columns, keyed by their ORIGINAL integer indices, coerced to float."""
    ext = os.path.splitext(path)[1].lower()
    wanted = sorted(set([COL_TIME, COL_CAM1, COL_EXC1, COL_EXC2, COL_DIO2,
                         COL_ISO, COL_SIG] + list(COL_FEVENTS)))

    if ext in (".xlsx", ".xls"):
        full = pd.read_excel(path, header=None, dtype=str)
        ncol = full.shape[1]
        labels = _combine_headers(full.iloc[:n_header], ncol)
        use = [c for c in wanted if c < ncol]
        df = full.iloc[n_header:, use].apply(pd.to_numeric, errors="coerce")
        df.columns = use
        return labels, df.reset_index(drop=True), ncol, ncol

    head_block, head_ncol = _header_block_csv(path, n_header)
    data_ncol = _max_fields(path, skip_lines=n_header)
    labels = _combine_headers(head_block, max(head_ncol, data_ncol))
    use = [c for c in wanted if c < data_ncol]

    # time needs full precision; the rest is fine in float32 (halves memory on
    # multi-million-row files).
    dtypes = {c: np.float32 for c in use}
    dtypes[COL_TIME] = np.float64

    # Three tiers, cheapest first. Tier 2 handles a stray non-numeric cell;
    # tier 3 handles a row wider than the scan suggested (usecols is what the
    # C parser refuses to reconcile on ragged files, so it goes last).
    try:
        df = pd.read_csv(path, header=None, names=range(data_ncol),
                         skiprows=n_header, usecols=use, dtype=dtypes,
                         low_memory=False)
    except (ValueError, pd.errors.ParserError) as e1:
        R.say(f"  [info] numeric read failed ({type(e1).__name__}: "
              f"{str(e1)[:60]}); retrying as text")
        try:
            df = pd.read_csv(path, header=None, names=range(data_ncol),
                             skiprows=n_header, usecols=use, dtype=str,
                             low_memory=False)
            df = df.apply(pd.to_numeric, errors="coerce")
        except (ValueError, pd.errors.ParserError) as e2:
            R.say(f"  [info] retry failed ({type(e2).__name__}); reading all "
                  f"columns (slower, more memory)")
            df = pd.read_csv(path, header=None, names=range(data_ncol),
                             skiprows=n_header, dtype=str, low_memory=False)
            df = df[[c for c in use if c in df.columns]].apply(
                pd.to_numeric, errors="coerce")
    return labels, df.reset_index(drop=True), head_ncol, data_ncol


def _combine_headers(block, ncol):
    labels = []
    for col in range(ncol):
        parts = []
        for r in range(block.shape[0]):
            v = str(block.iloc[r, col]).strip() if col < block.shape[1] else ""
            if v not in ("", "nan", "-", "---"):
                parts.append(v)
        labels.append(" | ".join(parts) if parts else f"col_{col}")
    return labels


def section_A(path):
    R.head("A. FILE STRUCTURE & COLUMN MAP")
    t0 = _time.time()
    labels, df, head_ncol, data_ncol = load_session(path)
    R.say(f"  file            : {os.path.abspath(path)}")
    R.say(f"  size            : {os.path.getsize(path)/1e6:,.1f} MB")
    R.say(f"  columns: header rows {head_ncol}, widest data row {data_ncol}")
    if data_ncol < head_ncol:
        R.say(f"    (header is wider than the data -- normal for this export: the "
              f"trailing\n     F-event columns are only written on rows that carry "
              f"an event)")
    R.say(f"  data rows       : {len(df):,}   (loaded in {_time.time()-t0:.1f} s)")
    R.say("")
    R.say("  resolved header labels for the columns this script uses:")
    for c in sorted(df.columns):
        R.say(f"    col {c:<3d} -> {labels[c]}")

    # The one assumption worth machine-checking: that ISO/SIG are the ROI 1
    # EXC1/EXC2 pair and not ROI 0.
    def _lab(c):
        return labels[c].upper() if c < len(labels) else ""

    missing = [c for c in (COL_TIME, COL_CAM1, COL_EXC1, COL_EXC2, COL_DIO2,
                          COL_ISO, COL_SIG) if c not in df.columns]
    if missing:
        R.add("FAIL", "A", f"required column(s) {missing} are not present in the "
                           f"data rows (widest data row has {data_ncol} fields). "
                           "Check the column map.")
    missing_ev = [c for c in COL_FEVENTS if c not in df.columns]
    if missing_ev:
        R.say(f"  note: F-event column(s) {missing_ev} absent from the data rows "
              f"-- no events were written to them in this session")

    ok_iso = ("ROI 1" in _lab(COL_ISO)) and ("EXC 1" in _lab(COL_ISO))
    ok_sig = ("ROI 1" in _lab(COL_SIG)) and ("EXC 2" in _lab(COL_SIG))
    if ok_iso and ok_sig:
        R.add("PASS", "A", "column map matches ROI 1 / EXC1(iso) + EXC2(signal)")
    else:
        R.add("FAIL", "A",
              f"column map does NOT match the expected labels "
              f"(col {COL_ISO}='{labels[COL_ISO]}', col {COL_SIG}='{labels[COL_SIG]}'). "
              f"Fix COL_ISO/COL_SIG before trusting anything below.")
    return labels, df


# =============================================================================
# B. MASTER TIMEBASE
# =============================================================================
def _stream_stats(t_stream):
    """Monotonicity + spacing of ONE acquisition stream."""
    if len(t_stream) < 3:
        return None
    d = np.diff(t_stream)
    med = np.median(d)
    return {"n": len(t_stream), "med": med,
            "back": int(np.sum(d < 0)), "dup": int(np.sum(d == 0)),
            "gaps": int(np.sum(d > 1.5 * med)),
            "maxd": d.max(), "d": d, "t": t_stream}


def section_B(df):
    """Timebase integrity, tested PER STREAM.

    This export does not write one row per instant. Photometry rows (ISO/SIG,
    ~25 ms apart) and digital rows (CAM/EXC/DI-O, ~2 ms apart) are written in
    separate alternating blocks, so consecutive ROWS are not in time order even
    when every stream is individually perfect. Differencing the raw time column
    therefore reports backward jumps, duplicates and gaps that are artefacts of
    the row ordering and mean nothing.

    What actually matters is whether each stream the analysis indexes into is
    monotonic and evenly spaced, because every downstream step (de-interleaving,
    np.interp, edge extraction) works within a stream, never across rows. That
    is what this checks."""
    R.head("B. TIMEBASE INTEGRITY (tested per stream, not per row)")
    t = df[COL_TIME].to_numpy(dtype=float)
    finite = np.isfinite(t)
    if finite.sum() < len(t):
        R.add("WARN", "B", f"{len(t)-finite.sum():,} rows have no timestamp")
    t_all = t[finite]

    R.say(f"  span            : {t_all[0]:.4f} - {t_all[-1]:.4f} s "
          f"({(t_all[-1]-t_all[0])/60:.2f} min)")
    R.say(f"  rows            : {len(t_all):,}  "
          f"({len(t_all)/(t_all[-1]-t_all[0]):.1f} rows/s)")

    # --- identify the streams -------------------------------------------------
    streams = []
    for col, lab in ((COL_ISO, "isosbestic rows"), (COL_SIG, "signal rows")):
        if col in df.columns:
            st = _stream_stats(t[df[col].notna().to_numpy() & finite])
            if st:
                streams.append((lab, st, True))     # True = analysis depends on it
    dig_cols = [c for c in (COL_CAM1, COL_EXC1, COL_EXC2, COL_DIO2)
                if c in df.columns]
    if dig_cols:
        m = np.zeros(len(df), dtype=bool)
        for c in dig_cols:
            m |= df[c].notna().to_numpy()
        st = _stream_stats(t[m & finite])
        if st:
            streams.append(("digital rows", st, True))

    R.say("")
    R.say(f"  {'stream':<18s} {'rows':>10s} {'median dt':>11s} {'max dt':>10s} "
          f"{'backward':>9s} {'duplicate':>10s} {'gaps':>7s}")
    for lab, st, _ in streams:
        R.say(f"  {lab:<18s} {st['n']:>10,} {st['med']*1000:>9.4f} ms "
              f"{st['maxd']*1000:>8.4f} ms {st['back']:>9,} {st['dup']:>10,} "
              f"{st['gaps']:>7,}")

    # --- verdicts, per stream -------------------------------------------------
    bad = [(lab, st) for lab, st, _ in streams if st["back"] or st["dup"]]
    for lab, st in bad:
        R.add("FAIL", "B", f"{lab}: {st['back']:,} backward jump(s), "
                           f"{st['dup']:,} duplicate timestamp(s) WITHIN the "
                           "stream -- this stream's timestamps are not usable")
    for lab, st, _ in streams:
        if st["gaps"] and not (st["back"] or st["dup"]):
            idx = np.where(st["d"] > 1.5 * st["med"])[0]
            R.say("")
            R.say(f"  {lab}: {st['gaps']:,} interval(s) > 1.5x this stream's "
                  f"median ({st['med']*1000:.3f} ms)")
            for i in idx[:5]:
                R.say(f"    t={st['t'][i]:.3f} s : {st['d'][i]*1000:.2f} ms "
                      f"(~{int(round(st['d'][i]/st['med']))-1} sample(s) missing)")
            if len(idx) > 5:
                R.say(f"    ... and {len(idx)-5} more")
            R.add("WARN", "B", f"{lab}: {st['gaps']:,} gap(s) -- samples were "
                               "dropped; check they do not sit on events")
    if not bad and not any(st["gaps"] for _, st, _ in streams):
        R.add("PASS", "B", "every stream is monotonic, evenly spaced, no gaps")

    # --- row ORDER: reported, never a verdict --------------------------------
    dt = np.diff(t_all)
    n_back, n_dup = int(np.sum(dt < 0)), int(np.sum(dt == 0))
    R.say("")
    R.say("  row-order note (informational -- nothing downstream uses row order):")
    R.say(f"    consecutive-row dt: {n_back:,} negative, {n_dup:,} zero, "
          f"median {np.median(dt)*1000:.4f} ms")
    if (n_back or n_dup) and not bad:
        R.say("    Rows are not in time order, but every stream above is clean.")
        R.say("    That is block-buffered writing: the streams are flushed in")
        R.say("    alternating chunks. Expected for this export; not a defect,")
        R.say("    and it cannot affect de-interleaving, interpolation or edge")
        R.say("    extraction, all of which index within a stream.")
    return t_all


# =============================================================================
# C. DE-INTERLEAVING
# =============================================================================
def extract_channel(df, value_col):
    sub = df[[COL_TIME, value_col]].copy()
    sub.columns = ["time", "value"]
    sub = sub.dropna()
    return sub["time"].to_numpy(dtype=float), sub["value"].to_numpy(dtype=float)


def section_C(df):
    R.head("C. DE-INTERLEAVING (the two LEDs are NOT simultaneous)")
    iso_t, iso_v = extract_channel(df, COL_ISO)
    sig_t, sig_v = extract_channel(df, COL_SIG)

    if len(iso_t) < 10 or len(sig_t) < 10:
        R.add("FAIL", "C", f"a channel came back nearly empty "
                           f"(iso={len(iso_t)}, sig={len(sig_t)}) -- wrong columns?")
        return None

    iso_fs = 1.0 / np.median(np.diff(iso_t))
    sig_fs = 1.0 / np.median(np.diff(sig_t))
    R.say(f"  isosbestic : {len(iso_v):>9,} samples  ~{iso_fs:6.2f} Hz  "
          f"[{iso_t[0]:.4f} - {iso_t[-1]:.4f} s]")
    R.say(f"  signal     : {len(sig_v):>9,} samples  ~{sig_fs:6.2f} Hz  "
          f"[{sig_t[0]:.4f} - {sig_t[-1]:.4f} s]")

    ratio = len(iso_v) / len(sig_v)
    R.say(f"  iso/sig sample ratio : {ratio:.4f}   (expect ~1.000)")

    # strict alternation check
    merged_t = np.concatenate([iso_t, sig_t])
    lab = np.concatenate([np.zeros(len(iso_t)), np.ones(len(sig_t))])
    order = np.argsort(merged_t, kind="mergesort")
    lab = lab[order]
    n_viol = int(np.sum(np.diff(lab) == 0))
    R.say(f"  alternation violations (two samples of the same channel in a row): "
          f"{n_viol:,}")

    # signed iso -> signal offset
    j = np.searchsorted(iso_t, sig_t) - 1
    good = j >= 0
    offset = np.median(sig_t[good] - iso_t[j[good]])
    R.say(f"  median iso->signal offset : {offset*1000:.3f} ms "
          f"(expect ~half the per-channel period = {500/sig_fs:.1f} ms)")

    if abs(ratio - 1.0) > 0.05:
        R.add("WARN", "C", f"iso/sig sample counts differ by {abs(1-ratio)*100:.1f}%")
    if n_viol > 0.01 * len(sig_v):
        R.add("WARN", "C", f"{n_viol:,} alternation violations -- the interleaving "
                           "is not clean; check for dropped frames")
    if offset <= 0 or offset > 1.5 / sig_fs:
        R.add("WARN", "C", f"iso->signal offset {offset*1000:.1f} ms is not a "
                           f"plausible half-frame at {sig_fs:.1f} Hz per channel")
    else:
        R.add("PASS", "C", f"channels de-interleave cleanly, "
                           f"{sig_fs:.2f} Hz per channel, offset {offset*1000:.1f} ms")
    return iso_t, iso_v, sig_t, sig_v, sig_fs


# =============================================================================
# D. PER-CHANNEL SIGNAL QUALITY
# =============================================================================
def detect_steps(t, v, bin_s=STEP_BIN_S, z=STEP_Z, min_frac=STEP_MIN_FRAC):
    """Bin-median the trace, difference it, flag robustly-large jumps.
    Returns (times, signed amplitudes in raw units, robust z scores)."""
    if len(t) < 10:
        return np.array([]), np.array([]), np.array([])
    edges = np.arange(t[0], t[-1] + bin_s, bin_s)
    idx = np.clip(np.searchsorted(edges, t) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    med = np.full(nb, np.nan)
    order = np.argsort(idx, kind="mergesort")
    idx_s, v_s = idx[order], v[order]
    bounds = np.searchsorted(idx_s, np.arange(nb + 1))
    for b in range(nb):
        seg = v_s[bounds[b]:bounds[b + 1]]
        if seg.size:
            med[b] = np.median(seg)
    ok = np.isfinite(med)
    med_ok, ctr = med[ok], (edges[:-1] + bin_s / 2)[ok]
    d = np.diff(med_ok)
    s = robust_sd(d)
    if not np.isfinite(s) or s == 0:
        return np.array([]), np.array([]), np.array([])
    zz = d / s
    base = np.median(med_ok)
    hit = (np.abs(zz) > z) & (np.abs(d) > abs(min_frac * base))
    return ctr[1:][hit], d[hit], zz[hit]


def channel_quality(name, t, v, fs):
    R.say("")
    R.say(f"  --- {name} ---")
    vmin, vmax, vmed = np.min(v), np.max(v), np.median(v)
    R.say(f"    min / median / max : {vmin:,.2f} / {vmed:,.2f} / {vmax:,.2f}")

    # saturation / clipping: fraction of samples pinned at the extreme value
    n_hi = int(np.sum(v >= vmax - 1e-9))
    n_lo = int(np.sum(v <= vmin + 1e-9))
    pin_hi, pin_lo = 100.0 * n_hi / v.size, 100.0 * n_lo / v.size
    R.say(f"    samples at max/min : {n_hi:,} ({pin_hi:.4f}%) / "
          f"{n_lo:,} ({pin_lo:.4f}%)")
    # a genuine rail means MANY samples share the extreme value; one sample at
    # the max is just the maximum.
    if (n_hi > 20 and pin_hi > CLIP_WARN_PCT) or (n_lo > 20 and pin_lo > CLIP_WARN_PCT):
        R.add("WARN", "D", f"{name}: {max(pin_hi,pin_lo):.2f}% of samples sit at an "
                           "extreme value -- possible detector saturation/clipping")
    if np.any(v <= 0):
        R.add("WARN", "D", f"{name}: {int(np.sum(v<=0)):,} non-positive samples "
                           "(dF/F is undefined where the fitted baseline <= 0)")

    # bleaching: first vs last 60 s of the trace
    w = min(60.0, (t[-1] - t[0]) / 10)
    a = np.median(v[t <= t[0] + w])
    b = np.median(v[t >= t[-1] - w])
    R.say(f"    first {w:.0f}s median {a:,.2f} -> last {w:.0f}s median {b:,.2f} "
          f"({100*(b-a)/a:+.1f}%)")

    # noise vs drift
    hp = v - lowpass(v, fs, SLOW_HZ)
    lp = lowpass(v, fs, SLOW_HZ)
    R.say(f"    fast noise (robust SD of >{SLOW_HZ} Hz) : {robust_sd(hp):,.3f}")
    R.say(f"    slow drift (peak-to-peak of <{SLOW_HZ} Hz): "
          f"{np.ptp(lp):,.3f} ({100*np.ptp(lp)/abs(vmed):.1f}% of median)")

    # dead / flat segments
    win = max(3, int(round(DEAD_WIN_S * fs)))
    n_full = (len(v) // win) * win
    if n_full >= win:
        blocks = v[:n_full].reshape(-1, win)
        flat = np.sum(np.std(blocks, axis=1) == 0)
        if flat:
            R.add("FAIL", "D", f"{name}: {flat} block(s) of {DEAD_WIN_S:.0f} s with "
                               "zero variance -- detector dropped out")
        else:
            R.say(f"    flat {DEAD_WIN_S:.0f}s blocks : 0")

    # step-downs
    st, sa, sz = detect_steps(t, v)
    R.say(f"    abrupt steps detected : {len(st)}")
    for tt, aa, zz in list(zip(st, sa, sz))[:10]:
        R.say(f"      t={tt:8.1f} s  {aa:+,.2f} a.u. ({100*aa/vmed:+.2f}% of median, "
              f"z={zz:+.1f})")
    if len(st) > 10:
        R.say(f"      ... and {len(st)-10} more")
    return st, sa


def section_D(iso_t, iso_v, sig_t, sig_v, fs):
    R.head("D. PER-CHANNEL SIGNAL QUALITY")
    R.say(f"  (a 'step' = jump between adjacent {STEP_BIN_S:.0f} s "
          f"medians that is >{STEP_Z} robust-z AND >{STEP_MIN_FRAC*100:.1f}% "
          f"of the median)")
    channel_quality("isosbestic 405", iso_t, iso_v, fs)
    st, sa = channel_quality("signal 465/470", sig_t, sig_v, fs)
    if len(st):
        R.add("WARN", "D", f"signal: {len(st)} abrupt step(s) detected; section F "
                           "reports whether the isosbestic correction removes them")
    return st


# =============================================================================
# E. ISOSBESTIC USABILITY
# =============================================================================
def section_E(iso_t, iso_v, sig_t, sig_v, fs):
    R.head("E. IS THE ISOSBESTIC A USABLE CONTROL?")
    x = np.interp(sig_t, iso_t, iso_v)     # iso onto the signal's own timestamps
    y = sig_v

    r_raw = float(np.corrcoef(x, y)[0, 1])
    b_ols, a_ols = np.polyfit(x, y, 1)
    b_rob, a_rob, engine = robust_fit(x, y, ROBUST_METHOD)
    gap = 100 * abs(b_rob - b_ols) / (abs(b_ols) + 1e-12)

    # r on the raw traces is dominated by shared BLEACHING. The more informative
    # number for motion artefacts is r after both slow components are removed.
    xh = x - lowpass(x, fs, SLOW_HZ)
    yh = y - lowpass(y, fs, SLOW_HZ)
    r_fast = float(np.corrcoef(xh, yh)[0, 1])

    R.say(f"  paired samples        : {len(x):,} (iso interpolated onto signal times)")
    R.say(f"  Pearson r, raw        : {r_raw:+.4f}  (r^2 = {r_raw**2:.4f})")
    R.say(f"  Pearson r, >{SLOW_HZ} Hz : {r_fast:+.4f}  <- shared FAST artefact only")
    R.say(f"  OLS    : signal = {a_ols:10.2f} + {b_ols:+.4f} * iso")
    R.say(f"  Robust : signal = {a_rob:10.2f} + {b_rob:+.4f} * iso   [{engine}]")
    R.say(f"  OLS vs robust slope gap : {gap:.1f}%")
    R.say("")
    R.say("  Reading it:")
    R.say("   - raw r high but fast r ~0 -> the control only tracks bleaching;")
    R.say("     subtraction will not remove motion. Not fatal, but say so in Methods.")
    R.say("   - r <= 0 -> the control does not track the artefact. STOP.")
    R.say("   - large slope gap -> real transients bias OLS; use robust (default).")
    R.say("   - iGluSnFR caveat: 405 is not guaranteed glutamate-independent for")
    R.say("     cpGFP sensors, so part of r may be inverted real signal.")

    if not np.isfinite(r_raw) or r_raw <= 0:
        R.add("FAIL", "E", f"isosbestic does not co-vary with the signal "
                           f"(r={r_raw:+.3f}) -- regression correction is invalid")
    elif r_raw < R_WARN:
        R.add("WARN", "E", f"weak isosbestic-signal correlation (r={r_raw:+.3f})")
    else:
        R.add("PASS", "E", f"isosbestic tracks the signal (r={r_raw:+.3f}, "
                           f"fast-only r={r_fast:+.3f})")
    if gap > SLOPE_GAP_WARN:
        R.add("INFO", "E", f"OLS/robust slope gap {gap:.1f}% -- robust regression is "
                           "doing real work here, keep it")
    return x, y, (a_rob, b_rob)


# =============================================================================
# F. dF/F PREVIEW
# =============================================================================
def section_F(t, iso_on_t, sig, fs, steps):
    R.head("F. dF/F PREVIEW (does the correction flatten the steps?)")
    iso_f = lowpass(iso_on_t, fs, LOWPASS_HZ)
    sig_f = lowpass(sig, fs, LOWPASS_HZ)
    a, b, engine = robust_fit(iso_f, sig_f, ROBUST_METHOD)
    fitted = a * iso_f + b
    R.say(f"  low-pass {LOWPASS_HZ} Hz, robust fit [{engine}]: "
          f"slope={a:.4f} intercept={b:.2f}")
    if np.any(fitted <= 0):
        R.add("FAIL", "F", f"fitted baseline crosses zero at "
                           f"{int(np.sum(fitted<=0)):,} samples -- dF/F undefined there")
        fitted = np.where(fitted <= 0, np.nan, fitted)
    dff = (sig_f - fitted) / fitted

    R.say(f"  dF/F  median {100*np.nanmedian(dff):+.3f} %  |  robust SD "
          f"{100*robust_sd(dff):.3f} %  |  range "
          f"{100*np.nanmin(dff):+.2f} .. {100*np.nanmax(dff):+.2f} %")

    if len(steps):
        R.say("")
        R.say("  step-by-step, raw signal vs corrected dF/F "
              "(10 s median before vs after):")
        for tt in steps[:10]:
            pre = (t >= tt - 10) & (t < tt)
            post = (t > tt) & (t <= tt + 10)
            if pre.sum() < 5 or post.sum() < 5:
                continue
            raw_step = (np.median(sig[post]) - np.median(sig[pre])) / np.median(sig[pre])
            dff_step = np.nanmedian(dff[post]) - np.nanmedian(dff[pre])
            if abs(raw_step) < 0.005:
                tail = "   (step too small to judge)"
            else:
                tail = f"   (removed {(1-abs(dff_step)/abs(raw_step))*100:5.1f}%)"
            R.say(f"    t={tt:8.1f} s : raw {100*raw_step:+7.2f} %  ->  "
                  f"dF/F {100*dff_step:+7.2f} %{tail}")
        R.say("")
        R.say("  A step that survives correction is NOT a shared artefact:")
        R.say("  differential bleaching, a fibre/patch-cord shift, or a real")
        R.say("  change. Single-isosbestic subtraction cannot separate these.")
    return dff


# =============================================================================
# G. DIGITAL LINES
# =============================================================================
def rising_edges(df, col, t):
    x = df[col].ffill().fillna(0).to_numpy(dtype=float)
    hi = x > 0.5
    idx = np.flatnonzero(hi[1:] & ~hi[:-1]) + 1
    return t[idx] if len(t) == len(x) else df[COL_TIME].to_numpy(dtype=float)[idx]


def describe_train(name, ft):
    if len(ft) < 3:
        R.say(f"  {name:<22s}: {len(ft)} edges -- too few to characterise")
        return None
    d = np.diff(ft)
    med = np.median(d)
    mean_int = (ft[-1] - ft[0]) / (len(ft) - 1)
    n_gap = int(np.sum(d > 1.5 * med))
    R.say(f"  {name:<22s}: {len(ft):>9,} edges | {ft[0]:.4f} - {ft[-1]:.4f} s | "
          f"median {med*1000:.4f} ms | TRUE {1/mean_int:.5f} Hz | gaps {n_gap}")
    if n_gap:
        idx = np.where(d > 1.5 * med)[0]
        for i in idx[:5]:
            R.say(f"      gap after edge {i:,} (t={ft[i]:.3f} s): "
                  f"{d[i]*1000:.1f} ms ~ {int(round(d[i]/med))-1} missing pulse(s)")
        if len(idx) > 5:
            R.say(f"      ... and {len(idx)-5} more")
    return {"n": len(ft), "med": med, "mean_int": mean_int, "gaps": n_gap}


def section_G(df, t):
    R.head("G. DIGITAL / TTL LINES")
    trains = {}
    for name, col in (("CAM #1 (exposure)", COL_CAM1),
                      ("EXC #1 (iso gate)", COL_EXC1),
                      ("EXC #2 (sig gate)", COL_EXC2),
                      ("DI/O #2 (cam TTL)", COL_DIO2)):
        if col not in df.columns:
            R.say(f"  {name:<22s}: column {col} not in file")
            continue
        ft = rising_edges(df, col, t)
        trains[col] = (ft, describe_train(name, ft))

    sync = trains.get(SYNC_COL, (np.array([]), None))
    ft_sync, info = sync
    if info is None:
        R.add("FAIL", "G", f"no usable edge train on the sync column {SYNC_COL}")
        return None

    # A gap breaks frame index == edge index for every later frame.
    if info["gaps"]:
        R.add("FAIL", "G", f"{info['gaps']} gap(s) in the sync train -- frame index N "
                           "!= edge index N after the first gap, so every later onset "
                           "would be mapped to the wrong pulse")
    else:
        R.add("PASS", "G", f"sync train has no gaps ({info['n']:,} edges, "
                           f"index N == frame N is safe)")

    true_fps = 1.0 / info["mean_int"]
    drift = (true_fps - NOMINAL_FPS) / NOMINAL_FPS
    total_drift = drift * (ft_sync[-1] - ft_sync[0])
    R.say("")
    R.say(f"  nominal fps {NOMINAL_FPS:.3f} vs TRUE {true_fps:.5f} Hz "
          f"-> frame/{NOMINAL_FPS:.0f} accumulates {total_drift:+.2f} s by the end")
    if abs(total_drift) > 0.5:
        R.add("WARN", "G", f"frame/{NOMINAL_FPS:.0f} would misplace late events by "
                           f"{total_drift:+.2f} s -- always look frames up in the "
                           "pulse train (htr_sync.corrected_onsets)")

    # identity check: is the "camera TTL" independent of the LED gate?
    cmp_ = trains.get(COMPARE_COL, (np.array([]), None))
    ft_cmp = cmp_[0]
    R.say("")
    R.say("  TRAIN IDENTITY (is the sync line independent of the LED gate?)")
    if len(ft_cmp) == len(ft_sync) and len(ft_sync):
        dmax = float(np.abs(ft_sync - ft_cmp).max())
        R.say(f"    col {SYNC_COL} vs col {COMPARE_COL}: same edge count, "
              f"max |dt| = {dmax*1000:.4f} ms")
        if dmax < 2.5 * np.median(np.diff(t)):
            R.add("WARN", "G", f"sync line and LED gate are indistinguishable "
                               f"({dmax*1000:.2f} ms apart = ~1 master sample). "
                               "Camera frame-lock is ASSUMED, not demonstrated; "
                               "settle it with a landmark visible in both streams")
        else:
            R.add("PASS", "G", "sync line differs from the LED gate -> plausibly a "
                               "real, independent camera input")
    else:
        R.say(f"    different edge counts ({len(ft_sync):,} vs {len(ft_cmp):,}) "
              "-> independent lines")
        R.add("PASS", "G", "sync line is independent of the LED gate")
    return ft_sync


# =============================================================================
# H. F-EVENT COLUMNS
# =============================================================================
def section_H(df, t):
    R.head("H. F1-F4 EVENT COLUMNS")
    any_found = False
    for c in COL_FEVENTS:
        if c not in df.columns:
            R.say(f"  col {c}: not present in file")
            continue
        v = df[c]
        n = int(v.notna().sum())
        if n == 0:
            R.say(f"  col {c}: empty")
            continue
        any_found = True
        idx = np.flatnonzero(v.notna().to_numpy())
        R.say(f"  col {c}: {n:,} entries | first t={t[idx[0]]:.2f} s | "
              f"last t={t[idx[-1]]:.2f} s | values "
              f"{[float(u) for u in sorted(pd.unique(v.dropna()))[:5]]}")
    if not any_found:
        R.say("  no F-event markers in this file (fine if events come from the "
              "scoring sheet instead)")


# =============================================================================
# I. VIDEO INTEGRITY
# =============================================================================
def section_I(video_path, ft_sync):
    R.head("I. VIDEO INTEGRITY")
    if not video_path:
        R.say("  VIDEO_PATH empty -- skipped.")
        return
    if not os.path.exists(video_path):
        R.add("WARN", "I", f"video not found: {video_path}")
        return
    try:
        import cv2
    except ImportError:
        R.add("WARN", "I", "opencv-python not installed (py -m pip install "
                           "opencv-python) -- video section skipped")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        R.add("FAIL", "I", f"OpenCV cannot open {video_path}")
        return

    meta_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    R.say(f"  file            : {os.path.abspath(video_path)}")
    R.say(f"  metadata        : {meta_n:,} frames @ {meta_fps:.6f} fps "
          f"({meta_n/meta_fps if meta_fps else float('nan'):,.2f} s)")
    R.say(f"  decoding every {VIDEO_STRIDE} frame(s) at {VIDEO_THUMB}x{VIDEO_THUMB} "
          f"grey -- this is the slow part, be patient")

    diffs, brights, dups, seps, fracs = [], [], [], [], []
    grid_y, grid_x = np.mgrid[0:VIDEO_THUMB, 0:VIDEO_THUMB]
    prev = prev_full = None
    n_dec = 0
    t0 = _time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_dec += 1
        if VIDEO_MAX_FRAMES and n_dec >= VIDEO_MAX_FRAMES:
            break
        if (n_dec - 1) % VIDEO_STRIDE:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(gray, (VIDEO_THUMB, VIDEO_THUMB),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        brights.append(float(g.mean()))
        if prev is not None:
            signed = g - prev
            absd = np.abs(signed)
            diffs.append(float(absd.mean()))
            fracs.append(float((absd > 25).mean()))
            # centroid of pixels that got BRIGHTER (animal arrived) vs DARKER
            # (animal left). Adjacent for real movement, far apart for a splice.
            wp = np.clip(signed, 0, None)
            wn = np.clip(-signed, 0, None)
            sp, sn = wp.sum(), wn.sum()
            if sp > 1e-6 and sn > 1e-6:
                cpx, cpy = (wp * grid_x).sum() / sp, (wp * grid_y).sum() / sp
                cnx, cny = (wn * grid_x).sum() / sn, (wn * grid_y).sum() / sn
                seps.append(float(np.hypot(cpx - cnx, cpy - cny)))
            else:
                seps.append(0.0)
            # duplicate test at FULL resolution: with real sensor noise two
            # frames are essentially never bit-identical unless the encoder
            # repeated one. Doing this on the thumbnail would false-positive
            # every time the animal sits still.
            dups.append(bool(np.array_equal(gray, prev_full)))
        prev, prev_full = g, gray
        if n_dec % 20000 == 0:
            R.say(f"    ... {n_dec:,} frames ({_time.time()-t0:.0f} s)")
    cap.release()

    d = np.asarray(diffs)
    br = np.asarray(brights)
    R.say(f"  decoded         : {n_dec:,} frames in {_time.time()-t0:.0f} s")
    if meta_n and abs(n_dec - meta_n) > 2:
        R.add("WARN", "I", f"metadata claims {meta_n:,} frames, decoding yields "
                           f"{n_dec:,} ({n_dec-meta_n:+,}). Trust the decoded count; "
                           "AVI headers are frequently wrong")
    if d.size < 10:
        R.add("WARN", "I", "too few frames decoded to test continuity")
        return

    sep = np.asarray(seps)
    frac = np.asarray(fracs)
    med = np.median(d)
    R.say(f"  frame-to-frame difference: median {med:.3f}, "
          f"99.9th pct {np.percentile(d, 99.9):.3f} "
          f"(grey levels, {VIDEO_THUMB}x{VIDEO_THUMB})")
    R.say(f"  largest differences (informational -- these are normally just "
          f"vigorous movement):")
    for i in np.argsort(d)[-5:][::-1]:
        R.say(f"    frame {(i+1)*VIDEO_STRIDE:>8,}  diff={d[i]:6.2f}  "
              f"displacement={sep[i]:5.1f} px  changed={100*frac[i]:5.2f}% of pixels")

    # --- CUT CANDIDATES ---------------------------------------------------
    # Two independent signatures, either one qualifies:
    #   (a) the animal jumps: large difference AND the vacated and arrived
    #       regions are far apart (impossible in one 50 ms frame interval);
    #   (b) the whole scene changes: most pixels change at once.
    big = d > np.percentile(d, CUT_DIFF_PCT)
    is_cut = big & ((sep > CUT_SEP_PX) | (frac > CUT_FRAC_GLOBAL))
    is_cut[:EDGE_SKIP] = False
    is_cut[max(0, len(is_cut) - EDGE_SKIP):] = False
    cut_i = np.flatnonzero(is_cut)
    R.say(f"  cut candidates (top {100-CUT_DIFF_PCT:.1f}% of differences AND "
          f"displacement >{CUT_SEP_PX:.0f} px or >{CUT_FRAC_GLOBAL*100:.0f}% of "
          f"pixels changed): {len(cut_i)}")
    for i in cut_i[:20]:
        f_idx = (i + 1) * VIDEO_STRIDE
        line = (f"    frame {f_idx:>8,}  diff={d[i]:6.2f}  "
                f"displacement={sep[i]:5.1f} px  changed={100*frac[i]:5.2f}%")
        if ft_sync is not None and f_idx < len(ft_sync):
            line += f"  ~photometry t={ft_sync[f_idx]:9.2f} s"
        R.say(line)
    if len(cut_i) > 20:
        R.say(f"    ... and {len(cut_i)-20} more")
    if len(cut_i):
        R.add("WARN", "I", f"{len(cut_i)} candidate splice(s) -- OPEN THE VIDEO AT "
                           "THESE FRAMES. If a block of frames was removed, frame "
                           "index stops matching TTL index after it and every later "
                           "onset is wrong")
    else:
        R.add("PASS", "I", "no frame-to-frame change shows the signature of a "
                           "splice (large jumps present are consistent with animal "
                           "movement)")

    # first/last frames, reported separately: a big change here is start-up
    # settling (auto-exposure, gain), not a mid-recording cut
    if len(d) > 2 * EDGE_SKIP:
        e = max(d[:EDGE_SKIP].max(), d[-EDGE_SKIP:].max())
        if e > np.percentile(d, 99.9):
            R.say(f"  note: a large change sits within {EDGE_SKIP} frames of the "
                  f"start/end (diff={e:.2f}) -- normally camera settling, not a cut")

    # --- FROZEN / DUPLICATED FRAMES: encoders pad dropped frames by repeating
    #     the previous one. Same consequence: index drift.
    dup = np.asarray(dups, dtype=bool)  # bit-identical at full resolution
    n_dup = int(dup.sum())
    runs = []
    i = 0
    while i < dup.size:
        if dup[i]:
            j = i
            while j + 1 < dup.size and dup[j + 1]:
                j += 1
            if (j - i + 1) >= FROZEN_RUN:
                runs.append((i, j - i + 1))
            i = j + 1
        else:
            i += 1
    R.say(f"  bit-identical consecutive frames (full resolution): {n_dup} "
          f"(runs of >={FROZEN_RUN}: {len(runs)})")
    for i, ln in runs[:10]:
        R.say(f"    frames {(i+1)*VIDEO_STRIDE:,} .. "
              f"{(i+ln)*VIDEO_STRIDE:,}  ({ln} repeats)")
    if len(runs) > 10:
        R.say(f"    ... and {len(runs)-10} more runs")
    if runs:
        R.add("WARN", "I", f"{len(runs)} run(s) of duplicated frames "
                           f"({sum(l for _, l in runs)} frames total) -- typical of an "
                           "encoder padding dropped frames. Video frame N then stops "
                           "corresponding to TTL pulse N")
    elif n_dup:
        R.add("INFO", "I", f"{n_dup} isolated identical frame pairs -- expected when "
                           "the animal is completely still")
    still = 100.0 * np.mean(d < 0.1 * med)
    R.say(f"  low-motion frames (<10% of median diff): {still:.1f}% "
          "(informational: immobility, not an error)")

    # --- black / blank frames
    dark = np.flatnonzero(br < 5)
    if len(dark):
        R.add("WARN", "I", f"{len(dark)} essentially black frame(s) "
                           f"(first at frame {dark[0]*VIDEO_STRIDE:,})")

    # --- reconciliation against the TTL train
    if ft_sync is not None and len(ft_sync):
        extra = n_dec - len(ft_sync)
        R.say("")
        R.say(f"  decoded frames {n_dec:,} - sync edges {len(ft_sync):,} = {extra:+,} "
              f"({abs(extra)/NOMINAL_FPS:.2f} s unaccounted)")
        if extra == 0:
            R.add("PASS", "I", "video frame count matches the sync train exactly")
        elif abs(extra) <= 2:
            R.add("INFO", "I", f"{extra:+d} frame(s) difference -- boundary effect, "
                               "inspect the first/last frames before fixing an offset")
        else:
            R.add("WARN", "I", f"{extra:+,} frames vs sync edges. Whether they LEAD or "
                               f"TRAIL decides everything: trailing is harmless, "
                               f"leading shifts every index by {abs(extra)} frames "
                               f"({abs(extra)/NOMINAL_FPS:.2f} s). This file cannot "
                               "tell you which -- use a landmark in both streams")
    return d


# =============================================================================
# PLOTS
# =============================================================================
def make_plots(tag, iso_t, iso_v, sig_t, sig_v, t, dff, ft_sync, vdiff):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        R.say("\n[info] matplotlib not available -- no figures")
        return
    n = 3 + (1 if vdiff is not None and len(vdiff) else 0)
    fig, ax = plt.subplots(n, 1, figsize=(12, 2.6 * n))
    ax = np.atleast_1d(ax)

    ax[0].plot(iso_t, iso_v, lw=0.4, color="tab:blue", label="isosbestic 405")
    ax[0].plot(sig_t, sig_v, lw=0.4, color="tab:green", label="signal 465/470")
    ax[0].set_ylabel("a.u.")
    ax[0].set_title(f"[{tag}] raw traces")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, 100 * dff, lw=0.4, color="tab:purple")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_ylabel("dF/F (%)")
    ax[1].set_title("corrected dF/F (robust iso regression)")

    if ft_sync is not None and len(ft_sync) > 2:
        d = np.diff(ft_sync) * 1000
        ax[2].plot(ft_sync[1:], d, lw=0.3, color="k")
        ax[2].set_ylabel("TTL interval (ms)")
        ax[2].set_title("sync-pulse intervals (flat line = no dropped frames)")
    ax[2].set_xlabel("time (s)")

    if n == 4:
        ax[3].plot(np.arange(len(vdiff)) * VIDEO_STRIDE, vdiff, lw=0.3, color="tab:red")
        ax[3].set_ylabel("frame diff")
        ax[3].set_xlabel("video frame")
        ax[3].set_title("frame-to-frame image difference (spikes = candidate cuts)")

    fig.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    tag = os.path.splitext(os.path.basename(PHOTOMETRY_PATH))[0]
    R.say("#" * 78)
    R.say(f"# SESSION QC : {tag}")
    R.say(f"# scipy={HAVE_SCIPY}  statsmodels={HAVE_STATSMODELS}")
    R.say("#" * 78)

    labels, df = section_A(PHOTOMETRY_PATH)
    t_master = section_B(df)

    deint = section_C(df)
    dff = vdiff = None
    if deint is None:
        R.verdict()
        return
    iso_t, iso_v, sig_t, sig_v, fs = deint

    steps = section_D(iso_t, iso_v, sig_t, sig_v, fs)
    x, y, _ = section_E(iso_t, iso_v, sig_t, sig_v, fs)
    dff = section_F(sig_t, x, y, fs, steps)

    ft_sync = section_G(df, df[COL_TIME].to_numpy(dtype=float))
    section_H(df, df[COL_TIME].to_numpy(dtype=float))
    vdiff = section_I(VIDEO_PATH, ft_sync)

    R.verdict()

    if SAVE_REPORT:
        out = os.path.abspath(f"qc_{tag}.txt")
        try:
            with open(out, "w") as fh:
                fh.write("\n".join(R.lines))
            print(f"\nreport -> {out}")
        except OSError as e:
            print(f"\n[warn] could not write report ({e})")

    if SHOW_PLOTS:
        make_plots(tag, iso_t, iso_v, sig_t, sig_v, sig_t, dff, ft_sync, vdiff)


if __name__ == "__main__":
    main()