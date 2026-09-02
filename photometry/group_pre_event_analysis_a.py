# =============================================================================
# group_peri_event_analysis.py
# -----------------------------------------------------------------------------
# PURPOSE
#   Across-animal, across-condition peri-event analysis of PFC iGluSnFR dF/F
#   around head-twitch-response (HTR) onsets, including fixed-window mean,
#   pre-twitch peak amplitude/latency, and signed + positive AUC.
#
#   This is the GROUP-LEVEL companion to pfc_peri_event_analysis.py. The
#   per-session pipeline is byte-for-byte the same arithmetic (same loader,
#   same 3 Hz low-pass, same Tukey-biweight robust iso regression, same Lerner
#   dF/F, same baseline [-5,-2] s, same epoch grid). The only thing that changes
#   is the UNIT OF ANALYSIS.
#
#     per-session script : unit = one twitch   -> pseudoreplication
#     this script        : unit = one ANIMAL   -> legitimate within-subject test
#
#   Aggregation chain (each step is an unweighted mean of the level below):
#     twitches  ->  session mean  ->  animal x condition cell  ->  condition
#   An animal with 40 twitches and an animal with 4 therefore count equally at
#   the group level. That is deliberate: it is what makes n = 5, not n = 200.
#   The per-cell twitch count is carried through and printed, because a cell
#   built on 2 twitches is a much noisier estimate than one built on 30 and you
#   should be able to see that.
#
#   Design assumed: 5 animals x 4 conditions (quipazine alone; quipazine + DN1;
#   quipazine + DN13/DN1 DN13_1; quipazine + DN13/DN1 DN13_2), fully within-subject.
#
# -----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES *NOT* FIX
#
#   * SELECTION ON THE EVENT. This is the largest interpretive problem in the
#     whole design and no code can remove it. If the nanobody changes HTR RATE
#     (which is the point of an mGlu2/3 PAM), then the twitches that survive in
#     the nanobody conditions are not a random subsample of the twitches in the
#     quipazine-alone condition. Any condition difference in peri-twitch
#     glutamate is then confounded with "which twitches still occur". The
#     comparison is conditional on a twitch having happened. Report HTR COUNTS
#     per animal x condition (this script prints and saves them) alongside the
#     photometry result; a rate effect with no peri-event effect is a perfectly
#     coherent - and arguably more likely - outcome.
#
#   * CROSS-SESSION SCALE. The three conditions are separate recordings
#     Fibre coupling, expression, bleaching state and
#     baseline autofluorescence all differ between them, so ABSOLUTE dF/F is not
#     comparable across conditions or animals. This is why the default group
#     metric is baseline-normalised (see NORMALIZE below) and why the absolute
#     baseline level is reported but never tested across conditions.
#
#   * THE MOTION CONFOUND, if RESPONSE_WIN straddles onset. Handled here by
#     epoching the ISOSBESTIC over the identical windows and running the
#     identical group statistics on it (section 7 / Fig 3). Read that figure
#     before you read the signal figure. If the iso shows a deflection with the
#     same latency and sign structure, the "response" is movement.
#
#   * CLOCK ALIGNMENT. Inherited unchanged from htr_sync. FRAME_OFFSET is set
#     PER SESSION in the manifest because it was never validated and was
#     reported to differ between recordings. An offset error is a systematic
#     time shift of every onset in that session, which at a ~1 s window is not
#     a small error.
#
# -----------------------------------------------------------------------------
# ONE DELIBERATE DIFFERENCE FROM THE PER-SESSION SCRIPTS (read this)
#   Exclusions use a single span, [BASELINE_WIN[0], max(end of any window)],
#   so that the SAME set of twitches is kept for every measurement window.
#   Otherwise the pre-twitch and peri-onset numbers would come from different
#   twitch sets and could not be compared. With the peri-onset window included
#   the span is [-5, +0.75] s, which is wider than the [-5, -0.5] s used by the
#   older pre-twitch-only script. A few twitches that survived there will be
#   dropped here. That is expected; it is not a bug, and the dropped ones are
#   printed with their reason.
# =============================================================================

import os
import csv
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- sync layer (unchanged, from 4_htr_sync.py) ------------------------------
from htr_sync import corrected_onsets
try:
    from htr_sync import report_sync
    HAVE_REPORT_SYNC = True
except Exception:
    HAVE_REPORT_SYNC = False

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

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY_STATS = True
except Exception:
    HAVE_SCIPY_STATS = False


# =============================================================================
# CONFIG A -- THE MANIFEST (this is the part you edit per experiment)
# =============================================================================
# One entry per RECORDING. animal + condition define the cell; if an animal has
# two recordings in the same condition they are averaged (session means, equally
# weighted). Paths may be absolute or relative to DATA_DIR.
#
# frame_offset: passed straight to htr_sync.corrected_onsets. 0 = surplus video
# frames trailing. Set per session; it is still unvalidated.

DATA_DIR = r"."          # e.g. r"C:\Users\...\photometry"; "." = same folder

# Easy animal-level switch. Put animal IDs in this set to skip ALL sessions from
# those animals without deleting anything from the manifest. Use set() to include all.
# EXCLUDE_ANIMALS = set()   # <- uncomment and fill this to exclude one animal

CONDITIONS = ["QUIP", "QUIP_DN1", "QUIP_DN13_1", "QUIP_DN13_2"]  # fixes order everywhere
CONDITION_LABELS = {                                  # for figures only
    "QUIP":        "quipazine",
    "QUIP_DN1":    "quipazine + DN1",
    "QUIP_DN13_1": "quipazine + DN13/DN1 (DN13_1)",
    "QUIP_DN13_2": "quipazine + DN13/DN1 (DN13_2)",
}
CONDITION_COLORS = {
    "QUIP":        "tab:purple",
    "QUIP_DN1":    "tab:orange",
    "QUIP_DN13_1": "tab:blue",
    "QUIP_DN13_2": "tab:green",
}

SESSIONS = [
    # ---- animal 1 -----------------------------------------------------------
    dict(animal="A1", condition="QUIP",
         photometry=r"photometry_recording_A1_QUIP.csv",
         scoring=r"Scoring_A1_C22.xlsx",
         frame_offset=0),
    dict(animal="A1", condition="QUIP_DN1",
         photometry=r"photometry_recording_A1_DN1.csv",
         scoring=r"Scoring_A1_DN1.xlsx",
         frame_offset=0),
    dict(animal="A1", condition="QUIP_DN13_1",
         photometry=r"photometry_recording_A1_DN13_1.csv",
         scoring=r"Scoring_A1_DN13_1.xlsx",
         frame_offset=0),
    dict(animal="A1", condition="QUIP_DN13_2",
         photometry=r"photometry_recording_A1_DN13_2.csv",
         scoring=r"Scoring_A1_DN13_2.xlsx",
         frame_offset=0),

    # ---- animal 2 -----------------------------------------------------------
    dict(animal="A2", condition="QUIP",
             photometry=r"photometry_recording_A2_QUIP.csv",
             scoring=r"Scoring_A2_C22.xlsx",
             frame_offset=0),
    dict(animal="A2", condition="QUIP_DN1",
             photometry=r"photometry_recording_A2_DN1.csv",
             scoring=r"Scoring_A2_DN1.xlsx",
             frame_offset=0),
    dict(animal="A2", condition="QUIP_DN13_1",
             photometry=r"photometry_recording_A2_DN13_1.csv",
             scoring=r"Scoring_A2_DN13_1.xlsx",
             frame_offset=0),
    dict(animal="A2", condition="QUIP_DN13_2",
             photometry=r"photometry_recording_A2_DN13_2.csv",
             scoring=r"Scoring_A2_DN13_2.xlsx",
             frame_offset=0),

    # ---- animal 3 -----------------------------------------------------------
    dict(animal="A3", condition="QUIP",
              photometry=r"photometry_recording_A3_QUIP.csv",
              scoring=r"Scoring_A3_C22.xlsx",
              frame_offset=0),
    dict(animal="A3", condition="QUIP_DN1",
              photometry=r"photometry_recording_A3_DN1.csv",
              scoring=r"Scoring_A3_DN1.xlsx",
              frame_offset=0),
    dict(animal="A3", condition="QUIP_DN13_1",
              photometry=r"photometry_recording_A3_DN13_1.csv",
              scoring=r"Scoring_A3_DN13_1.xlsx",
              frame_offset=0),
    dict(animal="A3", condition="QUIP_DN13_2",
              photometry=r"photometry_recording_A3_DN13_2.csv",
              scoring=r"Scoring_A3_DN13_2.xlsx",
              frame_offset=0),

    # ---- animal 4 -----------------------------------------------------------
    dict(animal="A4", condition="QUIP",
              photometry=r"photometry_recording_A4_QUIP.csv",
              scoring=r"Scoring_A4_C22.xlsx",
              frame_offset=0),
    dict(animal="A4", condition="QUIP_DN1",
              photometry=r"photometry_recording_A4_DN1.csv",
              scoring=r"Scoring_A4_DN1.xlsx",
              frame_offset=0),
    dict(animal="A4", condition="QUIP_DN13_1",
              photometry=r"photometry_recording_A4_DN13_1.csv",
              scoring=r"Scoring_A4_DN13_1.xlsx",
              frame_offset=0),
    dict(animal="A4", condition="QUIP_DN13_2",
              photometry=r"photometry_recording_A4_DN13_2.csv",
              scoring=r"Scoring_A4_DN13_2.xlsx",
              frame_offset=0),

    # ---- animal 5 -----------------------------------------------------------
    dict(animal="A5", condition="QUIP",
                  photometry=r"photometry_recording_A5_QUIP.csv",
                  scoring=r"Scoring_A5_C22.xlsx",
                  frame_offset=0),
    dict(animal="A5", condition="QUIP_DN1",
                  photometry=r"photometry_recording_A5_DN1.csv",
                  scoring=r"Scoring_A5_DN1.xlsx",
                  frame_offset=0),
    dict(animal="A5", condition="QUIP_DN13_1",
                  photometry=r"photometry_recording_A5_DN13_1.csv",
                  scoring=r"Scoring_A5_DN13_1.xlsx",
                  frame_offset=0),
    dict(animal="A5", condition="QUIP_DN13_2",
                  photometry=r"photometry_recording_A5_DN13_2.csv",
                  scoring=r"Scoring_A5_DN13_2.xlsx",
                  frame_offset=0),
    # Rows with an empty photometry path are skipped with a warning, so you can
    # build the manifest incrementally.
]

# =============================================================================
# CONFIG B -- ANALYSIS PARAMETERS
#   These MUST be identical across every session or the sessions are not
#   comparable. They are set once here, globally, on purpose: there is no
#   per-session override for anything below this line except frame_offset.
# =============================================================================

# --- photometry format (DORIC, two header rows, time-multiplexed) ---
N_HEADER_ROWS = 2
TIME_COL      = 0     # Time(s)
ISO_COL       = 7     # ROI 1 | CAM 1 EXC 1  (isosbestic, 405)
SIG_COL       = 8     # ROI 1 | CAM 1 EXC 2  (signal, 465/470)

# --- preprocessing (identical to step 4) ---
LOWPASS_HZ        = 3.0
ROBUST_METHOD     = "tukey"
REMOVE_SLOW_DRIFT = False
SLOW_DRIFT_HZ     = 0.001

# --- epoching ---
BASELINE_WIN = (-5.0, -2.0)
EPOCH_WIN    = (-8.0,  4.0)
GRID_HZ      = 20.0

# --- measurement windows -----------------------------------------------------
# BOTH are computed for every twitch. PRIMARY_WINDOW is the one that gets the
# figures and the headline statistics; the other is reported as a secondary,
# explicitly exploratory number. Keeping both costs nothing and stops the
# window from being chosen after seeing the group result.
WINDOWS = {
    "pre_twitch": (-2.00, -0.50),   # a priori, pre-movement, motion-clean
    "peri_onset": (-0.25,  0.75),   # post-hoc, contains the movement
}
PRIMARY_WINDOW = "pre_twitch"       

# --- pre-twitch peak / AUC analysis -------------------------------------------
# These metrics are measured on EVERY kept twitch, then aggregated with the same
# hierarchy as the traces: twitch -> session -> animal x condition.
PEAK_AUC_WINDOW = "pre_twitch"
# "positive" integrates only signal above the twitch's baseline (0 after
# baseline subtraction); "signed" integrates positive and negative deviations.
# Both are always saved/reported. This setting only chooses the headline AUC plot.
AUC_MODE = "signed"         # "positive" or "signed"

# A peak is defined operationally as the MAXIMUM sample in the fixed window after
# the existing 3-Hz low-pass filter. No data-dependent prominence threshold is
# used. Peak latency is relative to corrected HTR onset.

# --- cross-animal normalisation ----------------------------------------------
#   "zscore"   : (epoch - baseline mean) / baseline SD   [recommended]
#   "subtract" : (epoch - baseline mean), in % dF/F
# Absolute % dF/F is not commensurate across animals or across days (different
# coupling, expression, bleaching), so averaging raw % across animals lets the
# animal with the brightest fibre dominate the mean. z-scoring against each
# twitch's own baseline removes that gain factor. The cost: the z denominator
# is baseline NOISE, so a quiet baseline inflates z, and with slow drift kept
# (REMOVE_SLOW_DRIFT=False) part of that SD is drift rather than noise.
# Both metrics are computed and saved regardless; this only sets the primary.
NORMALIZE  = "zscore"
ZSCORE_SD  = "baseline"   # "baseline" = per-twitch baseline SD (matches the
                          # per-session script). "session" = SD of the whole
                          # session dF/F -- a more stable denominator, less
                          # sensitive to a quiet 3 s baseline. Try both; they
                          # should agree in sign if the effect is real.

EXCLUDE_FLAGGED = True    # drop twitches carrying a scorer note. Verify the
                          # note inventory htr_sync prints -- if it lists
                          # numbers, SCORE_NOTE_COL is pointing at the wrong
                          # column and this is either a no-op or drops all.

# --- statistics ---
RUN_CLUSTER_PERM = True   # time-resolved cluster permutation over the epoch
CLUSTER_ALPHA    = 0.05
MAX_PERMS        = 20000  # exhaustive is used whenever it is smaller than this

SAVE_TABLES = False

# Compact Excel output for Prism / reporting. This is independent of SAVE_TABLES:
# you can keep the large CSV exports off and still get one clean .xlsx workbook.
SAVE_PRISM_EXCEL = True
PRISM_EXCEL_FILENAME = "group_prism_results.xlsx"

OUT_DIR     = r"."        # where CSVs / Excel output go


# =============================================================================
# 1. PER-SESSION PREPROCESSING  (identical arithmetic to the per-session script)
# =============================================================================
def load_two_header_file(path, n_header):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        full = pd.read_excel(path, header=None, dtype=str)
    else:
        with open(path, "r", newline="") as fh:
            max_cols = max((len(row) for row in csv.reader(fh)), default=0)
        full = pd.read_csv(path, header=None, dtype=str,
                           names=range(max_cols), engine="python")
    header_block = full.iloc[:n_header].reset_index(drop=True)
    df = full.iloc[n_header:].reset_index(drop=True)
    combined = []
    for col in range(header_block.shape[1]):
        parts = [str(header_block.iloc[r, col]).strip()
                 for r in range(n_header)
                 if str(header_block.iloc[r, col]).strip() not in ("", "nan", "-")]
        combined.append(" | ".join(parts) if parts else f"col_{col}")
    df.columns = combined
    return df, combined


def extract_channel(df, time_idx, value_idx):
    sub = df.iloc[:, [time_idx, value_idx]].copy()
    sub.columns = ["time", "value"]
    sub["time"]  = pd.to_numeric(sub["time"],  errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    return sub.dropna(subset=["time", "value"]).reset_index(drop=True)


def _sanitize_channel_time(sub, name, gap_factor=20.0):
    """Remove only *obvious isolated* timestamp spikes, then require monotonicity.

    Do NOT sort timestamps: sorting can silently reinsert a corrupt value at a
    plausible later time. A row is removed only when it makes a huge forward jump
    and the very next row returns to the expected local sequence. Any remaining
    non-monotonicity raises, because interpolation would otherwise be unsafe.
    """
    sub = sub.reset_index(drop=True).copy()
    if len(sub) < 3:
        return sub

    t = sub["time"].to_numpy(float)
    dt = np.diff(t)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    if len(pos) == 0:
        raise ValueError(f"{name}: no positive timestamp intervals")
    med = float(np.median(pos))

    # Isolated 'teleport' timestamp: t[i] jumps far ahead, t[i+1] returns to
    # approximately one normal sample after t[i-1].
    bad = np.zeros(len(t), dtype=bool)
    for i in range(1, len(t) - 1):
        jump = t[i] - t[i - 1]
        bridge = t[i + 1] - t[i - 1]
        if (jump > gap_factor * med and t[i + 1] < t[i]
                and 0 < bridge < gap_factor * med):
            bad[i] = True

    if bad.any():
        idx = np.where(bad)[0]
        for i in idx[:10]:
            print(f"    [time QC] {name}: dropping isolated bad row {i}: "
                  f"t={t[i]:.6f}s between {t[i-1]:.6f} and {t[i+1]:.6f}s")
        if len(idx) > 10:
            print(f"    [time QC] {name}: ... plus {len(idx)-10} more")
        sub = sub.loc[~bad].reset_index(drop=True)

    t2 = sub["time"].to_numpy(float)
    d2 = np.diff(t2)
    bad2 = np.where(d2 <= 0)[0]
    if len(bad2):
        i = int(bad2[0])
        raise ValueError(
            f"{name}: timestamps still non-monotonic after QC at rows {i}/{i+1} "
            f"({t2[i]:.6f} -> {t2[i+1]:.6f} s). Refusing np.interp().")
    return sub


def load_and_align(path):
    """Load only time + 405 + signal columns, then align 405 to signal timestamps.

    This is numerically equivalent to the old loader but avoids reading the whole
    Doric file as strings (important for ~100 MB recordings). Timestamp QC is
    performed before interpolation because np.interp requires increasing x.
    """
    ext = os.path.splitext(path)[1].lower()
    needed = sorted({TIME_COL, ISO_COL, SIG_COL})
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, usecols=needed, skiprows=N_HEADER_ROWS)
    else:
        raw = pd.read_csv(path, header=None, usecols=needed,
                          skiprows=N_HEADER_ROWS, low_memory=False)

    # Column labels remain the original integer positions when usecols receives
    # integer indices (0, 7, 8), so select by those labels directly.
    for c in needed:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    iso = raw[[TIME_COL, ISO_COL]].copy()
    iso.columns = ["time", "value"]
    iso = iso.dropna(subset=["time", "value"]).reset_index(drop=True)
    sig = raw[[TIME_COL, SIG_COL]].copy()
    sig.columns = ["time", "value"]
    sig = sig.dropna(subset=["time", "value"]).reset_index(drop=True)

    iso = _sanitize_channel_time(iso, "isosbestic")
    sig = _sanitize_channel_time(sig, "signal")
    if len(iso) < 2 or len(sig) < 2:
        raise ValueError("A channel came back nearly empty -- check ISO_COL/SIG_COL.")

    t        = sig["time"].to_numpy(float)
    sig_vals = sig["value"].to_numpy(float)
    iso_on_t = np.interp(t, iso["time"].to_numpy(float), iso["value"].to_numpy(float))
    fs_sig   = 1.0 / np.median(np.diff(t))
    print(f"    photometry: {len(sig)} sig samples | per-channel ~{fs_sig:.2f} Hz "
          f"| span {t[0]:.1f}-{t[-1]:.1f} s")
    return t, iso_on_t, sig_vals


def lowpass(x, fs, cutoff_hz):
    if cutoff_hz is None or cutoff_hz <= 0:
        return x.copy()
    if HAVE_SCIPY and cutoff_hz < fs / 2:
        b, a = butter(2, cutoff_hz / (fs / 2), btype="low")
        return filtfilt(b, a, x)
    win = max(1, int(round(fs / cutoff_hz)))
    if win % 2 == 0:
        win += 1
    return np.convolve(x, np.ones(win) / win, mode="same")


def _irls_numpy(x, y, method="tukey", n_iter=50, tol=1e-9):
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
            w = np.where(np.abs(u) < 1.0, (1.0 - u**2)**2, 0.0)
        else:
            w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X + 1e-12 * np.eye(2), XtW @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta[0], beta[1]


def robust_fit(iso, sig, method="tukey"):
    if HAVE_STATSMODELS:
        norm = sm.robust.norms.TukeyBiweight() if method == "tukey" else sm.robust.norms.HuberT()
        res = sm.RLM(sig, sm.add_constant(iso), M=norm).fit()
        return res.params[1], res.params[0], "statsmodels RLM"
    a, b = _irls_numpy(iso, sig, method=method)
    return a, b, "numpy IRLS"


def compute_traces(path):
    """Return (t, dff, iso_filt, meta).

    dff      : the step-4 corrected signal, (sig - fitted_iso)/fitted_iso
    iso_filt : the low-passed isosbestic on the same timebase, kept RAW (not
               regressed, not dF/F'd) so it can serve as an independent motion
               control. It is turned into a fractional change per epoch, against
               that epoch's own baseline, in build_epochs().
    """
    t, iso_raw, sig_raw = load_and_align(path)
    fs  = 1.0 / np.median(np.diff(t))
    iso = lowpass(iso_raw, fs, LOWPASS_HZ)
    sig = lowpass(sig_raw, fs, LOWPASS_HZ)
    a_r, b_r, engine = robust_fit(iso, sig, ROBUST_METHOD)
    fitted = a_r * iso + b_r
    if np.any(fitted <= 0):
        print("    WARNING: fitted baseline crosses zero; dF/F unsafe there.")
    dff = (sig - fitted) / fitted
    if REMOVE_SLOW_DRIFT:
        dff = dff - lowpass(dff, fs, SLOW_DRIFT_HZ)
    print(f"    robust fit ({engine}): slope={a_r:.4f} intercept={b_r:.4f} | "
          f"slow drift {'REMOVED' if REMOVE_SLOW_DRIFT else 'KEPT'}")
    meta = dict(fs=fs, slope=a_r, intercept=b_r, engine=engine,
                dff_sd=float(np.nanstd(dff)), n_samples=len(t),
                t_start=float(t[0]), t_end=float(t[-1]))
    return t, dff, iso, meta


# =============================================================================
# 2. EXCLUSIONS  (single span, shared by all measurement windows)
# =============================================================================
EXCL_SPAN = (BASELINE_WIN[0], max(w[1] for w in WINDOWS.values()))


def flag_exclusions(ev, t, verbose=True):
    """(a) another onset inside the analysed span -> contaminated;
       (b) epoch not fully inside the recording;
       (c) optional: carries a scorer note.
       The span is EXCL_SPAN, i.e. the union over all measurement windows, so
       every window is scored on the identical twitch set."""
    onsets = ev["onset_s"].values
    span_lo, span_hi = EXCL_SPAN
    t0, t1 = t[0], t[-1]
    has_note = "note" in ev.columns
    reasons = []
    for i, o in enumerate(onsets):
        reason = ""
        for j, o2 in enumerate(onsets):
            if j == i:
                continue
            if (o + span_lo) <= o2 <= (o + span_hi):
                reason = f"neighbour HTR at {o2:.1f}s in analysed span"
                break
        if not reason and not (t0 <= o + span_lo and o + span_hi <= t1):
            reason = "analysis window outside recording"
        if not reason and EXCLUDE_FLAGGED and has_note and ev.iloc[i]["note"]:
            reason = f"scorer note: {ev.iloc[i]['note']}"
        reasons.append(reason)
    ev = ev.copy()
    ev["reason"]   = reasons
    ev["excluded"] = ev["reason"] != ""
    if verbose:
        n_excl = int(ev["excluded"].sum())
        print(f"    exclusions: {n_excl} / {len(ev)} removed "
              f"(span [{span_lo:+.2f}, {span_hi:+.2f}] s)")
        for _, row in ev[ev["excluded"]].iterrows():
            print(f"      HTR {row.get('htr_idx','?')} @ {row['onset_s']:.1f}s "
                  f"-> {row['reason']}")
    return ev


# =============================================================================
# 3. EPOCHING
# =============================================================================
# Round the event grid so exact window boundaries do not flip inclusion because
# of floating-point representation (e.g. -2.00000000000002).
GRID      = np.round(np.arange(EPOCH_WIN[0], EPOCH_WIN[1] + 1e-9,
                               1.0 / GRID_HZ), 10)
BASE_MASK = (GRID >= BASELINE_WIN[0]) & (GRID < BASELINE_WIN[1])
WIN_MASKS = {k: (GRID >= v[0]) & (GRID < v[1]) for k, v in WINDOWS.items()}


def fixed_window_auc(trace, window, positive=False):
    """Trapezoidal AUC over the exact [lo, hi] interval in GRID units.

    Boundary values are linearly interpolated on the already filtered epoch.
    This avoids losing one 50-ms interval merely because the scalar-response mask
    is right-exclusive. `positive=True` integrates max(trace, 0).
    """
    lo, hi = window
    trace = np.asarray(trace, float)
    good = np.isfinite(GRID) & np.isfinite(trace)
    if good.sum() < 2 or lo < GRID[good].min() or hi > GRID[good].max():
        return np.nan
    inner = GRID[(GRID > lo) & (GRID < hi)]
    tt = np.concatenate(([lo], inner, [hi]))
    yy = np.interp(tt, GRID[good], trace[good])
    if positive:
        yy = np.maximum(yy, 0.0)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(yy, tt))
    return float(np.trapz(yy, tt))   # NumPy < 2.0 compatibility


def build_epochs(ev, t, dff, iso, session_dff_sd):
    """Per kept twitch, return the normalised signal epoch, the isosbestic
    control epoch, and the scalar response in every measurement window.

    Signal epochs are stored twice:
      *_sub : (dF/F - baseline mean), in %      -- physical units, animal-specific gain
      *_z   : (dF/F - baseline mean) / SD       -- gain-free, used for group averaging

    Isosbestic epochs are stored as fractional change against the epoch's own
    baseline mean iso, in %. This is NOT a dF/F; it is a raw-channel motion
    readout, and that is exactly what it is for.
    """
    kept = ev[~ev["excluded"]].reset_index(drop=True)
    rows, mat_sub, mat_z, mat_iso = [], [], [], []

    for _, r in kept.iterrows():
        o = r["onset_s"]
        ep_sig = np.interp(o + GRID, t, dff, left=np.nan, right=np.nan)
        ep_iso = np.interp(o + GRID, t, iso, left=np.nan, right=np.nan)

        bm = np.nanmean(ep_sig[BASE_MASK])
        bs = np.nanstd(ep_sig[BASE_MASK])
        denom = bs if ZSCORE_SD == "baseline" else session_dff_sd
        if not np.isfinite(denom) or denom <= 0:
            denom = np.nan

        tr_sub = (ep_sig - bm) * 100.0
        tr_z   = (ep_sig - bm) / denom

        bmi = np.nanmean(ep_iso[BASE_MASK])
        tr_iso = (ep_iso - bmi) / bmi * 100.0 if np.isfinite(bmi) and bmi != 0 \
                 else np.full_like(ep_iso, np.nan)

        row = dict(onset_s=o,
                   htr_idx=r.get("htr_idx", np.nan),
                   frame=r.get("frame", np.nan),
                   baseline_mean_dff_pct=bm * 100.0,
                   baseline_sd_dff_pct=bs * 100.0)
        for k, m in WIN_MASKS.items():
            # Existing fixed-window mean response
            row[f"resp_{k}_sub_pct"] = np.nanmean(tr_sub[m])
            row[f"resp_{k}_z"]       = np.nanmean(tr_z[m])
            row[f"resp_{k}_abs_pct"] = np.nanmean(ep_sig[m]) * 100.0
            row[f"iso_{k}_pct"]      = np.nanmean(tr_iso[m])

            # Peak amplitude + latency in the SAME fixed window.
            # GRID is the actual integration/latency timebase.
            tw = GRID[m]
            subw = tr_sub[m]
            zw   = tr_z[m]

            if np.any(np.isfinite(subw)):
                ip = int(np.nanargmax(subw))
                row[f"peak_{k}_sub_pct"] = float(subw[ip])
                row[f"peak_{k}_latency_s"] = float(tw[ip])
            else:
                row[f"peak_{k}_sub_pct"] = np.nan
                row[f"peak_{k}_latency_s"] = np.nan

            if np.any(np.isfinite(zw)):
                ipz = int(np.nanargmax(zw))
                row[f"peak_{k}_z"] = float(zw[ipz])
                # z and sub traces differ only by a positive scale factor, so
                # latency should match; keep a QC copy to make that testable.
                row[f"peak_{k}_latency_z_s"] = float(tw[ipz])
            else:
                row[f"peak_{k}_z"] = np.nan
                row[f"peak_{k}_latency_z_s"] = np.nan

            # AUC on the baseline-centered trace over the EXACT configured
            # interval. Signed AUC is conventional; positive AUC integrates only
            # excursions above baseline.
            row[f"auc_signed_{k}_sub_pct_s"] = fixed_window_auc(
                tr_sub, WINDOWS[k], positive=False)
            row[f"auc_positive_{k}_sub_pct_s"] = fixed_window_auc(
                tr_sub, WINDOWS[k], positive=True)
            row[f"auc_signed_{k}_z_s"] = fixed_window_auc(
                tr_z, WINDOWS[k], positive=False)
            row[f"auc_positive_{k}_z_s"] = fixed_window_auc(
                tr_z, WINDOWS[k], positive=True)
        rows.append(row)
        mat_sub.append(tr_sub)
        mat_z.append(tr_z)
        mat_iso.append(tr_iso)

    per_twitch = pd.DataFrame(rows)
    to_arr = lambda L: (np.array(L) if L else np.empty((0, len(GRID))))
    return per_twitch, to_arr(mat_sub), to_arr(mat_z), to_arr(mat_iso)


# =============================================================================
# 4. SESSION DRIVER
# =============================================================================
def _resolve(p):
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.join(DATA_DIR, p)


def process_session(sess, verbose_sync=False):
    ph = _resolve(sess["photometry"])
    sc = _resolve(sess["scoring"])
    tag = f"{sess['animal']} / {sess['condition']}"
    print(f"\n--- {tag} ---")
    print(f"    {os.path.basename(ph)}  +  {os.path.basename(sc)}")

    t, dff, iso, meta = compute_traces(ph)

    # Validate the camera TTL train for EVERY session. report_sync() and
    # corrected_onsets() share htr_sync's in-memory cache, so this does not cause
    # a second photometry read. A gap would make frame N != edge N thereafter.
    if HAVE_REPORT_SYNC:
        report_sync(ph, video_n_frames=None)
    elif verbose_sync:
        print("    [warn] report_sync unavailable; sync-gap validation skipped")

    ev = corrected_onsets(ph, sc, frame_offset=sess.get("frame_offset", 0))
    n_scored = len(ev)
    ev = flag_exclusions(ev, t)
    per_twitch, mat_sub, mat_z, mat_iso = build_epochs(
        ev, t, dff, iso, session_dff_sd=meta["dff_sd"])

    print(f"    kept {len(per_twitch)} / {n_scored} twitches | "
          f"session dF/F SD = {meta['dff_sd']*100:.3f} %")

    per_twitch.insert(0, "condition", sess["condition"])
    per_twitch.insert(0, "animal", sess["animal"])
    per_twitch.insert(2, "session", os.path.splitext(os.path.basename(ph))[0])

    return dict(animal=sess["animal"], condition=sess["condition"],
                session=os.path.splitext(os.path.basename(ph))[0],
                n_scored=n_scored, n_kept=len(per_twitch),
                per_twitch=per_twitch,
                mean_sub=np.nanmean(mat_sub, axis=0) if len(mat_sub) else np.full(len(GRID), np.nan),
                mean_z=np.nanmean(mat_z,   axis=0) if len(mat_z)   else np.full(len(GRID), np.nan),
                mean_iso=np.nanmean(mat_iso, axis=0) if len(mat_iso) else np.full(len(GRID), np.nan),
                meta=meta)


# =============================================================================
# 5. STATISTICS  (unit = animal; exact permutation where the design allows)
# =============================================================================
def signflip_p(x):
    """Exact (or sampled) two-sided sign-flip permutation test of mean = 0.
    Exhaustive when 2**n <= MAX_PERMS. Hard floor: p >= 2/2**n, because the
    all-plus and all-minus flips both reproduce the observed |mean|."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan, n, np.nan
    obs = abs(np.mean(x))
    if 2 ** n <= MAX_PERMS:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(MAX_PERMS, n))
    null = np.abs((signs * x).mean(axis=1))
    p = float(np.mean(null >= obs - 1e-12))
    return p, n, 2.0 / (2.0 ** n)


def condition_perm_p(M):
    """Within-animal label-permutation test of 'all conditions equal'.
    M is (n_animals x k_conditions), complete cases only. Statistic = spread of
    the condition means. Exhaustive when (k!)**n <= MAX_PERMS; otherwise a
    reproducible Monte-Carlo sample of MAX_PERMS label arrangements is used.
    With 4 conditions x 5 animals, 24**5 = 7,962,624 arrangements, so the
    default MAX_PERMS=20,000 uses the sampled version."""
    M = np.asarray(M, float)
    n, k = M.shape
    stat = lambda A: float(np.sum((A.mean(axis=0) - A.mean()) ** 2))
    obs = stat(M)
    perms = list(itertools.permutations(range(k)))
    total = len(perms) ** n
    exact = total <= MAX_PERMS
    if exact:
        assign = itertools.product(perms, repeat=n)
        null = [stat(np.array([M[i, list(p)] for i, p in enumerate(a)]))
                for a in assign]
    else:
        rng = np.random.default_rng(0)
        null = []
        for _ in range(MAX_PERMS):
            A = np.array([M[i, list(perms[rng.integers(len(perms))])] for i in range(n)])
            null.append(stat(A))
    null = np.asarray(null)
    return float(np.mean(null >= obs - 1e-12)), n, len(null), exact, total


def _tstat(M):
    n = M.shape[0]
    m = np.nanmean(M, axis=0)
    s = np.nanstd(M, axis=0, ddof=1) / np.sqrt(n)
    return np.where(s > 0, m / np.where(s > 0, s, 1.0), 0.0)


def _clusters(tvals, thresh):
    mask = np.abs(tvals) > thresh
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j, float(np.sum(tvals[i:j + 1]))))
            i = j + 1
        else:
            i += 1
    return out


def cluster_perm_1samp(X, alpha=CLUSTER_ALPHA):
    """Cluster-mass sign-flip permutation over the epoch. X = (n_animals x n_t).
    Answers 'is there a time interval where the mean differs from zero',
    correcting over time. Same p-floor as above: 2/2**n."""
    X = np.asarray(X, float)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    n = X.shape[0]
    if n < 3:
        return [], np.nan
    thresh = scipy_stats.t.ppf(1 - alpha / 2, n - 1) if HAVE_SCIPY_STATS else 2.5
    obs = _clusters(_tstat(X), thresh)
    if 2 ** n <= MAX_PERMS:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(MAX_PERMS, n))
    null = []
    for s in signs:
        cl = _clusters(_tstat(X * s[:, None]), thresh)
        null.append(max([abs(c[2]) for c in cl], default=0.0))
    null = np.asarray(null)
    res = [(GRID[a], GRID[b], mass, float(np.mean(null >= abs(mass) - 1e-12)))
           for a, b, mass in obs]
    return res, 2.0 / (2.0 ** n)


# =============================================================================
# 6. AGGREGATION
# =============================================================================
def aggregate(results):
    """session -> animal x condition cell. Sessions within a cell are averaged
    with equal weight (not pooled by twitch), so a cell is one number per
    animal regardless of how many recordings it contains."""
    per_twitch_all = pd.concat([r["per_twitch"] for r in results],
                               ignore_index=True) if results else pd.DataFrame()

    sess_rows = []
    for r in results:
        pt = r["per_twitch"]
        row = dict(animal=r["animal"], condition=r["condition"],
                   session=r["session"], n_scored=r["n_scored"], n_kept=r["n_kept"],
                   fs=r["meta"]["fs"], robust_slope=r["meta"]["slope"],
                   session_dff_sd_pct=r["meta"]["dff_sd"] * 100.0,
                   duration_s=r["meta"]["t_end"] - r["meta"]["t_start"],
                   htr_rate_per_min=r["n_scored"] /
                                    ((r["meta"]["t_end"] - r["meta"]["t_start"]) / 60.0),
                   baseline_mean_dff_pct=np.nanmean(pt["baseline_mean_dff_pct"])
                                          if len(pt) else np.nan)
        for k in WINDOWS:
            for suf in ("sub_pct", "z", "abs_pct"):
                col = f"resp_{k}_{suf}"
                row[col] = np.nanmean(pt[col]) if len(pt) else np.nan
            row[f"iso_{k}_pct"] = np.nanmean(pt[f"iso_{k}_pct"]) if len(pt) else np.nan

            # New peak / AUC metrics: first average twitch-level measurements
            # within a session; sessions are then equally weighted in per_cell.
            for col in (
                f"peak_{k}_sub_pct", f"peak_{k}_z",
                f"peak_{k}_latency_s", f"peak_{k}_latency_z_s",
                f"auc_signed_{k}_sub_pct_s", f"auc_positive_{k}_sub_pct_s",
                f"auc_signed_{k}_z_s", f"auc_positive_{k}_z_s",
            ):
                row[col] = np.nanmean(pt[col]) if len(pt) else np.nan
        sess_rows.append(row)
    per_session = pd.DataFrame(sess_rows)

    num = [c for c in per_session.columns
           if c not in ("animal", "condition", "session")]
    per_cell = (per_session.groupby(["animal", "condition"], as_index=False)[num]
                .mean(numeric_only=True))
    per_cell["n_sessions"] = (per_session.groupby(["animal", "condition"])
                              .size().reset_index(drop=True).values)

    # traces: session means -> cell means
    traces = {}
    for key in ("mean_sub", "mean_z", "mean_iso"):
        d = {}
        for r in results:
            d.setdefault((r["animal"], r["condition"]), []).append(r[key])
        traces[key] = {k: np.nanmean(np.array(v), axis=0) for k, v in d.items()}
    return per_twitch_all, per_session, per_cell, traces


def cell_matrix(per_cell, col):
    """(animals x conditions) matrix in CONDITIONS order, animals sorted.
    Returns the full matrix plus the complete-case subset used for the
    within-subject tests."""
    animals = sorted(per_cell["animal"].unique())
    M = np.full((len(animals), len(CONDITIONS)), np.nan)
    for i, a in enumerate(animals):
        for j, c in enumerate(CONDITIONS):
            v = per_cell[(per_cell["animal"] == a) & (per_cell["condition"] == c)][col]
            if len(v):
                M[i, j] = v.values[0]
    complete = np.all(np.isfinite(M), axis=1)
    return animals, M, complete


def trace_matrix(traces, key, condition, animals):
    rows = [traces[key].get((a, condition), np.full(len(GRID), np.nan))
            for a in animals]
    return np.array(rows)


# =============================================================================
# 7. REPORTING
# =============================================================================
def bar(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_scalar_metric(per_cell, col, title, unit):
    """Animal-level within-subject report for one scalar metric."""
    animals, M, complete = cell_matrix(per_cell, col)
    bar(title)
    print(f"metric: {col} | unit = {unit} | statistical unit = ANIMAL")

    for j, c in enumerate(CONDITIONS):
        v = M[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            print(f"\n{c:>16s}: no data")
            continue
        sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        print(f"\n{c:>16s}: n={len(v)} | mean {np.mean(v):+.4f} "
              f"+/- {sem:.4f} SEM | median {np.median(v):+.4f}")

    Mc = M[complete]
    print(f"\ncomplete-case animals: {Mc.shape[0]} / {len(animals)}")
    if Mc.shape[0] >= 3:
        p, n, nperm, exact, total = condition_perm_p(Mc)
        mode = "exact" if exact else "Monte-Carlo"
        print(f"{mode} within-animal condition permutation: p={p:.4g} "
              f"({nperm:,} tested of {total:,} possible arrangements; n={n})")
        if HAVE_SCIPY_STATS:
            fr = scipy_stats.friedmanchisquare(*[Mc[:, j] for j in range(Mc.shape[1])])
            print(f"Friedman: chi2={fr.statistic:.3f}, p={fr.pvalue:.4g}")
        print("pairwise paired sign-flips on differences (UNCORRECTED):")
        for a, b in itertools.combinations(range(len(CONDITIONS)), 2):
            d = Mc[:, a] - Mc[:, b]
            p2, n2, fl = signflip_p(d)
            print(f"  {CONDITIONS[a]:>16s} - {CONDITIONS[b]:<16s} "
                  f"mean diff {np.mean(d):+.4f} | p={p2:.4g} (floor {fl:.4g})")
    else:
        print("too few complete animals for a within-subject test")
    return animals, M, complete


def prism_metric_specs():
    """Metrics exported to the compact Prism workbook.

    Each metric is already at the ANIMAL x CONDITION level in per_cell.  The
    animal-level sheets are therefore the correct inputs for repeated-measures
    statistics in Prism; the Summary sheet is descriptive only (mean/SEM/n).
    """
    win = PRIMARY_WINDOW
    pa_win = PEAK_AUC_WINDOW
    response_col = (f"resp_{win}_z" if NORMALIZE == "zscore"
                    else f"resp_{win}_sub_pct")
    peak_col = (f"peak_{pa_win}_z" if NORMALIZE == "zscore"
                else f"peak_{pa_win}_sub_pct")
    signed_auc_col = (f"auc_signed_{pa_win}_z_s" if NORMALIZE == "zscore"
                      else f"auc_signed_{pa_win}_sub_pct_s")
    positive_auc_col = (f"auc_positive_{pa_win}_z_s" if NORMALIZE == "zscore"
                        else f"auc_positive_{pa_win}_sub_pct_s")

    response_unit = ("baseline-z" if NORMALIZE == "zscore"
                     else "% dF/F (baseline-subtracted)")
    peak_unit = ("baseline-z" if NORMALIZE == "zscore"
                 else "% dF/F above baseline")
    auc_unit = "z*s" if NORMALIZE == "zscore" else "% dF/F*s"

    return [
        dict(sheet="Response", metric="Pre-twitch response",
             col=response_col, unit=response_unit),
        dict(sheet="Peak amplitude", metric="Pre-twitch peak amplitude",
             col=peak_col, unit=peak_unit),
        dict(sheet="Signed AUC", metric="Pre-twitch signed AUC",
             col=signed_auc_col, unit=auc_unit),
        dict(sheet="Positive AUC", metric="Pre-twitch positive AUC",
             col=positive_auc_col, unit=auc_unit),
        dict(sheet="Peak latency", metric="Pre-twitch peak latency",
             col=f"peak_{pa_win}_latency_s",
             unit="s relative to HTR onset"),
    ]


def export_prism_excel(per_cell):
    """Write one compact .xlsx with descriptive summaries and Prism-ready data.

    Workbook layout
    ---------------
    Summary:
        one row per metric x condition with n, mean, SEM and median.

    One sheet per metric:
        rows = animals, columns = conditions. This is the table to copy/import
        into a Prism Column table with matched/repeated-measures rows.

    Means/SEMs are deliberately NOT used as the input for inferential statistics;
    Prism should receive the individual animal values and calculate them itself.
    """
    specs = prism_metric_specs()
    summary_rows = []
    animal_tables = {}

    for spec in specs:
        animals, M, _ = cell_matrix(per_cell, spec["col"])

        # Wide animal-level table: one matched row per animal, one condition/column.
        wide = pd.DataFrame(M,
                            columns=[CONDITION_LABELS[c] for c in CONDITIONS])
        wide.insert(0, "Animal", animals)
        animal_tables[spec["sheet"]] = wide

        # The exact mean / SEM values represented by the black summary markers
        # in the paired per-animal figures.
        for j, c in enumerate(CONDITIONS):
            v = M[:, j]
            v = v[np.isfinite(v)]
            n = len(v)
            mean = float(np.mean(v)) if n else np.nan
            sem = (float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
            median = float(np.median(v)) if n else np.nan
            summary_rows.append(dict(
                Metric=spec["metric"],
                Unit=spec["unit"],
                Condition=CONDITION_LABELS[c],
                n=n,
                Mean=mean,
                SEM=sem,
                Median=median,
            ))

    summary = pd.DataFrame(summary_rows)
    out = os.path.abspath(os.path.join(OUT_DIR, PRISM_EXCEL_FILENAME))

    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            for sheet_name, df in animal_tables.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            # Minimal formatting for readability; openpyxl is already a dependency
            # of htr_sync.py for reading the scoring sheets.
            wb = writer.book
            for ws in wb.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                for cell in ws[1]:
                    cell.font = cell.font.copy(bold=True)
                for column_cells in ws.columns:
                    letter = column_cells[0].column_letter
                    max_len = max(len(str(cell.value)) if cell.value is not None else 0
                                  for cell in column_cells)
                    ws.column_dimensions[letter].width = min(max(max_len + 2, 11), 34)
                # Keep numerical output comfortably precise for Prism copy/paste.
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        if isinstance(cell.value, float):
                            cell.number_format = "0.000000"

        print(f"\nsaved Prism-ready Excel -> {out}")
        print("  Summary: Mean / SEM / n for each metric x condition")
        print("  Metric sheets: individual animal values (use THESE for Prism statistics)")
        return out
    except ImportError as e:
        print(f"\n[warn] Excel export unavailable ({e}). Install openpyxl.")
    except OSError as e:
        print(f"\n[warn] could not save Prism Excel ({e})")
    return None


def report(per_session, per_cell, traces):
    win = PRIMARY_WINDOW
    metric = f"resp_{win}_z" if NORMALIZE == "zscore" else f"resp_{win}_sub_pct"
    unit   = "baseline-z" if NORMALIZE == "zscore" else "% dF/F (baseline-subtracted)"
    animals, M, complete = cell_matrix(per_cell, metric)

    bar("DESIGN / COVERAGE")
    cov = per_cell.pivot(index="animal", columns="condition", values="n_kept")
    cov = cov.reindex(columns=CONDITIONS)
    print("twitches kept per cell:")
    print(cov.to_string())
    print("\nHTR count SCORED per cell (before exclusions) -- this is itself a "
          "\nresult, and a condition effect on it confounds everything below:")
    print(per_cell.pivot(index="animal", columns="condition", values="n_scored")
          .reindex(columns=CONDITIONS).to_string())
    print(f"\ncomplete-case animals for within-subject tests: "
          f"{int(complete.sum())} / {len(animals)}")
    if complete.sum() < len(animals):
        for a, ok in zip(animals, complete):
            if not ok:
                print(f"  dropped from RM tests (missing a condition): {a}")

    bar(f"PRIMARY WINDOW = {win} {WINDOWS[win]} s | metric = {unit} | unit = ANIMAL")
    if WINDOWS[win][1] > 0:
        print("[!] This window CONTAINS the movement. Nothing below is glutamate "
              "until the\n    isosbestic section is read and found flat.")
    if win == "peri_onset":
        print("[!] Post-hoc window (moved after the a priori [-2,-0.5] returned "
              "null). Exploratory.")

    for j, c in enumerate(CONDITIONS):
        v = M[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            print(f"\n{c:>16s}: no data")
            continue
        p, n, floor = signflip_p(v)
        sem = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        print(f"\n{c:>16s}: n={len(v)} animals | mean {np.mean(v):+.3f} "
              f"+/- {sem:.3f} SEM | median {np.median(v):+.3f} | "
              f"{int(np.sum(v > 0))}/{len(v)} up")
        if np.isfinite(p):
            print(f"{'':>16s}  exact sign-flip vs 0: p = {p:.4g} "
                  f"(floor at n={n}: p >= {floor:.4g})")
        if HAVE_SCIPY_STATS and len(v) > 1:
            tt = scipy_stats.ttest_1samp(v, 0.0)
            print(f"{'':>16s}  one-sample t: t={tt.statistic:+.2f}, p={tt.pvalue:.4g} "
                  f"(normality untestable at this n)")

    bar("CONDITION EFFECT (within-subject, complete cases)")
    Mc = M[complete]
    if Mc.shape[0] < 3:
        print("too few complete animals for a within-subject test.")
    else:
        p, n, nperm, exact, total = condition_perm_p(Mc)
        mode = "exact" if exact else "Monte-Carlo"
        print(f"{mode} within-animal label permutation "
              f"({nperm:,} tested of {total:,} possible arrangements, "
              f"n={n} animals):\n  p = {p:.4g}")
        if HAVE_SCIPY_STATS:
            fr = scipy_stats.friedmanchisquare(*[Mc[:, j] for j in range(Mc.shape[1])])
            print(f"Friedman: chi2={fr.statistic:.3f}, p={fr.pvalue:.4g}")
        print("\npairwise (paired, exact sign-flip on the differences):")
        for a, b in itertools.combinations(range(len(CONDITIONS)), 2):
            d = Mc[:, a] - Mc[:, b]
            p2, n2, fl = signflip_p(d)
            print(f"  {CONDITIONS[a]:>16s} - {CONDITIONS[b]:<16s} "
                  f"mean diff {np.mean(d):+.3f} | p = {p2:.4g} (floor {fl:.4g})")
        pair_floor = 2.0 / (2.0 ** Mc.shape[0])
        print(f"\n[!] Pairwise tests are UNCORRECTED. At n={Mc.shape[0]} the exact "
              f"sign-flip p-floor is {pair_floor:.4g}. Report effect sizes and "
              "CIs alongside p-values; do not hunt for stars.")

    # ---------------- isosbestic control ----------------
    bar("ISOSBESTIC CONTROL (same windows, same animals, same tests)")
    print("The isosbestic is not glutamate-sensitive (approximately -- iGluSnFR "
          "is cpGFP-based,\nso the 405 control can carry a small inverted "
          "fraction of the real signal). A\ndeflection here at the same latency "
          "means movement, not transmitter.")
    _, Mi, comp_i = cell_matrix(per_cell, f"iso_{win}_pct")
    for j, c in enumerate(CONDITIONS):
        v = Mi[:, j]
        v = v[np.isfinite(v)]
        if len(v) < 2:
            print(f"{c:>16s}: n<2, no test")
            continue
        p, n, floor = signflip_p(v)
        sem = np.std(v, ddof=1) / np.sqrt(len(v))
        print(f"{c:>16s}: {np.mean(v):+.4f} +/- {sem:.4f} % | sign-flip p = {p:.4g}")

    # ---------------- secondary window ----------------
    for k in WINDOWS:
        if k == win:
            continue
        bar(f"SECONDARY WINDOW = {k} {WINDOWS[k]} s (reported, not headline)")
        m2 = f"resp_{k}_z" if NORMALIZE == "zscore" else f"resp_{k}_sub_pct"
        _, M2, comp2 = cell_matrix(per_cell, m2)
        for j, c in enumerate(CONDITIONS):
            v = M2[:, j]
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            p, n, _ = signflip_p(v)
            print(f"{c:>16s}: {np.mean(v):+.3f} "
                  f"+/- {np.std(v, ddof=1)/np.sqrt(len(v)):.3f} | p = {p:.4g}")

    # ---------------- time-resolved ----------------
    if RUN_CLUSTER_PERM:
        bar("TIME-RESOLVED CLUSTER PERMUTATION (corrected over the epoch)")
        key = "mean_z" if NORMALIZE == "zscore" else "mean_sub"
        for c in CONDITIONS:
            X = trace_matrix(traces, key, c, animals)
            X = X[np.all(np.isfinite(X), axis=1)]
            if X.shape[0] < 3:
                print(f"{c:>16s}: n<3, skipped")
                continue
            cl, floor = cluster_perm_1samp(X)
            if not cl:
                print(f"{c:>16s}: no suprathreshold cluster")
            for (t0, t1, mass, p) in cl:
                print(f"{c:>16s}: cluster {t0:+.2f} to {t1:+.2f} s | "
                      f"mass {mass:+.1f} | p = {p:.4g} (floor {floor:.4g})")
        print("\nThis is the right test for 'when does it change'. The fixed "
              "windows above are\nnot -- they were chosen, not found.")

    # ---------------- pre-twitch peak + AUC ----------------
    pa_win = PEAK_AUC_WINDOW
    peak_col = (f"peak_{pa_win}_z" if NORMALIZE == "zscore"
                else f"peak_{pa_win}_sub_pct")
    peak_unit = "baseline-z" if NORMALIZE == "zscore" else "% dF/F above baseline"
    report_scalar_metric(per_cell, peak_col,
                         f"PRE-TWITCH PEAK AMPLITUDE {WINDOWS[pa_win]} s", peak_unit)

    auc_signed_col = (f"auc_signed_{pa_win}_z_s" if NORMALIZE == "zscore"
                      else f"auc_signed_{pa_win}_sub_pct_s")
    auc_pos_col = (f"auc_positive_{pa_win}_z_s" if NORMALIZE == "zscore"
                   else f"auc_positive_{pa_win}_sub_pct_s")
    auc_unit = "z*s" if NORMALIZE == "zscore" else "% dF/F*s"
    report_scalar_metric(per_cell, auc_signed_col,
                         f"PRE-TWITCH SIGNED AUC {WINDOWS[pa_win]} s", auc_unit)
    report_scalar_metric(per_cell, auc_pos_col,
                         f"PRE-TWITCH POSITIVE AUC {WINDOWS[pa_win]} s", auc_unit)

    latency_col = f"peak_{pa_win}_latency_s"
    report_scalar_metric(per_cell, latency_col,
                         f"PRE-TWITCH PEAK LATENCY {WINDOWS[pa_win]} s",
                         "seconds relative to corrected HTR onset")

    return animals, M, metric, unit


# =============================================================================
# 8. FIGURES
# =============================================================================
def paired_metric_figure(per_cell, col, ylabel, title):
    animals, Mx, _ = cell_matrix(per_cell, col)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    xs = np.arange(len(CONDITIONS))
    for i, a in enumerate(animals):
        y = Mx[i]
        ax.plot(xs, y, "-o", color="0.55", lw=0.9, ms=4, alpha=0.8, zorder=2)
        if np.isfinite(y[0]):
            ax.annotate(a, (xs[0] - 0.08, y[0]), fontsize=7, ha="right", va="center")
    mean = np.nanmean(Mx, axis=0)
    n = np.sum(np.isfinite(Mx), axis=0)
    sem = np.nanstd(Mx, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.errorbar(xs, mean, yerr=sem, fmt="s", ms=9, capsize=6, lw=1.6,
                color="k", zorder=3, label="mean +/- SEM (animals)")
    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                       rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def figures(traces, animals, M, metric, unit, per_cell):
    key   = "mean_z" if NORMALIZE == "zscore" else "mean_sub"
    win   = PRIMARY_WINDOW
    ylab  = "baseline-z" if NORMALIZE == "zscore" else "dF/F - baseline (%)"

    # --- Fig 1: grand average per condition, animals as the error unit -------
    fig1, ax = plt.subplots(figsize=(9.5, 5.2))
    for c in CONDITIONS:
        X = trace_matrix(traces, key, c, animals)
        ok = np.any(np.isfinite(X), axis=1)
        X = X[ok]
        if X.shape[0] == 0:
            continue
        m = np.nanmean(X, axis=0)
        s = (np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
             if X.shape[0] > 1 else np.zeros_like(m))
        ax.fill_between(GRID, m - s, m + s, alpha=0.20, color=CONDITION_COLORS[c])
        ax.plot(GRID, m, lw=1.6, color=CONDITION_COLORS[c],
                label=f"{CONDITION_LABELS[c]} (n={X.shape[0]} animals)")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvspan(*BASELINE_WIN, color="grey", alpha=0.10, label="baseline")
    ax.axvspan(*WINDOWS[win], color="k", alpha=0.08, label=f"{win} window")
    ax.set_xlabel("time from HTR onset (s)")
    ax.set_ylabel(ylab)
    ax.set_title("Peri-HTR PFC glutamate, averaged across ANIMALS "
                 "(shading = SEM across animals)")
    ax.legend(loc="upper left", fontsize=8)
    fig1.tight_layout()

    # --- Fig 2: per-animal paired plot across conditions ---------------------
    fig2, ax2 = plt.subplots(figsize=(6.0, 5.4))
    xs = np.arange(len(CONDITIONS))
    for i, a in enumerate(animals):
        y = M[i]
        ax2.plot(xs, y, "-o", color="0.5", lw=0.9, ms=4, alpha=0.8, zorder=2)
        if np.isfinite(y[0]):
            ax2.annotate(a, (xs[0] - 0.08, y[0]), fontsize=7, ha="right", va="center")
    mean = np.nanmean(M, axis=0)
    sem  = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(M), axis=0))
    ax2.errorbar(xs, mean, yerr=sem, fmt="s", ms=9, capsize=6, lw=1.6,
                 color="k", zorder=3, label="mean +/- SEM (animals)")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                        rotation=15, ha="right")
    ax2.set_ylabel(f"{win} response ({unit})")
    ax2.set_title(f"Per-animal {win} response {WINDOWS[win]} s\n"
                  f"lines = animals (within-subject); no stars by design")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    # --- Fig 3: isosbestic control, same layout as Fig 1 ---------------------
    fig3, ax3 = plt.subplots(figsize=(9.5, 4.6))
    for c in CONDITIONS:
        X = trace_matrix(traces, "mean_iso", c, animals)
        ok = np.any(np.isfinite(X), axis=1)
        X = X[ok]
        if X.shape[0] == 0:
            continue
        m = np.nanmean(X, axis=0)
        s = (np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
             if X.shape[0] > 1 else np.zeros_like(m))
        ax3.fill_between(GRID, m - s, m + s, alpha=0.20, color=CONDITION_COLORS[c])
        ax3.plot(GRID, m, lw=1.4, color=CONDITION_COLORS[c],
                 label=CONDITION_LABELS[c])
    ax3.axvline(0, color="k", lw=1, ls="--")
    ax3.axhline(0, color="grey", lw=0.6)
    ax3.axvspan(*WINDOWS[win], color="k", alpha=0.08)
    ax3.set_xlabel("time from HTR onset (s)")
    ax3.set_ylabel("isosbestic change (% of epoch baseline)")
    ax3.set_title("CONTROL: 405 isosbestic, same epochs. "
                  "Flat here = the signal figure is not movement.")
    ax3.legend(fontsize=8)
    fig3.tight_layout()

    # --- Fig 4: per-condition panels with individual animals -----------------
    fig4, axes = plt.subplots(1, len(CONDITIONS), figsize=(4.2 * len(CONDITIONS), 4.2),
                              sharey=True)
    axes = np.atleast_1d(axes)
    for j, c in enumerate(CONDITIONS):
        a4 = axes[j]
        X = trace_matrix(traces, key, c, animals)
        for i in range(X.shape[0]):
            if np.any(np.isfinite(X[i])):
                a4.plot(GRID, X[i], lw=0.7, color="0.65", alpha=0.9)
        ok = np.any(np.isfinite(X), axis=1)
        if ok.sum():
            a4.plot(GRID, np.nanmean(X[ok], axis=0), lw=1.8,
                    color=CONDITION_COLORS[c])
        a4.axvline(0, color="k", lw=0.9, ls="--")
        a4.axhline(0, color="grey", lw=0.6)
        a4.axvspan(*WINDOWS[win], color="k", alpha=0.08)
        a4.set_title(f"{CONDITION_LABELS[c]} (n={int(ok.sum())})", fontsize=10)
        a4.set_xlabel("time from HTR onset (s)")
    axes[0].set_ylabel(ylab)
    fig4.suptitle("Individual animals (grey) and condition mean", fontsize=11)
    fig4.tight_layout()

    # --- Fig 5: pre-twitch peak amplitude ------------------------------------
    pa_win = PEAK_AUC_WINDOW
    peak_col = (f"peak_{pa_win}_z" if NORMALIZE == "zscore"
                else f"peak_{pa_win}_sub_pct")
    peak_unit = "peak amplitude (baseline-z)" if NORMALIZE == "zscore" \
                else "peak amplitude (% dF/F above baseline)"
    paired_metric_figure(
        per_cell, peak_col, peak_unit,
        f"Pre-twitch peak amplitude {WINDOWS[pa_win]} s\nlines = animals")

    # --- Fig 6: pre-twitch AUC -----------------------------------------------
    if AUC_MODE not in ("positive", "signed"):
        raise ValueError("AUC_MODE must be 'positive' or 'signed'")
    auc_col = (f"auc_{AUC_MODE}_{pa_win}_z_s" if NORMALIZE == "zscore"
               else f"auc_{AUC_MODE}_{pa_win}_sub_pct_s")
    auc_unit = f"{AUC_MODE} AUC (z*s)" if NORMALIZE == "zscore" \
               else f"{AUC_MODE} AUC (% dF/F*s)"
    paired_metric_figure(
        per_cell, auc_col, auc_unit,
        f"Pre-twitch {AUC_MODE} AUC {WINDOWS[pa_win]} s\nlines = animals")

    # --- Fig 7: peak latency --------------------------------------------------
    paired_metric_figure(
        per_cell, f"peak_{pa_win}_latency_s",
        "peak latency from HTR onset (s)",
        f"Pre-twitch peak latency {WINDOWS[pa_win]} s\nlines = animals")

    plt.show()


# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    print("=" * 78)
    print("GROUP PERI-EVENT ANALYSIS -- unit of analysis = ANIMAL")
    print(f"baseline {BASELINE_WIN} s | windows {WINDOWS} | primary = "
          f"{PRIMARY_WINDOW}")
    print(f"normalisation = {NORMALIZE} (z denominator: {ZSCORE_SD}) | "
          f"slow drift {'removed' if REMOVE_SLOW_DRIFT else 'kept'}")
    print(f"peak/AUC window = {PEAK_AUC_WINDOW} {WINDOWS[PEAK_AUC_WINDOW]} | "
          f"headline AUC = {AUC_MODE}")
    print("=" * 78)

    results, skipped = [], []
    if EXCLUDE_ANIMALS:
        print(f"excluded animals by config: {sorted(EXCLUDE_ANIMALS)}")
    for sess in SESSIONS:
        if sess["animal"] in EXCLUDE_ANIMALS:
            skipped.append((sess["animal"], sess["condition"],
                            "animal excluded by EXCLUDE_ANIMALS"))
            continue
        if not sess.get("photometry") or not sess.get("scoring"):
            skipped.append((sess["animal"], sess["condition"], "empty path"))
            continue
        if sess["condition"] not in CONDITIONS:
            raise ValueError(f"unknown condition {sess['condition']!r}")
        if not os.path.exists(_resolve(sess["photometry"])):
            skipped.append((sess["animal"], sess["condition"], "photometry not found"))
            continue
        if not os.path.exists(_resolve(sess["scoring"])):
            skipped.append((sess["animal"], sess["condition"], "scoring not found"))
            continue
        try:
            results.append(process_session(sess))
        except Exception as e:
            skipped.append((sess["animal"], sess["condition"], f"failed: {e}"))
            print(f"    [!] FAILED: {e}")

    if skipped:
        print("\nskipped sessions:")
        for a, c, why in skipped:
            print(f"  {a} / {c}: {why}")
    if not results:
        print("\n[!] nothing loaded -- fill in the manifest.")
        return

    per_twitch_all, per_session, per_cell, traces = aggregate(results)
    animals, M, metric, unit = report(per_session, per_cell, traces)

    if SAVE_PRISM_EXCEL:
        export_prism_excel(per_cell)

    if SAVE_TABLES:
        for name, df in [("per_twitch_all", per_twitch_all),
                         ("per_session", per_session),
                         ("per_animal_condition", per_cell)]:
            out = os.path.abspath(os.path.join(OUT_DIR, f"group_{name}.csv"))
            try:
                df.to_csv(out, index=False)
                print(f"\nsaved -> {out}")
            except OSError as e:
                print(f"\n[warn] could not save {name} ({e})")
        # group traces, long format, one row per (condition, animal, time)
        rows = []
        key = "mean_z" if NORMALIZE == "zscore" else "mean_sub"
        for c in CONDITIONS:
            X = trace_matrix(traces, key, c, animals)
            Xi = trace_matrix(traces, "mean_iso", c, animals)
            for i, a in enumerate(animals):
                for ti, tt in enumerate(GRID):
                    rows.append(dict(condition=c, animal=a, time_s=tt,
                                     signal=X[i, ti], isosbestic=Xi[i, ti]))
        out = os.path.abspath(os.path.join(OUT_DIR, "group_traces_long.csv"))
        try:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"saved -> {out}")
        except OSError as e:
            print(f"[warn] could not save traces ({e})")

    figures(traces, animals, M, metric, unit, per_cell)


if __name__ == "__main__":
    main()