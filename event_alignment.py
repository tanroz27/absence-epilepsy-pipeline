"""
event_alignment.py
------------------
Aligns signals to LP 3-5 Hz oscillation onset times and produces
Figure 5F equivalent + pre-oscillation pupil analysis.

NORMALIZATION PHILOSOPHY (matching Dennis's Figure 5F):
  - Pupil: use pup_norm directly from .mat (already % of session max).
    If only pupil_area is available: sqrt(area) / session_99th_pct * 100.
    Session-wide normalization is applied ONCE before alignment.
    NO per-trial z-scoring. NO per-trial baseline subtraction.
    These operations were NOT in Dennis's shared scripts and created
    an artifactual sharp transition at t=0 in earlier versions.
  - All other signals: mean ± 95% bootstrapped CI across events.
  - Window: [-2, +2] s matching Dennis's Figure 5F x-axis.

Pre-ictal pupil analysis (Rozendal thesis):
  Separate from Figure 5F. Uses the same session-normalized pupil,
  tests whether the slope in [-3, 0] s is significantly negative.
  No additional normalization is applied.
"""

import os
import numpy as np
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
from scipy.stats import sem, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────
# Core alignment helper
# ─────────────────────────────────────────────────────────────────

def align_signal_to_events(
        signal   : np.ndarray,
        sig_t    : np.ndarray,
        event_t  : List[float],
        pre_s    : float = 2.0,
        post_s   : float = 2.0,
        target_hz: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract [-pre_s, +post_s] windows of signal around each event time.
    Resamples to target_hz for a common time axis.

    Returns
    -------
    time_axis : shape (n_samples,) — seconds relative to event onset
    matrix    : shape (n_events, n_samples) — NaN where out of bounds
    """
    dt   = 1.0 / target_hz
    t_ax = np.arange(-pre_s, post_s + dt/2, dt)
    mat  = np.full((len(event_t), len(t_ax)), np.nan)

    interp_fn = interp1d(sig_t, signal, kind='linear',
                         bounds_error=False, fill_value=np.nan)

    for i, t0 in enumerate(event_t):
        mat[i, :] = interp_fn(t_ax + t0)

    return t_ax, mat


def _bootstrap_ci(matrix: np.ndarray, n_boot: int = 1000,
                  ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    95% bootstrapped confidence interval across rows of matrix.
    Matches Dennis's 'Shaded regions in graphs indicate 95%
    bootstrapped confidence interval.'
    """
    rng   = np.random.default_rng(42)
    n     = matrix.shape[0]
    boot_means = np.full((n_boot, matrix.shape[1]), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.nanmean(matrix[idx], axis=0)
    lo = np.nanpercentile(boot_means, (1 - ci) / 2 * 100,       axis=0)
    hi = np.nanpercentile(boot_means, (1 - (1 - ci) / 2) * 100, axis=0)
    return lo, hi


def _session_normalize_pupil(pupil: np.ndarray) -> np.ndarray:
    """
    Convert pupil area → percent of session maximum.
    Matches pup_norm in Dennis's .mat files.
    If the input is already 0-100 (pup_norm), returns unchanged.
    """
    pup = np.sqrt(np.maximum(pupil, 0.0))   # area → proportional to diameter
    p99 = np.nanpercentile(pup, 99)
    if p99 > 0:
        pup = pup / p99 * 100
    return pup


# ─────────────────────────────────────────────────────────────────
# Vm spectrogram (for FFT panel in Fig 5F)
# ─────────────────────────────────────────────────────────────────

def compute_spectrogram_matrix(
        vm        : np.ndarray,
        t         : np.ndarray,
        event_t   : List[float],
        pre_s     : float = 2.0,
        post_s    : float = 2.0,
        window_s  : float = 1.25,
        overlap   : float = 0.9,
        ds_hz     : float = 40.0,
        freq_max  : float = 10.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Average time-frequency spectrogram of Vm aligned to events.
    Downsamples Vm to ds_hz before computing to match Dennis's
    40 Hz downsampling step.

    Returns: (f_ax, t_ax, mean_psd) or (None, None, None) if failed.
    """
    if vm is None or t is None or len(event_t) == 0:
        return None, None, None

    dt_orig   = t[1] - t[0]
    ds_factor = max(1, int(round(1.0 / dt_orig / ds_hz)))
    vm_ds = vm[::ds_factor]
    t_ds  = t[::ds_factor]
    fs_ds = 1.0 / (t_ds[1] - t_ds[0])

    nperseg  = int(round(window_s * fs_ds))
    noverlap = int(round(nperseg * overlap))

    psd_list = []
    for t0 in event_t:
        idx0 = np.searchsorted(t_ds, t0)
        i0   = idx0 - int(round(pre_s  * fs_ds))
        i1   = idx0 + int(round(post_s * fs_ds))
        if i0 < 0 or i1 > len(vm_ds):
            continue
        try:
            f, _, Sxx = spectrogram(vm_ds[i0:i1], fs=fs_ds,
                                    nperseg=nperseg, noverlap=noverlap,
                                    window='hamming')
            fmask = (f >= 0.5) & (f <= freq_max)
            psd_list.append(Sxx[fmask])
        except Exception:
            continue

    if not psd_list:
        return None, None, None

    min_t    = min(p.shape[1] for p in psd_list)
    mean_psd = np.nanmean(np.stack([p[:, :min_t] for p in psd_list]), axis=0)
    t_ax     = np.linspace(-pre_s, post_s, min_t)
    return f[fmask], t_ax, mean_psd


# ─────────────────────────────────────────────────────────────────
# Figure 5F equivalent
# ─────────────────────────────────────────────────────────────────

def compute_lp_psth(
        thalamic_units: list,
        event_t: list,
        pre_s: float = 2.0,
        post_s: float = 2.0,
        bin_s: float = 0.025,   # 25 ms — matches Dennis's bin size
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute LP population peri-stimulus firing rate aligned to events.
    For each unit: bin spikes in [-pre_s, +post_s] around each event.
    Average across units and events. Matches the orange LP trace in
    Dennis's Figure 5F.

    Returns: (time_axis, mean_firing_rate_spks_per_s)
    """
    if not thalamic_units or not event_t:
        return None, None

    bin_edges = np.arange(-pre_s, post_s + bin_s, bin_s)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Collect per-unit, per-event firing rate histograms
    all_unit_rates = []

    for spk_times in thalamic_units:
        spk = np.sort(np.asarray(spk_times).ravel())
        if len(spk) < 2:
            continue
        unit_event_rates = np.zeros((len(event_t), n_bins))
        for ei, t0 in enumerate(event_t):
            # Spikes relative to this event
            rel = spk - t0
            rel_in_window = rel[(rel >= -pre_s) & (rel <= post_s)]
            counts, _ = np.histogram(rel_in_window, bins=bin_edges)
            unit_event_rates[ei] = counts / bin_s   # convert to spikes/s
        all_unit_rates.append(np.mean(unit_event_rates, axis=0))  # avg across events

    if not all_unit_rates:
        return None, None

    # Average across units
    mean_rate = np.mean(np.stack(all_unit_rates), axis=0)
    return bin_centers, mean_rate


def figure_5f_equivalent(
        osc_onsets    : List[float],
        vm_clipped    : Optional[np.ndarray],
        t_vm          : Optional[np.ndarray],
        pupil         : Optional[np.ndarray],
        t_pupil       : Optional[np.ndarray],
        whisk         : Optional[np.ndarray],
        t_whisk       : Optional[np.ndarray],
        wheel         : Optional[np.ndarray],
        t_wheel       : Optional[np.ndarray],
        thalamic_units: Optional[List[np.ndarray]] = None,
        save_path     : str = 'figure5F_equivalent.png',
        pre_s         : float = 2.0,
        post_s        : float = 2.0,
) -> None:
    # LP population firing rate — the signal that defines the oscillation onset
    lp_t, lp_rate = (compute_lp_psth(thalamic_units, osc_onsets, pre_s, post_s)
                     if thalamic_units else (None, None))
    """
    Reproduce Dennis's Figure 5F layout:
      Row 1: Pupil (pup_norm, % max) — session-normalized, no z-score
      Row 2: Whisking motion energy
      Row 3: Walking speed
      Row 4: V1 Vm (spike-clipped)
      Row 5: Vm FFT power (spectrogram, 0.5–10 Hz)

    All traces: mean (bold) ± 95% bootstrapped CI (shaded), n events shown.
    Vertical dashed line at t = 0 = oscillation onset.
    """
    print(f"\n[figure_5f] {len(osc_onsets)} events, window [{-pre_s}, +{post_s}] s")

    beh_hz = 125.0   # behavioral signals
    vm_hz  = 500.0   # Vm (after downsampling)

    # ── Align all signals ────────────────────────────────────────
    def _align(sig, tt, hz):
        if sig is None or tt is None:
            return None, None
        return align_signal_to_events(sig, tt, osc_onsets,
                                      pre_s=pre_s, post_s=post_s,
                                      target_hz=hz)

    # Pupil: session-normalize first, then align — no further processing
    pup_norm = None
    if pupil is not None:
        # If pupil values are already 0-100 range (pup_norm from .mat), use as-is
        # Otherwise apply session normalization
        pmax = np.nanpercentile(np.sqrt(np.maximum(pupil, 0)), 99)
        if np.nanmax(pupil) > 200 or pmax > 10:
            # Looks like raw area — normalize
            pup_norm = _session_normalize_pupil(pupil)
        else:
            pup_norm = pupil.copy()

    t_pup, mat_pup = _align(pup_norm,    t_pupil, beh_hz)
    t_wsk, mat_wsk = _align(whisk,       t_whisk, beh_hz)
    t_wlk, mat_wlk = _align(wheel,       t_wheel, beh_hz)
    t_vm_,  mat_vm  = _align(vm_clipped, t_vm,    vm_hz)

    # Vm spectrogram
    freqs, t_spec, mean_psd = compute_spectrogram_matrix(
        vm_clipped, t_vm, osc_onsets, pre_s=pre_s, post_s=post_s)

    # ── Build figure ─────────────────────────────────────────────
    n_rows = 6
    fig = plt.figure(figsize=(9, 13))
    fig.suptitle(
        f'LP 3–5 Hz oscillation onset alignment  (N = {len(osc_onsets)} events)',
        fontsize=11, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.55,
                           left=0.13, right=0.91, top=0.94, bottom=0.06)
    axs = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    def _draw(ax, t_ax, mat, ylabel, color, units=''):
        if t_ax is None or mat is None:
            ax.set_visible(False)
            return
        valid = ~np.all(np.isnan(mat), axis=1)
        m = mat[valid]
        if len(m) == 0:
            ax.set_visible(False)
            return
        mn      = np.nanmean(m, axis=0)
        lo, hi  = _bootstrap_ci(m)
        ax.fill_between(t_ax, lo, hi, color=color, alpha=0.25)
        ax.plot(t_ax, mn, color=color, lw=2.0)
        ax.axvline(0, color='gray', lw=1.0, ls='--')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(t_ax[0], t_ax[-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        n_valid = int(valid.sum())
        ax.text(0.98, 0.92, f'n={n_valid}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='gray')

    _draw(axs[0], t_pup, mat_pup, 'Pupil\n(% max)',    '#555555')
    _draw(axs[1], t_wsk, mat_wsk, 'Whisking\n(a.u.)',   '#1f77b4')
    _draw(axs[2], t_wlk, mat_wlk, 'Walking\n(cm/s)',    '#d62728')
    _draw(axs[3], t_vm_,  mat_vm,  'V1 Vm\n(mV)',        'k')

    # LP population firing rate (the signal defining the oscillation)
    ax_lp = axs[4]
    if lp_t is not None and lp_rate is not None:
        ax_lp.fill_between(lp_t, 0, lp_rate, color='#e07800', alpha=0.4)
        ax_lp.plot(lp_t, lp_rate, color='#e07800', lw=1.8)
        ax_lp.axvline(0, color='gray', lw=1.0, ls='--')
        ax_lp.set_ylabel('LP firing\nrate (spks/s)', fontsize=9)
        ax_lp.set_xlim(-pre_s, post_s)
        ax_lp.text(0.98, 0.92, f'n={len(thalamic_units)} units',
                   transform=ax_lp.transAxes, ha='right', va='top',
                   fontsize=7, color='gray')
    else:
        ax_lp.set_visible(False)
    ax_lp.spines['top'].set_visible(False)
    ax_lp.spines['right'].set_visible(False)
    ax_lp.tick_params(labelsize=8)

    # FFT panel
    ax = axs[5]
    if freqs is not None and mean_psd is not None:
        im = ax.pcolormesh(t_spec, freqs,
                           10 * np.log10(mean_psd + 1e-30),
                           cmap='jet', shading='auto')
        ax.axhline(3, color='white', lw=0.7, ls='--', alpha=0.6)
        ax.axhline(5, color='white', lw=0.7, ls='--', alpha=0.6)
        ax.axvline(0, color='white', lw=1.0, ls='--')
        ax.set_ylim(freqs[0], freqs[-1])
        ax.set_xlim(t_spec[0], t_spec[-1])
        cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label('dB/Hz', fontsize=7);  cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, 'Vm spectrogram not available',
                transform=ax.transAxes, ha='center', color='gray')
    ax.set_ylabel('Freq\n(Hz)', fontsize=9)
    ax.set_xlabel('Time from alpha burst onset (s)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[figure_5f] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# Pre-oscillation pupil analysis  (Rozendal thesis)
# ─────────────────────────────────────────────────────────────────

def preictal_pupil_analysis(
        osc_onsets    : List[float],
        pupil         : np.ndarray,
        t_pupil       : np.ndarray,
        pre_ictal_s   : float = 2.0,
        baseline_pre_s: float = 1.0,
        pupil_lag_s   : float = 0.75,
        target_hz     : float = 120.0,
        save_path     : str   = 'pre_oscillation_pupil.png',
) -> Dict:
    """
    Tanner's thesis: characterize pupil dynamics before 3-5 Hz onset.

    PUPIL LAG CORRECTION (pupil_lag_s = 0.75s):
    The pupil lags the underlying neuromodulatory state by ~0.75s
    (Reimer et al. 2016; noted in Rozendal thesis prospectus and
    Dennis's Figure 4A caption).

    This means:
      - Pupil at t = +0.75s reflects neural state at t = 0 (onset)
      - Pupil at t = -1.25s reflects neural state at t = -2s

    The pre-oscillation slope is therefore measured from
    [-pre_ictal_s, +pupil_lag_s] — i.e., the window ends 0.75s AFTER
    the oscillation onset to capture the full lag-corrected constriction.

    Normalization: session-wide only. No per-trial z-scoring.
    Baseline: [-pre_ictal_s - baseline_pre_s, -pre_ictal_s].
    """
    print(f"\n[preictal_pupil] {len(osc_onsets)} events  "
          f"(pupil lag correction: +{pupil_lag_s}s)")

    # Session-normalize once
    pmax = np.nanpercentile(np.sqrt(np.maximum(pupil, 0)), 99)
    if np.nanmax(pupil) > 200 or pmax > 10:
        pup = _session_normalize_pupil(pupil)
    else:
        pup = pupil.copy()

    # Align: post window extends to pupil_lag_s after onset so we capture
    # the full lag-corrected pre-oscillation constriction in the pupil signal
    t_wide   = pre_ictal_s + baseline_pre_s
    post_win = max(pre_ictal_s, pupil_lag_s + 1.0)   # enough post-onset for display
    t_ax, mat = align_signal_to_events(
        pup, t_pupil, osc_onsets,
        pre_s=t_wide, post_s=post_win,
        target_hz=target_hz
    )

    # Baseline correction: mean of [-3.0, -2.0s] — a window well before the
    # pre-oscillation period, ensuring baseline and signal don't overlap.
    # Dennis uses [-2.0, -1.5s] (personal communication, April 2026) but
    # that window overlaps with the pre-oscillation pupil change in RN51.
    # Both approaches are reported; see Rozendal thesis methods.
    base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
    for i in range(mat.shape[0]):
        bval = np.nanmean(mat[i, base_mask])
        if not np.isnan(bval):
            mat[i, :] -= bval

    # Drop events with > 30% NaN
    ok = np.array([np.isnan(mat[i]).mean() < 0.30
                   for i in range(mat.shape[0])])
    mat = mat[ok]
    n   = mat.shape[0]
    print(f"  {n} events retained")

    mn      = np.nanmean(mat, axis=0)
    lo, hi  = _bootstrap_ci(mat)

    # Pre-oscillation slope: measured from -pre_ictal_s to +pupil_lag_s
    # (the lag-corrected window that captures the full pre-oscillation signal)
    pre_mask = (t_ax >= -pre_ictal_s) & (t_ax <= pupil_lag_s)
    t_pre    = t_ax[pre_mask]
    slopes   = []
    for row in mat[:, pre_mask]:
        ok_pts = ~np.isnan(row)
        if ok_pts.sum() >= 4:
            slopes.append(np.polyfit(t_pre[ok_pts], row[ok_pts], 1)[0])
    slopes = np.array(slopes)

    # Pre-oscillation mean value per event (vs 0 baseline)
    preictal_vals = np.nanmean(mat[:, pre_mask], axis=1)

    t_stat_v, p_val   = ttest_1samp(preictal_vals[~np.isnan(preictal_vals)], 0)
    t_stat_s, p_slope = (ttest_1samp(slopes, 0) if len(slopes) > 2
                         else (np.nan, np.nan))

    print(f"  Mean pre-oscillation Δpupil : {np.nanmean(preictal_vals):+.3f}%  p={p_val:.4f}")
    print(f"  Mean pre-oscillation slope  : {np.nanmean(slopes):+.4f} %/s  p={p_slope:.4f}")
    print(f"  (slope window: −{pre_ictal_s:.0f}s to +{pupil_lag_s}s, lag-corrected)")
    if np.nanmean(slopes) < 0:
        print(f"  → Pupil CONSTRICTS before oscillation onset ✓")

    # ── Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f'Pre-oscillation Pupil Analysis  (N = {n} events)\n'
        f'Slope: {np.nanmean(slopes):+.3f} %/s  p={p_slope:.4f}  |  '
        f'Δpupil: {np.nanmean(preictal_vals):+.3f}%  p={p_val:.4f}  |  '
        f'[0.75s lag correction applied]',
        fontsize=10, fontweight='bold')

    # Panel 1: mean ± 95% CI trace
    # Shaded region = lag-corrected pre-oscillation window [-pre_ictal_s, +pupil_lag_s]
    ax = axes[0]
    ax.axvspan(-pre_ictal_s, pupil_lag_s, alpha=0.07, color='red',
               label=f'Lag-corrected pre-oscillation\n[−{pre_ictal_s:.0f}s, +{pupil_lag_s}s]')
    ax.fill_between(t_ax, lo, hi, alpha=0.25, color='k')
    ax.plot(t_ax, mn, 'k', lw=2)
    ax.axvline(0,           color='gray', lw=1.2, ls='--', label='Oscillation onset')
    ax.axvline(pupil_lag_s, color='red',  lw=1.0, ls=':',
               label=f'+{pupil_lag_s}s (pupil lag)')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=9)
    ax.set_ylabel('Pupil (% max, baseline-corrected)', fontsize=9)
    ax.set_title(f'Mean ± 95% CI  (pupil lag = {pupil_lag_s}s)', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    # Panel 2: slope distribution
    ax = axes[1]
    if len(slopes):
        ax.hist(slopes, bins=min(20, len(slopes)),
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='k', lw=1.2, ls='--')
        ax.axvline(np.nanmean(slopes), color='red', lw=2,
                   label=f'Mean={np.nanmean(slopes):.3f}\np={p_slope:.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Pre-oscillation slope (%/s)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('Pre-oscillation slope distribution\n(H₀: slope = 0)', fontsize=10)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    # Panel 3: per-event Δpupil sorted
    ax = axes[2]
    sd = np.sort(preictal_vals[~np.isnan(preictal_vals)])
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in sd]
    ax.bar(range(len(sd)), sd, color=colors, width=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.text(0.02, 0.97,
            f'Mean = {np.nanmean(preictal_vals):+.3f}%\n'
            f'p = {p_val:.4f}\n'
            f'{np.mean(preictal_vals < 0)*100:.0f}% events constrict',
            transform=ax.transAxes, fontsize=8, va='top', color='red')
    ax.set_xlabel('Event (sorted)', fontsize=9)
    ax.set_ylabel('Δ Pupil (% max)', fontsize=9)
    ax.set_title('Per-event pupil change\n(pre-oscillation vs. baseline)', fontsize=10)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[preictal_pupil] Saved → {save_path}")

    return {
        't_ax'            : t_ax,
        'mean_pupil'      : mn,
        'ci_lo'           : lo,
        'ci_hi'           : hi,
        'matrix'          : mat,
        'n_events'        : n,
        'slopes'          : slopes,
        'slope_preictal'  : float(np.nanmean(slopes)),
        'p_slope'         : float(p_slope),
        'pre_oscillation_vals'   : preictal_vals,
        'mean_delta'      : float(np.nanmean(preictal_vals)),
        'p_delta'         : float(p_val),
        'pct_const'       : float(np.mean(preictal_vals < 0) * 100),
        'ttest_preictal'  : type('R', (), {'statistic': t_stat_v,
                                           'pvalue': p_val})(),
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS 2: Whisking pre-oscillation analysis
# ─────────────────────────────────────────────────────────────────

def preoscillation_whisking_analysis(
        osc_onsets    : List[float],
        whisk         : np.ndarray,
        t_whisk       : np.ndarray,
        pre_s         : float = 2.0,
        post_s        : float = 2.0,
        pupil_lag_s   : float = 0.75,
        target_hz     : float = 120.0,
        save_path     : str   = 'preoscillation_whisking.png',
) -> Dict:
    """
    Analysis 2: Whisking dynamics before 3-5 Hz oscillation onset.
    Uses same normalization approach as the significant pupil result:
    session-wide normalization, baseline [-3, -2s].
    Expected: whisking decreases before oscillation onset.
    """
    print(f"\n[pre_oscillation_whisk] {len(osc_onsets)} events")

    # Session-normalize: percent of session 99th percentile
    wh = np.asarray(whisk).ravel().astype(float)
    w99 = np.nanpercentile(wh, 99)
    if w99 > 0:
        wh = wh / w99 * 100

    # Align with extra pre window for baseline
    t_wide = pre_s + 1.0   # 1s extra for baseline window
    t_ax, mat = align_signal_to_events(
        wh, t_whisk, osc_onsets,
        pre_s=t_wide, post_s=post_s,
        target_hz=target_hz
    )

    # Baseline: [-3, -2s] — matching pupil analysis
    base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
    for i in range(mat.shape[0]):
        bval = np.nanmean(mat[i, base_mask])
        if not np.isnan(bval):
            mat[i, :] -= bval

    ok = np.array([np.isnan(mat[i]).mean() < 0.30 for i in range(mat.shape[0])])
    mat = mat[ok]
    n = mat.shape[0]
    print(f"  {n} events retained")

    mn     = np.nanmean(mat, axis=0)
    lo, hi = _bootstrap_ci(mat)

    # Pre-oscillation slope: [-2s, +0.75s] lag-corrected
    pre_mask  = (t_ax >= -pre_s) & (t_ax <= pupil_lag_s)
    t_pre     = t_ax[pre_mask]
    slopes    = []
    for row in mat[:, pre_mask]:
        ok_pts = ~np.isnan(row)
        if ok_pts.sum() >= 4:
            slopes.append(np.polyfit(t_pre[ok_pts], row[ok_pts], 1)[0])
    slopes = np.array(slopes)

    prevals = np.nanmean(mat[:, pre_mask], axis=1)
    _, p_val   = ttest_1samp(prevals[~np.isnan(prevals)], 0)
    _, p_slope = ttest_1samp(slopes, 0) if len(slopes) > 2 else (np.nan, np.nan)

    print(f"  Mean pre-oscillation Δwhisk: {np.nanmean(prevals):+.3f}%  p={p_val:.4f}")
    print(f"  Mean pre-oscillation slope : {np.nanmean(slopes):+.4f} %/s  p={p_slope:.4f}")
    if np.nanmean(slopes) < 0:
        print(f"  → Whisking DECREASES before oscillation onset ✓")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f'Pre-oscillation Whisking Analysis  (N = {n} events)\n'
        f'Slope: {np.nanmean(slopes):+.3f} %/s  p={p_slope:.4f}  |  '
        f'Δwhisk: {np.nanmean(prevals):+.3f}%  p={p_val:.4f}',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ax.axvspan(-pre_s, 0, alpha=0.07, color='blue',
               label=f'Pre-oscillation window [−{pre_s:.0f}s, 0s]')
    ax.fill_between(t_ax, lo, hi, alpha=0.25, color='steelblue')
    ax.plot(t_ax, mn, color='steelblue', lw=2)
    ax.axvline(0, color='gray', lw=1.2, ls='--', label='Oscillation onset')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=9)
    ax.set_ylabel('Whisking (% max, baseline-corrected)', fontsize=9)
    ax.set_title('Mean ± 95% CI', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    if len(slopes):
        ax.hist(slopes, bins=min(20, len(slopes)),
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='k', lw=1.2, ls='--')
        ax.axvline(np.nanmean(slopes), color='red', lw=2,
                   label=f'Mean={np.nanmean(slopes):.3f}\np={p_slope:.4f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Pre-oscillation slope (%/s)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('Slope distribution\n(H₀: slope = 0)', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[2]
    sd = np.sort(prevals[~np.isnan(prevals)])
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in sd]
    ax.bar(range(len(sd)), sd, color=colors, width=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.text(0.02, 0.97,
            f'Mean = {np.nanmean(prevals):+.3f}%\np = {p_val:.4f}\n'
            f'{np.mean(prevals < 0)*100:.0f}% events decrease',
            transform=ax.transAxes, fontsize=8, va='top', color='red')
    ax.set_xlabel('Event (sorted)', fontsize=9)
    ax.set_ylabel('Δ Whisking (% max)', fontsize=9)
    ax.set_title('Per-event whisking change', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[pre_oscillation_whisk] Saved → {save_path}")

    return {
        't_ax': t_ax, 'mean_whisk': mn, 'matrix': mat, 'n_events': n,
        'slopes': slopes, 'slope_mean': float(np.nanmean(slopes)),
        'p_slope': float(p_slope), 'prevals': prevals,
        'mean_delta': float(np.nanmean(prevals)), 'p_delta': float(p_val),
        'pct_decrease': float(np.mean(prevals < 0) * 100),
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS 3: Combined pupil + whisking
# ─────────────────────────────────────────────────────────────────

def combined_pupil_whisking_analysis(
        pupil_results : Dict,
        whisk_results : Dict,
        label         : str  = '',
        save_path     : str  = 'combined_pupil_whisking.png',
) -> Dict:
    """
    Analysis 3: Test whether pupil + whisking together predict oscillation onset.

    Approach: per-event composite score = mean of z-scored pupil slope
    + z-scored whisking slope. Test if composite score < 0.

    Also plots pupil vs. whisking slope scatter to show correlation.
    """
    print(f"\n[combined_analysis] {label}")

    p_slopes = pupil_results.get('slopes', np.array([]))
    w_slopes = whisk_results.get('slopes', np.array([]))

    if len(p_slopes) == 0 or len(w_slopes) == 0:
        print("  Missing slope data — skipping.")
        return {}

    # Match lengths (events must overlap)
    n = min(len(p_slopes), len(w_slopes))
    p_slopes = p_slopes[:n]
    w_slopes = w_slopes[:n]

    # Z-score each
    def _zscore(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10)

    p_z = _zscore(p_slopes)
    w_z = _zscore(w_slopes)

    # Composite: mean of the two z-scores per event
    composite = (p_z + w_z) / 2
    _, p_combined = ttest_1samp(composite[~np.isnan(composite)], 0)

    # Correlation between pupil and whisking slopes
    valid = ~(np.isnan(p_slopes) | np.isnan(w_slopes))
    if valid.sum() > 3:
        corr = np.corrcoef(p_slopes[valid], w_slopes[valid])[0, 1]
        from scipy.stats import pearsonr
        _, p_corr = pearsonr(p_slopes[valid], w_slopes[valid])
    else:
        corr, p_corr = np.nan, np.nan

    print(f"  Composite score mean : {np.nanmean(composite):+.4f}  p={p_combined:.4f}")
    print(f"  Pupil-whisk correlation: r={corr:.3f}  p={p_corr:.4f}")
    if np.nanmean(composite) < 0 and p_combined < 0.05:
        print(f"  → Combined signal SIGNIFICANT ✓")

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f'{label}  —  Analysis 3: Combined Pupil + Whisking\n'
        f'Composite score p={p_combined:.4f}  |  '
        f'Pupil-whisk correlation r={corr:.3f} p={p_corr:.4f}',
        fontsize=10, fontweight='bold')

    # Panel 1: Composite score distribution
    ax = axes[0]
    ax.hist(composite[~np.isnan(composite)], bins=20,
            color='purple', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', lw=1.2, ls='--')
    ax.axvline(np.nanmean(composite), color='red', lw=2,
               label=f'Mean={np.nanmean(composite):.3f}\np={p_combined:.4f}')
    ax.set_xlabel('Composite score (z)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('Combined pupil+whisking score\n(H₀: score = 0)', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Pupil slope vs. whisking slope scatter
    ax = axes[1]
    ax.scatter(p_slopes[valid], w_slopes[valid],
               alpha=0.5, s=20, color='purple')
    xlim = np.nanpercentile(np.abs(p_slopes[valid]), 98)
    ylim = np.nanpercentile(np.abs(w_slopes[valid]), 98)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlim(-xlim*1.2, xlim*1.2)
    ax.set_ylim(-ylim*1.2, ylim*1.2)
    ax.set_xlabel('Pupil pre-oscillation slope (%/s)', fontsize=9)
    ax.set_ylabel('Whisking pre-oscillation slope (%/s)', fontsize=9)
    ax.set_title(f'Slope correlation\nr={corr:.3f}  p={p_corr:.4f}', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Mean traces overlay (pupil + whisking, both normalized)
    ax = axes[2]
    p_t   = pupil_results.get('t_ax')
    p_mn  = pupil_results.get('mean_pupil') if 'mean_pupil' in pupil_results else pupil_results.get('mn')
    w_t   = whisk_results.get('t_ax')
    w_mn  = whisk_results.get('mean_whisk')

    if p_t is not None and p_mn is not None:
        p_mn_z = _zscore(p_mn)
        ax.plot(p_t, p_mn_z, color='#555555', lw=2, label='Pupil')
    if w_t is not None and w_mn is not None:
        w_mn_z = _zscore(w_mn)
        ax.plot(w_t, w_mn_z, color='steelblue', lw=2, label='Whisking')

    ax.axvline(0, color='gray', lw=1.2, ls='--', label='Oscillation onset')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=9)
    ax.set_ylabel('Signal (z-score)', fontsize=9)
    ax.set_title('Mean traces\n(both z-scored for overlay)', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[combined_analysis] Saved → {save_path}")

    return {
        'composite': composite, 'p_combined': float(p_combined),
        'corr': float(corr), 'p_corr': float(p_corr),
        'n_events': n,
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS 3 (REVISED): Whisking as predictor — stratified analysis
# ─────────────────────────────────────────────────────────────────

def whisking_as_predictor_analysis(
        osc_onsets  : List[float],
        whisk       : np.ndarray,
        t_whisk     : np.ndarray,
        pupil       : Optional[np.ndarray] = None,
        t_pupil     : Optional[np.ndarray] = None,
        pre_s       : float = 2.0,
        pupil_lag_s : float = 0.75,
        target_hz   : float = 120.0,
        save_path   : str   = 'whisking_as_predictor.png',
) -> Dict:
    """
    Analysis 3 (revised): Tests whisking as a predictor of oscillation onset.

    All detected oscillation events are included regardless of whisking state.
    Events are then split into:
      - "Whisking-decrease" events: whisking slope < 0 in [-2s, +0.75s]
      - "No-decrease" events: slope >= 0

    For each group, compare:
      1. Whisking trace shape
      2. Pupil trace shape (if available)

    This answers: are oscillations specifically associated with whisking decrease,
    or do they occur equally during any state?

    Also computes: what fraction of events are preceded by whisking decrease?
    Sensitivity (true positive rate) for using whisking decrease as a predictor.
    """
    print(f"\n[whisking_predictor] {len(osc_onsets)} events (all, unfiltered)")

    # Session-normalize whisking
    wh = np.asarray(whisk).ravel().astype(float)
    w99 = np.nanpercentile(wh, 99)
    if w99 > 0:
        wh = wh / w99 * 100

    # Align
    t_wide = pre_s + 1.0
    t_ax, mat = align_signal_to_events(
        wh, t_whisk, osc_onsets,
        pre_s=t_wide, post_s=pre_s,
        target_hz=target_hz
    )

    # Baseline correct
    base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
    for i in range(mat.shape[0]):
        bval = np.nanmean(mat[i, base_mask])
        if not np.isnan(bval):
            mat[i, :] -= bval

    ok = np.array([np.isnan(mat[i]).mean() < 0.30 for i in range(mat.shape[0])])
    mat = mat[ok]
    n = mat.shape[0]

    # Compute per-event whisking slope in [-pre_s, +pupil_lag_s]
    pre_mask = (t_ax >= -pre_s) & (t_ax <= pupil_lag_s)
    t_pre    = t_ax[pre_mask]
    slopes   = []
    for row in mat[:, pre_mask]:
        ok_pts = ~np.isnan(row)
        slopes.append(np.polyfit(t_pre[ok_pts], row[ok_pts], 1)[0]
                      if ok_pts.sum() >= 4 else np.nan)
    slopes = np.array(slopes)

    # Split: decrease vs no decrease
    decrease_idx    = np.where(slopes < 0)[0]
    no_decrease_idx = np.where(slopes >= 0)[0]
    n_decrease    = len(decrease_idx)
    n_no_decrease = len(no_decrease_idx)
    sensitivity   = n_decrease / n * 100

    print(f"  Events with whisking decrease : {n_decrease}/{n} ({sensitivity:.1f}%)")
    print(f"  Events without decrease       : {n_no_decrease}/{n} ({100-sensitivity:.1f}%)")

    # Pupil alignment for each group
    pup_decrease = pup_nodecrease = None
    if pupil is not None and t_pupil is not None:
        pup = np.asarray(pupil).ravel().astype(float)
        p99 = np.nanpercentile(np.sqrt(np.maximum(pup, 0)), 99)
        if p99 > 0:
            pup = np.sqrt(np.maximum(pup, 0)) / p99 * 100

        t_pax, pmat = align_signal_to_events(
            pup, t_pupil, osc_onsets,
            pre_s=t_wide, post_s=pre_s, target_hz=target_hz
        )
        for i in range(pmat.shape[0]):
            bv = np.nanmean(pmat[i, base_mask])
            if not np.isnan(bv):
                pmat[i, :] -= bv
        pmat = pmat[ok]

        pup_decrease    = pmat[decrease_idx]    if len(decrease_idx)    else None
        pup_nodecrease  = pmat[no_decrease_idx] if len(no_decrease_idx) else None

    # ── Figure ───────────────────────────────────────────────────
    n_cols = 3 if pupil is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    fig.suptitle(
        f'Analysis 3: Whisking as Predictor of Oscillation Onset\n'
        f'N = {n} events  |  Decrease: {n_decrease} ({sensitivity:.0f}%)  |  '
        f'No decrease: {n_no_decrease} ({100-sensitivity:.0f}%)',
        fontsize=10, fontweight='bold'
    )

    # Panel 1: whisking traces — decrease vs no-decrease
    ax = axes[0]
    if len(decrease_idx):
        mn_d = np.nanmean(mat[decrease_idx], axis=0)
        lo_d, hi_d = _bootstrap_ci(mat[decrease_idx])
        ax.fill_between(t_ax, lo_d, hi_d, alpha=0.20, color='#d62728')
        ax.plot(t_ax, mn_d, color='#d62728', lw=2,
                label=f'Whisking ↓ (n={n_decrease}, {sensitivity:.0f}%)')
    if len(no_decrease_idx):
        mn_n = np.nanmean(mat[no_decrease_idx], axis=0)
        lo_n, hi_n = _bootstrap_ci(mat[no_decrease_idx])
        ax.fill_between(t_ax, lo_n, hi_n, alpha=0.20, color='steelblue')
        ax.plot(t_ax, mn_n, color='steelblue', lw=2,
                label=f'Whisking → or ↑ (n={n_no_decrease}, {100-sensitivity:.0f}%)')
    ax.axvline(0, color='gray', lw=1.2, ls='--', label='Oscillation onset')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=9)
    ax.set_ylabel('Whisking (% max, baseline-corrected)', fontsize=9)
    ax.set_title('Whisking traces by group\n(all events included)', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: slope histogram with clear split at 0
    ax = axes[1]
    ax.hist(slopes[slopes < 0],  bins=15, color='#d62728',
            alpha=0.7, edgecolor='white', label=f'Decrease (n={n_decrease})')
    ax.hist(slopes[slopes >= 0], bins=15, color='steelblue',
            alpha=0.7, edgecolor='white', label=f'No decrease (n={n_no_decrease})')
    ax.axvline(0, color='k', lw=1.5, ls='--')
    ax.set_xlabel('Pre-oscillation whisking slope (%/s)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'Slope distribution\n{sensitivity:.0f}% of oscillations '
                 f'preceded by\nwhisking decrease', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 3: Pupil comparison if available
    if n_cols == 3 and pup_decrease is not None:
        ax = axes[2]
        if len(pup_decrease):
            mn_pd = np.nanmean(pup_decrease, axis=0)
            lo_pd, hi_pd = _bootstrap_ci(pup_decrease)
            ax.fill_between(t_pax, lo_pd, hi_pd, alpha=0.20, color='#d62728')
            ax.plot(t_pax, mn_pd, color='#d62728', lw=2,
                    label=f'Whisking ↓ events (n={len(pup_decrease)})')
        if pup_nodecrease is not None and len(pup_nodecrease):
            mn_pn = np.nanmean(pup_nodecrease, axis=0)
            lo_pn, hi_pn = _bootstrap_ci(pup_nodecrease)
            ax.fill_between(t_pax, lo_pn, hi_pn, alpha=0.20, color='steelblue')
            ax.plot(t_pax, mn_pn, color='steelblue', lw=2,
                    label=f'No whisking ↓ (n={len(pup_nodecrease)})')
        ax.axvline(0, color='gray', lw=1.2, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlabel('Time from oscillation onset (s)', fontsize=9)
        ax.set_ylabel('Pupil (% max, baseline-corrected)', fontsize=9)
        ax.set_title('Pupil: events with vs.\nwithout whisking decrease', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[whisking_predictor] Saved → {save_path}")

    return {
        'n_events': n, 'n_decrease': n_decrease,
        'n_no_decrease': n_no_decrease,
        'sensitivity': sensitivity,
        'slopes': slopes,
        'mat': mat, 't_ax': t_ax,
    }


# ─────────────────────────────────────────────────────────────────
# GRAND AVERAGE WHISKING across recordings
# ─────────────────────────────────────────────────────────────────

def grand_average_whisking(
        all_results : list,
        pre_s       : float = 2.0,
        post_s      : float = 2.0,
        save_path   : str   = 'grand_average_whisking.png',
) -> None:
    """
    Pool whisking-aligned traces across all recordings.
    Produces the grand-average whisking PETH equivalent to the
    grand-average pupil figure.
    """
    pooled     = []
    labels     = []
    common_t   = None
    n_per_rec  = []

    for r in all_results:
        wr = r.get('whisk_results', {})
        if not wr or wr.get('matrix') is None:
            continue
        mat = wr['matrix'].copy()
        t   = wr['t_ax']

        valid = ~np.all(np.isnan(mat), axis=1)
        mat = mat[valid]
        if len(mat) == 0:
            continue

        pooled.append(mat)
        labels.append(r['label'])
        n_per_rec.append(len(mat))
        if common_t is None:
            common_t = t

    if not pooled or common_t is None:
        print("  Not enough whisking data for grand average.")
        return

    all_mat = np.vstack(pooled)
    n_total = len(all_mat)
    mn      = np.nanmean(all_mat, axis=0)
    lo, hi  = _bootstrap_ci(all_mat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'Grand-Average Pre-oscillation Whisking\n'
        f'{" | ".join(labels)}  —  N = {n_total} events  [no lag correction]',
        fontsize=12, fontweight='bold'
    )

    # Panel 1: Grand mean ± 95% CI
    ax = axes[0]
    ax.axvspan(-pre_s, 0, alpha=0.07, color='steelblue',
               label='Pre-oscillation window')
    ax.fill_between(common_t, lo, hi, alpha=0.25, color='steelblue')
    ax.plot(common_t, mn, color='steelblue', lw=2.5)
    ax.axvline(0, color='k', lw=1.4, ls='--', label='Oscillation onset')
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=10)
    ax.set_ylabel('Whisking (% max, baseline-corrected)', fontsize=10)
    ax.set_title('Grand-average whisking\naligned to 3–5 Hz onset', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Per-recording mean traces overlaid
    ax = axes[1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    start = 0
    for i, (mat_i, lbl, n_i) in enumerate(zip(pooled, labels, n_per_rec)):
        mn_i = np.nanmean(mat_i, axis=0)
        ax.plot(common_t, mn_i, color=colors[i % len(colors)],
                lw=2, label=f'{lbl} (n={n_i})')
    ax.plot(common_t, mn, color='k', lw=2.5, ls='--', label='Grand mean')
    ax.axvline(0, color='gray', lw=1.2, ls='--')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=10)
    ax.set_ylabel('Whisking (% max, baseline-corrected)', fontsize=10)
    ax.set_title('Per-recording mean traces\n(grand mean in black)', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[grand_avg_whisking] Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════
# SUPPLEMENTARY ANALYSES 1-6
# ═════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# ANALYSIS S1: Pupil constriction probability (Dennis Figure 4B)
# ─────────────────────────────────────────────────────────────────

def pupil_constriction_probability(
        osc_onsets  : List[float],
        pupil       : np.ndarray,
        t_pupil     : np.ndarray,
        window_s    : float = 0.5,
        n_perms     : int   = 1000,
        target_hz   : float = 120.0,
        save_path   : str   = 'pupil_constriction_prob.png',
) -> Dict:
    """
    Direct replication of Dennis's Figure 4B.
    For each oscillation event: did pupil decrease in the [window_s] 
    before onset? Compare frequency to random time points.
    Dennis found ~80% of events preceded by constriction.
    """
    print(f"\n[S1 constriction_prob] {len(osc_onsets)} events, window={window_s}s")

    # Session-normalize
    pup = np.sqrt(np.maximum(np.asarray(pupil).ravel(), 0))
    p99 = np.nanpercentile(pup, 99)
    if p99 > 0:
        pup = pup / p99 * 100

    interp_fn = interp1d(t_pupil, pup, bounds_error=False, fill_value=np.nan)

    def _constricted(t0):
        """True if pupil decreased in [-window_s, 0] before t0."""
        t_start = t0 - window_s
        t_mid   = t0 - window_s / 2
        early = np.nanmean(interp_fn(np.linspace(t_start, t_mid, 20)))
        late  = np.nanmean(interp_fn(np.linspace(t_mid,   t0,    20)))
        return (late < early), (early - late)  # True = constricted, magnitude

    # Real events
    real_results = [_constricted(t0) for t0 in osc_onsets]
    real_binary  = np.array([r[0] for r in real_results])
    real_mag     = np.array([r[1] for r in real_results])
    real_prob    = np.nanmean(real_binary) * 100

    # Permutation: random time points
    rng = np.random.default_rng(42)
    t_min = t_pupil[0] + window_s + 0.5
    t_max = t_pupil[-1] - 0.5
    perm_probs = []
    for _ in range(n_perms):
        fake = rng.uniform(t_min, t_max, size=len(osc_onsets))
        fb   = np.array([_constricted(t)[0] for t in fake])
        perm_probs.append(np.nanmean(fb) * 100)
    perm_probs = np.array(perm_probs)
    p_perm = np.mean(perm_probs >= real_prob)

    print(f"  Constriction probability: {real_prob:.1f}%")
    print(f"  Permutation null mean:    {np.mean(perm_probs):.1f}%")
    print(f"  Permutation p-value:      {p_perm:.4f}")
    print(f"  Dennis's value:           ~80%")

    # Figure (matches Dennis Fig 4B style)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f'S1: Pupil Constriction Probability Before Oscillation Onset\n'
        f'(Replication of Dennis Fig 4B)\n'
        f'Constriction in {window_s}s before onset: {real_prob:.1f}%  '
        f'(permutation p = {p_perm:.4f})',
        fontsize=10, fontweight='bold')

    # Panel 1: real vs. shuffled (boxplot style like Dennis)
    ax = axes[0]
    ax.boxplot([perm_probs, [real_prob]],
               labels=['Shuffled\n(n=1000)', 'Real\nonsets'],
               widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='#dddddd'),
               medianprops=dict(color='k', lw=2))
    ax.scatter([2], [real_prob], color='red', s=80, zorder=5)
    ax.set_ylabel('Pupil constriction probability (%)', fontsize=10)
    ax.set_title(f'Real ({real_prob:.1f}%) vs. shuffled\n'
                 f'(Dennis: ~80%)', fontsize=10)
    ax.axhline(50, color='gray', lw=0.8, ls='--', label='Chance')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: per-event constriction magnitude
    ax = axes[1]
    sort_idx = np.argsort(real_mag)[::-1]
    colors_m = ['#d62728' if real_binary[i] else '#1f77b4'
                for i in sort_idx]
    ax.bar(range(len(real_mag)), real_mag[sort_idx], color=colors_m, width=0.9)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Event (sorted by constriction magnitude)', fontsize=9)
    ax.set_ylabel(f'Pupil change in {window_s}s before onset (% max)', fontsize=9)
    ax.set_title(f'Per-event constriction\n'
                 f'{real_prob:.0f}% constrict (red)', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[S1] Saved → {save_path}")

    return {
        'prob_real': real_prob, 'perm_null': perm_probs,
        'p_perm': p_perm, 'binary': real_binary, 'magnitude': real_mag
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS S2: Oscillation event duration distribution (Dennis Fig 4C)
# ─────────────────────────────────────────────────────────────────

def oscillation_duration_distribution(
        osc_onsets      : List[float],
        thalamic_units  : List[np.ndarray],
        v1_onsets_raw   = None,   # threshold-crossing onsets from detect_v1_oscillations
        v1_offsets      = None,   # corresponding offsets (when alpha power drops)
        save_path       : str = 'osc_duration_dist.png',
) -> Dict:
    """
    Replication of Dennis Figure 4C.

    Duration measurement strategy:
      PRIMARY  — V1 spectrogram-based: offset - threshold_crossing_onset.
                 Matches Nestvogel & McCormick's definition exactly.
                 Used when v1_onsets_raw and v1_offsets are provided AND a
                 V1 event can be matched to each thalamic onset within 1 s.
      FALLBACK — Burst-pair span: last_spike_of_last_burst - first_spike_of_first_burst.
                 Used when no V1 match exists (preserves backward-compatibility).

    The two methods are plotted separately so the difference is transparent.
    """
    from thalamic_burst_detection import (detect_bursts_single_unit,
                                          detect_alpha_oscillations_single_unit,
                                          ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT)

    use_v1 = (v1_onsets_raw is not None and v1_offsets is not None
              and len(v1_onsets_raw) > 0 and len(v1_offsets) > 0)

    print(f"\n[S2 duration_dist] {len(osc_onsets)} thalamic events  "
          f"| V1 events available: {len(v1_onsets_raw) if use_v1 else 0}")

    # ── V1-based durations (primary) ────────────────────────────────────────
    v1_durations   = []
    n_matched      = 0
    unmatched_onsets = []

    if use_v1:
        v1_on  = np.array(v1_onsets_raw)
        v1_off = np.array(v1_offsets)
        for t0 in osc_onsets:
            diffs = np.abs(v1_on - t0)
            best  = np.argmin(diffs)
            if diffs[best] <= 2.0:          # match window: 2 seconds
                dur = float(v1_off[best] - v1_on[best])
                if 0.1 < dur < 10.0:
                    v1_durations.append(dur)
                    n_matched += 1
                else:
                    unmatched_onsets.append(t0)
            else:
                unmatched_onsets.append(t0)
        print(f"  V1 match: {n_matched}/{len(osc_onsets)} thalamic events matched "
              f"({len(unmatched_onsets)} unmatched → burst-span fallback)")

    # ── Burst-span durations (fallback for unmatched, or primary if no V1) ──
    burst_durations = []
    targets = unmatched_onsets if use_v1 else osc_onsets

    if targets:
        for spk_times in thalamic_units:
            st = np.sort(np.asarray(spk_times).ravel())
            _, true_bursts = detect_bursts_single_unit(st)
            if not true_bursts:
                continue

            ibi = np.array([true_bursts[i+1][0] - true_bursts[i][-1]
                            for i in range(len(true_bursts)-1)])
            alpha_idx = np.where(
                (ibi >= ALPHA_LOWER_LIMIT) & (ibi <= ALPHA_UPPER_LIMIT)
            )[0]
            if len(alpha_idx) == 0:
                continue

            event_groups = []
            cur = [alpha_idx[0], alpha_idx[0]+1]
            for k in range(1, len(alpha_idx)):
                if alpha_idx[k] == alpha_idx[k-1] + 1:
                    cur.append(alpha_idx[k]+1)
                else:
                    event_groups.append(sorted(set(cur)))
                    cur = [alpha_idx[k], alpha_idx[k]+1]
            event_groups.append(sorted(set(cur)))

            for grp in event_groups:
                onset_t = true_bursts[grp[0]][0]
                # Only include if this onset is in our target list
                if any(abs(onset_t - t0) < 0.05 for t0 in targets):
                    first_burst = true_bursts[grp[0]]
                    last_burst  = true_bursts[grp[-1]]
                    dur = last_burst[-1] - first_burst[0]
                    if 0.1 < dur < 10.0:
                        burst_durations.append(dur)

    # ── Combine and report ──────────────────────────────────────────────────
    all_durations = v1_durations + burst_durations
    durations     = np.array(all_durations)

    if len(durations) == 0:
        print("  No durations computed.")
        return {}

    median_dur    = np.median(durations)
    median_v1     = np.median(v1_durations)     if v1_durations     else None
    median_burst  = np.median(burst_durations)  if burst_durations  else None

    print(f"  Overall median duration : {median_dur:.3f} s  (N&M reported: ~1.0 s)")
    if median_v1    is not None: print(f"  V1-based median         : {median_v1:.3f} s  (n={len(v1_durations)})")
    if median_burst is not None: print(f"  Burst-span median       : {median_burst:.3f} s  (n={len(burst_durations)})")

    # ── Figure ──────────────────────────────────────────────────────────────
    n_panels = 2 if (v1_durations and burst_durations) else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    method_label = "V1 spectrogram (threshold-to-threshold)" if use_v1 else "Burst-pair span"
    fig.suptitle(
        f'S2: Oscillation Event Duration Distribution\n'
        f'(Replication of Nestvogel & McCormick Fig 4C)\n'
        f'Primary method: {method_label}',
        fontsize=10, fontweight='bold')

    bins = np.arange(0, 5, 0.1)

    # Panel 1: primary method (V1 or burst-span)
    ax = axes[0]
    primary_durs = np.array(v1_durations if v1_durations else burst_durations)
    primary_med  = np.median(primary_durs)
    ax.hist(primary_durs, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(primary_med, color='red',  lw=2.5,
               label=f'This pipeline median = {primary_med:.2f} s')
    ax.axvline(1.0,         color='gray', lw=2.0, ls='--',
               label='N&M reported median = ~1.0 s')
    ax.set_xlabel('Oscillation duration (s)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('V1 spectrogram-based duration\n(threshold-crossing to threshold-drop)'
                 if v1_durations else 'Burst-pair span duration', fontsize=9)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: burst-span comparison (only shown if both methods ran)
    if n_panels == 2:
        ax2 = axes[1]
        bs  = np.array(burst_durations)
        ax2.hist(bs, bins=bins, color='orange', edgecolor='white', alpha=0.8)
        ax2.axvline(np.median(bs), color='red',  lw=2.5,
                    label=f'Burst-span median = {np.median(bs):.2f} s')
        ax2.axvline(1.0,           color='gray', lw=2.0, ls='--',
                    label='N&M reported median = ~1.0 s')
        ax2.set_xlabel('Oscillation duration (s)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Burst-pair span duration\n(fallback / comparison)', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

    return {
        'durations'      : durations,
        'v1_durations'   : np.array(v1_durations),
        'burst_durations': np.array(burst_durations),
        'median_overall' : median_dur,
        'median_v1'      : median_v1,
        'median_burst'   : median_burst,
        'n_matched'      : n_matched,
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS S3: Whisking offset latency to oscillation onset (Dennis Fig 4E/H)
# ─────────────────────────────────────────────────────────────────

def whisking_offset_latency(
        osc_onsets : List[float],
        whisk      : np.ndarray,
        t_whisk    : np.ndarray,
        max_lag_s  : float = 3.0,
        thresh_pct : float = 20.0,
        save_path  : str   = 'whisk_offset_latency.png',
) -> Dict:
    """
    For each oscillation event preceded by whisking, compute the time
    from whisking offset to oscillation onset.
    Dennis Figure 4H: distribution peaks ~0.5-1s after whisking offset.
    Only events where whisking was clearly above threshold before onset
    are included (matching Dennis's criterion).
    """
    print(f"\n[S3 whisk_offset_latency] {len(osc_onsets)} events")

    wh = np.asarray(whisk).ravel().astype(float)
    w99 = np.nanpercentile(wh, 99)
    if w99 > 0:
        wh = wh / w99 * 100
    thresh = thresh_pct   # % of max

    interp_fn = interp1d(t_whisk, wh, bounds_error=False, fill_value=np.nan)

    latencies = []
    for t0 in osc_onsets:
        # Look back max_lag_s before onset
        t_back = np.arange(t0 - max_lag_s, t0, 0.01)
        if len(t_back) == 0:
            continue
        w_trace = interp_fn(t_back)

        # Find last time whisking was above threshold
        above = np.where(w_trace > thresh)[0]
        if len(above) == 0:
            continue  # No whisking before onset — skip

        last_above_idx = above[-1]
        t_offset       = t_back[last_above_idx]
        latency        = t0 - t_offset
        if 0 < latency <= max_lag_s:
            latencies.append(latency)

    latencies = np.array(latencies)
    n_with_whisk = len(latencies)
    pct_with_whisk = n_with_whisk / len(osc_onsets) * 100

    print(f"  Events with prior whisking: {n_with_whisk}/{len(osc_onsets)} ({pct_with_whisk:.0f}%)")
    if len(latencies):
        print(f"  Median latency: {np.median(latencies):.3f} s  (Dennis: ~0.5-1 s)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f'S3: Whisking Offset → Oscillation Onset Latency\n'
        f'(Replication of Dennis Fig 4E/H)\n'
        f'{n_with_whisk}/{len(osc_onsets)} events preceded by whisking  |  '
        f'Median latency = {np.median(latencies):.2f} s',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    if len(latencies):
        ax.hist(latencies, bins=np.arange(0, max_lag_s + 0.1, 0.1),
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.median(latencies), color='red', lw=2.5,
                   label=f'Median = {np.median(latencies):.2f} s')
        ax.axvspan(0.5, 1.0, alpha=0.10, color='orange',
                   label="Dennis's peak range (0.5–1 s)")
    ax.set_xlabel('Time from whisking offset to oscillation onset (s)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('Latency distribution', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.bar(['With prior\nwhisking', 'No prior\nwhisking'],
           [n_with_whisk, len(osc_onsets) - n_with_whisk],
           color=['steelblue', '#aaaaaa'], edgecolor='k', width=0.5)
    ax.set_ylabel('N events', fontsize=9)
    ax.set_title(f'Event breakdown\n({pct_with_whisk:.0f}% preceded by whisking)', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[S3] Saved → {save_path}")

    return {'latencies': latencies, 'n_with_whisk': n_with_whisk,
            'pct_with_whisk': pct_with_whisk}


# ─────────────────────────────────────────────────────────────────
# ANALYSIS S4: Stratification by baseline pupil level
# ─────────────────────────────────────────────────────────────────

def pupil_baseline_stratification(
        osc_onsets  : List[float],
        pupil       : np.ndarray,
        t_pupil     : np.ndarray,
        baseline_s  : float = 2.0,
        target_hz   : float = 120.0,
        save_path   : str   = 'pupil_baseline_strat.png',
) -> Dict:
    """
    Split events into high vs. low baseline pupil quartiles at onset.
    Tests: do events starting from higher arousal show larger pre-event decrease?
    Mechanistic prediction: higher baseline → larger withdrawal → longer oscillation.
    """
    print(f"\n[S4 baseline_strat] {len(osc_onsets)} events")

    pup = np.sqrt(np.maximum(np.asarray(pupil).ravel(), 0))
    p99 = np.nanpercentile(pup, 99)
    if p99 > 0:
        pup = pup / p99 * 100

    interp_fn = interp1d(t_pupil, pup, bounds_error=False, fill_value=np.nan)

    # Baseline pupil = mean in [-2.0, -1.5s] before each onset
    baseline_vals = np.array([
        np.nanmean(interp_fn(np.linspace(t0 - 2.0, t0 - 1.5, 30)))
        for t0 in osc_onsets
    ])

    # Align all events
    t_wide = baseline_s + 1.0
    t_ax, mat = align_signal_to_events(
        pup, t_pupil, osc_onsets,
        pre_s=t_wide, post_s=baseline_s,
        target_hz=target_hz
    )
    base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
    for i in range(mat.shape[0]):
        bv = np.nanmean(mat[i, base_mask])
        if not np.isnan(bv):
            mat[i, :] -= bv

    ok = ~np.isnan(baseline_vals)
    mat = mat[ok]; baseline_vals = baseline_vals[ok]

    # Quartile split: Q1 (lowest 25%) vs Q4 (highest 25%)
    q1 = np.percentile(baseline_vals, 25)
    q4 = np.percentile(baseline_vals, 75)
    low_idx  = np.where(baseline_vals <= q1)[0]
    high_idx = np.where(baseline_vals >= q4)[0]

    print(f"  Low baseline (Q1 ≤ {q1:.1f}%): n={len(low_idx)}")
    print(f"  High baseline (Q4 ≥ {q4:.1f}%): n={len(high_idx)}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f'S4: Pre-oscillation Pupil Stratified by Baseline Level\n'
        f'Low baseline Q1 (n={len(low_idx)}) vs. High baseline Q4 (n={len(high_idx)})',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    if len(high_idx):
        mn_h = np.nanmean(mat[high_idx], axis=0)
        lo_h, hi_h = _bootstrap_ci(mat[high_idx])
        ax.fill_between(t_ax, lo_h, hi_h, alpha=0.20, color='#d62728')
        ax.plot(t_ax, mn_h, color='#d62728', lw=2,
                label=f'High baseline Q4\n(n={len(high_idx)}, ≥{q4:.0f}%)')
    if len(low_idx):
        mn_l = np.nanmean(mat[low_idx], axis=0)
        lo_l, hi_l = _bootstrap_ci(mat[low_idx])
        ax.fill_between(t_ax, lo_l, hi_l, alpha=0.20, color='steelblue')
        ax.plot(t_ax, mn_l, color='steelblue', lw=2,
                label=f'Low baseline Q1\n(n={len(low_idx)}, ≤{q1:.0f}%)')
    ax.axvline(0, color='gray', lw=1.2, ls='--')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from onset (s)', fontsize=9)
    ax.set_ylabel('Pupil (% max, baseline-corrected)', fontsize=9)
    ax.set_title('Pupil traces by baseline level', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Pre-event Δpupil per quartile
    pre_mask = (t_ax >= -2.0) & (t_ax <= 0.75)
    ax = axes[1]
    groups = {'Low Q1': mat[low_idx][:, pre_mask].mean(axis=1) if len(low_idx) else np.array([]),
              'High Q4': mat[high_idx][:, pre_mask].mean(axis=1) if len(high_idx) else np.array([])}
    positions = [1, 2]
    data_to_plot = [groups['Low Q1'], groups['High Q4']]
    data_to_plot = [d[~np.isnan(d)] for d in data_to_plot]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor='#dddddd'),
                    medianprops=dict(color='red', lw=2))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Low Q1\nbaseline', 'High Q4\nbaseline'])
    ax.set_ylabel('Mean pre-oscillation Δpupil (%)', fontsize=9)
    ax.set_title('Do higher-baseline events\nshow larger decrease?', fontsize=10)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Baseline vs. Δpupil scatter
    ax = axes[2]
    pre_vals = np.nanmean(mat[:, pre_mask], axis=1)
    ok2 = ~np.isnan(pre_vals)
    ax.scatter(baseline_vals[ok2], pre_vals[ok2],
               alpha=0.4, s=15, color='purple')
    if ok2.sum() > 3:
        from scipy.stats import pearsonr
        r, p = pearsonr(baseline_vals[ok2], pre_vals[ok2])
        z = np.polyfit(baseline_vals[ok2], pre_vals[ok2], 1)
        xline = np.linspace(baseline_vals[ok2].min(), baseline_vals[ok2].max(), 100)
        ax.plot(xline, np.polyval(z, xline), 'r-', lw=2,
                label=f'r={r:.3f}, p={p:.4f}')
        print(f"  Baseline vs Δpupil correlation: r={r:.3f}, p={p:.4f}")
    ax.set_xlabel('Baseline pupil (% max)', fontsize=9)
    ax.set_ylabel('Mean pre-oscillation Δpupil (%)', fontsize=9)
    ax.set_title('Baseline level vs. pre-event change', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[S4] Saved → {save_path}")

    return {'baseline_vals': baseline_vals, 'low_idx': low_idx, 'high_idx': high_idx}


# ─────────────────────────────────────────────────────────────────
# ANALYSIS S5: Baseline pupil vs. oscillation duration
# ─────────────────────────────────────────────────────────────────

def pupil_vs_oscillation_duration(
        osc_onsets      : List[float],
        thalamic_units  : List[np.ndarray],
        pupil           : np.ndarray,
        t_pupil         : np.ndarray,
        v1_onsets_raw   = None,
        v1_offsets      = None,
        v1_matched_only : bool = True,
        save_path       : str = 'pupil_vs_duration.png',
) -> Dict:
    """
    Correlate pre-event pupil level (or constriction magnitude)
    against oscillation duration.
    Mechanistic prediction (Dennis Discussion + McCormick & Bal 1997):
    deeper neuromodulatory withdrawal → deeper thalamic hyperpolarization
    → more LTS cycles → longer oscillation.
    If pupil proxies arousal/neuromodulation, pre-event pupil drop
    should negatively correlate with oscillation duration.

    Duration source: V1 spectrogram-based (threshold-to-threshold) when
    v1_onsets_raw/v1_offsets are provided; burst-pair span otherwise.

    v1_matched_only: if True (default), only include events with a V1 match.
    This avoids mixing V1-based and burst-span durations in the correlation,
    which would confound the analysis with a methodological artifact.
    """
    from thalamic_burst_detection import (detect_bursts_single_unit,
                                          detect_alpha_oscillations_single_unit,
                                          ALPHA_LOWER_LIMIT, ALPHA_UPPER_LIMIT)
    from scipy.stats import pearsonr, spearmanr

    print(f"\n[S5 pupil_vs_duration] {len(osc_onsets)} events")

    # ── Build duration map ───────────────────────────────────────────────────
    event_duration_map = {}

    # Primary: V1-based durations
    use_v1 = (v1_onsets_raw is not None and v1_offsets is not None
              and len(v1_onsets_raw) > 0)
    if use_v1:
        v1_on  = np.array(v1_onsets_raw)
        v1_off = np.array(v1_offsets)
        for t0 in osc_onsets:
            diffs = np.abs(v1_on - t0)
            best  = np.argmin(diffs)
            if diffs[best] <= 2.0:
                dur = float(v1_off[best] - v1_on[best])
                if 0.1 < dur < 10.0:
                    event_duration_map[t0] = dur

    # Fallback: burst-span for unmatched events (skipped if v1_matched_only=True)
    unmatched = [t0 for t0 in osc_onsets if t0 not in event_duration_map]
    if unmatched and not v1_matched_only:
        burst_dur_map = {}
        for spk_times in thalamic_units:
            st = np.sort(np.asarray(spk_times).ravel())
            _, true_bursts = detect_bursts_single_unit(st)
            if not true_bursts:
                continue
            ibi = np.array([true_bursts[i+1][0] - true_bursts[i][-1]
                            for i in range(len(true_bursts)-1)])
            alpha_idx = np.where(
                (ibi >= ALPHA_LOWER_LIMIT) & (ibi <= ALPHA_UPPER_LIMIT)
            )[0]
            if not len(alpha_idx):
                continue

            cur = [alpha_idx[0], alpha_idx[0]+1]
            groups = []
            for k in range(1, len(alpha_idx)):
                if alpha_idx[k] == alpha_idx[k-1] + 1:
                    cur.append(alpha_idx[k]+1)
                else:
                    groups.append(sorted(set(cur)))
                    cur = [alpha_idx[k], alpha_idx[k]+1]
            groups.append(sorted(set(cur)))

            for grp in groups:
                onset_t = true_bursts[grp[0]][0]
                dur     = true_bursts[grp[-1]][-1] - onset_t
                if 0.1 < dur < 10.0:
                    closest = osc_onsets[
                        np.argmin(np.abs(np.array(osc_onsets) - onset_t))]
                    if abs(closest - onset_t) < 0.1 and closest in unmatched:
                        if closest not in burst_dur_map:
                            burst_dur_map[closest] = []
                        burst_dur_map[closest].append(dur)

        for t0, durs in burst_dur_map.items():
            event_duration_map[t0] = max(durs)

    # Take duration per event (V1-based where matched, burst-span as fallback)
    durations = np.array([event_duration_map[t]
                          for t in osc_onsets if t in event_duration_map])
    matched_onsets = np.array([t for t in osc_onsets if t in event_duration_map])

    if len(durations) < 5:
        print("  Too few matched events for correlation.")
        return {}

    # Pupil measures
    pup = np.sqrt(np.maximum(np.asarray(pupil).ravel(), 0))
    p99 = np.nanpercentile(pup, 99)
    if p99 > 0:
        pup = pup / p99 * 100
    interp_fn = interp1d(t_pupil, pup, bounds_error=False, fill_value=np.nan)

    # Baseline pupil and pre-event constriction magnitude
    baseline_pup = np.array([
        np.nanmean(interp_fn(np.linspace(t0 - 2.0, t0 - 1.5, 20)))
        for t0 in matched_onsets])
    onset_pup = np.array([
        np.nanmean(interp_fn(np.linspace(t0 - 0.5, t0, 20)))
        for t0 in matched_onsets])
    constriction_mag = baseline_pup - onset_pup   # positive = constricted

    ok = ~(np.isnan(baseline_pup) | np.isnan(durations))
    baseline_pup_c = baseline_pup[ok]; durations_c = durations[ok]
    constriction_c = constriction_mag[ok]

    r_base, p_base    = pearsonr(baseline_pup_c, durations_c)
    r_const, p_const  = pearsonr(constriction_c, durations_c)
    r_sp, p_sp        = spearmanr(constriction_c, durations_c)

    print(f"  N matched events: {len(durations_c)}")
    print(f"  Baseline pupil vs. duration: r={r_base:.3f}, p={p_base:.4f}")
    print(f"  Constriction mag vs. duration: r={r_const:.3f}, p={p_const:.4f} (Pearson)")
    print(f"  Constriction mag vs. duration: rho={r_sp:.3f}, p={p_sp:.4f} (Spearman)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f'S5: Pre-oscillation Pupil vs. Oscillation Duration\n'
        f'(Tests LTS mechanism: deeper withdrawal → more cycles)\n'
        f'N = {len(durations_c)} matched events',
        fontsize=10, fontweight='bold')

    for ax, x, r, p, xlabel, title in [
        (axes[0], baseline_pup_c, r_base, p_base,
         'Baseline pupil (% max)',
         f'Baseline level vs. duration\nr={r_base:.3f}, p={p_base:.4f}'),
        (axes[1], constriction_c, r_const, p_const,
         'Pre-oscillation constriction magnitude (%)',
         f'Constriction magnitude vs. duration\nr={r_const:.3f}, p={p_const:.4f}')
    ]:
        ax.scatter(x, durations_c, alpha=0.4, s=20, color='purple')
        z = np.polyfit(x, durations_c, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), 'r-', lw=2,
                label=f'r={r:.3f}, p={p:.4f}')
        ax.axhline(1.0, color='gray', lw=0.8, ls='--',
                   label="Dennis's median (1.0 s)")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Oscillation duration (s)', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[S5] Saved → {save_path}")

    return {'r_baseline': r_base, 'p_baseline': p_base,
            'r_constriction': r_const, 'p_constriction': p_const,
            'r_spearman': r_sp, 'p_spearman': p_sp,
            'n': len(durations_c)}


# ─────────────────────────────────────────────────────────────────
# ANALYSIS S6: Circular time-shift null distribution test
# ─────────────────────────────────────────────────────────────────

def circular_shift_null_test(
        osc_onsets  : List[float],
        pupil       : np.ndarray,
        t_pupil     : np.ndarray,
        n_shifts    : int   = 1000,
        shift_range : tuple = (5, 50),
        pre_s       : float = 2.0,
        pupil_lag_s : float = 0.75,
        target_hz   : float = 120.0,
        save_path   : str   = 'circular_shift_null.png',
) -> Dict:
    """
    Rigorous null test: circularly shift pupil trace by random offsets
    (5-50 s) and recompute pre-oscillation slope at each shift.
    Compare observed slope distribution to null.
    More conservative than 1-sample t-test against zero.
    Maris & Oostenveld 2007 framework.
    """
    print(f"\n[S6 circular_shift] {len(osc_onsets)} events, {n_shifts} shifts")

    pup = np.sqrt(np.maximum(np.asarray(pupil).ravel(), 0))
    p99 = np.nanpercentile(pup, 99)
    if p99 > 0:
        pup = pup / p99 * 100

    recording_duration = t_pupil[-1] - t_pupil[0]
    dt = np.median(np.diff(t_pupil))

    def _compute_mean_slope(sig, tt, onsets):
        """Compute mean pre-oscillation slope across events."""
        t_wide = pre_s + 1.0
        t_ax, mat = align_signal_to_events(
            sig, tt, onsets,
            pre_s=t_wide, post_s=pre_s, target_hz=target_hz)
        base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
        pre_mask  = (t_ax >= -pre_s) & (t_ax <= pupil_lag_s)
        t_pre     = t_ax[pre_mask]
        for i in range(mat.shape[0]):
            bv = np.nanmean(mat[i, base_mask])
            if not np.isnan(bv):
                mat[i, :] -= bv
        slopes = []
        for row in mat[:, pre_mask]:
            ok = ~np.isnan(row)
            if ok.sum() >= 4:
                slopes.append(np.polyfit(t_pre[ok], row[ok], 1)[0])
        return np.nanmean(slopes) if slopes else np.nan

    # Observed slope
    obs_slope = _compute_mean_slope(pup, t_pupil, osc_onsets)
    print(f"  Observed mean slope: {obs_slope:+.4f} %/s")

    # Null distribution via circular shifts
    rng = np.random.default_rng(42)
    null_slopes = []
    n_samples = len(pup)

    for _ in range(n_shifts):
        shift_s = rng.uniform(shift_range[0], shift_range[1])
        if rng.random() > 0.5:
            shift_s = -shift_s
        shift_samples = int(round(shift_s / dt))
        pup_shifted   = np.roll(pup, shift_samples)
        s = _compute_mean_slope(pup_shifted, t_pupil, osc_onsets)
        if not np.isnan(s):
            null_slopes.append(s)

    null_slopes = np.array(null_slopes)
    p_val = np.mean(null_slopes <= obs_slope)  # one-tailed (slope < 0)

    print(f"  Null mean slope: {np.mean(null_slopes):+.4f} %/s")
    print(f"  Circular-shift p-value: {p_val:.4f}")
    print(f"  Interpretation: "
          + ("Non-random pre-oscillation pupil dynamics ✓"
             if p_val < 0.05 else "Cannot distinguish from null"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f'S6: Circular Time-Shift Null Distribution Test\n'
        f'(Maris & Oostenveld 2007 framework)\n'
        f'Observed slope = {obs_slope:+.4f} %/s  |  '
        f'p = {p_val:.4f} ({n_shifts} circular shifts)',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ax.hist(null_slopes, bins=40, color='gray', edgecolor='white',
            alpha=0.7, label=f'Null distribution\n({n_shifts} shifts)')
    ax.axvline(obs_slope, color='red', lw=2.5,
               label=f'Observed = {obs_slope:+.4f} %/s')
    ax.axvline(np.percentile(null_slopes, 5), color='orange', lw=1.5,
               ls='--', label='5th percentile null')
    ax.set_xlabel('Mean pre-oscillation slope (%/s)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'Observed vs. circular-shift null\n'
                 f'p = {p_val:.4f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.ecdf = ax.hist(null_slopes, bins=40, density=True, cumulative=True,
                      color='gray', alpha=0.5, label='Null CDF')
    ax.axvline(obs_slope, color='red', lw=2.5,
               label=f'Observed = {obs_slope:+.4f}')
    ax.axhline(0.05, color='orange', lw=1, ls='--', label='α = 0.05')
    ax.set_xlabel('Mean pre-oscillation slope (%/s)', fontsize=9)
    ax.set_ylabel('Cumulative probability', fontsize=9)
    ax.set_title('Cumulative null distribution', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[S6] Saved → {save_path}")

    return {'obs_slope': obs_slope, 'null_slopes': null_slopes,
            'p_val': p_val, 'n_shifts': n_shifts}
