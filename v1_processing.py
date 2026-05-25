"""
v1_processing.py
-----------------
1. Spike clipping  – port of getspikes_and_clip_2021.m
2. 3-5 Hz oscillation detection from V1 Vm – port of entire_recording_alpha_detect_and_pool_2021_2.m
   (automated version: replaces the interactive ginput/inputdlg with threshold-based onset detection)

Reference:
  Nestvogel & McCormick, Neuron 2022
"""

import numpy as np
from scipy.signal import spectrogram, find_peaks
from scipy.signal.windows import hamming
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional


# ── Spike detection parameters (matching Dennis's script) ───────────────────
SPIKE_THRESHOLD_MV  = -10.0   # mV  – everything above this = in a spike
DVDT_THRESHOLD      = 25000   # V/s – threshold for AP onset (=25 V/s)
ARTIFACT_MAX_MV     = 100.0   # mV  – reject APs > this (artifacts)
PRE_POST_MS         = 1.0     # ms  – window around spike peak for analysis


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SPIKE CLIPPING
# ══════════════════════════════════════════════════════════════════════════════

def clip_spikes(vm: np.ndarray, t: np.ndarray
                ) -> Tuple[np.ndarray, dict]:
    """
    Detect and remove APs from membrane potential trace.
    Returns the clipped trace and a dict of AP statistics.

    Port of getspikes_and_clip_2021.m – same algorithm, same parameters.

    Parameters
    ----------
    vm : 1D array, membrane potential in mV
    t  : 1D array, time in seconds (same length as vm)

    Returns
    -------
    vm_clipped   : vm with spikes replaced by linear interpolation
    ap_stats     : dict with keys:
                     spike_times, n_spikes, firing_rate,
                     mean_halfwidth_ms, mean_threshold_mV
    """
    vm   = np.asarray(vm, dtype=float)
    t    = np.asarray(t,  dtype=float)

    dt   = t[1] - t[0]
    fs   = 1.0 / dt

    # ── Step 1: find intervals above -10 mV ─────────────────────────────────
    above = (vm >= SPIKE_THRESHOLD_MV).astype(float)

    # onset indices: rising edge  (0→1)
    padded      = np.concatenate([[0.0], above, [0.0]])
    d_padded    = np.diff(padded)
    onset_inds  = np.where(d_padded > 0)[0]
    offset_inds = np.where(d_padded < 0)[0] - 1   # last sample still above thresh

    # Keep only intervals with > 5 samples (filter noise)
    valid = [(on, off) for on, off in zip(onset_inds, offset_inds)
             if (off - on) > 5]
    if not valid:
        return vm.copy(), _empty_ap_stats()

    onset_inds  = np.array([v[0] for v in valid])
    offset_inds = np.array([v[1] for v in valid])

    # ── Step 2: find peak (max) in each interval ─────────────────────────────
    spike_peak_inds = []
    for on, off in zip(onset_inds, offset_inds):
        seg = vm[on:off + 1]
        peak_local = np.argmax(seg)
        spike_peak_inds.append(on + peak_local)

    # ── Step 3: reject artifacts (peak > 100 mV) ────────────────────────────
    keep = [vm[pi] < ARTIFACT_MAX_MV for pi in spike_peak_inds]
    spike_peak_inds = [spike_peak_inds[i] for i in range(len(spike_peak_inds)) if keep[i]]
    onset_inds      = onset_inds[keep]
    offset_inds     = offset_inds[keep]

    if not spike_peak_inds:
        return vm.copy(), _empty_ap_stats()

    spike_times = t[spike_peak_inds]

    # ── Step 4: AP threshold via dV/dt = 25 V/s ─────────────────────────────
    pre_post_samp = int(round(PRE_POST_MS * 1e-3 / dt))

    thresholds_mV    = []
    halfwidths_ms    = []
    slow_rise_count  = 0

    for pi in spike_peak_inds:
        lo = max(0, pi - pre_post_samp)
        hi = min(len(vm), pi + pre_post_samp + 1)

        seg_vm = vm[lo:hi]
        seg_t  = t[lo:hi]

        if len(seg_t) < 3:
            slow_rise_count += 1
            continue

        dvdt = np.gradient(seg_vm, seg_t)

        thresh_inds = np.where(dvdt > DVDT_THRESHOLD)[0]
        if len(thresh_inds) == 0:
            slow_rise_count += 1
            continue

        thresh_ind = thresh_inds[0]
        thresh_mV  = seg_vm[thresh_ind]
        peak_mV    = vm[pi]

        thresholds_mV.append(thresh_mV)

        # Half-max amplitude
        half_max_mV = thresh_mV + (peak_mV - thresh_mV) / 2.0

        clipped_seg = seg_vm.copy()
        clipped_seg[seg_vm > half_max_mV] = half_max_mV
        clipped_seg[seg_vm < half_max_mV] = -30.0  # sentinel

        # First half-max crossing (onset)
        onset_half = np.where(clipped_seg != -30.0)[0]
        if len(onset_half) == 0:
            continue
        t_onset_half = seg_t[onset_half[0]]

        # Second half-max crossing (descending, offset)
        rev = clipped_seg[::-1]
        offset_half = np.where(rev != -30.0)[0]
        if len(offset_half) == 0:
            continue
        t_offset_half = seg_t[len(seg_t) - 1 - offset_half[0]]

        hw_ms = (t_offset_half - t_onset_half) * 1000.0
        if hw_ms > 0:
            halfwidths_ms.append(hw_ms)

    mean_hw  = float(np.mean(halfwidths_ms))  if halfwidths_ms   else np.nan
    mean_thr = float(np.mean(thresholds_mV))  if thresholds_mV   else np.nan
    n_spikes = len(spike_peak_inds)
    duration = t[-1] - t[0]
    fr       = n_spikes / duration if duration > 0 else 0.0

    # ── Step 5: clip spikes from Vm trace ────────────────────────────────────
    # Use ±1×AP_halfwidth (pre) and ±3.5×AP_halfwidth (post) from peak
    # If halfwidth is nan, fall back to ±2 ms
    hw_s = (mean_hw / 1000.0) if not np.isnan(mean_hw) else 0.002

    vm_clipped = vm.copy()
    clip_inds_list = []

    for pi in spike_peak_inds:
        pre_samp  = max(1, int(round(hw_s * 1.0 / dt)))
        post_samp = max(1, int(round(hw_s * 3.5 / dt)))
        lo = max(0, pi - pre_samp)
        hi = min(len(vm), pi + post_samp + 1)
        clip_inds_list.extend(range(lo, hi))

    clip_inds = np.array(sorted(set(clip_inds_list)))
    if len(clip_inds) > 0:
        vm_clipped[clip_inds] = np.nan

    # Linear interpolation across NaN regions
    nan_mask = np.isnan(vm_clipped)
    if nan_mask.any():
        valid_mask = ~nan_mask
        if valid_mask.sum() >= 2:
            interp_fn = interp1d(t[valid_mask], vm_clipped[valid_mask],
                                 kind='linear', bounds_error=False,
                                 fill_value='extrapolate')
            vm_clipped[nan_mask] = interp_fn(t[nan_mask])

    ap_stats = {
        'spike_times'      : spike_times,
        'n_spikes'         : n_spikes,
        'firing_rate'      : fr,
        'mean_halfwidth_ms': mean_hw,
        'mean_threshold_mV': mean_thr,
        'slow_rise_count'  : slow_rise_count,
    }

    return vm_clipped, ap_stats


def _empty_ap_stats():
    return {
        'spike_times': np.array([]),
        'n_spikes': 0, 'firing_rate': 0.0,
        'mean_halfwidth_ms': np.nan, 'mean_threshold_mV': np.nan,
        'slow_rise_count': 0
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  3-5 Hz OSCILLATION DETECTION FROM V1 Vm
# ══════════════════════════════════════════════════════════════════════════════

def detect_v1_oscillations(
        vm_clipped : np.ndarray,
        t          : np.ndarray,
        threshold_sd: float = 1.0,
        min_duration_s: float = 0.5,
        window_s: float = 4.0,
        overlap: float = 0.9,
        downsample_to_hz: float = 40.0,
        refine_onset: bool = True,
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Automated 3-5 Hz oscillation detection from spike-clipped V1 membrane
    potential. Port of entire_recording_alpha_detect_and_pool_2021_2.m.

    The original script uses a manual ginput curation step; here we replace
    that with an automated onset-refinement step (find the point of steepest
    rise in RMS power leading up to the threshold crossing).

    Parameters
    ----------
    vm_clipped      : spike-clipped membrane potential (mV)
    t               : time vector (s)
    threshold_sd    : mean + N*std threshold for alpha power (default 1.0 = Dennis)
    min_duration_s  : minimum oscillation duration (s)
    window_s        : FFT window length (s)  – Dennis uses 4 s
    overlap         : fractional overlap between windows (Dennis uses 0.9)
    downsample_to_hz: downsample rate before spectrogram (Dennis uses 40 Hz)
    refine_onset    : if True, refine onset to point of steepest RMS rise

    Returns
    -------
    osc_onsets   : list of oscillation onset times (s)
    osc_offsets  : list of oscillation offset times (s)
    T_spec       : spectrogram time axis (s)
    F_spec       : spectrogram frequency axis (Hz)
    P_alpha_mean : mean alpha-band power over time (for threshold inspection)
    """
    dt_orig  = t[1] - t[0]
    fs_orig  = 1.0 / dt_orig

    # ── Downsample ────────────────────────────────────────────────────────────
    ds_factor = max(1, int(round(fs_orig / downsample_to_hz)))
    vm_ds = vm_clipped[::ds_factor]
    t_ds  = t[::ds_factor]
    fs_ds = 1.0 / (t_ds[1] - t_ds[0])

    # ── Spectrogram ───────────────────────────────────────────────────────────
    win_samp  = int(round(window_s * fs_ds))
    win_samp  = max(win_samp, 8)
    noverlap  = int(round(win_samp * overlap))
    nfft      = win_samp  # frequency resolution = 1/window_s

    win_fn = hamming(win_samp)

    F_spec, T_spec, Sxx = spectrogram(
        vm_ds, fs=fs_ds, window=win_fn, noverlap=noverlap,
        nfft=nfft, scaling='density'
    )

    # Average power in 3-5 Hz band
    alpha_band = (F_spec >= 3.0) & (F_spec <= 5.0)
    P_alpha = Sxx[alpha_band, :]
    P_alpha_mean = P_alpha.mean(axis=0)

    # ── Threshold ─────────────────────────────────────────────────────────────
    thresh = P_alpha_mean.mean() + threshold_sd * P_alpha_mean.std()

    above_thresh = (P_alpha_mean >= thresh).astype(float)
    padded = np.concatenate([[0.0], above_thresh, [0.0]])
    d = np.diff(padded)
    cand_on  = np.where(d > 0)[0]
    cand_off = np.where(d < 0)[0] - 1

    # ── Duration filter ───────────────────────────────────────────────────────
    min_samp = min_duration_s / (T_spec[1] - T_spec[1 - 1] if len(T_spec) > 1 else 1)
    # simpler: use actual time
    dt_spec = T_spec[1] - T_spec[0] if len(T_spec) > 1 else 1.0

    osc_onsets  = []
    osc_offsets = []

    for on_i, off_i in zip(cand_on, cand_off):
        duration = (off_i - on_i) * dt_spec
        if duration < min_duration_s:
            continue

        t_on  = float(T_spec[min(on_i,  len(T_spec) - 1)])
        t_off = float(T_spec[min(off_i, len(T_spec) - 1)])

        # ── Refine onset to steepest rise ────────────────────────────────────
        if refine_onset:
            # Look at RMS power in a 2-s window leading up to the threshold cross
            look_back = 2.0  # seconds
            look_start = max(0, t_on - look_back)
            win_mask = (T_spec >= look_start) & (T_spec <= t_on)
            if win_mask.sum() >= 3:
                p_win = P_alpha_mean[win_mask]
                t_win = T_spec[win_mask]
                slope = np.gradient(p_win, t_win)
                best  = np.argmax(slope)
                t_on  = float(t_win[best])

        osc_onsets.append(t_on)
        osc_offsets.append(t_off)

    return osc_onsets, osc_offsets, T_spec, F_spec, P_alpha_mean
