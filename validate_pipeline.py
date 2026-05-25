"""
validate_pipeline.py
====================
Three independent validation checks for the pre-ictal pupil pipeline.

CHECK 1 — Permutation test
    Randomly shuffle onset times 500x within the recording duration.
    If the pupil constriction is real, it disappears with shuffled onsets.
    If it's a pipeline artifact, it persists with shuffled onsets.
    Result: real pupil trace must fall OUTSIDE the shuffled null distribution.

CHECK 2 — Benchmark against Dennis's pre-computed PETH
    The .mat files contain Dennis's own aligned pupil traces for whisking
    onsets (pupil_during_PETHwhiskbouts_mat, PETH_time_whisk_onset).
    We re-compute the same alignment from scratch using our pipeline
    and compare the two. If they match, our alignment code is correct.
    This is the most direct validation — Dennis's own output as ground truth.

CHECK 3 — Raw trace inspection
    Plot the raw pupil recording with oscillation onset times overlaid
    as vertical lines. Visually verify that the average reflects what
    you see by eye — no algorithmic trickery can fake this.

Usage:
    python validate_pipeline.py --mat_file /path/to/file.mat
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

sys.path.insert(0, os.path.dirname(__file__))
from load_data               import load_mat, extract_recording_vars
from thalamic_burst_detection import detect_all_units
from event_alignment          import (align_signal_to_events,
                                      _session_normalize_pupil,
                                      _bootstrap_ci)

OUTPUT_DIR = './output_figures'


# ─────────────────────────────────────────────────────────────────
# CHECK 1: Permutation test
# ─────────────────────────────────────────────────────────────────

def permutation_test(osc_onsets, pupil_norm, t_pupil,
                     pre_s=3.0, post_s=3.0,
                     n_perms=500, label='', save_path=None):
    """
    Shuffle onset times 500x within the recording duration.
    Compare real pupil trace to the null distribution.

    If the pre-ictal constriction is real:
      → real trace falls below the shuffled null in pre-ictal window
      → shuffled traces cluster around zero (flat)

    If it's a pipeline artifact:
      → shuffled traces show the same constriction shape
    """
    print(f"\n[CHECK 1] Permutation test  ({n_perms} shuffles)...")

    t_min = t_pupil[0] + pre_s + 1.0   # keep onsets away from edges
    t_max = t_pupil[-1] - post_s - 1.0

    # Real alignment
    t_ax, real_mat = align_signal_to_events(
        pupil_norm, t_pupil, osc_onsets,
        pre_s=pre_s, post_s=post_s, target_hz=120.0)

    valid = ~np.all(np.isnan(real_mat), axis=1)
    real_mat = real_mat[valid]
    n_events = len(real_mat)

    # Baseline-correct each trial (drift removal only — same as main pipeline)
    base_mask = (t_ax >= -pre_s) & (t_ax < -(pre_s * 0.6))
    for i in range(len(real_mat)):
        b = np.nanmean(real_mat[i, base_mask])
        if not np.isnan(b):
            real_mat[i] -= b

    real_mean = np.nanmean(real_mat, axis=0)

    # Permuted alignments
    rng = np.random.default_rng(42)
    perm_means = np.full((n_perms, len(t_ax)), np.nan)

    for p in range(n_perms):
        fake_onsets = rng.uniform(t_min, t_max, size=n_events)
        _, perm_mat = align_signal_to_events(
            pupil_norm, t_pupil, fake_onsets,
            pre_s=pre_s, post_s=post_s, target_hz=120.0)
        v = ~np.all(np.isnan(perm_mat), axis=1)
        pm = perm_mat[v]
        for i in range(len(pm)):
            b = np.nanmean(pm[i, base_mask])
            if not np.isnan(b):
                pm[i] -= b
        perm_means[p] = np.nanmean(pm, axis=0)

    # Null distribution bounds (2.5–97.5 percentile across shuffles)
    null_lo = np.nanpercentile(perm_means, 2.5,  axis=0)
    null_hi = np.nanpercentile(perm_means, 97.5, axis=0)
    null_mn = np.nanmean(perm_means, axis=0)

    # p-value at pre-ictal peak: what fraction of shuffles are more extreme?
    pre_mask = (t_ax >= -pre_s) & (t_ax < 0)
    real_preictal_mean = np.nanmean(real_mean[pre_mask])
    perm_preictal_means = np.nanmean(perm_means[:, pre_mask], axis=1)
    p_perm = np.mean(perm_preictal_means <= real_preictal_mean)

    print(f"  Real pre-ictal mean: {real_preictal_mean:+.3f}%")
    print(f"  Permutation p-value: {p_perm:.4f}  "
          f"({'SIGNIFICANT' if p_perm < 0.05 else 'not significant'})")
    print(f"  Interpretation: "
          + ("Real signal — constriction disappears with shuffled onsets ✓"
             if p_perm < 0.05
             else "WARNING: signal may be a pipeline artifact"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'{label}  —  CHECK 1: Permutation Test\n'
        f'Real pre-ictal mean = {real_preictal_mean:+.3f}%  '
        f'(permutation p = {p_perm:.4f})',
        fontsize=11, fontweight='bold')

    # Left: real vs null distribution
    ax = axes[0]
    for pm in perm_means[::10]:   # plot every 10th shuffle
        ax.plot(t_ax, pm, color='#aaaaaa', lw=0.3, alpha=0.3)
    ax.fill_between(t_ax, null_lo, null_hi,
                    color='gray', alpha=0.20, label='Null 95% range')
    ax.plot(t_ax, null_mn, color='gray', lw=1.0, ls='--', label='Null mean')
    ax.plot(t_ax, real_mean, color='red', lw=2.5, label='Real onsets')
    ax.axvline(0,  color='k',    lw=1.0, ls='--')
    ax.axhline(0,  color='gray', lw=0.5, ls=':')
    ax.axvspan(-pre_s, 0, alpha=0.06, color='red')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=10)
    ax.set_ylabel('Δ Pupil (%, baseline-corrected)', fontsize=10)
    ax.set_title('Real trace vs. 500 shuffled null traces', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: distribution of pre-ictal means across shuffles
    ax = axes[1]
    ax.hist(perm_preictal_means, bins=30,
            color='gray', edgecolor='white', alpha=0.7,
            label='Shuffled null distribution')
    ax.axvline(real_preictal_mean, color='red', lw=2.5,
               label=f'Real = {real_preictal_mean:+.3f}%')
    ax.axvline(np.nanpercentile(perm_preictal_means, 2.5),
               color='gray', lw=1.5, ls='--', alpha=0.8, label='2.5th percentile')
    ax.set_xlabel('Mean pre-ictal pupil change (%)', fontsize=10)
    ax.set_ylabel('Count (shuffles)', fontsize=10)
    ax.set_title(f'Permutation null distribution\np = {p_perm:.4f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.close()

    return {'p_perm': p_perm, 'real_mean': real_preictal_mean,
            'null_lo': null_lo, 'null_hi': null_hi}


# ─────────────────────────────────────────────────────────────────
# CHECK 2: Benchmark against Dennis's pre-computed PETH
# ─────────────────────────────────────────────────────────────────

def benchmark_against_dennis_peth(data, pupil_norm, t_pupil,
                                   label='', save_path=None):
    """
    Dennis pre-computed whisking-aligned pupil PETH in the .mat files:
      pupil_during_PETHwhiskbouts_mat  — matrix of aligned pupil trials
      PETH_time_whisk_onset            — time axis Dennis used

    We re-compute the same alignment from scratch using our pipeline
    and overlay the two. If they match, our alignment code is correct.

    This is the strongest possible validation: Dennis's own output
    is the ground truth.
    """
    print(f"\n[CHECK 2] Benchmarking against Dennis's pre-computed PETH...")

    # Pull Dennis's pre-computed data
    dennis_mat   = data.get('pupil_during_PETHwhiskbouts_mat')
    dennis_t     = data.get('PETH_time_whisk_onset')
    whisk_onsets = data.get('time_onset_whiskbout_small_removed',
                   data.get('bout_on'))

    if dennis_mat is None:
        print("  pupil_during_PETHwhiskbouts_mat not found in .mat — skipping.")
        return None
    if whisk_onsets is None:
        print("  Whisking onset times not found — skipping.")
        return None

    dennis_mat   = np.array(dennis_mat).squeeze()
    whisk_onsets = np.array(whisk_onsets).ravel()

    # Infer Dennis's window from his time axis or matrix shape
    if dennis_t is not None:
        dennis_t = np.array(dennis_t).ravel()
        pre_s_d  = abs(dennis_t[0])
        post_s_d = dennis_t[-1]
    else:
        # Default: Dennis typically uses [-2, +2] s for whisk PETH
        pre_s_d, post_s_d = 2.0, 2.0
        n_samp = dennis_mat.shape[1] if dennis_mat.ndim == 2 else len(dennis_mat)
        dennis_t = np.linspace(-pre_s_d, post_s_d, n_samp)

    # Our re-computed alignment
    t_ax, our_mat = align_signal_to_events(
        pupil_norm, t_pupil, list(whisk_onsets),
        pre_s=pre_s_d, post_s=post_s_d,
        target_hz=abs(1.0 / (dennis_t[1] - dennis_t[0])) if len(dennis_t) > 1 else 120.0
    )

    our_valid = ~np.all(np.isnan(our_mat), axis=1)
    our_mat   = our_mat[our_valid]

    # Handle pup_norm scale: Dennis's matrix may be raw area or normalized
    # Just compare shapes and normalized means
    dennis_mean = np.nanmean(dennis_mat, axis=0) if dennis_mat.ndim == 2 else dennis_mat
    our_mean    = np.nanmean(our_mat, axis=0)

    # Normalize both to zero-mean for shape comparison
    dennis_norm = dennis_mean - np.nanmean(dennis_mean)
    our_norm    = our_mean    - np.nanmean(our_mean)

    # Scale our to match Dennis's amplitude range
    d_scale = np.nanstd(dennis_norm)
    o_scale = np.nanstd(our_norm)
    if o_scale > 0 and d_scale > 0:
        our_norm_scaled = our_norm * (d_scale / o_scale)
    else:
        our_norm_scaled = our_norm

    # Correlation between the two mean traces (shape match)
    valid_both = ~(np.isnan(dennis_norm) | np.isnan(our_norm))
    if valid_both.sum() > 5:
        corr = np.corrcoef(dennis_norm[valid_both], our_norm[valid_both])[0, 1]
    else:
        corr = np.nan

    print(f"  Dennis PETH shape: {dennis_mat.shape}")
    print(f"  Our re-computed shape: {our_mat.shape}")
    print(f"  Shape correlation (Dennis vs ours): r = {corr:.4f}")
    print(f"  Interpretation: "
          + (f"Good match (r>{0.80:.2f}) — alignment code validated ✓"
             if corr > 0.80
             else f"Poor match (r={corr:.3f}) — check time axis / pupil variable"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'{label}  —  CHECK 2: Benchmark Against Dennis\'s Pre-computed PETH\n'
        f'Shape correlation r = {corr:.4f}  '
        f'(N Dennis={dennis_mat.shape[0] if dennis_mat.ndim==2 else "1"}, '
        f'N ours={len(our_mat)})',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    lo_d, hi_d = _bootstrap_ci(dennis_mat) if dennis_mat.ndim == 2 else (dennis_mean, dennis_mean)
    lo_o, hi_o = _bootstrap_ci(our_mat)
    ax.fill_between(dennis_t, lo_d - np.nanmean(lo_d),
                    hi_d - np.nanmean(hi_d), alpha=0.25, color='steelblue')
    ax.fill_between(t_ax, lo_o - np.nanmean(lo_o),
                    hi_o - np.nanmean(hi_o), alpha=0.25, color='red')
    ax.plot(dennis_t, dennis_norm, color='steelblue', lw=2,
            label="Dennis's pre-computed")
    ax.plot(t_ax, our_norm_scaled, color='red', lw=2, ls='--',
            label='Our re-computed')
    ax.axvline(0, color='k', lw=1.0, ls='--')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Time from whisking onset (s)', fontsize=10)
    ax.set_ylabel('Δ Pupil (normalized)', fontsize=10)
    ax.set_title('Whisking-aligned pupil: Dennis vs. ours\n(shape comparison, both mean-subtracted)', fontsize=9)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Scatter of Dennis mean vs. our mean (point-by-point)
    ax = axes[1]
    min_len = min(len(dennis_norm), len(our_norm_scaled))
    ax.scatter(dennis_norm[:min_len], our_norm_scaled[:min_len],
               alpha=0.4, s=8, color='k')
    lim = max(np.nanmax(np.abs(dennis_norm)), np.nanmax(np.abs(our_norm_scaled))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1.5, label='Identity')
    ax.set_xlabel("Dennis's mean trace (normalized)", fontsize=10)
    ax.set_ylabel('Our mean trace (normalized)', fontsize=10)
    ax.set_title(f'Point-by-point agreement\nr = {corr:.4f}', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.close()

    return {'corr': corr}


# ─────────────────────────────────────────────────────────────────
# CHECK 3: Raw trace inspection
# ─────────────────────────────────────────────────────────────────

def raw_trace_inspection(pupil_norm, t_pupil, osc_onsets,
                          whisk=None, t_whisk=None,
                          label='', save_path=None,
                          window_s=120.0):
    """
    Plot a representative segment of the raw pupil recording with
    oscillation onset times marked. Inspect visually whether the
    average reflects what you see by eye.

    Also plots 10 individual aligned trials so you can verify the
    average is not driven by a few outlier events.
    """
    print(f"\n[CHECK 3] Raw trace inspection...")

    # Find a segment containing multiple oscillation events
    onsets = np.sort(osc_onsets)
    # Pick segment centered on a cluster of events
    mid = onsets[len(onsets) // 2]
    t0 = max(t_pupil[0], mid - window_s / 2)
    t1 = min(t_pupil[-1], mid + window_s / 2)
    mask = (t_pupil >= t0) & (t_pupil <= t1)
    onsets_in_window = onsets[(onsets >= t0) & (onsets <= t1)]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f'{label}  —  CHECK 3: Raw Trace Inspection\n'
        f'Representative {int(window_s)}s segment with '
        f'{len(onsets_in_window)} oscillation onsets marked',
        fontsize=11, fontweight='bold')

    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                      left=0.08, right=0.97, top=0.88, bottom=0.08)

    # Top row: raw pupil trace with onset markers
    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(t_pupil[mask], pupil_norm[mask], color='#555555', lw=0.8)
    for t_on in onsets_in_window:
        ax_raw.axvline(t_on, color='red', lw=0.7, alpha=0.6)
    ax_raw.set_xlabel('Time (s)', fontsize=9)
    ax_raw.set_ylabel('Pupil (% max)', fontsize=9)
    ax_raw.set_title('Raw pupil with oscillation onsets (red). '
                     'Do the vertical lines fall at pupil dips?', fontsize=9)
    ax_raw.spines['top'].set_visible(False); ax_raw.spines['right'].set_visible(False)

    # Whisking if available
    if whisk is not None and t_whisk is not None:
        ax2 = ax_raw.twinx()
        wh_mask = (t_whisk >= t0) & (t_whisk <= t1)
        ax2.plot(t_whisk[wh_mask], whisk[wh_mask],
                 color='steelblue', lw=0.6, alpha=0.6)
        ax2.set_ylabel('Whisking (a.u.)', color='steelblue', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=7)

    # Middle row: 10 individual aligned trials
    t_ax, mat = align_signal_to_events(
        pupil_norm, t_pupil, osc_onsets, pre_s=3.0, post_s=3.0, target_hz=120.0)

    valid = ~np.all(np.isnan(mat), axis=1)
    mat_v = mat[valid]
    base_mask = (t_ax >= -3.0) & (t_ax < -2.0)
    for i in range(len(mat_v)):
        b = np.nanmean(mat_v[i, base_mask])
        if not np.isnan(b):
            mat_v[i] -= b

    # Show 10 random individual trials
    rng = np.random.default_rng(0)
    idx_show = rng.choice(len(mat_v), size=min(10, len(mat_v)), replace=False)
    ax_trials = fig.add_subplot(gs[1, :])
    colors_t = plt.cm.tab10(np.linspace(0, 1, len(idx_show)))
    for k, idx in enumerate(idx_show):
        ax_trials.plot(t_ax, mat_v[idx], color=colors_t[k], lw=1.0, alpha=0.7)
    ax_trials.plot(t_ax, np.nanmean(mat_v, axis=0), 'k', lw=2.5, label='Mean')
    ax_trials.axvline(0, color='k', lw=1.0, ls='--')
    ax_trials.axhline(0, color='gray', lw=0.5, ls=':')
    ax_trials.axvspan(-3, 0, alpha=0.05, color='red')
    ax_trials.set_xlabel('Time from oscillation onset (s)', fontsize=9)
    ax_trials.set_ylabel('Δ Pupil (%)', fontsize=9)
    ax_trials.set_title('10 individual aligned trials (colored) + mean (black). '
                        'Is pre-ictal constriction consistent across events?', fontsize=9)
    ax_trials.legend(fontsize=8)
    ax_trials.spines['top'].set_visible(False); ax_trials.spines['right'].set_visible(False)

    # Bottom row: histogram of pupil values at -1.5s vs. +1.5s
    ax_hl = fig.add_subplot(gs[2, 0])
    idx_pre  = np.argmin(np.abs(t_ax - (-1.5)))
    idx_post = np.argmin(np.abs(t_ax - 1.5))
    pre_vals  = mat_v[:, idx_pre]
    post_vals = mat_v[:, idx_post]
    ax_hl.hist(pre_vals[~np.isnan(pre_vals)],
               bins=20, alpha=0.6, color='blue', edgecolor='white',
               label=f't=−1.5s  (mean={np.nanmean(pre_vals):+.2f}%)')
    ax_hl.hist(post_vals[~np.isnan(post_vals)],
               bins=20, alpha=0.6, color='orange', edgecolor='white',
               label=f't=+1.5s  (mean={np.nanmean(post_vals):+.2f}%)')
    ax_hl.axvline(0, color='k', lw=1.0, ls='--')
    ax_hl.set_xlabel('Δ Pupil (%)', fontsize=9)
    ax_hl.set_ylabel('Count', fontsize=9)
    ax_hl.set_title('Distribution of pupil values\nat t=−1.5s vs. +1.5s', fontsize=9)
    ax_hl.legend(fontsize=8)
    ax_hl.spines['top'].set_visible(False); ax_hl.spines['right'].set_visible(False)

    # Event-by-event heatmap
    ax_heat = fig.add_subplot(gs[2, 1])
    sort_idx = np.argsort(np.nanmean(mat_v[:, (t_ax >= -1) & (t_ax < 0)], axis=1))
    im = ax_heat.imshow(
        mat_v[sort_idx],
        aspect='auto', cmap='RdBu_r', vmin=-30, vmax=30,
        extent=[t_ax[0], t_ax[-1], 0, len(mat_v)]
    )
    ax_heat.axvline(0, color='k', lw=1.5, ls='--')
    ax_heat.set_xlabel('Time from onset (s)', fontsize=9)
    ax_heat.set_ylabel('Event (sorted by\npre-ictal level)', fontsize=9)
    ax_heat.set_title('All events heatmap\n(sorted by pre-ictal pupil)', fontsize=9)
    plt.colorbar(im, ax=ax_heat, label='Δ Pupil (%)', fraction=0.04)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

    n_constrict = np.sum(np.nanmean(mat_v[:, (t_ax >= -1.5) & (t_ax < 0)], axis=1) < 0)
    print(f"  {n_constrict}/{len(mat_v)} events show pre-ictal constriction "
          f"({n_constrict/len(mat_v)*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def run_validation(mat_path: str):
    label = os.path.splitext(os.path.basename(mat_path))[0]
    out   = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  VALIDATION: {label}")
    print(f"{'='*60}")

    raw  = load_mat(mat_path)
    data = extract_recording_vars(raw)

    # Detect LP oscillation onsets
    if 'thalamic_units' not in data:
        print("  No thalamic units — cannot run.")
        return
    _, _, osc_onsets = detect_all_units(data['thalamic_units'], verbose=False)
    print(f"  {len(osc_onsets)} oscillation events detected")

    if not osc_onsets or data.get('pupil') is None:
        print("  Missing data — skipping.")
        return

    # Session-normalize pupil once
    pupil = np.asarray(data['pupil']).ravel()
    t_pup = np.asarray(data['pupil_t']).ravel()
    pmax  = np.nanpercentile(np.sqrt(np.maximum(pupil, 0)), 99)
    if np.nanmax(pupil) > 200 or pmax > 10:
        from event_alignment import _session_normalize_pupil
        pupil_norm = _session_normalize_pupil(pupil)
    else:
        pupil_norm = pupil.copy()

    # CHECK 1: Permutation test
    permutation_test(
        osc_onsets, pupil_norm, t_pup,
        label=label,
        save_path=os.path.join(out, 'check1_permutation.png')
    )

    # CHECK 2: Benchmark against Dennis's PETH
    benchmark_against_dennis_peth(
        raw, pupil_norm, t_pup,
        label=label,
        save_path=os.path.join(out, 'check2_dennis_peth_benchmark.png')
    )

    # CHECK 3: Raw trace inspection
    raw_trace_inspection(
        pupil_norm, t_pup, osc_onsets,
        whisk=data.get('whisk'), t_whisk=data.get('whisk_t'),
        label=label,
        save_path=os.path.join(out, 'check3_raw_inspection.png')
    )

    print(f"\n  Validation figures saved → {out}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_file', required=True,
                        help='Path to .mat recording file')
    args = parser.parse_args()
    run_validation(args.mat_file)
