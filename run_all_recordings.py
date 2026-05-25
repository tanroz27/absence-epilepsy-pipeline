"""
run_all_recordings.py
=====================
Runs the full Nestvogel Figure 5F replication + pre-oscillation pupil pipeline
across all three LP recordings.

Usage:
    python run_all_recordings.py

Outputs (in ./output_figures/):
    {recording}_fig5F.png            — Figure 5F equivalent
    {recording}_preictal_pupil.png   — 4-panel pre-oscillation pupil analysis
    grand_average_preictal_pupil.png — Grand average across recordings
    pipeline_summary.txt             — Numeric results table
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp, sem as scipy_sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from load_data               import load_mat, extract_recording_vars
from thalamic_burst_detection import detect_all_units
from v1_processing            import clip_spikes, detect_v1_oscillations
from event_alignment          import (align_signal_to_events,
                                      compute_spectrogram_matrix,
                                      figure_5f_equivalent,
                                      preictal_pupil_analysis,
                                      preoscillation_whisking_analysis,
                                      whisking_as_predictor_analysis,
                                      grand_average_whisking,
                                      pupil_constriction_probability,
                                      oscillation_duration_distribution,
                                      whisking_offset_latency,
                                      pupil_baseline_stratification,
                                      pupil_vs_oscillation_duration,
                                      circular_shift_null_test)

# ─────────────────────────────────────────────────────────────────
# RECORDINGS
# ─────────────────────────────────────────────────────────────────

RECORDINGS = [
    {
        'label': 'DPV20_rec5',
        'path': '/Volumes/home/Dennis_to_Kevin/LP/'
                'DPV20_rec5_analyzed_wspikes_NRPX_sync_corrected.mat',
    },
    {
        'label': 'DPV21_rec6',
        'path': '/Volumes/home/Dennis_to_Kevin/LP/'
                'DPV21_rec6_analyzed_wspikes_and_NRPX_sync_corrected.mat',
    },
    {
        'label': 'RN51_rec3',
        'path': '/Volumes/home/Dennis_to_Kevin/LP/'
                'RN51_rec3_L23_LP_NRPX_sync_corrected.mat',
    },
]

OUTPUT_DIR = Path('./output_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

PRE_S  = 2.0   # seconds — Dennis: "max -2s to +2s" (personal communication)
POST_S = 2.0   # seconds


# ─────────────────────────────────────────────────────────────────
# PER-RECORDING PIPELINE
# ─────────────────────────────────────────────────────────────────

def process_recording(label: str, mat_path: str) -> dict:
    """Full pipeline for one recording. Returns result dict."""
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    if not os.path.exists(mat_path):
        print(f"  FILE NOT FOUND: {mat_path}")
        return {'label': label, 'error': 'file not found'}

    # 1. Load
    raw  = load_mat(mat_path)
    data = extract_recording_vars(raw)

    # 2. Thalamic burst detection (LP units)
    if 'thalamic_units' not in data:
        print("  ERROR: thalamic_units not found after extraction.")
        return {'label': label, 'error': 'no thalamic units'}

    print(f"\n  Detecting LP 3-5 Hz oscillations across "
          f"{len(data['thalamic_units'])} units ...")
    _, _, osc_onsets = detect_all_units(data['thalamic_units'], verbose=False)
    print(f"  → {len(osc_onsets)} oscillation events")

    if len(osc_onsets) == 0:
        print("  No events — skipping alignment.")
        return {'label': label, 'osc_onsets': [], 'n_events': 0}

    # V1 Vm pipeline — used only for spectrogram-based duration measurement
    # (matches Nestvogel & McCormick's threshold-to-threshold duration definition)
    v1_onsets_raw, v1_offsets = [], []
    if data.get('vm') is not None and data.get('t') is not None:
        print("\n  Spike clipping V1 Vm for spectrogram duration measurement ...")
        try:
            vm_clipped, ap_stats = clip_spikes(data['vm'], data['t'])
            print(f"  → {ap_stats['n_spikes']} spikes clipped")
            # refine_onset=False: threshold-crossing onset matches N&M's definition
            v1_onsets_raw, v1_offsets, _, _, _ = detect_v1_oscillations(
                vm_clipped, data['t'], refine_onset=False)
            print(f"  → {len(v1_onsets_raw)} V1 oscillation events detected")
        except Exception as e:
            print(f"  Warning: V1 pipeline failed ({e}); "
                  f"duration will use burst-span fallback")

    # 3. Figure 5F equivalent
    print("\n  Plotting Figure 5F equivalent ...")
    fig5f_path = str(OUTPUT_DIR / f'{label}_fig5F.png')
    figure_5f_equivalent(
        osc_onsets     = osc_onsets,
        vm_clipped     = data.get('vm'),
        t_vm           = data.get('t'),
        pupil          = data.get('pupil'),
        t_pupil        = data.get('pupil_t'),
        whisk          = data.get('whisk'),
        t_whisk        = data.get('whisk_t', data.get('t')),
        wheel          = data.get('wheel'),
        t_wheel        = data.get('wheel_t', data.get('t')),
        thalamic_units = data.get('thalamic_units'),
        save_path      = fig5f_path,
        pre_s          = PRE_S,
        post_s         = POST_S,
    )

    # 4. Pre-oscillation pupil analysis (thesis core)
    pupil_results = {}
    if data.get('pupil') is not None and data.get('pupil_t') is not None:
        print("\n  Pre-oscillation pupil analysis ...")
        pupil_path = str(OUTPUT_DIR / f'{label}_preictal_pupil.png')
        raw_results = preictal_pupil_analysis(
            osc_onsets     = osc_onsets,
            pupil          = data['pupil'],
            t_pupil        = data['pupil_t'],
            pre_ictal_s    = 2.0,    # Dennis: max -2s to +2s window
            baseline_pre_s = 1.0,    # enough room for -2.0 to -1.5s baseline
            pupil_lag_s    = 0.75,   # Reimer et al. 2016
            save_path      = pupil_path,
        )
        # Keep raw_results directly — combined analysis needs per-event slopes
        pupil_results = raw_results
    else:
        print("  No pupil data — skipping pre-oscillation analysis.")

    # ── Analysis 2: Whisking pre-oscillation ────────────────────
    whisk_results = {}
    if data.get('whisk') is not None and data.get('whisk_t') is not None:
        print("\n  Pre-oscillation whisking analysis ...")
        whisk_path = str(OUTPUT_DIR / f'{label}_preoscillation_whisking.png')
        whisk_results = preoscillation_whisking_analysis(
            osc_onsets  = osc_onsets,
            whisk       = data['whisk'],
            t_whisk     = data['whisk_t'],
            pre_s       = 2.0,
            pupil_lag_s = 0.75,
            save_path   = whisk_path,
        )
    else:
        print("  No whisking data — skipping.")

    # ── Analysis 3: Whisking as predictor (stratified) ──────────
    combined_results = {}
    if data.get('whisk') is not None:
        print("\n  Whisking-as-predictor analysis ...")
        combined_path = str(OUTPUT_DIR / f'{label}_whisking_as_predictor.png')
        combined_results = whisking_as_predictor_analysis(
            osc_onsets  = osc_onsets,
            whisk       = data['whisk'],
            t_whisk     = data['whisk_t'],
            pupil       = data.get('pupil'),
            t_pupil     = data.get('pupil_t'),
            pre_s       = 2.0,
            pupil_lag_s = 0.75,
            save_path   = combined_path,
        )

    # ── Supplementary analyses S1-S6 ─────────────────────────────
    sup_dir = OUTPUT_DIR / label
    sup_dir.mkdir(exist_ok=True)

    if data.get('pupil') is not None and len(osc_onsets):
        print("\n  S1: Pupil constriction probability (Dennis Fig 4B) ...")
        pupil_constriction_probability(
            osc_onsets, data['pupil'], data['pupil_t'],
            save_path=str(sup_dir / 'S1_pupil_constriction_prob.png'))

    if len(osc_onsets) and data.get('thalamic_units'):
        print("\n  S2: Oscillation duration distribution (Dennis Fig 4C) ...")
        oscillation_duration_distribution(
            osc_onsets, data['thalamic_units'],
            v1_onsets_raw=v1_onsets_raw if v1_onsets_raw else None,
            v1_offsets=v1_offsets if v1_offsets else None,
            save_path=str(sup_dir / 'S2_osc_duration.png'))

    if data.get('whisk') is not None and len(osc_onsets):
        print("\n  S3: Whisking offset latency ...")
        whisking_offset_latency(
            osc_onsets, data['whisk'], data['whisk_t'],
            save_path=str(sup_dir / 'S3_whisk_offset_latency.png'))

    if data.get('pupil') is not None and len(osc_onsets):
        print("\n  S4: Baseline pupil stratification ...")
        pupil_baseline_stratification(
            osc_onsets, data['pupil'], data['pupil_t'],
            save_path=str(sup_dir / 'S4_baseline_stratification.png'))

        if data.get('thalamic_units'):
            print("\n  S5: Pupil vs. oscillation duration ...")
            pupil_vs_oscillation_duration(
                osc_onsets, data['thalamic_units'],
                data['pupil'], data['pupil_t'],
                v1_onsets_raw=v1_onsets_raw if v1_onsets_raw else None,
                v1_offsets=v1_offsets if v1_offsets else None,
                save_path=str(sup_dir / 'S5_pupil_vs_duration.png'))

        print("\n  S6: Circular shift null test ...")
        circular_shift_null_test(
            osc_onsets, data['pupil'], data['pupil_t'],
            n_shifts=500,
            save_path=str(sup_dir / 'S6_circular_shift_null.png'))

    return {
        'label'           : label,
        'osc_onsets'      : osc_onsets,
        'n_events'        : len(osc_onsets),
        'pupil_results'   : pupil_results,
        'whisk_results'   : whisk_results,
        'combined_results': combined_results,
        'data'            : data,
    }


# ─────────────────────────────────────────────────────────────────
# GRAND AVERAGE ACROSS RECORDINGS
# ─────────────────────────────────────────────────────────────────

def grand_average_pupil(results: list, pre_s=3.0, post_s=3.0):
    """
    Pool pre-oscillation pupil traces across all recordings and compute the
    grand-average aligned pupil signature (the key thesis figure).
    """
    from event_alignment import align_signal_to_events

    all_matrices  = []
    all_slopes    = []
    all_deltas    = []
    common_t_axis = None
    labels        = []

    for r in results:
        if r.get('error') or r.get('data') is None:
            continue
        data   = r['data']
        onsets = r['osc_onsets']
        if not onsets or data.get('pupil') is None:
            continue

        t_ax, mat = align_signal_to_events(
            signal=data['pupil'],
            sig_t=data['pupil_t'],
            event_t=onsets,
            pre_s=pre_s,
            post_s=post_s,
            target_hz=120.0,
        )

        # Z-score each trial
        valid = ~np.all(np.isnan(mat), axis=1)
        mat = mat[valid].copy()
        for i in range(len(mat)):
            mu, sd = np.nanmean(mat[i]), np.nanstd(mat[i])
            if sd > 0:
                mat[i] = (mat[i] - mu) / sd

        # Pre-oscillation slope (linear fit over [-pre_s, 0])
        pre_mask = t_ax < 0
        pre_t    = t_ax[pre_mask]
        for row in mat[:, pre_mask]:
            ok = ~np.isnan(row)
            if ok.sum() > 5:
                m = np.polyfit(pre_t[ok], row[ok], 1)[0]
                all_slopes.append(m)

        # Δ pupil: mean[-0.5, 0] − mean[-pre_s, -(pre_s-1)]
        base_m = (t_ax >= -2.0) & (t_ax < -1.5)   # Dennis: mean of -2.0 to -1.5s
        on_m   = (t_ax >= -0.5) & (t_ax <= 0.75)  # lag-corrected to +0.75s
        for row in mat:
            all_deltas.append(np.nanmean(row[on_m]) - np.nanmean(row[base_m]))

        all_matrices.append(mat)
        labels.append(r['label'])
        if common_t_axis is None:
            common_t_axis = t_ax

    if not all_matrices or common_t_axis is None:
        print("  Not enough pupil data for grand average.")
        return

    pooled = np.vstack(all_matrices)
    slopes = np.array(all_slopes)
    deltas = np.array(all_deltas)

    mn_g  = np.nanmean(pooled, axis=0)
    sem_g = np.nanstd(pooled, axis=0) / np.sqrt(len(pooled))

    _, p_slope = ttest_1samp(slopes[~np.isnan(slopes)], 0)
    _, p_delta = ttest_1samp(deltas[~np.isnan(deltas)], 0)

    print(f"\n  ══ GRAND AVERAGE (N={len(pooled)} events) ══")
    print(f"  Slope  : {np.nanmean(slopes):+.4f} ± {scipy_sem(slopes):.4f} z/s  "
          f"[p = {p_slope:.4f}]")
    print(f"  Δ Pupil: {np.nanmean(deltas):+.4f} ± {scipy_sem(deltas):.4f} z    "
          f"[p = {p_delta:.4f}]")
    print(f"  % events with constriction: {np.mean(deltas < 0)*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Grand-Average Pre-oscillation Pupil Signature  [0.75s lag correction applied]\n'
        f'{" | ".join(labels)}  —  N = {len(pooled)} events',
        fontsize=12, fontweight='bold'
    )

    # Panel 1: grand-mean trace
    ax = axes[0]
    ax.axvspan(-0.5, 0.75, alpha=0.08, color='red',
               label='Lag-corrected peri-oscillation window [−0.5s, +0.75s]')
    ax.fill_between(common_t_axis, mn_g - sem_g, mn_g + sem_g,
                    alpha=0.30, color='purple')
    ax.plot(common_t_axis, mn_g, color='purple', lw=2.5)
    ax.axvline(0,    color='k',   lw=1.4, ls='--', label='Oscillation onset')
    ax.axvline(0.75, color='red', lw=1.0, ls=':',  label='+0.75s (pupil lag)')
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Time from oscillation onset (s)', fontsize=10)
    ax.set_ylabel('Pupil (z-score)', fontsize=10)
    ax.set_title('Grand-average pupil\naligned to 3–5 Hz onset\n'
                 '[0.75s lag correction applied]', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: slope histogram
    ax = axes[1]
    ax.hist(slopes[~np.isnan(slopes)], bins=25,
            color='purple', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', lw=1.2, ls='--')
    ax.axvline(np.nanmean(slopes), color='red', lw=2,
               label=f'Mean = {np.nanmean(slopes):.3f}\np = {p_slope:.4f}')
    ax.set_xlabel('Pre-oscillation slope (z/s)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Pre-oscillation slope\ndistribution  (H₀: slope = 0)', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: effect-size bar chart
    ax = axes[2]
    means_ = [np.nanmean(slopes), np.nanmean(deltas)]
    sems_  = [scipy_sem(slopes[~np.isnan(slopes)]),
              scipy_sem(deltas[~np.isnan(deltas)])]
    ps_    = [p_slope, p_delta]
    xlbls  = ['Pre-oscillation\nslope (z/s)', 'Δ Pupil\n(z)']
    bars = ax.bar(xlbls, means_, yerr=sems_,
                  color=['#9467bd', '#e377c2'],
                  edgecolor='k', capsize=6, width=0.5)
    ax.axhline(0, color='k', lw=0.8)
    for bar, p in zip(bars, ps_):
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*'   if p < 0.05 else 'ns')
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(s for s in sems_) * 0.2,
                sig, ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('Effect size (z-score)', fontsize=10)
    ax.set_title('Pre-oscillation changes\n(mean ± SEM)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = str(OUTPUT_DIR / 'grand_average_preictal_pupil.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────

def print_summary(results: list, summary_path: Path):
    lines = []
    lines.append("PIPELINE SUMMARY — ROZENDAL THESIS")
    lines.append("=" * 70)
    lines.append(f"{'Recording':<22} {'N events':>8}  {'Δ pupil':>10}  "
                 f"{'p(Δ)':>7}  {'slope':>10}  {'p(s)':>7}  {'%const':>7}")
    lines.append("-" * 70)

    for r in results:
        pr = r.get('pupil_results', {})
        n  = r.get('n_events', '?')
        d  = pr.get('mean_delta',    pr.get('mean_delta',    float('nan')))
        pd = pr.get('p_delta',       pr.get('ttest_preictal', type('x',(),{'pvalue':float('nan')})).pvalue if pr.get('ttest_preictal') else float('nan'))
        s  = pr.get('slope_preictal', float('nan'))
        ps = pr.get('p_slope',        float('nan'))
        pc = pr.get('pct_const',      pr.get('pct_const', float('nan')))
        lines.append(
            f"  {r['label']:<20} {str(n):>8}  {d:>+10.4f}  "
            f"{pd:>7.4f}  {s:>+10.4f}  {ps:>7.4f}  {pc:>6.1f}%"
        )

    lines.append("=" * 70)
    text = "\n".join(lines)
    print("\n" + text)

    with open(summary_path, 'w') as f:
        f.write(text + "\n")
    print(f"\n  Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 62)
    print("  Nestvogel Fig 5F Replication + Preictal Pupil Pipeline")
    print("  Rozendal Thesis — University of Oregon / Yale MD")
    print("═" * 62)

    results = []
    for rec in RECORDINGS:
        r = process_recording(rec['label'], rec['path'])
        results.append(r)

    # Grand average
    print("\n" + "=" * 62)
    print("  Grand Average")
    print("=" * 62)
    grand_average_pupil(results)

    # Grand average whisking
    valid_whisk = [r for r in results if r.get('whisk_results')]
    if valid_whisk:
        grand_average_whisking(
            valid_whisk,
            save_path=str(OUTPUT_DIR / 'grand_average_whisking.png')
        )

    # Summary table
    print_summary(results, OUTPUT_DIR / 'pipeline_summary.txt')

    print(f"\n  All outputs → {OUTPUT_DIR.resolve()}")
    print("  Done.\n")


if __name__ == '__main__':
    main()
