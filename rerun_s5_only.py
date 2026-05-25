"""
rerun_s5_only.py
----------------
Regenerates S5 (pupil vs. oscillation duration) figures only,
using V1-matched events exclusively (v1_matched_only=True).

Loads only the five variables needed — skips the raw 50 kHz Vm
trace (V / Vtt) which is the main cause of slow load times.

Run from your pipeline folder:
    python3 rerun_s5_only.py
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

from thalamic_burst_detection import detect_all_units
from v1_processing            import clip_spikes, detect_v1_oscillations
from event_alignment          import pupil_vs_oscillation_duration

# ── Recording paths ──────────────────────────────────────────────────────────
RECORDINGS = [
    {
        'label': 'DPV20_rec5',
        'path' : '/Volumes/home/Dennis_to_Kevin/LP/DPV20_rec5_analyzed_wspikes_NRPX_sync_corrected.mat',
    },
    {
        'label': 'DPV21_rec6',
        'path' : '/Volumes/home/Dennis_to_Kevin/LP/DPV21_rec6_analyzed_wspikes_and_NRPX_sync_corrected.mat',
    },
    {
        'label': 'RN51_rec3',
        'path' : '/Volumes/home/Dennis_to_Kevin/LP/RN51_rec3_L23_LP_NRPX_sync_corrected.mat',
    },
]

# Only these variables are needed for S5 — skips raw 50 kHz V/Vtt
NEEDED_VARS = [
    'dLGN_cells',
    'V_dwsmpl_medfiltered',
    'Vtt_downsmpl',
    'pupil_area',
    'Wtt',
]

OUTPUT_BASE = Path('output_figures')


def load_targeted(filepath: str) -> dict:
    print(f"  Loading (targeted): {filepath}")
    try:
        raw = sio.loadmat(
            filepath,
            squeeze_me=True,
            struct_as_record=False,
            variable_names=NEEDED_VARS,
        )
        loaded = [k for k in raw if not k.startswith('_')]
        print(f"  Loaded: {loaded}")
        return raw
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


def extract_s5_vars(raw: dict) -> dict:
    def sq(x):
        if x is None:
            return None
        return np.asarray(x).squeeze().ravel().astype(float)

    out = {}

    if 'V_dwsmpl_medfiltered' in raw:
        out['vm'] = sq(raw['V_dwsmpl_medfiltered'])
        out['t']  = sq(raw.get('Vtt_downsmpl'))

    if 'dLGN_cells' in raw:
        raw_units = raw['dLGN_cells']
        if isinstance(raw_units, np.ndarray) and raw_units.dtype == object:
            out['thalamic_units'] = [
                np.asarray(raw_units[i]).ravel().astype(float)
                for i in range(raw_units.size)
            ]
        elif isinstance(raw_units, list):
            out['thalamic_units'] = [
                np.asarray(u).ravel().astype(float) for u in raw_units
            ]
        else:
            out['thalamic_units'] = [np.asarray(raw_units).ravel().astype(float)]

    if 'pupil_area' in raw:
        out['pupil'] = sq(raw['pupil_area'])

    if 'Wtt' in raw:
        wtt = sq(raw['Wtt'])
        if 'pupil' in out and wtt is not None and len(wtt) == len(out['pupil']):
            out['pupil_t'] = wtt
        elif 't' in out and out['t'] is not None and 'pupil' in out:
            n = len(out['pupil'])
            out['pupil_t'] = np.linspace(out['t'][0], out['t'][-1], n)

    print(f"  Extracted: { {k: (len(v) if isinstance(v, list) else np.shape(v)) for k, v in out.items()} }")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
for rec in RECORDINGS:
    label = rec['label']
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    sup_dir = OUTPUT_BASE / label
    sup_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw  = load_targeted(rec['path'])
        data = extract_s5_vars(raw)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    if data.get('pupil') is None or data.get('pupil_t') is None:
        print("  No pupil data — skipping.")
        continue

    if not data.get('thalamic_units'):
        print("  No thalamic units — skipping.")
        continue

    print(f"\n  Detecting LP oscillations ...")
    _, _, osc_onsets = detect_all_units(data['thalamic_units'], verbose=False)
    print(f"  → {len(osc_onsets)} oscillation events")

    if not osc_onsets:
        print("  No events — skipping.")
        continue

    v1_onsets_raw, v1_offsets = [], []
    if data.get('vm') is not None and data.get('t') is not None:
        print(f"\n  V1 spike clipping and oscillation detection ...")
        try:
            vm_clipped, ap_stats = clip_spikes(data['vm'], data['t'])
            print(f"  → {ap_stats['n_spikes']} spikes clipped")
            v1_onsets_raw, v1_offsets, _, _, _ = detect_v1_oscillations(
                vm_clipped, data['t'], refine_onset=False)
            print(f"  → {len(v1_onsets_raw)} V1 events detected")
        except Exception as e:
            print(f"  WARNING: V1 pipeline failed ({e})")

    print(f"\n  Generating S5 figure (V1-matched events only) ...")
    pupil_vs_oscillation_duration(
        osc_onsets,
        data['thalamic_units'],
        data['pupil'],
        data['pupil_t'],
        v1_onsets_raw   = v1_onsets_raw if v1_onsets_raw else None,
        v1_offsets      = v1_offsets    if v1_offsets    else None,
        v1_matched_only = True,
        save_path       = str(sup_dir / 'S5_pupil_vs_duration.png'),
    )

print(f"\n{'='*60}")
print("  Done. S5 figures → output_figures/<recording>/S5_pupil_vs_duration.png")
print(f"{'='*60}\n")
