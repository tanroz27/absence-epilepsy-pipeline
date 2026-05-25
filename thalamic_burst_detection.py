"""
thalamic_burst_detection.py
----------------------------
Faithful Python port of Dennis Nestvogel's
identify_3_to_5_Hz_bursts_function_ftf_2021.m

Dennis's algorithm uses findpeaks() on a binary ISI vector.
MATLAB's findpeaks() returns strict local maxima — a peak at index i
requires arr[i] > arr[i-1] AND arr[i] > arr[i+1].
For a binary 0/1 vector this means only ISOLATED 1s are detected
(a 1 flanked on both sides by 0). Runs of consecutive 1s yield no peaks.

Practical consequence: this effectively detects 2-spike LTS bursts
(single short ISI preceded and followed by longer ISIs). This is
intentional — Dennis is looking for the minimum criterion for an LTS
burst (2 spikes ≤ 4 ms apart, preceded by ≥ 100 ms silence).

IBI (inter-burst interval) in Dennis's code:
  delta = true_bursts{i+1}(1) - true_bursts{i}(end)
  = first spike of next burst - LAST spike of current burst
  For 2-spike bursts: last spike ≈ first spike + 2-4 ms
  Alpha range: 200-330 ms (3-5 Hz)
"""

import numpy as np
from typing import List, Tuple

# Constants — match Dennis's script exactly
ISI_BURST_CUTOFF  = 0.004   # 4 ms
MIN_SILENCE       = 0.100   # 100 ms
ALPHA_LOWER_LIMIT = 0.200   # 200 ms = 5 Hz
ALPHA_UPPER_LIMIT = 0.330   # 330 ms ≈ 3 Hz  (Dennis uses 0.33)


def _findpeaks_binary(c: np.ndarray) -> np.ndarray:
    """
    Python equivalent of MATLAB findpeaks() on a binary 0/1 vector.
    Returns 0-based indices of isolated 1s (flanked by 0 on both sides).
    This matches MATLAB's strict local-maximum definition.
    """
    c = np.asarray(c, dtype=int)
    peaks = []
    for i in range(1, len(c) - 1):
        if c[i] == 1 and c[i-1] == 0 and c[i+1] == 0:
            peaks.append(i)
    # Edge cases: first and last element
    if len(c) >= 2:
        if c[0] == 1 and c[1] == 0:
            peaks.append(0)
        if c[-1] == 1 and c[-2] == 0:
            peaks.append(len(c) - 1)
    return np.array(sorted(peaks), dtype=int)


def detect_bursts_single_unit(
        spike_times: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Port of the burst-detection section of Dennis's script.

    Steps (line-by-line from MATLAB):
      1. Compute ISI vector
      2. Binary c: 1 where ISI <= 4 ms
      3. findpeaks(c) → locs_on   (burst onset ISI indices)
      4. findpeaks(fliplr(c)) → locs_off  (burst offset ISI indices)
      5. bursts = spike_times[locs_on(i) : locs_off(i)+1]  (MATLAB 1-based)
      6. true_bursts = bursts where preceding spike is >= 100 ms earlier

    Returns (all_bursts, true_bursts) as lists of spike-time arrays.
    """
    spk = np.sort(spike_times.ravel())
    if len(spk) < 2:
        return [], []

    isi = np.diff(spk)                          # length N-1
    c   = (isi <= ISI_BURST_CUTOFF).astype(int) # binary vector

    locs_on  = _findpeaks_binary(c)             # onset ISI indices
    if len(locs_on) == 0:
        return [], []

    # Offset: findpeaks on fliplr(c), then map back
    flip_c    = c[::-1]
    locs_off_flip = _findpeaks_binary(flip_c)
    # Map flipped indices back to original
    offset_ind = np.zeros(len(c), dtype=int)
    offset_ind[locs_off_flip] = 1
    offset_ind = offset_ind[::-1]
    locs_off   = np.where(offset_ind == 1)[0]

    if len(locs_off) != len(locs_on):
        # Safety fallback: use locs_on as locs_off (2-spike bursts always match)
        locs_off = locs_on.copy()

    # Construct burst spike arrays
    # Dennis: bursts_spike_IDs{i} = Neuron_A(locs_on(i):locs_off(i)+1)
    # MATLAB 1-based → spike index = ISI index + 1 for onset,
    #                               ISI index + 1 + 1 for offset (+1 to include end spike)
    # In 0-based Python: spk[locs_on[i] : locs_off[i]+2]
    all_bursts = []
    for on, off in zip(locs_on, locs_off):
        burst = spk[on : off + 2]   # +2 = MATLAB's locs_off(i)+1 in 1-based
        if len(burst) >= 2:
            all_bursts.append(burst)

    # True bursts: preceding spike must be >= 100 ms before first burst spike
    true_bursts = []
    for burst in all_bursts:
        first = burst[0]
        idx   = np.searchsorted(spk, first)
        if idx == 0:
            true_bursts.append(burst)   # no preceding spike → silent before
        elif first - spk[idx - 1] >= MIN_SILENCE:
            true_bursts.append(burst)

    return all_bursts, true_bursts


def detect_alpha_oscillations_single_unit(
        true_bursts: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Port of the 3-5 Hz oscillation detection section.

    IBI = true_bursts[i+1][0] - true_bursts[i][-1]
    Alpha pair: 0.200 <= IBI <= 0.330

    Groups consecutive alpha pairs into oscillation events.
    Onset = first spike of first burst in each event.
    """
    if len(true_bursts) < 2:
        return [], []

    # Dennis: delta_time_between_bursts{i} = burst{i+1}(1) - burst{i}(end)
    ibi = np.array([
        true_bursts[i+1][0] - true_bursts[i][-1]
        for i in range(len(true_bursts) - 1)
    ])

    alpha_wave_ind = np.where(
        (ibi >= ALPHA_LOWER_LIMIT) & (ibi <= ALPHA_UPPER_LIMIT)
    )[0]

    if len(alpha_wave_ind) == 0:
        return [], []

    alpha_wave_ind_neighbor = alpha_wave_ind + 1
    alpha_wave_ind_all = np.unique(
        np.concatenate([alpha_wave_ind, alpha_wave_ind_neighbor])
    )

    # Group consecutive alpha pairs into oscillation events
    # (port of Dennis's ISI_alpha_ind + findpeaks section)
    ISI_alpha_ind = np.diff(alpha_wave_ind)
    a = np.where(ISI_alpha_ind <= 1)[0]

    oscillation_onsets = []
    alpha_bursts = [true_bursts[i] for i in alpha_wave_ind_all]

    if len(a) == 0:
        # All alpha pairs are isolated (short oscillation = 2 bursts)
        for idx in alpha_wave_ind:
            oscillation_onsets.append(true_bursts[idx][0])
    else:
        # Group consecutive pairs
        # Mimics Dennis's findpeaks on padded ISI_alpha_ind vector
        c2 = np.zeros(len(ISI_alpha_ind) + 6, dtype=int)
        c2[a + 3] = 1
        locs_on2  = _findpeaks_binary(c2)
        locs_on2  = locs_on2 - 3

        flip_c2 = c2[::-1]
        locs_off2_flip = _findpeaks_binary(flip_c2)
        off_ind = np.zeros(len(c2), dtype=int)
        off_ind[locs_off2_flip] = 1
        off_ind = off_ind[::-1]
        locs_off2 = np.where(off_ind == 1)[0] - 3

        for on2, off2 in zip(locs_on2, locs_off2):
            # Burst indices in this oscillation event
            event_alpha_inds = alpha_wave_ind[on2:off2+1]
            if len(event_alpha_inds):
                onset_burst_idx = event_alpha_inds[0]
                oscillation_onsets.append(true_bursts[onset_burst_idx][0])

        # Isolated alpha pairs not caught by chain grouper
        chain_covered = set()
        for on2, off2 in zip(locs_on2, locs_off2):
            for k in range(on2, off2+1):
                chain_covered.add(k)
        for k, idx in enumerate(alpha_wave_ind):
            if k not in chain_covered:
                oscillation_onsets.append(true_bursts[idx][0])

    return alpha_bursts, sorted(oscillation_onsets)


def detect_all_units(
        thalamic_units: List[np.ndarray],
        verbose: bool = True,
) -> Tuple[List, List, List[float]]:
    """
    Run pipeline across all LP units, return pooled onset times.
    Deduplicates events within 50 ms (same event across units).
    """
    all_alpha_bursts = []
    all_osc_onsets   = []

    for i, spk in enumerate(thalamic_units):
        _, true_bursts = detect_bursts_single_unit(np.asarray(spk))
        alpha_b, onsets = detect_alpha_oscillations_single_unit(true_bursts)
        all_alpha_bursts.append(alpha_b)
        all_osc_onsets.append(onsets)

        if verbose:
            print(f"  Unit {i:3d}: {len(spk):6d} spikes | "
                  f"{len(true_bursts):4d} true bursts | "
                  f"{len(alpha_b):4d} alpha bursts | "
                  f"{len(onsets):4d} osc events")

    # Pool and deduplicate within 50 ms
    flat = sorted(t for u in all_osc_onsets for t in u)
    if not flat:
        return all_alpha_bursts, all_osc_onsets, []

    dedup = [flat[0]]
    for t in flat[1:]:
        if t - dedup[-1] >= 0.050:
            dedup.append(t)

    if verbose:
        print(f"\n  → {len(dedup)} pooled events (50-ms dedup)")

    return all_alpha_bursts, all_osc_onsets, dedup
