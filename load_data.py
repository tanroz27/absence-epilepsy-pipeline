"""
load_data.py  (updated for Dennis's LP .mat files)
---------------------------------------------------
Handles MATLAB v5/v6 (scipy) and v7.3 HDF5 (h5py) formats.
Correctly dereferences cell arrays (dLGN_cells) in HDF5 files.

Variable mapping for these specific recordings:
  V_dwsmpl_medfiltered  → vm   (spike-clipped, downsampled Vm)
  Vtt_downsmpl          → t    (time for spike-clipped Vm)
  V                     → vm_raw
  Vtt                   → t_raw
  dLGN_cells            → thalamic_units  (LP unit spike times)
  pupil_area            → pupil
  W_abs_smo             → whisk  (smoothed whisker motion energy)
  W                     → wheel  (wheel speed cm/s)
  Wtt                   → whisk_t / wheel_t
"""

import numpy as np
import scipy.io as sio
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────
# Low-level loaders
# ─────────────────────────────────────────────────────────────────

def _deref_cell_array(h5file, dataset_name: str) -> List[np.ndarray]:
    """
    Dereference an HDF5 object-reference cell array into a Python list
    of numpy arrays. This is the correct way to read MATLAB cell arrays
    (like dLGN_cells) from v7.3 .mat files.
    """
    import h5py
    ds = h5file[dataset_name]
    refs = ds[()].flatten()
    result = []
    for ref in refs:
        try:
            arr = h5file[ref][()].flatten().astype(float)
            result.append(arr)
        except Exception:
            result.append(np.array([]))
    return result


def load_mat(filepath: str) -> dict:
    """
    Load a .mat file. Returns a plain dict of {varname: value}.
    Tries scipy.io first (old format), then h5py (v7.3 HDF5).
    dLGN_cells cell arrays are properly dereferenced.
    """
    # ── scipy (MATLAB < 7.3) ─────────────────────────────────────
    try:
        raw = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in raw.items() if not k.startswith('_')}
        print(f"[load_mat] scipy: {filepath}")
        return data
    except Exception as e_scipy:
        pass

    # ── h5py (MATLAB v7.3 HDF5) ──────────────────────────────────
    try:
        import h5py
        data = {}

        with h5py.File(filepath, 'r') as f:

            # Load all simple numeric datasets
            def _load_numeric(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = obj[()]
                        # Skip pure reference arrays — handle below
                        if arr.dtype.names:          # structured → skip
                            return
                        data[name] = arr
                    except Exception:
                        pass

            f.visititems(_load_numeric)

            # dLGN_cells: object-reference cell array → list of spike arrays
            if 'dLGN_cells' in f:
                data['dLGN_cells'] = _deref_cell_array(f, 'dLGN_cells')

        print(f"[load_mat] h5py:  {filepath}")
        return data

    except Exception as e_h5:
        raise RuntimeError(
            f"Cannot load '{filepath}'.\n"
            f"  scipy: likely MATLAB v7.3 HDF5 — need h5py\n"
            f"  h5py:  {e_h5}"
        )


# ─────────────────────────────────────────────────────────────────
# Variable extraction (maps Dennis's specific names → standard keys)
# ─────────────────────────────────────────────────────────────────

def _sq(x):
    """Squeeze + flatten a numpy array, or return None."""
    if x is None:
        return None
    return np.asarray(x).squeeze().ravel().astype(float)


def extract_recording_vars(data: dict) -> dict:
    """
    Extract standard variables from a loaded .mat dict.

    Priority order for Vm: V_dwsmpl_medfiltered (spike-clipped) > CNTRL_spk_clipped > V
    """
    out = {}

    # ── Membrane potential (V1, spike-clipped preferred) ─────────
    if 'V_dwsmpl_medfiltered' in data:
        out['vm'] = _sq(data['V_dwsmpl_medfiltered'])
        out['t']  = _sq(data.get('Vtt_downsmpl', data.get('Vtt')))
    elif 'CNTRL_spk_clipped' in data:
        out['vm'] = _sq(data['CNTRL_spk_clipped'])
        out['t']  = _sq(data.get('Vtt'))
    elif 'V' in data:
        out['vm'] = _sq(data['V'])
        out['t']  = _sq(data.get('Vtt'))

    # ── Thalamic (LP) spike times ─────────────────────────────────
    # dLGN_cells already contains ONLY the LP-depth units — Dennis
    # filtered by dLGN_low/dLGN_high when constructing this variable.
    # good_clusters_index (320) → subset → dLGN_cells (59/46/41 LP units).
    # Do NOT apply an additional depth filter here.
    for key in ('dLGN_cells', 'LP_cells', 'thalamic_units'):
        if key in data:
            raw = data[key]
            if isinstance(raw, list):
                out['thalamic_units'] = [
                    np.asarray(u).ravel().astype(float) for u in raw]
            elif isinstance(raw, np.ndarray) and raw.dtype == object:
                out['thalamic_units'] = [
                    np.asarray(raw[i]).ravel().astype(float)
                    for i in range(raw.size)]
            else:
                out['thalamic_units'] = [np.asarray(raw).ravel().astype(float)]
            break

    # ── Pupil ─────────────────────────────────────────────────────
    for key in ('pup_norm', 'pupil_area', 'pupil'):
        if key in data:
            out['pupil'] = _sq(data[key])
            break

    # Pupil time: infer from whisking time (same camera) or Vm time
    if 'pupil' in out:
        for tkey in ('Wtt', 'Vtt_downsmpl', 'Vtt'):
            if tkey in data:
                t_cand = _sq(data[tkey])
                if t_cand is not None and len(t_cand) == len(out['pupil']):
                    out['pupil_t'] = t_cand
                    break
        if 'pupil_t' not in out and 't' in out:
            # Last resort: linear interpolation over Vm time span
            n = len(out['pupil'])
            out['pupil_t'] = np.linspace(out['t'][0], out['t'][-1], n)

    # ── Whisking (smoothed motion energy) ────────────────────────
    for key in ('whis_norm', 'whisk_ME', 'whisk'):
        if key in data:
            out['whisk'] = _sq(data[key])
            break

    # ── Walking (wheel speed) ─────────────────────────────────────
    for key in ('W_abs_smo', 'W', 'wheel', 'walk', 'locomotion'):
        if key in data:
            arr = _sq(data[key])
            if arr is not None and arr.ndim <= 1:
                out['wheel'] = arr
                break

    # ── Shared behavioral time axis ───────────────────────────────
    if 'Wtt' in data:
        out['whisk_t'] = _sq(data['Wtt'])
        out['wheel_t'] = out['whisk_t']  # wheel logged on same clock

    # ── Report ────────────────────────────────────────────────────
    print("\n[extract_recording_vars]")
    for k, v in out.items():
        if isinstance(v, list):
            print(f"  {k:20s}: list of {len(v)} arrays")
        elif v is not None:
            print(f"  {k:20s}: shape={np.shape(v)}")
    return out
