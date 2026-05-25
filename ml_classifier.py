"""
ml_classifier.py
================
Logistic Regression + LSTM classifier for pre-oscillation pupil prediction.
Intended to be called AFTER run_all_recordings.py has produced the results list.

Usage (from within run_all_recordings.py, after main() builds results):
    from ml_classifier import run_ml_pipeline
    run_ml_pipeline(results, output_dir=OUTPUT_DIR)

Or as a standalone script:
    python ml_classifier.py

Requires:
    pip install scikit-learn torch torchvision matplotlib numpy scipy

Outputs (in OUTPUT_DIR):
    ML_01_feature_importance.png     — Logistic regression coefficients
    ML_02_roc_curves.png             — ROC curves per recording + pooled
    ML_03_probability_timeline.png   — Predicted seizure probability over time
    ML_04_lstm_training.png          — LSTM training curve + ROC
    ML_05_combined_summary.png       — Summary panel for appendix figure
    ML_summary.txt                   — Numeric results table
"""

import os
import sys
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — all in one place
# ─────────────────────────────────────────────────────────────────────────────

PRE_S         = 2.0    # seconds of pre-event window used for features
POST_S        = 0.75   # seconds after onset (lag-corrected)
PUPIL_LAG_S   = 0.75   # Reimer et al. 2016
FS_TARGET     = 120.0  # Hz — common resampling rate
MIN_GAP_S     = 3.0    # minimum gap between negative sample and any event
N_NEG_MULT    = 3      # number of negative samples per positive event

LSTM_HIDDEN   = 32
LSTM_LAYERS   = 1
LSTM_DROPOUT  = 0.3
LSTM_LR       = 5e-3
LSTM_EPOCHS   = 120
LSTM_BATCH    = 16

OUTPUT_DIR    = Path('./output_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    'DPV20_rec5': '#2196F3',
    'DPV21_rec6': '#FF9800',
    'RN51_rec3':  '#9C27B0',
    'pooled':     '#333333',
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SIGNAL RESAMPLING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def resample_signal(sig, t, target_hz=120.0):
    """Linearly interpolate signal to a uniform target sampling rate."""
    t_new = np.arange(t[0], t[-1], 1.0 / target_hz)
    sig_new = np.interp(t_new, t, sig)
    return sig_new, t_new


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'baseline_pupil',      # mean pupil level 2.0–1.5 s before onset
    'pupil_slope',         # linear slope across full pre-event window
    'pupil_delta',         # peri-oscillation mean minus baseline mean
    'pupil_variability',   # std of pupil in pre-event window
    'whisk_slope',         # linear slope of whisking in pre-event window
    'whisk_mean',          # mean whisking in pre-event window
    'whisk_offset_flag',   # 1 if whisking stopped within 3 s before onset
]


def extract_features_for_event(pupil_rs, t_pupil_rs,
                                whisk_rs, t_whisk_rs,
                                event_time):
    """
    Extract a feature vector for a single event (real or negative sample).

    Pupil is lag-corrected: t_pupil_rs values already shifted by PUPIL_LAG_S
    before being passed here. All windows are therefore in lag-corrected time.

    Parameters
    ----------
    pupil_rs     : resampled pupil (% of session max)
    t_pupil_rs   : lag-corrected time axis for pupil
    whisk_rs     : resampled whisking motion energy
    t_whisk_rs   : time axis for whisking
    event_time   : onset time of this event (s)

    Returns
    -------
    features : np.ndarray of shape (7,), or None if insufficient data
    """
    # Masks relative to event_time
    base_m = ((t_pupil_rs >= event_time - 2.0) &
              (t_pupil_rs <  event_time - 1.5))
    pre_m  = ((t_pupil_rs >= event_time - PRE_S) &
              (t_pupil_rs <= event_time))
    peri_m = ((t_pupil_rs >= event_time - 0.5) &
              (t_pupil_rs <= event_time + POST_S))

    if base_m.sum() < 3 or pre_m.sum() < 5 or peri_m.sum() < 3:
        return None

    p_base = pupil_rs[base_m]
    p_pre  = pupil_rs[pre_m]
    t_pre  = t_pupil_rs[pre_m]
    p_peri = pupil_rs[peri_m]

    baseline_pupil    = float(np.nanmean(p_base))
    pupil_delta       = float(np.nanmean(p_peri) - np.nanmean(p_base))
    pupil_variability = float(np.nanstd(p_pre))
    ok                = ~np.isnan(p_pre)
    pupil_slope       = (float(np.polyfit(t_pre[ok] - event_time,
                                           p_pre[ok], 1)[0])
                         if ok.sum() > 3 else 0.0)

    # Whisking features
    if whisk_rs is not None and t_whisk_rs is not None:
        wm = ((t_whisk_rs >= event_time - PRE_S) &
              (t_whisk_rs <= event_time))
        if wm.sum() > 5:
            w_seg = whisk_rs[wm]
            t_seg = t_whisk_rs[wm]
            ok_w  = ~np.isnan(w_seg)
            whisk_mean  = float(np.nanmean(w_seg))
            whisk_slope = (float(np.polyfit(t_seg[ok_w] - event_time,
                                             w_seg[ok_w], 1)[0])
                           if ok_w.sum() > 3 else 0.0)
            # Offset flag: any whisking above threshold 0–3 s before onset
            wm_offset = ((t_whisk_rs >= event_time - 3.0) &
                         (t_whisk_rs <= event_time))
            WHISK_THRESH = 0.5
            whisk_offset_flag = float(
                np.any(whisk_rs[wm_offset] > WHISK_THRESH)
            )
        else:
            whisk_mean = whisk_slope = whisk_offset_flag = 0.0
    else:
        whisk_mean = whisk_slope = whisk_offset_flag = 0.0

    return np.array([
        baseline_pupil,
        pupil_slope,
        pupil_delta,
        pupil_variability,
        whisk_slope,
        whisk_mean,
        whisk_offset_flag,
    ], dtype=float)


def build_feature_matrix(result_dict):
    """
    Build (X, y) feature matrix + labels for one recording.

    Positive samples  (y=1): detected oscillation events
    Negative samples  (y=0): random time points ≥ MIN_GAP_S from any event
    """
    data       = result_dict['data']
    osc_onsets = result_dict['osc_onsets']

    if not osc_onsets or data.get('pupil') is None:
        return None, None

    pupil = data['pupil']
    t_pup = data['pupil_t'].copy()

    # Lag correction
    t_pup_lag = t_pup + PUPIL_LAG_S

    pupil_rs, t_pup_rs = resample_signal(pupil, t_pup_lag, FS_TARGET)

    whisk_rs = t_whisk_rs = None
    if data.get('whisk') is not None:
        whisk_rs, t_whisk_rs = resample_signal(
            data['whisk'], data['whisk_t'], FS_TARGET)

    # ── Positive samples ─────────────────────────────────────────────────────
    X_pos, y_pos = [], []
    for onset in osc_onsets:
        feat = extract_features_for_event(
            pupil_rs, t_pup_rs, whisk_rs, t_whisk_rs, onset)
        if feat is not None and not np.any(np.isnan(feat)):
            X_pos.append(feat)
            y_pos.append(1)

    if not X_pos:
        return None, None

    # ── Negative samples ─────────────────────────────────────────────────────
    # Sample random times, keeping ≥ MIN_GAP_S away from all real events
    t_start = t_pup_rs[0] + PRE_S + PUPIL_LAG_S
    t_end   = t_pup_rs[-1] - POST_S - 0.5
    onsets_arr = np.array(osc_onsets)

    rng = np.random.default_rng(42)
    n_neg_target = len(X_pos) * N_NEG_MULT
    X_neg, y_neg = [], []
    attempts = 0
    while len(X_neg) < n_neg_target and attempts < 50000:
        t_cand = rng.uniform(t_start, t_end)
        if np.all(np.abs(onsets_arr - t_cand) >= MIN_GAP_S):
            feat = extract_features_for_event(
                pupil_rs, t_pup_rs, whisk_rs, t_whisk_rs, t_cand)
            if feat is not None and not np.any(np.isnan(feat)):
                X_neg.append(feat)
                y_neg.append(0)
        attempts += 1

    X = np.vstack(X_pos + X_neg)
    y = np.array(y_pos + y_neg)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENCE EXTRACTION FOR LSTM
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences(result_dict, seq_len_s=2.0):
    """
    Extract fixed-length pupil (+ whisking) time series for each event.
    Returns (X_seq, y_seq) where X_seq has shape (n_events, timesteps, features).
    """
    data       = result_dict['data']
    osc_onsets = result_dict['osc_onsets']

    if not osc_onsets or data.get('pupil') is None:
        return None, None

    t_pup_lag = data['pupil_t'].copy() + PUPIL_LAG_S
    pupil_rs, t_pup_rs = resample_signal(data['pupil'], t_pup_lag, FS_TARGET)

    whisk_rs = t_whisk_rs = None
    if data.get('whisk') is not None:
        whisk_rs, t_whisk_rs = resample_signal(
            data['whisk'], data['whisk_t'], FS_TARGET)

    n_steps = int(seq_len_s * FS_TARGET)  # timesteps per sequence
    onsets_arr = np.array(osc_onsets)

    def get_seq(t_event, label):
        t0 = t_event - seq_len_s
        m  = (t_pup_rs >= t0) & (t_pup_rs <= t_event)
        if m.sum() < n_steps // 2:
            return None, None
        seg_p = pupil_rs[m]
        # Pad or trim to exact n_steps
        if len(seg_p) >= n_steps:
            seg_p = seg_p[-n_steps:]
        else:
            seg_p = np.pad(seg_p, (n_steps - len(seg_p), 0),
                           constant_values=np.nanmean(seg_p))
        # Z-score within sequence
        mu, sd = np.nanmean(seg_p), np.nanstd(seg_p)
        seg_p  = (seg_p - mu) / (sd + 1e-8)

        if whisk_rs is not None:
            mw = (t_whisk_rs >= t0) & (t_whisk_rs <= t_event)
            seg_w = whisk_rs[mw]
            if len(seg_w) >= n_steps:
                seg_w = seg_w[-n_steps:]
            else:
                seg_w = np.pad(seg_w, (n_steps - len(seg_w), 0),
                               constant_values=np.nanmean(seg_w))
            mu_w, sd_w = np.nanmean(seg_w), np.nanstd(seg_w)
            seg_w = (seg_w - mu_w) / (sd_w + 1e-8)
            seq = np.stack([seg_p, seg_w], axis=1)  # (n_steps, 2)
        else:
            seq = seg_p[:, None]  # (n_steps, 1)

        return seq, label

    X_pos, y_pos = [], []
    for onset in osc_onsets:
        seq, lbl = get_seq(onset, 1)
        if seq is not None:
            X_pos.append(seq)
            y_pos.append(lbl)

    rng = np.random.default_rng(42)
    t_start = t_pup_rs[0] + seq_len_s
    t_end   = t_pup_rs[-1] - 0.5
    X_neg, y_neg = [], []
    n_neg_target = len(X_pos) * N_NEG_MULT
    attempts = 0
    while len(X_neg) < n_neg_target and attempts < 50000:
        t_cand = rng.uniform(t_start, t_end)
        if np.all(np.abs(onsets_arr - t_cand) >= MIN_GAP_S):
            seq, lbl = get_seq(t_cand, 0)
            if seq is not None:
                X_neg.append(seq)
                y_neg.append(lbl)
        attempts += 1

    if not X_pos:
        return None, None

    X = np.stack(X_pos + X_neg, axis=0)     # (N, T, F)
    y = np.array(y_pos + y_neg, dtype=int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(all_results):
    """
    Trains logistic regression on pooled data using leave-one-recording-out CV.
    Returns per-recording ROC info, feature importances, and pooled predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    # Build per-recording (X, y)
    recording_data = []
    for r in all_results:
        if r.get('error') or r.get('data') is None:
            continue
        X, y = build_feature_matrix(r)
        if X is None:
            continue
        recording_data.append({'label': r['label'], 'X': X, 'y': y})

    if not recording_data:
        print("  [LR] No data available.")
        return None

    # Leave-one-recording-out
    roc_results = []
    for i, test_rec in enumerate(recording_data):
        train_recs = [r for j, r in enumerate(recording_data) if j != i]
        if not train_recs:
            # Single recording: use stratified k-fold instead
            X_all, y_all = test_rec['X'], test_rec['y']
            cv = StratifiedKFold(n_splits=min(5, int(np.sum(y_all))),
                                 shuffle=True, random_state=42)
            all_probs, all_true = [], []
            for tr, te in cv.split(X_all, y_all):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_all[tr])
                Xte = sc.transform(X_all[te])
                clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
                clf.fit(Xtr, y_all[tr])
                all_probs.extend(clf.predict_proba(Xte)[:, 1])
                all_true.extend(y_all[te])
            fpr, tpr, _ = roc_curve(all_true, all_probs)
            auc = roc_auc_score(all_true, all_probs)
            roc_results.append({
                'label': test_rec['label'],
                'fpr': fpr, 'tpr': tpr, 'auc': auc,
                'cv_type': 'k-fold'
            })
            continue

        X_train = np.vstack([r['X'] for r in train_recs])
        y_train = np.concatenate([r['y'] for r in train_recs])
        X_test  = test_rec['X']
        y_test  = test_rec['y']

        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)

        clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = float('nan')

        roc_results.append({
            'label': test_rec['label'],
            'fpr': fpr, 'tpr': tpr, 'auc': auc,
            'clf': clf, 'scaler': sc,
            'cv_type': 'LORO',
            'X_test': X_test, 'y_test': y_test, 'probs': probs,
        })

    # Train on ALL data for feature importance
    X_all = np.vstack([r['X'] for r in recording_data])
    y_all = np.concatenate([r['y'] for r in recording_data])
    sc_all = StandardScaler()
    clf_all = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    clf_all.fit(sc_all.fit_transform(X_all), y_all)

    return {
        'roc_results'     : roc_results,
        'clf_all'         : clf_all,
        'scaler_all'      : sc_all,
        'recording_data'  : recording_data,
        'X_all'           : X_all,
        'y_all'           : y_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. LSTM
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm(all_results):
    """
    Trains a simple 1-layer LSTM on pre-oscillation pupil + whisking sequences.
    Returns training history and ROC info.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_curve, roc_auc_score
    except ImportError:
        print("  [LSTM] PyTorch not installed. Run: pip install torch")
        return None

    # Pool sequences across all recordings
    all_X, all_y, all_labels = [], [], []
    for r in all_results:
        if r.get('error') or r.get('data') is None:
            continue
        X, y = extract_sequences(r)
        if X is None:
            continue
        all_X.append(X)
        all_y.append(y)
        all_labels.extend([r['label']] * len(y))

    if not all_X:
        print("  [LSTM] No sequence data.")
        return None

    X_np = np.vstack(all_X).astype(np.float32)   # (N, T, F)
    y_np = np.concatenate(all_y).astype(np.float32)

    # ── Model ─────────────────────────────────────────────────────────────────
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                                batch_first=True, dropout=dropout
                                if n_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            out = self.dropout(h[-1])
            return self.fc(out).squeeze(1)  # no sigmoid — BCEWithLogitsLoss applies it

    n_features = X_np.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Leave-one-recording-out on sequences
    unique_labels = list(dict.fromkeys(all_labels))
    labels_arr    = np.array(all_labels)

    roc_results   = []
    fold_losses   = []

    for held_out in unique_labels:
        train_mask = labels_arr != held_out
        test_mask  = labels_arr == held_out

        if test_mask.sum() == 0:
            continue

        Xtr = torch.tensor(X_np[train_mask])
        ytr = torch.tensor(y_np[train_mask])
        Xte = torch.tensor(X_np[test_mask])
        yte = y_np[test_mask]

        if len(torch.unique(ytr)) < 2:
            continue

        # Class weights — passed directly to BCEWithLogitsLoss
        pos_weight = torch.tensor(
            [(ytr == 0).sum().float() / (ytr == 1).sum().float()]
        )

        model    = LSTMClassifier(n_features, LSTM_HIDDEN,
                                  LSTM_LAYERS, LSTM_DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR,
                                     weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        dataset = TensorDataset(Xtr.to(device), ytr.to(device))
        loader  = DataLoader(dataset, batch_size=LSTM_BATCH, shuffle=True)

        epoch_losses = []
        model.train()
        for epoch in range(LSTM_EPOCHS):
            ep_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            epoch_losses.append(ep_loss / len(loader))

        fold_losses.append(epoch_losses)

        model.eval()
        with torch.no_grad():
            logits = model(Xte.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()

        if len(np.unique(yte)) < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(yte, probs)
            auc = roc_auc_score(yte, probs)
        except Exception:
            continue

        roc_results.append({
            'label': held_out,
            'fpr': fpr, 'tpr': tpr, 'auc': auc,
            'probs': probs, 'y_test': yte,
        })

    # Train final model on all data for timeline
    Xall_t = torch.tensor(X_np).to(device)
    yall_t = torch.tensor(y_np).to(device)
    pos_weight_all = torch.tensor(
        [(y_np == 0).sum() / (y_np == 1).sum()]
    ).float()

    model_final = LSTMClassifier(n_features, LSTM_HIDDEN,
                                 LSTM_LAYERS, LSTM_DROPOUT).to(device)
    opt_final = torch.optim.Adam(model_final.parameters(), lr=LSTM_LR,
                                  weight_decay=1e-4)
    crit_final = nn.BCEWithLogitsLoss(pos_weight=pos_weight_all)
    ds_final   = TensorDataset(Xall_t, yall_t)
    ld_final   = DataLoader(ds_final, batch_size=LSTM_BATCH, shuffle=True)

    final_losses = []
    model_final.train()
    for epoch in range(LSTM_EPOCHS):
        ep_loss = 0.0
        for xb, yb in ld_final:
            opt_final.zero_grad()
            pred = model_final(xb)
            loss = crit_final(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)
            opt_final.step()
            ep_loss += loss.item()
        final_losses.append(ep_loss / len(ld_final))

    return {
        'roc_results'  : roc_results,
        'fold_losses'  : fold_losses,
        'final_losses' : final_losses,
        'model_final'  : model_final,
        'device'       : device,
        'X_all'        : X_np,
        'y_all'        : y_np,
        'labels_all'   : labels_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PROBABILITY TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_probability_timeline(result_dict, clf, scaler, use_lstm=False,
                                  lstm_model=None, device=None):
    """
    Slide a prediction window across a recording and compute
    oscillation probability at every time point.
    Returns (t_timeline, prob_timeline, osc_onsets).
    """
    data       = result_dict['data']
    osc_onsets = result_dict['osc_onsets']

    if data.get('pupil') is None:
        return None, None, None

    t_pup_lag = data['pupil_t'].copy() + PUPIL_LAG_S
    pupil_rs, t_pup_rs = resample_signal(data['pupil'], t_pup_lag, FS_TARGET)

    whisk_rs = t_whisk_rs = None
    if data.get('whisk') is not None:
        whisk_rs, t_whisk_rs = resample_signal(
            data['whisk'], data['whisk_t'], FS_TARGET)

    # Step size 0.1 s
    step    = 0.1
    t_start = t_pup_rs[0] + PRE_S
    t_end   = t_pup_rs[-1] - 1.0
    t_grid  = np.arange(t_start, t_end, step)

    probs = []
    for t_ev in t_grid:
        if use_lstm and lstm_model is not None:
            try:
                import torch
                seq_len_s = PRE_S
                n_steps   = int(seq_len_s * FS_TARGET)
                m = (t_pup_rs >= t_ev - seq_len_s) & (t_pup_rs <= t_ev)
                seg_p = pupil_rs[m]
                if len(seg_p) < n_steps // 2:
                    probs.append(np.nan)
                    continue
                if len(seg_p) >= n_steps:
                    seg_p = seg_p[-n_steps:]
                else:
                    seg_p = np.pad(seg_p, (n_steps - len(seg_p), 0),
                                   constant_values=np.nanmean(seg_p))
                mu, sd = np.nanmean(seg_p), np.nanstd(seg_p)
                seg_p  = (seg_p - mu) / (sd + 1e-8)
                if whisk_rs is not None:
                    mw = (t_whisk_rs >= t_ev - seq_len_s) & (t_whisk_rs <= t_ev)
                    seg_w = whisk_rs[mw]
                    if len(seg_w) >= n_steps:
                        seg_w = seg_w[-n_steps:]
                    else:
                        seg_w = np.pad(seg_w, (n_steps - len(seg_w), 0),
                                       constant_values=np.nanmean(seg_w))
                    mu_w, sd_w = np.nanmean(seg_w), np.nanstd(seg_w)
                    seg_w = (seg_w - mu_w) / (sd_w + 1e-8)
                    seq = np.stack([seg_p, seg_w], axis=1)[None].astype(np.float32)
                else:
                    seq = seg_p[None, :, None].astype(np.float32)
                with torch.no_grad():
                    logit = lstm_model(torch.tensor(seq).to(device))
                    p = torch.sigmoid(logit).cpu().item()
                probs.append(p)
            except Exception:
                probs.append(np.nan)
        else:
            feat = extract_features_for_event(
                pupil_rs, t_pup_rs, whisk_rs, t_whisk_rs, t_ev)
            if feat is None or np.any(np.isnan(feat)):
                probs.append(np.nan)
            else:
                p = clf.predict_proba(
                    scaler.transform(feat[None]))[0, 1]
                probs.append(float(p))

    return t_grid, np.array(probs), np.array(osc_onsets)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(lr_out, save_path):
    if lr_out is None:
        return
    clf_all = lr_out['clf_all']
    coefs   = clf_all.coef_[0]
    order   = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#d32f2f' if c > 0 else '#1565C0' for c in coefs[order]]
    bars    = ax.barh(range(len(order)),
                      coefs[order], color=colors, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=11)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Logistic Regression Coefficient\n'
                  '(positive = increases seizure probability)', fontsize=11)
    ax.set_title('Feature Importance: Logistic Regression\n'
                 'trained on all recordings', fontsize=12, fontweight='bold')
    ax.text(0.98, 0.02,
            'Red = increases P(oscillation)\nBlue = decreases P(oscillation)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_roc_curves(lr_out, lstm_out, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('ROC Curves: Oscillation Onset Prediction\n'
                 'Leave-One-Recording-Out Cross-Validation',
                 fontsize=12, fontweight='bold')

    # ── LR panel ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance (AUC = 0.50)')
    if lr_out:
        for rr in lr_out['roc_results']:
            c = COLORS.get(rr['label'], '#888888')
            lbl = f"{rr['label']} (AUC = {rr['auc']:.2f})"
            ax.plot(rr['fpr'], rr['tpr'], color=c, lw=2.2, label=lbl)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Logistic Regression', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── LSTM panel ───────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance (AUC = 0.50)')
    if lstm_out:
        for rr in lstm_out['roc_results']:
            c = COLORS.get(rr['label'], '#888888')
            lbl = f"{rr['label']} (AUC = {rr['auc']:.2f})"
            ax.plot(rr['fpr'], rr['tpr'], color=c, lw=2.2, label=lbl)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('LSTM (pupil + whisking sequence)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_probability_timeline(all_results, lr_out, lstm_out, save_path):
    """
    One panel per recording showing P(oscillation) over time,
    with real event onset times marked as vertical lines.
    """
    valid = [r for r in all_results
             if not r.get('error') and r.get('data') is not None
             and r.get('osc_onsets')]

    if not valid:
        return

    n_rec = len(valid)
    fig, axes = plt.subplots(n_rec, 1, figsize=(14, 4 * n_rec))
    if n_rec == 1:
        axes = [axes]
    fig.suptitle('Predicted Oscillation Probability Over Time\n'
                 '(Logistic Regression — sliding 2 s window)',
                 fontsize=12, fontweight='bold')

    for ax, r in zip(axes, valid):
        clf = lstm_model = device = None
        if lr_out:
            clf    = lr_out['clf_all']
            scaler = lr_out['scaler_all']

        t_grid, probs, onsets = compute_probability_timeline(
            r, clf, scaler)

        if t_grid is None:
            continue

        c = COLORS.get(r['label'], '#333333')
        ax.fill_between(t_grid, 0, probs,
                        alpha=0.25, color=c)
        ax.plot(t_grid, probs, color=c, lw=1.4, label='P(oscillation)')
        ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6,
                   label='Decision threshold (0.5)')

        for onset in onsets:
            ax.axvline(onset, color='red', lw=0.9, alpha=0.6)

        # Shade first 20 s as example window
        t0_ex = onsets[0] - 5 if len(onsets) else t_grid[0]
        ax.axvspan(t0_ex, t0_ex + 20,
                   alpha=0.06, color='orange', label='Example window (20 s)')

        ax.set_ylabel('P(oscillation onset)', fontsize=10)
        ax.set_ylim([0, 1])
        ax.set_title(r['label'], fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time in recording session (s)', fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_lstm_training(lstm_out, save_path):
    if lstm_out is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('LSTM Training Diagnostics', fontsize=12, fontweight='bold')

    ax = axes[0]
    for i, losses in enumerate(lstm_out['fold_losses']):
        lbl = lstm_out['roc_results'][i]['label'] if i < len(
            lstm_out['roc_results']) else f'Fold {i+1}'
        c = COLORS.get(lbl, '#888888')
        ax.plot(losses, color=c, lw=1.8, label=lbl)
    ax.plot(lstm_out['final_losses'], color='k', lw=2.5, ls='--',
            label='Final model (all data)')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('BCE Loss', fontsize=10)
    ax.set_title('Training loss curves\n(leave-one-out folds)', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    for rr in lstm_out['roc_results']:
        c = COLORS.get(rr['label'], '#888888')
        ax.plot(rr['fpr'], rr['tpr'], color=c, lw=2.2,
                label=f"{rr['label']} AUC={rr['auc']:.2f}")
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('LSTM ROC\n(leave-one-recording-out)', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_combined_summary(lr_out, lstm_out, save_path):
    """
    Single-page appendix summary figure: 4 panels.
    Top left: feature importance
    Top right: LR ROC
    Bottom left: LSTM ROC
    Bottom right: AUC summary bar chart
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    fig.suptitle('ML Classifier Summary: Logistic Regression + LSTM\n'
                 'Pre-Oscillation Pupil Dynamics as Predictors of Thalamocortical Onset',
                 fontsize=12, fontweight='bold', y=0.98)

    # Panel A: Feature importance
    ax_a = fig.add_subplot(gs[0, 0])
    if lr_out:
        coefs = lr_out['clf_all'].coef_[0]
        order = np.argsort(np.abs(coefs))[::-1]
        colors_fi = ['#d32f2f' if c > 0 else '#1565C0' for c in coefs[order]]
        ax_a.barh(range(len(order)), coefs[order],
                  color=colors_fi, edgecolor='k', linewidth=0.4)
        ax_a.set_yticks(range(len(order)))
        ax_a.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=9)
        ax_a.axvline(0, color='k', lw=0.8)
        ax_a.set_xlabel('LR coefficient', fontsize=9)
        ax_a.set_title('A. Feature importance\n(Logistic Regression)',
                       fontsize=10, fontweight='bold', loc='left')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Panel B: LR ROC
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    if lr_out:
        for rr in lr_out['roc_results']:
            c = COLORS.get(rr['label'], '#888888')
            ax_b.plot(rr['fpr'], rr['tpr'], color=c, lw=2,
                      label=f"{rr['label']} ({rr['auc']:.2f})")
    ax_b.set_xlabel('FPR', fontsize=9)
    ax_b.set_ylabel('TPR', fontsize=9)
    ax_b.set_title('B. Logistic Regression ROC\n(leave-one-recording-out)',
                   fontsize=10, fontweight='bold', loc='left')
    ax_b.legend(fontsize=8, loc='lower right')
    ax_b.set_xlim([0, 1]); ax_b.set_ylim([0, 1.02])
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Panel C: LSTM ROC
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    if lstm_out:
        for rr in lstm_out['roc_results']:
            c = COLORS.get(rr['label'], '#888888')
            ax_c.plot(rr['fpr'], rr['tpr'], color=c, lw=2,
                      label=f"{rr['label']} ({rr['auc']:.2f})")
    ax_c.set_xlabel('FPR', fontsize=9)
    ax_c.set_ylabel('TPR', fontsize=9)
    ax_c.set_title('C. LSTM ROC\n(pupil + whisking sequence)',
                   fontsize=10, fontweight='bold', loc='left')
    ax_c.legend(fontsize=8, loc='lower right')
    ax_c.set_xlim([0, 1]); ax_c.set_ylim([0, 1.02])
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Panel D: AUC bar chart
    ax_d = fig.add_subplot(gs[1, 1])
    lr_aucs, lstm_aucs, rec_labels = [], [], []
    if lr_out and lstm_out:
        lr_dict   = {rr['label']: rr['auc'] for rr in lr_out['roc_results']}
        lstm_dict = {rr['label']: rr['auc'] for rr in lstm_out['roc_results']}
        for lbl in lr_dict:
            if lbl in lstm_dict:
                rec_labels.append(lbl.replace('_rec', '\nrec'))
                lr_aucs.append(lr_dict[lbl])
                lstm_aucs.append(lstm_dict[lbl])

    if rec_labels:
        x     = np.arange(len(rec_labels))
        width = 0.35
        ax_d.bar(x - width/2, lr_aucs,   width, label='Logistic Regression',
                 color='#2196F3', edgecolor='k', linewidth=0.5)
        ax_d.bar(x + width/2, lstm_aucs, width, label='LSTM',
                 color='#9C27B0', edgecolor='k', linewidth=0.5)
        ax_d.axhline(0.5, color='gray', lw=1.2, ls='--',
                     label='Chance (AUC = 0.5)')
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(rec_labels, fontsize=9)
        ax_d.set_ylabel('AUC', fontsize=9)
        ax_d.set_ylim([0, 1.05])
        ax_d.set_title('D. AUC comparison\nacross recordings + models',
                       fontsize=10, fontweight='bold', loc='left')
        ax_d.legend(fontsize=8)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────

def write_ml_summary(lr_out, lstm_out, save_path):
    lines = []
    lines.append("ML CLASSIFIER SUMMARY — ROZENDAL THESIS")
    lines.append("=" * 70)

    if lr_out:
        lines.append("\nLOGISTIC REGRESSION (Leave-One-Recording-Out):")
        lines.append(f"  Features: {FEATURE_NAMES}")
        for rr in lr_out['roc_results']:
            lines.append(f"  {rr['label']:<20}  AUC = {rr['auc']:.3f}  "
                         f"({rr.get('cv_type', 'LORO')})")
        clf = lr_out['clf_all']
        lines.append("\n  Feature coefficients (full model):")
        for name, coef in zip(FEATURE_NAMES, clf.coef_[0]):
            lines.append(f"    {name:<25} {coef:+.4f}")

    if lstm_out:
        lines.append("\nLSTM (Leave-One-Recording-Out):")
        lines.append(f"  Architecture: {LSTM_LAYERS}-layer LSTM, "
                     f"hidden={LSTM_HIDDEN}, dropout={LSTM_DROPOUT}")
        for rr in lstm_out['roc_results']:
            lines.append(f"  {rr['label']:<20}  AUC = {rr['auc']:.3f}")

    lines.append("\n" + "=" * 70)
    text = "\n".join(lines)
    print("\n" + text)
    with open(save_path, 'w') as f:
        f.write(text + "\n")
    print(f"\n  ML summary saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline(results: list, output_dir: Path = OUTPUT_DIR):
    """
    Main entry point. Call with the results list from run_all_recordings.main().

    Parameters
    ----------
    results    : list of result dicts from process_recording()
    output_dir : Path to save figures
    """
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "═" * 62)
    print("  ML Classifier Pipeline")
    print("  Logistic Regression + LSTM")
    print("═" * 62)

    # ── Logistic regression ───────────────────────────────────────────────────
    print("\n[1/5] Running logistic regression ...")
    lr_out = run_logistic_regression(results)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Running LSTM ...")
    lstm_out = run_lstm(results)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[3/5] Generating figures ...")
    plot_feature_importance(
        lr_out,
        output_dir / 'ML_01_feature_importance.png')

    plot_roc_curves(
        lr_out, lstm_out,
        output_dir / 'ML_02_roc_curves.png')

    plot_probability_timeline(
        results, lr_out, lstm_out,
        output_dir / 'ML_03_probability_timeline.png')

    plot_lstm_training(
        lstm_out,
        output_dir / 'ML_04_lstm_training.png')

    plot_combined_summary(
        lr_out, lstm_out,
        output_dir / 'ML_05_combined_summary.png')

    # ── Summary text ──────────────────────────────────────────────────────────
    print("\n[4/5] Writing summary ...")
    write_ml_summary(
        lr_out, lstm_out,
        output_dir / 'ML_summary.txt')

    print("\n[5/5] Done.")
    print(f"  Figures saved to {output_dir.resolve()}")
    print("\n  Files produced:")
    for fname in [
        'ML_01_feature_importance.png',
        'ML_02_roc_curves.png',
        'ML_03_probability_timeline.png',
        'ML_04_lstm_training.png',
        'ML_05_combined_summary.png',
        'ML_summary.txt',
    ]:
        print(f"    {fname}")

    return {'lr_out': lr_out, 'lstm_out': lstm_out}


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Standalone usage: loads recordings one at a time to conserve RAM,
    extracts features immediately, then discards raw data before loading next.
    Requires the same data drive as run_all_recordings.py.
    """
    import sys
    import gc
    sys.path.insert(0, os.path.dirname(__file__))

    from load_data                import load_mat, extract_recording_vars
    from thalamic_burst_detection import detect_all_units

    RECORDINGS = [
        ('DPV20_rec5',
         '/Volumes/home/Dennis_to_Kevin/LP/'
         'DPV20_rec5_analyzed_wspikes_NRPX_sync_corrected.mat'),
        ('DPV21_rec6',
         '/Volumes/home/Dennis_to_Kevin/LP/'
         'DPV21_rec6_analyzed_wspikes_and_NRPX_sync_corrected.mat'),
        ('RN51_rec3',
         '/Volumes/home/Dennis_to_Kevin/LP/'
         'RN51_rec3_L23_LP_NRPX_sync_corrected.mat'),
    ]

    # ── Memory-efficient loading ──────────────────────────────────
    # Load one recording at a time. Keep only lightweight result dicts
    # (feature matrices, onset times, small signal arrays for alignment).
    # Discard the full raw Vm (50M samples) immediately after extraction.

    # ── Hardcoded event counts from the main pipeline run ───────
    # These onset times were detected and saved by run_all_recordings.py.
    # We reload only the lightweight behavioural signals (pupil + whisk)
    # needed for ML features — Vm is never loaded, saving ~1.5 GB RAM.

    results = []
    for label, path in RECORDINGS:
        print(f"\nLoading {label} (behavioural signals only) ...")
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}")
            results.append({'label': label, 'error': 'file not found'})
            continue

        raw = load_mat(path)

        # Extract only the variables we need — skip Vm entirely
        def _sq(x):
            if x is None: return None
            return np.asarray(x).squeeze().ravel().astype(float)

        slim_data = {}
        for src, dst in [('pup_norm','pupil'), ('pupil_area','pupil'),
                         ('Wtt','pupil_t'), ('Wtt','whisk_t'),
                         ('W_abs_smo','whisk'), ('whisk_ME','whisk'),
                         ('W','wheel'), ('Wtt','wheel_t')]:
            if src in raw and dst not in slim_data:
                slim_data[dst] = _sq(raw.get(src))

        # Thalamic units needed only for burst detection (small arrays)
        thalamic_units = []
        for key in ('dLGN_cells', 'LP_cells', 'thalamic_units'):
            if key in raw:
                ru = raw[key]
                if isinstance(ru, list):
                    thalamic_units = [np.asarray(u).ravel().astype(float) for u in ru]
                elif hasattr(ru, 'dtype') and ru.dtype == object:
                    thalamic_units = [np.asarray(ru[i]).ravel().astype(float)
                                      for i in range(ru.size)]
                break

        del raw
        gc.collect()

        _, _, osc_onsets = detect_all_units(thalamic_units, verbose=False)
        del thalamic_units
        gc.collect()

        results.append({
            'label'     : label,
            'osc_onsets': osc_onsets,
            'n_events'  : len(osc_onsets),
            'data'      : slim_data,
        })
        print(f"  {label}: {len(osc_onsets)} events loaded")

    run_ml_pipeline(results)
