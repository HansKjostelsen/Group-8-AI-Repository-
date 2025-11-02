import os, re, glob, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- KONFIG --------------------

# Bytt disse to linjene hvis for å kjøre train_cut og test_cut
ROOTS_TRAIN = ["data/IDMT-ISA-ELECTRIC-ENGINE/train_cut"]
ROOTS_TEST  = ["data/IDMT-ISA-ELECTRIC-ENGINE/test"]

TARGET_SR  = 22_050
DURATION_S = 4.0
N_MFCC     = 40
HOP_LENGTH = 512
N_FFT      = 2048

LABEL_MAP = {"good": 0, "broken": 1, "heavyload": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Vekt og “tilt” (favoriser heavyload litt når sannsynligheten er nær good)
WEIGHTS = {LABEL_MAP["good"]: 1.0, LABEL_MAP["broken"]: 1.2, LABEL_MAP["heavyload"]: 3.0}
ALPHA   = 1.2

# -------------------- HJELPERE --------------------

def collect_items(roots):
    """Finn .wav-filer og avled label fra fil-/mappenavn."""
    items = []
    for root in roots:
        for w in glob.glob(str(Path(root) / "**" / "*.wav"), recursive=True):
            p = Path(w)
            lbl = None
            n, pn = p.name.lower(), p.parent.name.lower()
            for key in LABEL_MAP:
                if key in n or key in pn:
                    lbl = LABEL_MAP[key]; break
            if lbl is None:
                continue
            items.append((str(p), lbl))
    return items


def load_audio_fixed(path, sr=TARGET_SR, duration_s=DURATION_S):
    """Mono, resample og klipp/pad til fast lengde."""
    y, file_sr = sf.read(path, always_2d=False)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y=y, orig_sr=file_sr, target_sr=sr)
    target_len = int(sr * duration_s)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def mfcc_feature_vector(y, sr):
    """MFCC + deltas, CMVN pr. fil, robuste tids-statistikker."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    F = np.vstack([mfcc, d1, d2])

    stats = []
    for row in F:
        stats += [np.mean(row), np.std(row), np.median(row),
                  np.percentile(row, 10), np.percentile(row, 90)]
    return np.asarray(stats, dtype=np.float32)


def extra_spectral(y, sr):
    """Centroid, rolloff, bandwidth, ZCR, RMS (mean/std/median)."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))**2
    centroid  = librosa.feature.spectral_centroid(S=S, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    energy    = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    def summarize(M): return [np.mean(M), np.std(M), np.median(M)]
    return np.asarray(summarize(centroid) + summarize(rolloff) +
                      summarize(bandwidth) + summarize(zcr) +
                      summarize(energy), dtype=np.float32)


def build_features(items):
    """Ekstraher feature-vektor pr. fil."""
    X_rows, y_rows = [], []
    for path, lbl in items:
        try:
            y, sr = load_audio_fixed(path)
            vec = np.hstack([mfcc_feature_vector(y, sr), extra_spectral(y, sr)])
            X_rows.append(vec); y_rows.append(lbl)
        except Exception as e:
            print(f"[FEIL] {path}: {e}")
    return np.vstack(X_rows), np.array(y_rows, dtype=int)


def plot_confusion_norsk(cm, classes, normalize=False, save_path=None):
    """Norsk konfusjonsmatrise — samme stil som før (lilla/gul)."""
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(7.2, 6.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f" if not normalize else ".2f",
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar=False,
        annot_kws={"fontsize": 12}
    )
    plt.title("Konfusjonsmatrise", fontsize=22, fontweight="bold", pad=12)
    plt.xlabel("Predikert", fontsize=14)
    plt.ylabel("Fasit", fontsize=14)
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45, ha="right")
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# --- “Safe” validering: hopper over split hvis for få prøver/klasser
from collections import Counter
def safe_val_split(X, y, test_size=0.15, random_state=42):
    counts = Counter(y)
    if len(y) < 10 or min(counts.values()) < 2:
        print("⚠️  For få prøver per klasse → hopper over valideringssplitt.")
        return X, X, y, y  # signaliser “skippet” ved å returnere samme objekter
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# -------------------- HOVEDPROGRAM --------------------

if __name__ == "__main__":
    # 1) Finn filer
    train_items = collect_items(ROOTS_TRAIN)
    test_items  = collect_items(ROOTS_TEST)

    print("Train files:", len(train_items))
    print("Test files :", len(test_items))
    print("Train class dist:",
          {INV_LABEL_MAP[k]: v for k, v in Counter([l for _, l in train_items]).items()})
    print("Test  class dist:",
          {INV_LABEL_MAP[k]: v for k, v in Counter([l for _, l in test_items]).items()})

    # 2) Features
    print("Henter ut trekk (train)…")
    X_train, y_train = build_features(train_items)
    print("Henter ut trekk (test)…")
    X_test,  y_test  = build_features(test_items)

    # 3) Skalering
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4) Sikker validerings-splitt (hopper over hvis for lite)
    X_tr, X_val, y_tr, y_val = safe_val_split(X_train_s, y_train)

    # 5) Modeller
    rf  = RandomForestClassifier(
        n_estimators=900, max_features="sqrt", min_samples_leaf=2,
        random_state=42, class_weight=WEIGHTS
    )
    svm = SVC(kernel="rbf", C=6, gamma="scale",
              class_weight=WEIGHTS, probability=True)

    # 6) Modellvalg
    if X_tr is X_val:
        print("⚠️  Skipper modellvalg (for lite datasett) → bruker Random Forest.")
        best_name, best_model, best_f1 = "RF", rf, None
    else:
        best_name, best_model, best_f1 = None, None, -1
        for name, m in [("RF", rf), ("SVM", svm)]:
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)
            f1 = f1_score(y_val, pred, average="macro")
            print(f"{name} val macro-F1: {f1:.3f}")
            if f1 > best_f1:
                best_name, best_model, best_f1 = name, m, f1
        print(f"→ Bruker {best_name} (val macro-F1={best_f1:.3f})")

    # 7) Tren på hele train-settet og evaluer på test
    best_model.fit(X_train_s, y_train)
    base_pred = best_model.predict(X_test_s)

    # 8) Valgfri “tilt” mot heavyload ved proba
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test_s)
        idx_good, idx_hl = LABEL_MAP["good"], LABEL_MAP["heavyload"]
        y_pred = base_pred.copy()
        for i in range(len(base_pred)):
            if proba[i, idx_hl] > ALPHA * proba[i, idx_good]:
                y_pred[i] = idx_hl
    else:
        y_pred = base_pred

    # 9) Rapport + figur
    target_names = [INV_LABEL_MAP[i] for i in sorted(LABEL_MAP.values())]
    print("\nKlassifikasjonsrapport:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABEL_MAP.values()))
    plot_confusion_norsk(cm, target_names, normalize=False, save_path="konfusjonsmatrise.png")
