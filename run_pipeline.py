# ==============================================================
# PIEZO-ACOUSTIC FULL PIPELINE - RUN ALL (standalone, NO CNN)
# - CNN removed completely
# - FFT classifier uses HistGradientBoosting (HGB)
# - Saves graphs (confusion matrices, FFT example, pie chart) to PNG
# - Writes summary.json with metrics + environment
# ==============================================================

import importlib, sys, os, json, zipfile, warnings, subprocess, time, platform, argparse
importlib.invalidate_caches()
warnings.filterwarnings("ignore")

# ----------------------------
# 0) Robust auto-install helper
# ----------------------------
def _ensure(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
        return True
    except ModuleNotFoundError:
        print("[INFO] Installing %s ..." % pip_name)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q", "--no-warn-script-location"],
            check=False
        )
        importlib.invalidate_caches()
        try:
            importlib.import_module(import_name)
            print("[OK] %s imported." % import_name)
            return True
        except Exception as e:
            print("[ERROR] Failed importing %s: %s" % (import_name, str(e)))
            return False

_ensure("numpy")
_ensure("pandas")
_ensure("matplotlib")
_ensure("scipy")
_ensure("joblib")
_ensure("sklearn", "scikit-learn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

# ----------------------------
# Args / default output
# ----------------------------
def _default_output_dir():
    env_out = os.environ.get("PIEZO_OUT", "").strip()
    if env_out:
        return env_out
    if "google.colab" in sys.modules:
        return "/content/results_piezo"
    return "./saida_piezo"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=_default_output_dir(), help="Output directory")
ap.add_argument("--seed", type=int, default=42, help="Random seed")
ap.add_argument("--n_samples", type=int, default=100, help="Samples per medium")
ap.add_argument("--zip", action="store_true", help="Write publication zip at end")
ap.add_argument("--fft_storage", choices=["npz", "csv"], default="npz",
                help="Store FFTs as a single NPZ (default) or many CSV files")
ap.add_argument("--fft_repr", choices=["raw", "logf"], default="raw",
                help="FFT representation for HGB: raw (amplitude) or logf (log-frequency grid)")
ap.add_argument("--fft_points", type=int, default=4096,
                help="Number of points for FFT input to HGB (raw pad/cut; logf uses interpolation grid)")
args = ap.parse_args()

OUTPUT_DIR = os.path.abspath(args.out)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "dados_sinteticos", "dados_brutos"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "dados_reais"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "out_model"), exist_ok=True)

DIR_SYN_FFT = os.path.join(OUTPUT_DIR, "dados_sinteticos", "dados_brutos")
DIR_OUT = os.path.join(OUTPUT_DIR, "out_model")

print("OUTPUT_DIR =", OUTPUT_DIR)
print("FFT storage =", args.fft_storage)
print("FFT repr    =", args.fft_repr)
print("FFT points  =", int(args.fft_points))

# ----------------------------
# Synthetic media + colors
# ----------------------------
meios = {
    "agua": (1000, 1500, 1.0),
    "oleo": (850, 1400, 10),
    "ar": (1.2, 343, 0.02),
    "areia_seca": (1600, 300, 50),
    "areia_umida": (1800, 450, 70),
    "argila": (1900, 250, 200),
    "lama": (1750, 200, 400),
    "silte": (1650, 280, 150),
    "glicerina": (1260, 1900, 1200),
    "concreto": (2300, 3500, 1000),
}

color_map = {
    "agua": "#00bfff",
    "oleo": "#111111",
    "ar": "#bbbbbb",
    "areia_seca": "#ffe066",
    "areia_umida": "#0066cc",
    "argila": "#a0522d",
    "lama": "#556b2f",
    "silte": "#98fb98",
    "glicerina": "#daa520",
    "concreto": "#555555",
    "areia": "#ffd24d",
}

# ----------------------------
# 1) Generate synthetic data
# ----------------------------
np.random.seed(int(args.seed))

fs = 1e6
t = np.arange(0, 0.005, 1e-6)
espessura = 2e-3
n_samples = int(args.n_samples)

dados = []
t0 = time.time()

fft_bank = {}
FFT_BANK_PATH = None

for meio, (rho, c, visc) in meios.items():
    for i in range(n_samples):
        fr0 = c / (2.0 * espessura)
        fa0 = fr0 * 1.4
        fr = fr0 * (1.0 + 0.02 * np.random.randn())
        fa = fa0 * (1.0 + 0.02 * np.random.randn())

        v = (
            np.sin(2.0 * np.pi * fr * t) * np.exp(-1000.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * fa * t) * np.exp(-800.0 * t)
            + 0.05 * np.random.randn(len(t))
        )

        f = np.fft.rfftfreq(len(v), 1.0 / fs)
        A = np.abs(np.fft.rfft(v * np.hanning(len(v))))
        A /= (A.max() + 1e-9)

        peaks, _ = find_peaks(A, height=0.05, distance=100)
        if len(peaks) >= 2:
            frd, fad = f[peaks[0]], f[peaks[1]]
        elif len(peaks) == 1:
            frd, fad = f[peaks[0]], f[peaks[0]] * 1.2
        else:
            frd, fad = fr, fa

        k2 = (np.pi / 2.0) * ((fad - frd) / max(fad, 1e-12))
        Z = rho * c
        nome = "%s_%03d" % (meio, i)

        if args.fft_storage == "csv":
            np.savetxt(
                os.path.join(DIR_SYN_FFT, "fft_%s.csv" % nome),
                np.c_[f, A],
                delimiter=",",
                header="frequencia,amplitude",
                comments="",
            )
        else:
            fft_bank[nome] = np.c_[f, A].astype(np.float32)

        dados.append({"amostra": nome, "meio": meio, "f_r": frd, "f_a": fad, "k2": k2, "Z": Z})

df_sint = pd.DataFrame(dados)
df_sint.to_csv(os.path.join(OUTPUT_DIR, "banco_piezo_total.csv"), index=False)
print("Synthetic samples:", len(df_sint))

if args.fft_storage == "npz":
    FFT_BANK_PATH = os.path.join(DIR_SYN_FFT, "fft_bank.npz")
    np.savez_compressed(FFT_BANK_PATH, **fft_bank)
    print("FFT bank saved:", FFT_BANK_PATH)

# ----------------------------
# 2) Hybrid bank
# ----------------------------
df_hibrid = df_sint.copy()
df_hibrid["tipo"] = "synthetic"
df_hibrid.to_csv(os.path.join(OUTPUT_DIR, "banco_piezo_hibrido.csv"), index=False)
print("Hybrid bank created.")

# ----------------------------
# 3) Features model (RandomForest) + CM PNG
# ----------------------------
feat_cols = ["f_r", "f_a", "k2", "Z"]
X = df_hibrid[feat_cols]
y = df_hibrid["meio"]

le = LabelEncoder()
yid = le.fit_transform(y)

Xtr, Xte, ytr, yte = train_test_split(
    X, yid, test_size=0.2, stratify=yid, random_state=int(args.seed)
)

pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=300, random_state=int(args.seed)),
)
pipe.fit(Xtr, ytr)
yp = pipe.predict(Xte)
acc_feat = float(accuracy_score(yte, yp))
print("Accuracy (features):", acc_feat)

joblib.dump(
    {"pipeline": pipe, "label_encoder": le, "feat_cols": feat_cols},
    os.path.join(DIR_OUT, "model_features.joblib"),
)

cm = confusion_matrix(
    le.inverse_transform(yte),
    le.inverse_transform(yp),
    labels=le.classes_,
)
n_cls = len(le.classes_)
fig_w = max(7.0, 0.9 * n_cls)
fig_h = max(6.0, 0.8 * n_cls)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
ax.set_title("Confusion Matrix - Features (RandomForest)")
plt.tight_layout()
cm_features_png = os.path.join(DIR_OUT, "cm_features.png")
plt.savefig(cm_features_png, dpi=220)
plt.close()

# ----------------------------
# 4) FFT loaders + preprocessing
# ----------------------------
def preprocess_fft_for_hgb(X, mode="logz", crop=(10, 4096)):
    X = np.asarray(X)
    if X.ndim == 3:
        X2 = X[:, :, 0]
    else:
        X2 = X

    i0, i1 = crop
    i0 = max(0, int(i0))
    i1 = min(X2.shape[1], int(i1))
    if i1 > i0:
        X2 = X2[:, i0:i1]

    if mode in ("logz", "log"):
        X2 = np.log1p(np.maximum(X2, 0.0))

    if mode in ("logz", "z"):
        mu = X2.mean(axis=1, keepdims=True)
        sd = X2.std(axis=1, keepdims=True) + 1e-8
        X2 = (X2 - mu) / sd

    return X2.astype(np.float32)

def load_fft_tensor_from_csv(df, n_points=8192):
    X_list, y_list = [], []
    for nm, meio in zip(df["amostra"], df["meio"]):
        p = os.path.join(DIR_SYN_FFT, "fft_%s.csv" % nm)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        a = d.iloc[:, 1].values.astype(np.float32)
        a /= (a.max() + 1e-9)
        if len(a) >= n_points:
            a = a[:n_points]
        else:
            a = np.pad(a, (0, n_points - len(a)))
        X_list.append(a)
        y_list.append(meio)
    return np.array(X_list, dtype=np.float32), np.array(y_list)

def load_fft_tensor_from_npz(df, npz_path, n_points=8192):
    if (npz_path is None) or (not os.path.exists(npz_path)):
        return np.zeros((0, n_points), dtype=np.float32), np.array([])
    bank = np.load(npz_path, allow_pickle=False)
    X_list, y_list = [], []
    for nm, meio in zip(df["amostra"], df["meio"]):
        if nm not in bank.files:
            continue
        arr = bank[nm]
        a = arr[:, 1].astype(np.float32)
        a /= (a.max() + 1e-9)
        if len(a) >= n_points:
            a = a[:n_points]
        else:
            a = np.pad(a, (0, n_points - len(a)))
        X_list.append(a)
        y_list.append(meio)
    return np.array(X_list, dtype=np.float32), np.array(y_list)

def load_fft_tensor_logf_from_npz(df, npz_path, n_points=4096, fmin=100.0, fmax=5e5):
    if (npz_path is None) or (not os.path.exists(npz_path)):
        return np.zeros((0, n_points), dtype=np.float32), np.array([])
    bank = np.load(npz_path, allow_pickle=False)

    logf_grid = np.linspace(np.log10(fmin), np.log10(fmax), n_points)
    f_grid = (10 ** logf_grid).astype(np.float64)

    X_list, y_list = [], []
    for nm, meio in zip(df["amostra"], df["meio"]):
        if nm not in bank.files:
            continue
        arr = bank[nm]
        f = arr[:, 0].astype(np.float64)
        A = arr[:, 1].astype(np.float64)
        A = np.log1p(A)
        A_interp = np.interp(f_grid, f, A, left=A[0], right=A[-1]).astype(np.float32)
        X_list.append(A_interp)
        y_list.append(meio)

    return np.array(X_list, dtype=np.float32), np.array(y_list)

# ----------------------------
# 5) HGB on FFT + CM PNG
# ----------------------------
acc_fft = None
if args.fft_storage == "csv":
    Xf, yf = load_fft_tensor_from_csv(df_sint, n_points=int(args.fft_points))
else:
    if args.fft_repr == "raw":
        Xf, yf = load_fft_tensor_from_npz(df_sint, FFT_BANK_PATH, n_points=int(args.fft_points))
    else:
        Xf, yf = load_fft_tensor_logf_from_npz(df_sint, FFT_BANK_PATH, n_points=int(args.fft_points))

Xf2 = preprocess_fft_for_hgb(Xf, mode="logz", crop=(10, min(4096, int(args.fft_points))))

if (Xf2 is not None) and (len(Xf2) > 0):
    le_fft = LabelEncoder()
    y_fft = le_fft.fit_transform(yf).astype(np.int32)

    Xtrf, Xtef, ytrf, ytef = train_test_split(
        Xf2, y_fft, test_size=0.2, stratify=y_fft, random_state=int(args.seed)
    )

    hgb_fft = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=int(args.seed),
    )

    hgb_fft.fit(Xtrf, ytrf)
    ypf = hgb_fft.predict(Xtef)
    acc_fft = float(accuracy_score(ytef, ypf))
    print("Accuracy (FFT, HGB):", acc_fft)

    joblib.dump(
        {
            "model": hgb_fft,
            "label_encoder": le_fft,
            "n_points": int(Xf2.shape[1]),
            "repr": str(args.fft_repr),
            "fft_storage": str(args.fft_storage),
        },
        os.path.join(DIR_OUT, "model_fft_hgb.joblib"),
    )

    cmf = confusion_matrix(
        le_fft.inverse_transform(ytef),
        le_fft.inverse_transform(ypf),
        labels=le_fft.classes_,
    )

    n_cls2 = len(le_fft.classes_)
    fig_w2 = max(7.0, 0.9 * n_cls2)
    fig_h2 = max(6.0, 0.8 * n_cls2)

    fig, ax = plt.subplots(figsize=(fig_w2, fig_h2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cmf, display_labels=le_fft.classes_)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion Matrix - FFT (HGB)")
    plt.tight_layout()
    cm_fft_hgb_png = os.path.join(DIR_OUT, "cm_fft_hgb.png")
    plt.savefig(cm_fft_hgb_png, dpi=220)
    plt.close()
else:
    cm_fft_hgb_png = None
    print("No FFT tensors loaded; skipping HGB-FFT.")

# ----------------------------
# 6) Example inference + FFT example plot PNG
# ----------------------------
example_nm = df_sint.iloc[0]["amostra"]

if args.fft_storage == "csv":
    example_path = os.path.join(DIR_SYN_FFT, "fft_%s.csv" % example_nm)
    d = pd.read_csv(example_path)
    f_ex = d.iloc[:, 0].values.astype(np.float64)
    a_ex = d.iloc[:, 1].values.astype(np.float64)
    a_ex = a_ex / (a_ex.max() + 1e-9)
else:
    bank = np.load(FFT_BANK_PATH, allow_pickle=False)
    ex = bank[example_nm]
    f_ex = ex[:, 0].astype(np.float64)
    a_ex = ex[:, 1].astype(np.float64)
    a_ex = a_ex / (a_ex.max() + 1e-9)

peaks, _ = find_peaks(a_ex, height=0.05, distance=50)
if len(peaks) >= 2:
    frp, fap = f_ex[peaks[0]], f_ex[peaks[1]]
elif len(peaks) == 1:
    frp, fap = f_ex[peaks[0]], f_ex[peaks[0]] * 1.2
else:
    frp, fap = f_ex[0], f_ex[0] * 1.2

k2_ex = (np.pi / 2.0) * ((fap - frp) / max(fap, 1e-12))
Z_est = float(df_hibrid["Z"].mean())
arr = np.array([[frp, fap, k2_ex, Z_est]], dtype=np.float64)

pred_id = int(pipe.predict(arr)[0])
pred = le.inverse_transform([pred_id])[0]
try:
    proba = float(pipe.predict_proba(arr)[0].max())
except Exception:
    proba = 0.0

print("Example prediction:", pred, "| confidence:", proba)

plt.figure(figsize=(9, 3.6))
plt.plot(f_ex / 1000.0, a_ex, lw=2)
plt.axvline(frp / 1000.0, ls="--", lw=1)
plt.axvline(fap / 1000.0, ls="--", lw=1)
plt.title("%s - FFT example" % str(pred).capitalize())
plt.xlabel("Frequency (kHz)")
plt.ylabel("Amplitude (normalized)")
plt.grid(alpha=0.25)
plt.tight_layout()
fft_example_png = os.path.join(DIR_OUT, "fft_example.png")
plt.savefig(fft_example_png, dpi=220)
plt.close()

# ----------------------------
# 7) Pie chart: dataset composition (PNG)
# ----------------------------
cont = df_sint["meio"].value_counts()
labs = cont.index.tolist()
vals = cont.values.tolist()
colors = [color_map.get(lbl, None) for lbl in labs]

plt.figure(figsize=(6.5, 6.5))
plt.pie(vals, labels=labs, autopct="%1.1f%%", colors=colors)
plt.title("Synthetic dataset composition")
plt.tight_layout()
dataset_pie_png = os.path.join(DIR_OUT, "dataset_composition_pie.png")
plt.savefig(dataset_pie_png, dpi=220)
plt.close()

# ----------------------------
# 8) Summary report (NO CNN keys)
# ----------------------------
def _safe_ver(modname):
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__version__", None)
    except Exception:
        return None

summary = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "output_dir": OUTPUT_DIR,
    "seed": int(args.seed),
    "n_samples_per_medium": int(n_samples),
    "n_media": int(len(meios)),
    "n_total_samples": int(len(df_sint)),
    "fft_storage": str(args.fft_storage),
    "fft_repr": str(args.fft_repr),
    "fft_points": int(args.fft_points),
    "fft_bank_path": FFT_BANK_PATH,
    "features_model_accuracy": acc_feat,
    "fft_hgb_accuracy": acc_fft,
    "example_prediction": {"label": pred, "confidence": proba},
    "artifacts": {
        "bank_total_csv": os.path.join(OUTPUT_DIR, "banco_piezo_total.csv"),
        "bank_hybrid_csv": os.path.join(OUTPUT_DIR, "banco_piezo_hibrido.csv"),
        "model_features_joblib": os.path.join(DIR_OUT, "model_features.joblib"),
        "cm_features_png": cm_features_png,
        "model_fft_hgb_joblib": os.path.join(DIR_OUT, "model_fft_hgb.joblib"),
        "cm_fft_hgb_png": cm_fft_hgb_png,
        "fft_example_png": fft_example_png,
        "dataset_pie_png": dataset_pie_png,
    },
    "environment": {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": _safe_ver("numpy"),
        "pandas": _safe_ver("pandas"),
        "sklearn": _safe_ver("sklearn"),
        "scipy": _safe_ver("scipy"),
        "matplotlib": _safe_ver("matplotlib"),
    },
    "elapsed_sec": float(time.time() - t0),
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as fp:
    json.dump(summary, fp, indent=2)

# ----------------------------
# 9) Zip outputs (optional)
# ----------------------------
ZIP_PATH = os.path.join(OUTPUT_DIR, "saida_piezo_publicacao.zip")
if args.zip:
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for r, _, fs_ in os.walk(OUTPUT_DIR):
            for fz in fs_:
                if fz.lower().endswith(".zip"):
                    continue
                fullp = os.path.join(r, fz)
                relp = os.path.relpath(fullp, OUTPUT_DIR)
                z.write(fullp, arcname=relp)
    print("Final ZIP saved:", ZIP_PATH)

print("PIEZO pipeline finished successfully.")
