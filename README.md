# Piezo Pipeline (FFT + ML)

End-to-end pipeline for piezoelectric signal analysis and medium/material classification.

This repository provides a complete "run-all" workflow:
- Synthetic dataset generation (multi-class)
- FFT computation and peak extraction
- Feature engineering (physics-inspired summary features)
- Machine learning baselines:
  - RandomForest on engineered features
  - HistGradientBoosting (HGB) on FFT representation
- Automatic export of plots, trained models, and a summary report

The goal is to offer a reproducible, lightweight, local-first pipeline that can be cloned and executed with a single command.

---

## 1) Repository structure

```
piezo-pipeline/
  run_pipeline.py
  requirements.txt
  README.md
  LICENSE
  .gitignore

  artifacts/
    summary.json
    banco_piezo_total.csv
    banco_piezo_hibrido.csv
    model_features.joblib
    model_fft_hgb.joblib

  images/
    cm_features.png
    cm_fft_hgb.png
    dataset_composition_pie.png
    fft_example.png
```

### What each folder/file is for

- `run_pipeline.py`
  Main entrypoint. Running this script generates the dataset, trains the models, evaluates results, and exports outputs.

- `requirements.txt`
  Minimal dependencies to run locally with `pip`.

- `images/`
  Example output images included for README preview.

- `artifacts/`
  Example artifacts generated from one run (models + CSV + summary). These are included for demonstration and quick inspection.

- `.gitignore`
  Prevents committing large/generated outputs while keeping demo `images/` and `artifacts/` tracked.

---

## 2) Quickstart (clone and run)

### 2.1) Windows (CMD / PowerShell)

From the repository root folder:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

### 2.2) Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

---

## 3) What the pipeline produces

When you run:

```bash
python run_pipeline.py
```

the script will generate an output directory (typically `./saida_piezo/` or similar), containing:

- Trained models (`.joblib`)
- Confusion matrices (`.png`)
- FFT example plots (`.png`)
- Dataset composition plot (`.png`)
- CSV tables (synthetic dataset tables)
- `summary.json` with key metrics and run metadata

If your script uses a different output directory name, update this README accordingly.

---

## 4) Example results (tracked in this repo)

The images below are committed in `images/` so GitHub can render them directly.

### Confusion Matrix - Features (RandomForest)
![Confusion Matrix - Features](images/cm_features.png)

### Confusion Matrix - FFT (HistGradientBoosting)
![Confusion Matrix - FFT](images/cm_fft_hgb.png)

### Synthetic dataset composition
![Dataset composition](images/dataset_composition_pie.png)

### FFT example (signal)
![FFT example](images/fft_example.png)

---

## 5) Demo artifacts (tracked in this repo)

The folder `artifacts/` contains a small set of outputs generated from one successful run:

- `summary.json`
  Summary of metrics and configuration (useful for quick QA).

- `banco_piezo_total.csv`
  Full synthetic dataset table.

- `banco_piezo_hibrido.csv`
  Dataset table used for the hybrid/FFT workflow (if applicable).

- `model_features.joblib`
  Trained RandomForest model using engineered features.

- `model_fft_hgb.joblib`
  Trained HistGradientBoosting model using FFT representation.

These artifacts are included for demonstration only. Running the pipeline locally regenerates outputs.

---

## 6) Notes on interpretation

- The feature-based classifier (RandomForest) is expected to perform strongly on the synthetic dataset, because engineered features can capture discriminative structure.
- The FFT-based baseline (HGB) is intentionally more challenging and may show lower accuracy depending on:
  - frequency resolution
  - normalization strategy
  - peak selection strategy
  - dataset noise and overlap between classes

This is expected behavior and provides a realistic comparison between "engineered features" vs "raw spectral representation".

---

## 7) Reproducibility

For consistent results across machines:
- Use Python 3.11+
- Keep the same dependency versions (optionally pin versions in `requirements.txt`)
- Set a fixed random seed in the script (if not already set)

---

## 8) Troubleshooting

### 8.1) `pip install` fails
Upgrade pip and retry:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.2) Script runs but no outputs appear
Make sure you are running from the repository root:

```bash
python run_pipeline.py
```

If the output directory is created elsewhere, check the script's output path configuration.

### 8.3) `ModuleNotFoundError`
You likely did not activate the virtual environment:

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

---

Entendi. Entao toma uma versao **MAIS COMPLETA**, mas ainda **profissional** (sem virar texto infinito).
Substitui sua secao 9 inteira por esta aqui:

---

## 9) How to read the plots (detailed interpretation)

This section explains what each exported plot represents, why it exists in the pipeline, and what conclusions can be extracted from it.
All plots shown below are committed under `images/` so the repository can be inspected directly on GitHub without running the code.

---

### 9.1) Confusion Matrix - Features (RandomForest)

This confusion matrix reports the classification performance of the **feature-based model**, trained on engineered (physics-inspired) features extracted from the synthetic piezoelectric signals.

**What this model is using as input:**
Instead of learning directly from the raw waveform or spectrum, the model receives a compact set of engineered descriptors (summary statistics and peak-related quantities). This approach is typical in applied signal processing when interpretability and robustness are desired.

**How to interpret the matrix:**

* The **main diagonal** corresponds to correct classifications (true class = predicted class).
* **Off-diagonal cells** represent misclassifications, showing which classes the model confuses.
* A strong diagonal indicates the features capture discriminative structure between classes.

**What a strong result means here:**
On a synthetic dataset where classes were generated with controlled differences, engineered features tend to separate classes well. This makes RandomForest a strong baseline for a local-first, fast pipeline.

**Typical causes of residual confusion (if present):**

* Classes with similar spectral peak patterns (close resonance behavior)
* Overlapping parameter ranges in the synthetic generator
* Noise level and damping effects that reduce separation

**Takeaway:**
If the diagonal is dominant, the pipeline is correctly extracting stable features and the feature-based classifier is a reliable baseline for the current dataset configuration.

---

### 9.2) Confusion Matrix - FFT (HistGradientBoosting)

This confusion matrix reports the performance of the **FFT-based baseline**, trained on a spectral representation of the signal (instead of engineered features).

**Why this baseline exists:**
It provides a comparison between:

* **Feature engineering** (compact, interpretable, robust)
  vs
* **Raw spectral learning** (higher dimensional, potentially less stable, more sensitive)

**How to interpret the matrix:**

* A weaker diagonal compared to the feature-based model is expected in many setups.
* Confusions are informative: they reveal which classes share similar spectral signatures.

**Why FFT-only classification is harder:**

* FFT vectors can be high-dimensional and sensitive to:

  * windowing strategy
  * normalization choices
  * frequency resolution
  * peak alignment and small shifts in resonance
* Two classes may differ physically, but still generate similar spectral energy distributions under certain noise/damping conditions.

**What a “good” FFT baseline looks like:**

* Reasonable diagonal dominance
* Confusion concentrated only between physically similar classes
* No collapse into predicting a single dominant class

**Takeaway:**
This baseline helps validate whether the spectral representation alone contains enough information to separate classes, and it quantifies the performance gap between engineered features and raw spectral learning.

---

### 9.3) Synthetic dataset composition

This plot shows the distribution of samples per class in the synthetic dataset.

**Why it matters:**

* Class imbalance can artificially inflate or deflate metrics.
* A balanced dataset ensures:

  * fair training
  * fair confusion matrices
  * comparable per-class performance

**What to look for:**

* Ideally, classes should be approximately balanced.
* If imbalance exists, interpret confusion matrices with caution, because minority classes tend to show lower recall.

**Takeaway:**
This plot is a quick QA check that the dataset generator is producing the intended distribution.

---

### 9.4) FFT example (signal)

This plot shows an example FFT spectrum for a selected class (e.g., `"agua"`), including detected peaks when enabled by the pipeline.

**What the FFT represents:**

* The FFT converts the time-domain signal into frequency-domain energy distribution.
* Peaks correspond to dominant oscillation components (resonances and harmonics).

**What to look for:**

* **Main peak frequency** (dominant resonance)
* **Secondary peaks / harmonics** (structure beyond the main mode)
* **Noise floor level** (how clean the spectrum is)
* **Peak stability** across samples of the same class

**How this connects to the ML models:**

* The FFT baseline model uses spectral information directly.
* The feature-based model typically uses peak-related summaries derived from this spectrum.

**Takeaway:**
This plot validates that the FFT computation and peak extraction are working as expected and that the spectrum contains meaningful structure.

---

### 9.5) Practical conclusions (what these plots validate)

If all plots look consistent, the pipeline is validated end-to-end:

* dataset generation is producing meaningful multi-class data
* FFT and peak extraction are stable
* feature-based ML is learning separable structure
* FFT-based ML behaves as a realistic baseline
* outputs are exportable and reproducible locally

---

## 10) License

MIT License.
