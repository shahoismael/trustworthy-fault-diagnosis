# PICUP-FDD — Setup & Step-by-Step (Windows)

Follow in order. Copy-paste each command. Do it once; then you just run scripts.

---

## 1. Install Python (via Miniconda — recommended)
1. Download Miniconda (Windows 64-bit): https://www.anaconda.com/download/success
2. Install it (default options; tick "Add to PATH" if asked, otherwise use the "Anaconda Prompt").
3. Open **Anaconda Prompt** (Start menu → "Anaconda Prompt").

Verify:
```
conda --version
```

## 2. Create the project environment
```
conda create -n picup python=3.11 -y
conda activate picup
```
You should now see `(picup)` at the start of the prompt.

## 3. Install packages
```
cd C:\Users\Shaho\Desktop\claude_projects\R4\PICUP-FDD
pip install -r requirements.txt
```
This installs numpy, pandas, scikit-learn, matplotlib, PyTorch (CPU), SHAP, etc.

### (Optional) GPU version of PyTorch
Only if you have an NVIDIA GPU. First check:
```
nvidia-smi
```
If it prints a table, install the CUDA build (example for CUDA 12.1):
```
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
CPU is fine for TEP baselines; GPU just makes deep models faster.

## 4. Verify the install
```
python -c "import numpy,pandas,sklearn,torch,matplotlib;print('OK torch',torch.__version__,'cuda',torch.cuda.is_available())"
```
Expect `OK torch ... cuda True/False`. Either True or False is fine.

## 5. Confirm datasets are in place
Datasets already downloaded to `data/*/raw/`. Quick check:
```
dir data\tep_extended\raw
```
You should see the four `TEP_*.csv` files.

---

## 6. Run STEP 01 — explore TEP (first real result)
```
python src\step01_explore_tep.py
```
Expected: prints X shape `(~3790, 50, 52)`, 21 classes, 0 NaNs, and writes:
- `results\tep_class_distribution.csv`
- `results\tep_class_distribution.png`

(Tested — this already runs correctly on your data with `MAX_RUNS=10`. To use more data later, open `src\step01_explore_tep.py` and raise `MAX_RUNS` to 100+.)

---

## What each file does
- `src\config.py` — all paths and constants (edit nothing to start).
- `src\tep_loader.py` — memory-safe TEP loader → sliding windows `(N, 50, 52)`, leakage-free `groups`.
- `src\step01_explore_tep.py` — STEP 01: class balance + first plots.
- `requirements.txt` — dependency list.

## Roadmap (what I build next, in order)
1. **STEP 01 — explore** ✅ (done, runnable)
2. **STEP 02 — preprocess**: train/val/test split by simulation run (no leakage), standardize with train stats, cache to `data/*/processed/` as `.npy`/parquet.
3. **STEP 03 — baselines**: PCA/1D-CNN/LSTM on TEP; save metrics table to `results/`.
4. **STEP 04+ — contributions**: class-imbalance → uncertainty → open-set → sim→real (PRONTO) → interpretability. One script each, one results table each.
5. **Manuscript**: written last, from the real results tables.

## Troubleshooting
- `conda not recognized` → open **Anaconda Prompt**, not plain CMD.
- `pip install torch` slow/fails → run it alone: `pip install torch`.
- Out of memory on TEP → keep `MAX_RUNS` small (10–50); the loader already streams in chunks.
- Wrong folder → all commands assume you are in `C:\Users\Shaho\Desktop\claude_projects\R4\PICUP-FDD`.
