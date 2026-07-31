"""PICUP-FDD — master runner. One command runs the whole simulation.

Run:  python src/run_all.py

Executes steps 02 -> 08 in order. Each step is isolated: if one fails, the error
is logged and the pipeline continues, so you still get every result that works.
A full log is written to results/run_log.txt.

Deep/evidential torch variants (step03b, step05b) are optional and run only if
PyTorch is installed; the sklearn pipeline always produces real numbers.
"""
import importlib, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS

STEPS = [
    ("step02_preprocess_tep",     "main"),
    ("step03_baselines",          "main"),
    ("step03b_deep_baselines",    "main"),   # optional (needs torch)
    ("step04_imbalance",          "main"),
    ("step05_uncertainty_openset","main"),
    ("step05b_evidential",        "main"),   # optional (needs torch)
    ("step06_sim2real_pronto",    "main"),
    ("step07_interpretability",   "main"),
    ("step08_aggregate",          "main"),
]

def main():
    log = RESULTS / "run_log.txt"
    with open(log, "w", encoding="utf-8") as lf:
        for mod, fn in STEPS:
            t = time.time()
            banner = f"\n{'='*60}\n>>> {mod}\n{'='*60}"
            print(banner); lf.write(banner + "\n")
            try:
                m = importlib.import_module(mod)
                getattr(m, fn)()
                msg = f"[OK] {mod} in {time.time()-t:.1f}s"
            except ModuleNotFoundError as e:
                msg = f"[SKIP] {mod}: missing dependency ({e})"
            except FileNotFoundError as e:
                msg = f"[SKIP] {mod}: missing input ({e})"
            except Exception:
                msg = f"[FAIL] {mod}:\n{traceback.format_exc()}"
            print(msg); lf.write(msg + "\n"); lf.flush()
    print(f"\nDone. Log -> {log}\nSummary -> {RESULTS/'SUMMARY.md'}")

if __name__ == "__main__":
    main()
