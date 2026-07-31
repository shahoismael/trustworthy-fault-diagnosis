"""Central paths and constants for PICUP-FDD.
Import this everywhere so scripts stay portable across machines.
"""
from pathlib import Path

# Project root = parent of this src/ folder
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- Dataset raw locations ---
TEP_DIR       = DATA / "tep_extended" / "raw"
PRONTO_DIR    = DATA / "pronto_multiphase" / "raw"
INDPENSIM_DIR = DATA / "indpensim" / "raw"
SOFTSENSOR_DIR= DATA / "softsensor_deb_sru" / "raw"
STEEL_DIR     = DATA / "steel_plates" / "raw"

# --- Tennessee Eastman (Rieth extended) ---
TEP_FILES = {
    "faultfree_training": TEP_DIR / "TEP_FaultFree_Training.csv",
    "faultfree_testing":  TEP_DIR / "TEP_FaultFree_Testing.csv",
    "faulty_training":    TEP_DIR / "TEP_Faulty_Training.csv",
    "faulty_testing":     TEP_DIR / "TEP_Faulty_Testing.csv",
}
TEP_FEATURES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]  # 52
TEP_ID_COLS  = ["faultNumber", "simulationRun", "sample"]
# Fault onset sample index (Rieth dataset convention)
TEP_ONSET = {"training": 20, "testing": 160}

# --- Windowing defaults (used by baselines) ---
WINDOW = 50     # time steps per sample
STRIDE = 25     # hop between windows
SEED = 42

# --- Experiment protocol constants ---
RARE_FAULTS = [3, 9, 15]        # notoriously hard incipient TEP faults (imbalance focus)
UNKNOWN_FAULTS = [16, 17, 18]   # held out entirely during training -> open-set 'unknown'

# --- PRONTO (real plant) aligned/labelled process data ---
PRONTO_ALIGNED = PRONTO_DIR / "PRONTO benchmark case study" / "Pre-processed data" / "Aligned and labelled alarm and process data"
# long merged files carry a 'Fault' label + continuous process variables
PRONTO_MERGED_GLOB = "Testday*_merged.csv"
PRONTO_LABEL = "Fault"
PRONTO_PROC_VARS = ["AirIn","Air.T","Air.P","WaterIn","Water.T","Water.Density",
                    "Mixture.zone.P","riser.outlet.P","P.topsep","FR.topsep.gas",
                    "FR.topsep.liquid","P_3phase","Air.Valve","Water.level",
                    "Water.coalescer","Water.level.valve","water.tank.level"]
