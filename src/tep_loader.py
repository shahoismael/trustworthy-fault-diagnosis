"""Tennessee Eastman (Rieth extended) loader.

Memory-safe: reads the big CSVs in chunks and keeps only the simulation runs
you ask for, so it runs on a laptop even though the faulty files are GBs.

Returns sliding-window samples ready for CNN/LSTM baselines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import TEP_FILES, TEP_FEATURES, TEP_ONSET, WINDOW, STRIDE


def _read_runs(path, max_runs, faults=None, chunksize=200_000):
    """Stream a TEP CSV, keep rows with simulationRun <= max_runs (and given faults)."""
    keep = []
    usecols = ["faultNumber", "simulationRun", "sample"] + TEP_FEATURES
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        m = chunk["simulationRun"] <= max_runs
        if faults is not None:
            m &= chunk["faultNumber"].isin(faults)
        if m.any():
            keep.append(chunk.loc[m])
    if not keep:
        return pd.DataFrame(columns=usecols)
    return pd.concat(keep, ignore_index=True)


def _window_run(arr, label, window, stride):
    """Slide a window over one (fault,run) sequence -> list of (window, label)."""
    out = []
    n = arr.shape[0]
    for s in range(0, n - window + 1, stride):
        out.append(arr[s:s + window])
    if not out:
        return np.empty((0, window, arr.shape[1])), np.empty((0,), int)
    X = np.stack(out)
    y = np.full(len(out), label, dtype=int)
    return X, y


def load_tep(kind="training", max_runs=20, faults=None, window=WINDOW, stride=STRIDE):
    """Build windowed TEP samples.

    kind    : 'training' or 'testing'
    max_runs: how many simulationRuns to use per fault (controls dataset size)
    faults  : iterable of fault ids to include (None = 0..20)
    Returns X (N, window, 52), y (N,), groups (N,) run id for leakage-free splits.
    """
    onset = TEP_ONSET[kind]
    ff = _read_runs(TEP_FILES[f"faultfree_{kind}"], max_runs, faults={0} if faults is None or 0 in faults else set())
    fy_faults = None if faults is None else {f for f in faults if f != 0}
    fy = _read_runs(TEP_FILES[f"faulty_{kind}"], max_runs, faults=fy_faults)

    Xs, ys, gs = [], [], []
    # fault-free: label 0, use whole sequence
    for run, g in ff.groupby("simulationRun"):
        g = g.sort_values("sample")
        X, y = _window_run(g[TEP_FEATURES].to_numpy(np.float32), 0, window, stride)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), f"ff_{run}"))
    # faulty: keep post-onset samples, label = faultNumber
    for (fault, run), g in fy.groupby(["faultNumber", "simulationRun"]):
        g = g.sort_values("sample")
        g = g[g["sample"] >= onset]
        X, y = _window_run(g[TEP_FEATURES].to_numpy(np.float32), int(fault), window, stride)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), f"f{fault}_{run}"))

    X = np.concatenate([x for x in Xs if len(x)], axis=0)
    y = np.concatenate([a for a in ys if len(a)], axis=0)
    groups = np.concatenate([a for a in gs if len(a)], axis=0)
    return X, y, groups


def load_tep_ordered(kind="testing", max_runs=50, faults=None, window=WINDOW, stride=STRIDE):
    """Ordered test loader that also returns per-window metadata (fault, run,
    start-sample) so detection delay can be computed. No shuffling."""
    onset = TEP_ONSET[kind]
    ff = _read_runs(TEP_FILES[f"faultfree_{kind}"], max_runs, faults={0})
    fy = _read_runs(TEP_FILES[f"faulty_{kind}"], max_runs,
                    faults=None if faults is None else {f for f in faults if f != 0})
    Xs, ys, meta = [], [], []
    def add(g, fault, run, keep_from=None):
        g = g.sort_values("sample")
        if keep_from is not None:
            g = g[g["sample"] >= keep_from]
        arr = g[TEP_FEATURES].to_numpy(np.float32)
        starts = g["sample"].to_numpy()
        for s in range(0, arr.shape[0] - window + 1, stride):
            Xs.append(arr[s:s + window]); ys.append(fault)
            meta.append((int(fault), int(run), int(starts[s])))
    for run, g in ff.groupby("simulationRun"):
        add(g, 0, run)
    for (fault, run), g in fy.groupby(["faultNumber", "simulationRun"]):
        add(g, int(fault), run, keep_from=onset)
    X = np.stack(Xs); y = np.array(ys, int)
    meta = pd.DataFrame(meta, columns=["fault", "run", "start"])
    return X, y, meta


if __name__ == "__main__":
    X, y, g = load_tep("training", max_runs=2)
    print("X", X.shape, "y", y.shape, "classes", np.unique(y).size)
