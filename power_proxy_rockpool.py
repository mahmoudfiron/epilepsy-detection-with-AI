#!/usr/bin/env python3
"""
Activity-based "power proxy" for an externally-trained SNN using Rockpool.

Supports:
  (A) Single-vector probing
  (B) Dataset evaluation on a CSV:
      - Run sample-by-sample (vector-by-vector)
      - Collect spikes/activity per sample
      - Report dataset averages (mean/std/min/max)
      - Compute accuracy if labels are present
      - Compute activity-based manual power estimate (µW range)
      - Print per-neuron firing rate (24 neurons = 3 LIF layers × 8 neurons)

Important notes:
- Rockpool "native.lif" spike exposure varies by version; this script normalizes.
- Accurate Xylo µW/mW requires SynSense calibrated power model or hardware.
  The µW numbers here use explicit per-event energy assumptions (clearly labeled).
"""

import argparse
import sys
import numpy as np
import pickle
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# -------------------------
# Quant artifact loading
# -------------------------
def dequantize(q: np.ndarray, s: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) - z.astype(np.float32)) * s.astype(np.float32)

def load_artifacts(weights_path: str, scaler_path: str):
    pack = np.load(weights_path, allow_pickle=True)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    Ws, bs = [], []
    for i in range(4):
        kq = pack[f"k{i}_q"]
        bq = pack[f"b{i}_q"]

        sk = pack[f"s_k{i}"]
        zk = pack[f"z_k{i}"]
        sb = pack[f"s_b{i}"]
        zb = pack[f"z_b{i}"]

        W = dequantize(kq, sk, zk)
        b = dequantize(bq, sb, zb)

        Ws.append(W)
        bs.append(b)

    return Ws, bs, scaler

# -------------------------
# Input helpers
# -------------------------
def parse_input_vector(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError("Input vector must have exactly 6 comma-separated values.")
    return np.array([float(p) for p in parts], dtype=np.float32)

def make_clocked_input(x6: np.ndarray, steps: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build a clocked input array (T, 6).
    Default: constant value repeated, plus optional temporal Gaussian noise.
    """
    x = np.repeat(x6.reshape(1, 6), repeats=steps, axis=0).astype(np.float32)
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x

# -------------------------
# Rockpool net construction
# -------------------------
def build_rockpool_network(Ws, bs, dt: float, tau_mem: float, tau_syn: float, vth: float):
    """
    Build:
      Linear(6->8) -> LIF(8)
      Linear(8->8) -> LIF(8)
      Linear(8->8) -> LIF(8)
      Linear(8->1)
    """
    try:
        from rockpool.nn.modules import Linear
        from rockpool.nn.modules.native.lif import LIF
    except Exception as e:
        raise ImportError(
            "Rockpool is not available.\n"
            "Install it with: python -m pip install rockpool\n"
            f"Original error: {e}"
        )

    def make_lif(n: int):
        lif = LIF(n, dt=dt)  # minimal init for compatibility

        for name, value in [
            ("tau_mem", tau_mem),
            ("tau_syn", tau_syn),
            ("v_thresh", vth),
            ("spiking_output", True),
        ]:
            if hasattr(lif, name):
                try:
                    setattr(lif, name, value)
                except Exception:
                    pass

        if hasattr(lif, "threshold"):
            try:
                lif.threshold = vth
            except Exception:
                pass

        return lif

    lin0 = Linear((6, 8), weight=Ws[0], bias=bs[0], has_bias=True)
    lif0 = make_lif(8)

    lin1 = Linear((8, 8), weight=Ws[1], bias=bs[1], has_bias=True)
    lif1 = make_lif(8)

    lin2 = Linear((8, 8), weight=Ws[2], bias=bs[2], has_bias=True)
    lif2 = make_lif(8)

    lin3 = Linear((8, 1), weight=Ws[3], bias=bs[3], has_bias=True)

    modules = [lin0, lif0, lin1, lif1, lin2, lif2, lin3]
    lif_indices = [1, 3, 5]
    return modules, lif_indices

# -------------------------
# Robust module calling
# -------------------------
def _safe_call_module(m, current):
    out = m(current)
    if not isinstance(out, (tuple, list)):
        return out, None
    if len(out) == 0:
        raise RuntimeError("Module returned an empty tuple/list.")
    y = out[0]
    state = out[1] if len(out) >= 2 else None
    return y, state

def _get_spikes_from_lif(module, fallback_y: np.ndarray, steps_effective: int) -> np.ndarray:
    """
    Normalize spikes to either:
      - raster: (steps_effective, N) with 0/1
      - counts-like: (1, N) with 0..steps_effective
    """
    spk = getattr(module, "spikes", None)

    if spk is None:
        # fallback: if we don't have spikes, return zeros
        N = fallback_y.shape[-1] if getattr(fallback_y, "ndim", 0) else 1
        return np.zeros((steps_effective, N), dtype=np.float32)

    spk = np.array(spk)

    # Normalize common shapes -> 2D
    if spk.ndim == 3 and spk.shape[0] == 1:
        spk = spk[0]
    if spk.ndim == 3 and spk.shape[-1] == 1:
        spk = spk[:, :, 0]
    if spk.ndim == 1:
        spk = spk.reshape(1, -1)
    if spk.ndim != 2:
        spk = spk.reshape(1, -1) if spk.size else np.zeros((1, 1), dtype=np.float32)

    spk = spk.astype(np.float32)
    spk[spk < 0] = 0

    T_like, N = spk.shape

    # Case A: Raster (T,N) where T==steps
    if T_like == steps_effective:
        # Already binary-ish
        if spk.max() <= 1.0:
            return spk
        # Non-binary but per-timestep values: binarize
        return (spk > 0).astype(np.float32)

    # Case B: Counts-like (1,N) and counts are within [0, steps]
    if T_like == 1 and spk.max() <= steps_effective:
        return np.clip(spk, 0, steps_effective).astype(np.float32)

    # Otherwise: safest fallback (lower bound)
    return (spk > 0).astype(np.float32)

def run_and_collect(modules, lif_indices, x_clocked: np.ndarray, debug: bool = False):
    lif_spike_traces: Dict[int, np.ndarray] = {}
    current = x_clocked

    for idx, m in enumerate(modules):
        y, new_state = _safe_call_module(m, current)

        # Normalize y to (T, N)
        if isinstance(y, np.ndarray) and y.ndim == 3:
            y0 = y[0]
        else:
            y0 = y

        # Optional state update
        if new_state is not None and hasattr(m, "set_attributes"):
            try:
                m = m.set_attributes(new_state)
                modules[idx] = m
            except Exception:
                pass

        if idx in lif_indices:
            spk = _get_spikes_from_lif(modules[idx], np.array(y0), steps_effective=x_clocked.shape[0])
            lif_spike_traces[idx] = spk.copy()
            if debug:
                print(f"[DEBUG] LIF idx {idx}: spk shape={spk.shape}, sum={spk.sum()}, max={spk.max()}")

        current = y0

    final_output = np.array(current)
    return lif_spike_traces, final_output

# -------------------------
# Proxy rows
# -------------------------
@dataclass
class ProxyRow:
    lif_module_index: int
    neurons: int
    timesteps_effective: int
    total_spikes: float
    mean_firing_rate_hz: float
    fanout: float
    synaptic_events: float
    neuron_updates: float

def summarize_proxy(
    lif_spike_traces: Dict[int, np.ndarray],
    dt: float,
    fanouts: Dict[int, int],
    steps_effective: int
) -> List[ProxyRow]:
    """
    IMPORTANT:
    Some Rockpool versions expose module.spikes as (1, N) (counts-like).
    We treat that as spikes over the WHOLE window of 'steps_effective' timesteps.
    """
    rows: List[ProxyRow] = []
    duration_s = steps_effective * dt

    for idx, spk in lif_spike_traces.items():
        spk = np.array(spk, dtype=np.float32)
        if spk.ndim != 2:
            continue

        N = int(spk.shape[1])
        total_spikes = float(spk.sum())
        fr_hz = total_spikes / max(N * duration_s, 1e-12)

        fanout = float(fanouts.get(idx, 1))
        syn_events = total_spikes * fanout
        neuron_updates = float(N * steps_effective)

        rows.append(
            ProxyRow(
                lif_module_index=idx,
                neurons=N,
                timesteps_effective=steps_effective,
                total_spikes=total_spikes,
                mean_firing_rate_hz=fr_hz,
                fanout=fanout,
                synaptic_events=syn_events,
                neuron_updates=neuron_updates,
            )
        )

    return rows

def compute_activity_rates(rows: List[ProxyRow], dt: float, steps_effective: int):
    """
    Returns activity rates:
      spikes/s, syn_events/s, neuron_updates/s
    """
    if not rows:
        return 0.0, 0.0, 0.0

    total_syn_events = sum(r.synaptic_events for r in rows)
    total_neuron_updates = sum(r.neuron_updates for r in rows)
    total_spikes = sum(r.total_spikes for r in rows)

    duration_s = steps_effective * dt

    syn_per_s = total_syn_events / max(duration_s, 1e-12)
    neu_per_s = total_neuron_updates / max(duration_s, 1e-12)
    spk_per_s = total_spikes / max(duration_s, 1e-12)

    return spk_per_s, syn_per_s, neu_per_s

def manual_power_estimate_from_rates(
    syn_events_per_s: float,
    neuron_updates_per_s: float
) -> Dict[str, float]:
    """
    Manual activity-based power using explicit assumptions.

    Assumptions (Joules):
      E_syn in [5,10,20] pJ
      E_neuron in [1,2,5] pJ
    """
    E_syn_min, E_syn_typ, E_syn_max = 5e-12, 10e-12, 20e-12
    E_neu_min, E_neu_typ, E_neu_max = 1e-12, 2e-12, 5e-12

    P_min = syn_events_per_s * E_syn_min + neuron_updates_per_s * E_neu_min
    P_typ = syn_events_per_s * E_syn_typ + neuron_updates_per_s * E_neu_typ
    P_max = syn_events_per_s * E_syn_max + neuron_updates_per_s * E_neu_max

    return {
        "power_uW_min": P_min * 1e6,
        "power_uW_typ": P_typ * 1e6,
        "power_uW_max": P_max * 1e6,
    }

# -------------------------
# CSV dataset loading
# -------------------------
def load_csv_dataset(csv_path: str, label_col: str, input_dim: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Returns:
      X: (N, input_dim)
      y: (N,) or None if label_col not present
      feature_cols: list of used column names
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    y = None
    if label_col in df.columns:
        y = df[label_col].values.astype(np.int64)
        Xdf = df.drop(columns=[label_col])
    else:
        Xdf = df

    if Xdf.shape[1] < input_dim:
        raise ValueError(f"CSV needs at least {input_dim} feature columns, got {Xdf.shape[1]}")

    feature_cols = list(Xdf.columns[:input_dim])
    X = Xdf.iloc[:, :input_dim].values.astype(np.float32)

    return X, y, feature_cols

# -------------------------
# Prediction helper
# -------------------------
def predict_from_output(final_out: np.ndarray, threshold: float, last_k: int) -> int:
    """
    final_out expected shape (T,1) or (T,) or (1,)
    We compute mean over last_k timesteps for stability.
    """
    out = np.array(final_out).reshape(-1)
    if out.size == 0:
        return 0
    k = max(1, min(last_k, out.size))
    p_mean = float(out[-k:].mean())
    return int(p_mean >= threshold)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--weights", required=True)
    ap.add_argument("--scaler", required=True)

    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=5e-4)

    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--vth", type=float, default=0.2)
    ap.add_argument("--tau-mem", type=float, default=0.02)
    ap.add_argument("--tau-syn", type=float, default=0.02)

    # Single-vector mode
    ap.add_argument("--input-vector", type=str, default=None)

    # Dataset mode
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV dataset for vector-by-vector run")
    ap.add_argument("--label-col", type=str, default="label", help="Label column name if present")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for accuracy/pred")
    ap.add_argument("--last-k", type=int, default=200, help="Average last K timesteps for decision")
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, limit dataset to first N samples")
    ap.add_argument("--save-per-sample", type=str, default=None, help="Optional output CSV for per-sample stats")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    Ws, bs, scaler = load_artifacts(args.weights, args.scaler)
    fanouts = {1: 8, 3: 8, 5: 1}
    rng = np.random.default_rng(args.seed)

    # -------------------------
    # MODE 1: dataset CSV
    # -------------------------
    if args.csv is not None:
        X, y, feat_cols = load_csv_dataset(args.csv, args.label_col, input_dim=6)

        if args.max_samples and args.max_samples > 0:
            X = X[:args.max_samples]
            if y is not None:
                y = y[:args.max_samples]

        n = int(X.shape[0])
        duration_s = args.steps * args.dt

        per_sample_spikes: List[float] = []
        per_sample_spk_per_s: List[float] = []
        per_sample_syn_per_s: List[float] = []
        per_sample_neu_per_s: List[float] = []
        preds: List[int] = []

        # Per-layer totals across dataset
        per_layer_spikes_sum: Dict[int, float] = {1: 0.0, 3: 0.0, 5: 0.0}

        # Per-neuron spike totals across dataset (8 neurons per LIF layer)
        per_layer_neuron_spike_sums: Dict[int, np.ndarray] = {
            1: np.zeros(8, dtype=np.float64),
            3: np.zeros(8, dtype=np.float64),
            5: np.zeros(8, dtype=np.float64),
        }

        for i in range(n):
            x6 = X[i].astype(np.float32)
            x6_scaled = scaler.transform(x6.reshape(1, -1)).reshape(-1).astype(np.float32)
            x6_scaled = (args.gain * x6_scaled).astype(np.float32)

            x_clocked = make_clocked_input(x6_scaled, args.steps, args.noise_std, rng)

            modules, lif_indices = build_rockpool_network(
                Ws, bs, dt=args.dt, tau_mem=args.tau_mem, tau_syn=args.tau_syn, vth=args.vth
            )

            lif_traces, final_out = run_and_collect(
                modules, lif_indices, x_clocked, debug=(args.debug and i == 0)
            )
            rows = summarize_proxy(lif_traces, dt=args.dt, fanouts=fanouts, steps_effective=args.steps)

            # ---- accumulate per-neuron counts (MUST be inside the loop) ----
            for lif_idx, spk in lif_traces.items():
                spk = np.array(spk, dtype=np.float64)
                if spk.ndim == 1:
                    spk = spk.reshape(1, -1)

                # raster (T,N): sum over time -> (N,)
                if spk.shape[0] == args.steps:
                    per_neuron_counts = spk.sum(axis=0)
                else:
                    # counts-like (1,N)
                    per_neuron_counts = spk.reshape(-1)

                if per_neuron_counts.shape[0] == 8 and lif_idx in per_layer_neuron_spike_sums:
                    per_layer_neuron_spike_sums[lif_idx] += per_neuron_counts
            # --------------------------------------------------------------

            total_spikes = float(sum(r.total_spikes for r in rows)) if rows else 0.0
            spk_per_s, syn_per_s, neu_per_s = compute_activity_rates(rows, dt=args.dt, steps_effective=args.steps)

            per_sample_spikes.append(total_spikes)
            per_sample_spk_per_s.append(spk_per_s)
            per_sample_syn_per_s.append(syn_per_s)
            per_sample_neu_per_s.append(neu_per_s)

            # per-layer sum
            for r in rows:
                if r.lif_module_index in per_layer_spikes_sum:
                    per_layer_spikes_sum[r.lif_module_index] += r.total_spikes

            pred = predict_from_output(final_out, threshold=args.threshold, last_k=args.last_k)
            preds.append(pred)

        preds_arr = np.array(preds, dtype=np.int64)

        # ---- per-neuron firing rates (MUST be after the loop) ----
        print("\n=== PER-NEURON FIRING RATES (Hz) ===")
        total_duration_s = n * duration_s  # dataset time per neuron
        for lif_idx in [1, 3, 5]:
            counts = per_layer_neuron_spike_sums[lif_idx]
            hz = counts / max(total_duration_s, 1e-12)

            print(f"\nLIF idx {lif_idx} (8 neurons):")
            for j in range(8):
                print(f"  neuron {j:02d}: spikes={counts[j]:.0f}, rate={hz[j]:.3f} Hz")
            print(f"  [summary] min={hz.min():.3f} Hz, mean={hz.mean():.3f} Hz, max={hz.max():.3f} Hz")
        # ----------------------------------------------------------

        # Accuracy / confusion matrix if labels exist
        acc = None
        tn = fp = fn = tp = None
        if y is not None:
            y_arr = y.astype(np.int64)
            acc = float((preds_arr == y_arr).mean())
            tp = int(((preds_arr == 1) & (y_arr == 1)).sum())
            tn = int(((preds_arr == 0) & (y_arr == 0)).sum())
            fp = int(((preds_arr == 1) & (y_arr == 0)).sum())
            fn = int(((preds_arr == 0) & (y_arr == 1)).sum())
        else:
            y_arr = None

        # Aggregate activity
        spikes_mean = float(np.mean(per_sample_spikes))
        spikes_std = float(np.std(per_sample_spikes))
        spikes_min = float(np.min(per_sample_spikes))
        spikes_max = float(np.max(per_sample_spikes))

        spk_s_mean = float(np.mean(per_sample_spk_per_s))
        syn_s_mean = float(np.mean(per_sample_syn_per_s))
        neu_s_mean = float(np.mean(per_sample_neu_per_s))

        # Manual power estimate from averaged rates
        manual = manual_power_estimate_from_rates(
            syn_events_per_s=syn_s_mean,
            neuron_updates_per_s=neu_s_mean,
        )

        print("\n=== DATASET RUN (CSV) ===")
        print(f"CSV: {args.csv}")
        print(f"Features used (first 6): {feat_cols}")
        print(f"Samples: {n}")
        print(
            f"Controls: steps={args.steps}, dt={args.dt}, duration={duration_s:.6f}s | "
            f"gain={args.gain}, noise_std={args.noise_std}, vth={args.vth}, tau_mem={args.tau_mem}, tau_syn={args.tau_syn}"
        )
        print(f"Decision: threshold={args.threshold}, last_k={args.last_k}")

        if acc is not None:
            print("\n=== ACCURACY (on provided labels) ===")
            print(f"Accuracy: {acc*100:.2f}%")
            print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
        else:
            print("\n=== ACCURACY ===")
            print(f"No label column '{args.label_col}' found in CSV -> accuracy not computed.")

        print("\n=== ACTIVITY SUMMARY (per-sample over dataset) ===")
        print(f"Total spikes/sample: mean={spikes_mean:.3f}, std={spikes_std:.3f}, min={spikes_min:.3f}, max={spikes_max:.3f}")
        print("Mean rates (averaged over samples):")
        print(f"  spikes/sec:          {spk_s_mean:.3f}")
        print(f"  syn_events/sec:      {syn_s_mean:.3f}")
        print(f"  neuron_updates/sec:  {neu_s_mean:.3f}")

        print("\n=== MANUAL POWER ESTIMATE (from averaged activity) ===")
        print(f"Estimated Power (µW): min={manual['power_uW_min']:.3f}, typ={manual['power_uW_typ']:.3f}, max={manual['power_uW_max']:.3f}")
        print("NOTE: Uses assumed per-event energies; absolute values require SynSense calibration.")

        print("\n=== PER-LAYER SPIKE TOTALS (summed over dataset) ===")
        for idx in [1, 3, 5]:
            print(f"LIF idx {idx}: total spikes = {per_layer_spikes_sum[idx]:.3f}")

        # Optional save per-sample CSV
        if args.save_per_sample:
            import pandas as pd
            out_df = pd.DataFrame({
                "sample": np.arange(n, dtype=int),
                "total_spikes": per_sample_spikes,
                "spikes_per_s": per_sample_spk_per_s,
                "syn_events_per_s": per_sample_syn_per_s,
                "neuron_updates_per_s": per_sample_neu_per_s,
                "pred": preds_arr.astype(int),
            })
            if y_arr is not None:
                out_df["label"] = y_arr.astype(int)
                out_df["correct"] = (preds_arr == y_arr).astype(int)
            out_df.to_csv(args.save_per_sample, index=False)
            print(f"\n[saved] per-sample stats -> {args.save_per_sample}")

        return

    # -------------------------
    # MODE 2: single-vector (or random single sample)
    # -------------------------
    n_runs = 1

    spike_sums: List[float] = []
    syn_rates: List[float] = []
    neu_rates: List[float] = []
    spk_rates: List[float] = []
    proxy_list: List[float] = []

    for run_i in range(n_runs):
        if args.input_vector is not None:
            x6 = parse_input_vector(args.input_vector)
        else:
            x6 = rng.standard_normal(6).astype(np.float32)

        x6_scaled = scaler.transform(x6.reshape(1, -1)).reshape(-1).astype(np.float32)
        x6_scaled = (args.gain * x6_scaled).astype(np.float32)

        x_clocked = make_clocked_input(x6_scaled, args.steps, args.noise_std, rng)

        modules, lif_indices = build_rockpool_network(
            Ws, bs, dt=args.dt, tau_mem=args.tau_mem, tau_syn=args.tau_syn, vth=args.vth
        )

        lif_traces, final_out = run_and_collect(modules, lif_indices, x_clocked, debug=args.debug)
        rows = summarize_proxy(lif_traces, dt=args.dt, fanouts=fanouts, steps_effective=args.steps)

        total_spk = float(sum(r.total_spikes for r in rows)) if rows else 0.0
        spike_sums.append(total_spk)

        spk_per_s, syn_per_s, neu_per_s = compute_activity_rates(rows, dt=args.dt, steps_effective=args.steps)
        spk_rates.append(spk_per_s)
        syn_rates.append(syn_per_s)
        neu_rates.append(neu_per_s)

        # Simple proxy
        proxy_units_per_s = 1.0 * syn_per_s + 0.1 * neu_per_s + 0.01 * spk_per_s
        proxy_list.append(proxy_units_per_s)

        if args.debug:
            print(f"Run {run_i}: input raw = {x6}")
            print(f"Run {run_i}: input scaled*gain = {x6_scaled}")
            print(f"Final output shape: {np.array(final_out).shape}")

    duration_s = args.steps * args.dt

    print("\n=== Proxy Summary (Whole model) ===")
    print(f"Runs: {n_runs}")
    print(f"Params: steps={args.steps}, dt={args.dt}, duration={duration_s:.6f}s")
    print(f"Probe controls: gain={args.gain}, noise_std={args.noise_std}, vth={args.vth}, tau_mem={args.tau_mem}, tau_syn={args.tau_syn}")
    print(f"Total spikes per run: min={min(spike_sums):.2f}, max={max(spike_sums):.2f}, mean={float(np.mean(spike_sums)):.2f}")

    proxy_mean = float(np.mean(proxy_list))
    proxy_min = float(np.min(proxy_list))
    proxy_max = float(np.max(proxy_list))
    print(f"Power proxy (units/sec): min={proxy_min:.3e}, max={proxy_max:.3e}, mean={proxy_mean:.3e}")

    syn_s_mean = float(np.mean(syn_rates)) if syn_rates else 0.0
    neu_s_mean = float(np.mean(neu_rates)) if neu_rates else 0.0
    manual = manual_power_estimate_from_rates(syn_events_per_s=syn_s_mean, neuron_updates_per_s=neu_s_mean)

    print("\n=== Manual Power Estimate (Activity-Based) ===")
    print(f"Mean spikes/sec:         {float(np.mean(spk_rates)):.3f}")
    print(f"Mean syn_events/sec:     {syn_s_mean:.3f}")
    print(f"Mean neuron_updates/sec: {neu_s_mean:.3f}")
    print(f"Estimated Power (µW): min={manual['power_uW_min']:.3f}, typ={manual['power_uW_typ']:.3f}, max={manual['power_uW_max']:.3f}")
    print("NOTE: Uses assumed per-event energies; absolute values require SynSense calibration.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFULL TRACEBACK:\n" + traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
