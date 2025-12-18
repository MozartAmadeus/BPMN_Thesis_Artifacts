import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List

# ------------------------------------------------------------
# File: src/eval/single_value_prediction.py
# Purpose: Evaluate constant (single-value) baselines against EM data
# Output: Writes a text report to the specified path
# ------------------------------------------------------------

TARGETS: List[str] = ['Grade', 'QU1', 'QU2', 'QU3']
OUTPUT_TXT_PATH = r"C:\Dev\BPMNPredictor\results\Schreiben\SingleValuePrediction.txt"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE for numeric vectors."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² with safe handling when variance is zero (why: sklearn returns NaN)."""
    if y_true.size == 0:
        return float('nan')
    # If variance is zero, r2_score is undefined; return NaN to avoid misleading values
    if np.allclose(np.var(y_true), 0.0):
        return float('nan')
    return float(r2_score(y_true, y_pred))


def _format_line(label: str, value: float) -> str:
    return f"{label}: {value:.6f}"


def _evaluate_constant(y: np.ndarray, c: float) -> Dict[str, float]:
    """Compute RMSE and R² for constant prediction c on vector y."""
    y_pred = np.full_like(y, fill_value=c, dtype=float)
    return {
        'value': float(c),
        'rmse': _rmse(y, y_pred),
        'r2': _r2_safe(y, y_pred),
    }


def evaluate_single_value_baselines(em_combined: Dict[str, pd.DataFrame]) -> str:
    """
    Evaluate two constant baselines per AnonLevel on EM data:
    - Lowest value baseline: predict the minimum of the stacked targets.
    - Best single value baseline: predict the mean of the stacked targets (optimal for RMSE).

    Also compute per-target baselines using each target's own min/mean.

    Returns the full text report content (also written to OUTPUT_TXT_PATH).
    """
    lines: List[str] = []
    lines.append("Single Value Prediction Report")
    lines.append("================================\n")

    for anon_level, df in em_combined.items():
        # Collect stacked vector over the four targets (C1)
        y_all_parts: List[np.ndarray] = []
        for t in TARGETS:
            if t in df.columns:
                col = df[t].dropna().to_numpy(dtype=float)
                if col.size > 0:
                    y_all_parts.append(col)
        if len(y_all_parts) == 0:
            # Nothing to evaluate for this level
            lines.append(f"AnonLevel: {anon_level}")
            lines.append("No target data available.\n")
            continue

        y_all = np.concatenate(y_all_parts, axis=0)

        # Combined constants (C1)
        c_low_all = float(np.min(y_all))
        c_best_all = float(np.mean(y_all))

        res_low_all = _evaluate_constant(y_all, c_low_all)
        res_best_all = _evaluate_constant(y_all, c_best_all)

        lines.append(f"AnonLevel: {anon_level}")
        lines.append("-- Combined (C1: single constant shared across all 4 targets) --")
        lines.append(_format_line("Lowest constant (value)", res_low_all['value']))
        lines.append(_format_line("Lowest constant RMSE", res_low_all['rmse']))
        lines.append(_format_line("Lowest constant R2", res_low_all['r2']))
        lines.append(_format_line("Best constant (value)", res_best_all['value']))
        lines.append(_format_line("Best constant RMSE", res_best_all['rmse']))
        lines.append(_format_line("Best constant R2", res_best_all['r2']))
        lines.append("")

        # Per-target details
        lines.append("Per-Target Details")
        for t in TARGETS:
            if t not in df.columns:
                lines.append(f"  {t}: column missing, skipped.")
                continue
            y = df[t].dropna().to_numpy(dtype=float)
            if y.size == 0:
                lines.append(f"  {t}: no data, skipped.")
                continue

            c_low_t = float(np.min(y))
            c_best_t = float(np.mean(y))
            res_low_t = _evaluate_constant(y, c_low_t)
            res_best_t = _evaluate_constant(y, c_best_t)

            lines.append(f"  Target: {t}")
            lines.append(f"    Lowest constant (value): {res_low_t['value']:.6f}")
            lines.append(f"    Lowest constant RMSE: {res_low_t['rmse']:.6f}")
            lines.append(f"    Lowest constant R2: {res_low_t['r2']:.6f}")
            lines.append(f"    Best constant (value): {res_best_t['value']:.6f}")
            lines.append(f"    Best constant RMSE: {res_best_t['rmse']:.6f}")
            lines.append(f"    Best constant R2: {res_best_t['r2']:.6f}")
        lines.append("")
        lines.append("--------------------------------")
        lines.append("")

    report = "\n".join(lines)

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


# Example usage (keep commented; integrate into your pipeline as needed):
# em = combine_experts(load_em(ROOT_DIR))
# report_text = evaluate_single_value_baselines(em)
# print(report_text)
