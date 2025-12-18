
import os

import pandas as pd
import numpy as np
from typing import List, Dict

from src.evaluate_anonLevels import \
    evaluate_anonymization_effects
from src.om_a0_pdm_predict_em_a1_a2 import \
    combine_id_om_without_anon, \
    combine_id_em_without_anon

#Personal



# file: evaluation/system_performance_comparator.py

import pandas as pd
from typing import List, Dict


def compare_system_performance(
        df_sys_a: pd.DataFrame,
        df_sys_b: pd.DataFrame,
        df_sys_c: pd.DataFrame,
        metrics: List[str]
) -> Dict[str, any]:
    systems = {
        "Aalst": df_sys_a,
        "Aau": df_sys_b,
        "Prototype": df_sys_c
    }

    per_metric = {}
    system_scores = {k: [] for k in systems}

    for metric in metrics:
        values = {
            system: df[metric].dropna().mean()
            for system, df in systems.items()
        }

        best_system = max(values, key=values.get)
        best_value = values[best_system]

        diffs = {
            system: round(best_value - val, 4)
            for system, val in values.items()
        }

        for system, val in values.items():
            system_scores[system].append(val)

        per_metric[metric] = {
            "scores": {k: round(v, 4) for k, v in values.items()},
            "best": best_system,
            "delta_to_best": diffs
        }

    summary = {
        system: round(sum(vals) / len(vals), 4)
        for system, vals in system_scores.items()
    }

    ranked = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    best_overall = ranked[0][0]
    improvement_vs_others = {
        k: round(summary[best_overall] - v, 4)
        for k, v in summary.items() if k != best_overall
    }

    return {
        "per_metric": per_metric,
        "summary": summary,
        "best_overall": best_overall,
        "improvement_vs_others": improvement_vs_others
    }






def print_system_evaluation(
        om: Dict[str, Dict[str, pd.DataFrame]],
        em: Dict[str, pd.DataFrame],
        output_dir: str = "C:/Dev/BPMNPredictor/results/Personal"
):
    os.makedirs(output_dir, exist_ok=True)

    anon_levels = ["None", "Anon1", "Anon2"]

    metrics_om = [
        "Tasks", "SequenceFlows", "gatewayPositioning_fitness", "gatewayContent_precision", "task_precision",
        "event_precision", "AND_(Join)", "event_fitness", "XOR_(Split)", "flow_fitness", "gatewayPositioning_precision",
        "AND_(Split)", "task_fitness", "flow_precision", "Events", "HasErrors", "gatewayContent_fitness", "XOR_(Join)"
    ]
    metrics_em = ["Grade", "QU1", "QU2", "QU3"]

    group_cols = ['Group_Aalst', 'Group_Aau', 'Group_Prototype']
    system_labels = ['Aalst', 'Aau', 'Prototype']

    for anon_level in anon_levels:
        om_full = combine_id_om_without_anon(om, anon_level)
        em_full = combine_id_em_without_anon(em, anon_level)

        # Extract OM system DataFrames
        om_systems = {
            label: om_full[om_full[group_col] == 1]
            for label, group_col in zip(system_labels, group_cols)
        }

        # Extract EM system DataFrames
        em_systems = {
            label: em_full[em_full[group_col] == 1]
            for label, group_col in zip(system_labels, group_cols)
        }

        # Compare OM
        om_results = compare_system_performance(
            df_sys_a=om_systems["Aalst"],
            df_sys_b=om_systems["Aau"],
            df_sys_c=om_systems["Prototype"],
            metrics=metrics_om
        )

        # Compare EM
        em_results = compare_system_performance(
            df_sys_a=em_systems["Aalst"],
            df_sys_b=em_systems["Aau"],
            df_sys_c=em_systems["Prototype"],
            metrics=metrics_em
        )

        # Save or print the results
        # Save OM
        with open(os.path.join(output_dir, f"System_Comparison_OM_{anon_level}.txt"), "w", encoding="utf-8") as f:
            f.write(format_comparison_plaintext(om_results, anon_level, "OM"))

        # Save EM
        with open(os.path.join(output_dir, f"System_Comparison_EM_{anon_level}.txt"), "w", encoding="utf-8") as f:
            f.write(format_comparison_plaintext(em_results, anon_level, "EM"))





def format_comparison_plaintext(result: Dict, anon_level: str, domain: str) -> str:
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"{domain} Comparison – Anonymization Level: {anon_level}")
    lines.append(f"{'='*60}\n")

    lines.append("Per-Metric Comparison:")
    lines.append(f"{'Metric':<35} {'Aalst':>8} {'Aau':>8} {'Prototype':>8}   Best   ΔAalst     ΔAau     ΔPrototype")
    lines.append("-" * 80)

    for metric, data in result["per_metric"].items():
        scores = data["scores"]
        deltas = data["delta_to_best"]
        best = data["best"]

        lines.append(
            f"{metric:<35} "
            f"{scores['Aalst']:>8.4f} {scores['Aau']:>8.4f} {scores['Prototype']:>8.4f}   "
            f"{best:<5} "
            f"{deltas['Aalst']:>6.4f} {deltas['Aau']:>6.4f} {deltas['Prototype']:>6.4f}"
        )

    lines.append("\nSummary Scores:")
    summary = result["summary"]
    for system in ["Aalst", "Aau", "Prototype"]:
        lines.append(f"  System {system}: {summary[system]:.4f}")

    lines.append(f"\nBest Overall System: {result['best_overall']}")
    lines.append("Improvements over others:")
    for k, v in result["improvement_vs_others"].items():
        lines.append(f"  {result['best_overall']} vs {k}: +{v:.4f}")

    return "\n".join(lines)
