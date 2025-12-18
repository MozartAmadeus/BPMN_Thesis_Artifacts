import os

import pandas as pd
import numpy as np
from typing import List, Dict

#RQ3
#Anonymization Impact
#How does input anonymization affect the performance of the same LLM system, as measured by both expert ratings and automated metrics?
#Compare expert and output metrics across all anonymization levels
from src.om_a0_pdm_predict_em_a1_a2 import \
    combine_id_om_without_anon, \
    combine_id_em_without_anon


def evaluate_anonymization_effects(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        metrics: List[str],
        epsilon: float = 1e-8
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the impact of anonymization on a set of metrics between two dataframes.
    """
    df_before = df_before[metrics].copy()
    df_after = df_after[metrics].copy()

    # Drop rows with NaNs
    df_before.dropna(inplace=True)
    df_after = df_after.loc[df_before.index]  # ensure same rows

    results_per_metric = {}


    for metric in metrics:
        x_before = df_before[metric].astype(float)
        x_after = df_after[metric].astype(float)
        delta = x_after - x_before

        mean_abs = (delta.abs()).mean()
        mean_rel = abs(x_after.mean() - x_before.mean()) / (abs(x_before.mean()) + epsilon)
        msd = delta.mean()

        # Cohen's d
        std_before = x_before.std()
        std_after = x_after.std()
        pooled_std = np.sqrt((std_before**2 + std_after**2) / 2)
        cohen_d = msd / pooled_std if pooled_std > 0 else 0.0

        results_per_metric[metric] = {
            "MeanAbs": round(mean_abs, 4),
            "MeanRel": round(mean_rel, 4),
            "MSD": round(msd, 4),
            "CohenD": round(cohen_d, 4),
        }

    # Summary across all metrics
    summary = {
        stat: round(np.mean([v[stat] for v in results_per_metric.values()]), 4)
        for stat in ["MeanAbs", "MeanRel", "MSD", "CohenD"]
    }

    return {
        "per_metric": results_per_metric,
        "summary": summary

    }

'''
def print_anonymization_effects(om: Dict[str, Dict[str, pd.DataFrame]], em: Dict[str, pd.DataFrame], output_dir: str = "C:/Dev/BPMNPredictor/results/RQ3"):
    os.makedirs(output_dir, exist_ok=True)

    om_keyed = combine_id_om_without_anon(om)
    em_keyed = combine_id_em_without_anon(em)

    anon_pairs = [
        ("None", "Anon1"),
        ("None", "Anon2"),
        ("Anon1", "Anon2")
    ]

    for label, dataset in [("om", om_keyed), ("em", em_keyed)]:
        for anon_from, anon_to in anon_pairs:
            try:
                result = evaluate_anonymization_effects(dataset, anon_from, anon_to)
            except Exception as e:
                print(f"❌ Failed to evaluate {label.upper()} {anon_from} → {anon_to}: {e}")
                continue

            file_name = f"{label}_anonymization_{anon_from}_to_{anon_to}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"📦 Dataset: {label.upper()}\n")
                f.write(f"🔁 Comparison: {anon_from} → {anon_to}\n\n")

                f.write("📊 Per-Metric Changes:\n")
                for metric, values in result['per_metric'].items():
                    f.write(f"- {metric}:\n")
                    f.write(f"    • MeanAbs: {values['MeanAbs']:.4f}\n")
                    f.write(f"    • MeanRel: {values['MeanRel']:.4f}\n")
                    f.write(f"    • MSD: {values['MSD']:.4f}\n")
                    f.write(f"    • CohenD: {values['CohenD']:.4f}\n")

                f.write("\n📈 Summary Statistics Across Metrics:\n")
                for summary, value in result['summary'].items():
                    f.write(f"- {summary}: {value:.4f}\n")

            print(f"✅ Saved: {file_path}")

'''


def print_anonymization_effects(om: Dict[str, Dict[str, pd.DataFrame]], em: Dict[str, pd.DataFrame], output_dir: str = "C:/Dev/BPMNPredictor/results/RQ3"):
    """
    Evaluate and print anonymization effects between anonymization levels for both OM and EM data.
    Writes results to structured text files per comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    comparisons = [
        ("None", "Anon1"),
        ("None", "Anon2"),
        ("Anon1", "Anon2")
    ]
    #"gatewayPositioning_fitness","gatewayContent_precision","task_precision",
    #"event_precision","event_fitness","flow_fitness","gatewayPositioning_precision",
    #"task_fitness","flow_precision","gatewayContent_fitness"
    #"HasErrors",
    #"Events","Tasks", "SequenceFlows","XOR_(Join)","XOR_(Split)","AND_(Split)","AND_(Join)"
    metrics_om = [
        "gatewayPositioning_fitness","gatewayContent_precision","task_precision",
        "event_precision","event_fitness","flow_fitness","gatewayPositioning_precision",
        "task_fitness","flow_precision","gatewayContent_fitness"
    ]

    metrics_em = ["Grade", "QU1", "QU2", "QU3"]

    for base_level, compare_level in comparisons:
        # Prepare OM and EM
        om_base_full = combine_id_om_without_anon(om, base_level)
        om_compare_full = combine_id_om_without_anon(om, compare_level)
        em_base_full = combine_id_em_without_anon(em, base_level)
        em_compare_full = combine_id_em_without_anon(em, compare_level)

        om_result_full = evaluate_anonymization_effects(om_base_full, om_compare_full, metrics_om)
        em_result_full = evaluate_anonymization_effects(em_base_full, em_compare_full, metrics_em)

        output_path = os.path.join(output_dir, f"anonymization_effects_{base_level}_vs_{compare_level}_FULL.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"📦 Anonymization Comparison: {base_level} → {compare_level} | FULL\n\n")

            f.write("🔍 Output Metrics (OM):\n")
            f.write("📊 Per-Metric Changes:\n")
            for metric, values in om_result_full['per_metric'].items():
                f.write(f"- {metric}:\n")
                f.write(f"    • MeanAbs: {values['MeanAbs']:.4f}\n")
                f.write(f"    • MeanRel: {values['MeanRel']:.4f}\n")
                f.write(f"    • MSD: {values['MSD']:.4f}\n")
                f.write(f"    • CohenD: {values['CohenD']:.4f}\n")

            f.write("\n📈 Summary Statistics Across Metrics:\n")
            for summary, value in om_result_full['summary'].items():
                f.write(f"- {summary}: {value:.4f}\n")

            f.write("\n\n🧠 Expert Metrics (EM):\n")
            f.write("📊 Per-Metric Changes:\n")
            for metric, values in em_result_full['per_metric'].items():
                f.write(f"- {metric}:\n")
                f.write(f"    • MeanAbs: {values['MeanAbs']:.4f}\n")
                f.write(f"    • MeanRel: {values['MeanRel']:.4f}\n")
                f.write(f"    • MSD: {values['MSD']:.4f}\n")
                f.write(f"    • CohenD: {values['CohenD']:.4f}\n")

            f.write("\n📈 Summary Statistics Across Metrics:\n")
            for summary, value in em_result_full['summary'].items():
                f.write(f"- {summary}: {value:.4f}\n")


        print(f"✅ Saved: {output_path}")
        # Extract group columns to filter
        group_cols = ['Group_Aalst', 'Group_Aau', 'Group_Prototype']

        for group_col in group_cols:
            group_suffix = group_col.split('_')[-1]
            om_base = om_base_full[om_base_full[group_col] == 1]
            om_compare = om_compare_full[om_compare_full[group_col] == 1]
            em_base = em_base_full[em_base_full[group_col] == 1]
            em_compare = em_compare_full[em_compare_full[group_col] == 1]

            om_result = evaluate_anonymization_effects(om_base, om_compare, metrics_om)
            em_result = evaluate_anonymization_effects(em_base, em_compare, metrics_em)

            output_path = os.path.join(output_dir, f"anonymization_effects_{base_level}_vs_{compare_level}_{group_suffix}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"📦 Anonymization Comparison: {base_level} → {compare_level} | Group: {group_suffix}\n\n")

                f.write("🔍 Output Metrics (OM):\n")
                f.write("📊 Per-Metric Changes:\n")
                for metric, values in om_result['per_metric'].items():
                    f.write(f"- {metric}:\n")
                    f.write(f"    • MeanAbs: {values['MeanAbs']:.4f}\n")
                    f.write(f"    • MeanRel: {values['MeanRel']:.4f}\n")
                    f.write(f"    • MSD: {values['MSD']:.4f}\n")
                    f.write(f"    • CohenD: {values['CohenD']:.4f}\n")

                f.write("\n📈 Summary Statistics Across Metrics:\n")
                for summary, value in om_result['summary'].items():
                    f.write(f"- {summary}: {value:.4f}\n")

                f.write("\n\n🧠 Expert Metrics (EM):\n")
                f.write("📊 Per-Metric Changes:\n")
                for metric, values in em_result['per_metric'].items():
                    f.write(f"- {metric}:\n")
                    f.write(f"    • MeanAbs: {values['MeanAbs']:.4f}\n")
                    f.write(f"    • MeanRel: {values['MeanRel']:.4f}\n")
                    f.write(f"    • MSD: {values['MSD']:.4f}\n")
                    f.write(f"    • CohenD: {values['CohenD']:.4f}\n")

                f.write("\n📈 Summary Statistics Across Metrics:\n")
                for summary, value in em_result['summary'].items():
                    f.write(f"- {summary}: {value:.4f}\n")


            print(f"✅ Saved: {output_path}")