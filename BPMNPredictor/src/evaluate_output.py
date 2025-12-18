#RQ6

# src/evaluate_output.py
import json
import os
from typing import List, Dict
from collections import Counter
import numpy as np

from src.run_experiment import \
    auto_select_majority_winner_model_index_by_alpha


def parse_model_outputs(output_dir: str) -> List[Dict]:
    section_headers = {
        "Best Model Features:": "Best_Model_Features",
        "Removed Features History:": "Removed_Features_History",
        "Final Feature Weights:": "Final_Feature_Weights",
        "R² per Target:": "R2_per_Target",
        "RMSE per Target:": "RMSE_per_Target",
        "Final R²:": "Final_R2",
        "Final RMSE:": "Final_RMSE"
    }


    parsed_outputs = []

    for file_name in os.listdir(output_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(output_dir, file_name)
        with open(file_path, encoding="utf-8") as f:
            current_section = None
            data = {"file": file_name}

            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "AnonLevel:" in line:
                    try:
                        value_str = line.split("AnonLevel:", 1)[-1].strip()
                        data["AnonLevel"] = value_str
                    except Exception as e:
                        print(f"AnonLevel parse failed: {line}")
                    continue

                # Check for Final R² and Final RMSE directly in line
                if "Final R²:" in line:
                    try:
                        value_str = line.split("Final R²:", 1)[-1].strip()
                        data["Final_R2"] = float(value_str)
                    except ValueError:
                        print(f"Final_R2 parse failed: {line}")
                    continue

                if "Final RMSE:" in line:
                    try:
                        value_str = line.split("Final RMSE:", 1)[-1].strip()
                        data["Final_RMSE"] = float(value_str)
                    except ValueError:
                        print(f"Final_RMSE parse failed: {line}")
                    continue

                matched_section = False
                for header, key in section_headers.items():
                    if header in line:
                        current_section = key
                        matched_section = True

                        if key == "Best_Model_Features":
                            data[key] = []
                        elif key in [
                            "Removed_Features_History",
                            "Final_Feature_Weights",
                            "R2_per_Target",
                            "RMSE_per_Target"
                        ]:
                            data[key] = {}
                        break

                if matched_section:
                    continue

                if current_section is None:
                    continue

                if current_section == "Best_Model_Features":
                    if line.startswith("-"):
                        data[current_section].append(line.split("-", 1)[-1].strip())

                elif current_section in [
                    "Removed_Features_History",
                    "Final_Feature_Weights",
                    "R2_per_Target",
                    "RMSE_per_Target"
                ]:
                    if ":" in line:
                        key_part, value_part = line.split(":", 1)
                        try:
                            data[current_section][key_part.strip()] = float(value_part.strip())
                        except ValueError:
                            continue

        parsed_outputs.append(data)

    return parsed_outputs


def parse_multiple_rq_outputs(base_dir: str) -> List[Dict]:
    all_outputs = []
    for rq in ["RQ1", "RQ2", "RQ4", "RQ5"]: #RQ4 ist vlt klumpat weil es ja nicht direkt vorhersagt. i dont know
        rq_path = os.path.join(base_dir, rq)
        if os.path.exists(rq_path):
            outputs = parse_model_outputs(rq_path)
            all_outputs.extend(outputs)
        else:
            print(f"❌ Directory not found: {rq_path}")
    return all_outputs


def analyze_model_outputs(models: List[Dict]) -> Dict:
    r2s = [m["Final_R2"] for m in models if "Final_R2" in m]
    rmses = [m["Final_RMSE"] for m in models if "Final_RMSE" in m]
    best_index, best_alpha, _ = auto_select_majority_winner_model_index_by_alpha(r2s, rmses)
    best_model = models[best_index]

    all_best_features = [feat for m in models for feat in m.get("Best_Model_Features", [])]
    all_removed_features = [feat for m in models for feat in m.get("Removed_Features_History", {}).keys()]

    # Collect best R2 and RMSE models per target
    r2_per_target_best = {}
    rmse_per_target_best = {}

    for m in models:
        for target, val in m.get("R2_per_Target", {}).items():
            if target not in r2_per_target_best or val > r2_per_target_best[target]["score"]:
                r2_per_target_best[target] = {"score": val, "model": m}

        for target, val in m.get("RMSE_per_Target", {}).items():
            if target not in rmse_per_target_best or val < rmse_per_target_best[target]["score"]:
                rmse_per_target_best[target] = {"score": val, "model": m}

    return {
        "best_model": best_model,
        "best_index": best_index,
        "best_alpha": best_alpha,
        "avg_r2": np.mean(r2s),
        "std_r2": np.std(r2s),
        "avg_rmse": np.mean(rmses),
        "std_rmse": np.std(rmses),
        "top_best_features": Counter(all_best_features).most_common(15),
        "top_removed_features": Counter(all_removed_features).most_common(15),
        "r2_per_target_best": r2_per_target_best,
        "rmse_per_target_best": rmse_per_target_best
    }


def generate_and_save_report(base_dir: str):
    results = parse_multiple_rq_outputs(base_dir)
    report = analyze_model_outputs(results)

    os.makedirs(os.path.join(base_dir, "RQ6"), exist_ok=True)
    report_path = os.path.join(base_dir, "RQ6", "analysis_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("📊 Model Output Analysis Report\n")
        f.write("="*40 + "\n\n")

        f.write(f"Best Model File: {report['best_model']['file']}\n")
        f.write(f"AnonLevel: {report['best_model'].get('AnonLevel', 'N/A')}\n")
        f.write(f"Final R²: {report['best_model'].get('Final_R2', 'N/A')}\n")
        f.write(f"Final RMSE: {report['best_model'].get('Final_RMSE', 'N/A')}\n")
        f.write(f"Alpha Used: {report['best_alpha']:.5f}\n\n")

        f.write(f"Average R²: {report['avg_r2']:.4f}\n")
        f.write(f"Std Dev R²: {report['std_r2']:.4f}\n")
        f.write(f"Average RMSE: {report['avg_rmse']:.4f}\n")
        f.write(f"Std Dev RMSE: {report['std_rmse']:.4f}\n\n")

        f.write("Top Best Model Features:\n")
        for feat, count in report['top_best_features']:
            f.write(f"  - {feat}: {count}\n")
        f.write("\n")

        f.write("Top Removed Features:\n")
        for feat, count in report['top_removed_features']:
            f.write(f"  - {feat}: {count}\n")

        f.write("Best R² per Target:\n")
        for target, entry in report['r2_per_target_best'].items():
            model = entry['model']
            score = entry['score']
            f.write(f"  {target}: {score:.4f} (File: {model['file']})\n")
        f.write("\n")

        f.write("Lowest RMSE per Target:\n")
        for target, entry in report['rmse_per_target_best'].items():
            model = entry['model']
            score = entry['score']
            f.write(f"  {target}: {score:.4f} (File: {model['file']})\n")

    print(f"✅ Report saved to: {report_path}")

