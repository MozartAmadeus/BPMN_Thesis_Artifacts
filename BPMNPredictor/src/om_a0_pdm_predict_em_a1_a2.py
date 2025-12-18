import os
from typing import Dict, List
import pandas as pd
from src.run_experiment import \
    run_feature_elimination_multi, \
    auto_select_majority_winner_model_index_by_alpha


#RQ4
#Cross-Anonymization Prediction
#Can expert ratings on anonymized inputs be predicted using metrics from the original, non-anonymized input/output pair?
#-Predict Expert Ratings (A1, A2) from OM (A0) + PDM



def om_a0_pdm_predict_em_a1_a2(om_df: pd.DataFrame, em_df: pd.DataFrame, n_runs: int = 10) -> Dict:
    """Use OM from AnonLevel 'None' to predict EM from AnonLevel 'Anon1' or 'Anon2'.

    Both dataframes must have index = process_id_Group (no AnonLevel)
    """
    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    common_keys = om_df.index.intersection(em_df.index)
    if common_keys.empty:
        raise ValueError("❌ No overlapping process_id_Group keys between OM and EM")

    om_aligned = om_df.loc[common_keys].sort_index()
    em_aligned = em_df.loc[common_keys].sort_index()

    features = [
        col for col in om_aligned.columns
        if col not in target_columns and not col.startswith('AnonLevel_') and not col.startswith('process_id') and not col.startswith('Group_')
    ]

    X = om_aligned[features].apply(pd.to_numeric, errors='coerce')
    y = em_aligned[target_columns].apply(pd.to_numeric, errors='coerce')

    if X.isnull().any().any():
        raise ValueError("❌ Non-numeric values found in OM features")
    if y.isnull().any().any():
        raise ValueError("❌ Non-numeric values found in EM targets")

    history = run_feature_elimination_multi(
        X_raw=X,
        y_raw=y,
        target_columns=target_columns,
        initial_features=features,
        n_runs=n_runs
    )

    r2_list = [entry['best_model_r2'] for entry in history]
    rmse_list = [entry['best_model_rmse'] for entry in history]

    best_idx, best_alpha, _ = auto_select_majority_winner_model_index_by_alpha(r2_list, rmse_list)
    best_iteration = history[best_idx]
    removed_history = []
    for i in range(0, len(history)):
        prev_weights = history[i]['feature_weights']
        removed_feature = history[i]['removed_feature']
        weight_at_removal = prev_weights.get(removed_feature, None)
        removed_history.append((removed_feature, weight_at_removal))

    return {
        'best_model_features': best_iteration['features'],
        'removed_features_history': removed_history,
        'final_feature_weights': best_iteration['feature_weights'],
        'final_r2': best_iteration['mean_r2_all'],
        'final_rmse': best_iteration['mean_rmse_all'],
        'r2_per_target': best_iteration['best_model_r2_scores'],
        'rmse_per_target': best_iteration['best_model_rmse_scores']
    }

def combine_id_em_without_anon(em_combined: Dict[str, pd.DataFrame], anon_level: str) -> pd.DataFrame:
    df = em_combined[anon_level].copy()
    df['combined_id'] = df.apply(lambda row: f"{row['process_id']}_{row[[c for c in df.columns if c.startswith('Group_')]].idxmax().split('_')[1]}", axis=1)
    return df.set_index('combined_id')

def combine_id_om_without_anon(om_combined: Dict[str, Dict[str, pd.DataFrame]], anon_level: str) -> pd.DataFrame:
    from collections import defaultdict

    subgroup_map = {'A0': 'None', 'A1': 'Anon1', 'A2': 'Anon2'}
    merged: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for llm in om_combined:
        for subgroup, df in om_combined[llm].items():
            level_label = subgroup_map.get(subgroup, subgroup)
            merged[level_label].append(df)

    if anon_level not in merged:
        raise ValueError(f"❌ No OM data found for AnonLevel '{anon_level}'")

    frames = []
    for df in merged[anon_level]:
        df = df.copy()
        df['combined_id'] = df.apply(lambda row: f"{row['process_id']}_{row[[c for c in df.columns if c.startswith('Group_')]].idxmax().split('_')[1]}", axis=1)
        frames.append(df)

    return pd.concat(frames).set_index('combined_id')


def combine_pdm_om_features(om_df: pd.DataFrame, pdm_df: pd.DataFrame) -> pd.DataFrame:
    def extract_process_id(combined_id: str) -> str:
        for suffix in ['_Aalst', '_Aau', '_Prototype']:
            if combined_id.endswith(suffix):
                return combined_id[: -len(suffix)]
        raise ValueError(f"❌ Unexpected format for OM combined_id: '{combined_id}'")

    expanded_pdm_rows = []
    for combined_id in om_df.index:
        process_id = extract_process_id(combined_id)
        if process_id in pdm_df.index:
            pdm_row = pdm_df.loc[process_id]
            expanded_pdm_rows.append(pdm_row)
        else:
            raise ValueError(f"❌ PDM process_id '{process_id}' not found for OM combined_id '{combined_id}'")

    pdm_expanded_df = pd.DataFrame(expanded_pdm_rows, index=om_df.index)
    combined_df = pd.concat([om_df, pdm_expanded_df], axis=1)
    return combined_df


def print_om_a0_pdm_predict_em_a1_a2(om: Dict[str, Dict[str, pd.DataFrame]], em_combined: Dict[str, pd.DataFrame], pdm_df: pd.DataFrame, save_base_path: str = "C:/Dev/BPMNPredictor/results/RQ4"):
    pdm_df = pdm_df.set_index("process_id")
    #, "Anon2"
    for anon_level in ["Anon1"]:
        '''
        print(f"🔍 Running OM:A0 → EM:{anon_level} prediction")
        try:
            om_df = combine_id_om_without_anon(om, anon_level="None")
            em_df = combine_id_em_without_anon(em_combined, anon_level=anon_level)

            result_om = om_a0_pdm_predict_em_a1_a2(
                om_df, em_df)

            best_feats = set(result_om['best_model_features'])

            removed_history_filtered = [
                (feat, val) for feat, val in result_om['removed_features_history'] if feat not in best_feats
            ]

            output_lines = [
                f"\n📦 EM AnonLevel: {anon_level}\n",
                "🌟 Best Model Features:",
                *[f"  - {feat}" for feat in result_om['best_model_features']],
                "\n📉 Removed Features History:",
                *[f"  - {feat}: {val:.4f}" for feat, val in removed_history_filtered],
                "\n📊 Final Feature Weights:",
                *[f"  - {feat}: {val:.4f}" for feat, val in result_om['final_feature_weights'].items()],
                "\n📊 R² per Target:",
                *[f"  - {target}: {score:.4f}" for target, score in result_om['r2_per_target'].items()],
                "\n📉 RMSE per Target:",
                *[f"  - {target}: {score:.4f}" for target, score in result_om['rmse_per_target'].items()],
                f"\n🏆 Final R²: {result_om['final_r2']:.4f}",
                f"📉 Final RMSE: {result_om['final_rmse']:.4f}\n"
            ]

            output_text = "\n".join(output_lines)
            filename = os.path.join(save_base_path, f"om_a0_predict_em_{anon_level.lower()}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_text)
        except Exception as e:
            print(f"❌ Failed OM:A0 → EM:{anon_level} prediction: {e}")
        '''
        print(f"🔍 Running OM + PDM:A0 → EM:{anon_level} prediction")
        try:
            om_df = combine_id_om_without_anon(om, anon_level="None")
            em_df = combine_id_em_without_anon(em_combined, anon_level=anon_level)

            pdm_om_df = combine_pdm_om_features(om_df, pdm_df)

            result_pdf_om = om_a0_pdm_predict_em_a1_a2(
                pdm_om_df, em_df)

            best_feats = set(result_pdf_om['best_model_features'])

            removed_history_filtered_both = [
                (feat, val) for feat, val in result_pdf_om['removed_features_history'] if feat not in best_feats
            ]
            output_lines = [
                f"\n📦 EM AnonLevel: {anon_level}\n",
                "🌟 Best Model Features:",
                *[f"  - {feat}" for feat in result_pdf_om['best_model_features']],
                "\n📉 Removed Features History:",
                *[f"  - {feat}: {val:.4f}" for feat, val in removed_history_filtered_both],
                "\n📊 Final Feature Weights:",
                *[f"  - {feat}: {val:.4f}" for feat, val in result_pdf_om['final_feature_weights'].items()],
                "\n📊 R² per Target:",
                *[f"  - {target}: {score:.4f}" for target, score in result_pdf_om['r2_per_target'].items()],
                "\n📉 RMSE per Target:",
                *[f"  - {target}: {score:.4f}" for target, score in result_pdf_om['rmse_per_target'].items()],
                f"\n🏆 Final R²: {result_pdf_om['final_r2']:.4f}",
                f"📉 Final RMSE: {result_pdf_om['final_rmse']:.4f}\n"
            ]

            output_text = "\n".join(output_lines)
            filename = os.path.join(save_base_path, f"om_a0_pdm_predict_em_{anon_level.lower()}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_text)
        except Exception as e:
            print(f"❌ Failed OM:A0 → EM:{anon_level} prediction: {e}")