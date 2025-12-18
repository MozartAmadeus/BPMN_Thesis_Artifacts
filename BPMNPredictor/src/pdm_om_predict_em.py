import os

import pandas as pd
from typing import Dict, List

from src.data_loader import \
    create_combined_key_om, \
    create_combined_key_em_combined
from src.om_a0_pdm_predict_em_a1_a2 import \
    combine_id_om_without_anon, \
    combine_id_em_without_anon, \
    combine_pdm_om_features
from src.run_experiment import \
    run_feature_elimination_multi, \
    auto_select_majority_winner_model_index_by_alpha


#runs must be changed to 10
def pdm_om_predict_em(om_df: pd.DataFrame, em_df: pd.DataFrame, n_runs: int = 10) -> Dict:

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

def pdm_om_predict_em_with_anon(om_df: pd.DataFrame, em_df: pd.DataFrame, n_runs: int = 10) -> Dict:

    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    common_keys = om_df.index.intersection(em_df.index)
    if common_keys.empty:
        raise ValueError("❌ No overlapping process_id_Group keys between OM and EM")

    om_aligned = om_df.loc[common_keys].sort_index()
    em_aligned = em_df.loc[common_keys].sort_index()

    features = [
        col for col in om_aligned.columns
        #and not col.startswith('Group_')
        if col not in target_columns  and not col.startswith('process_id') and not col.startswith('Group_') and not col.startswith('AnonLevel_') and not col.endswith('_fitness') and not col.endswith('_precision')
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



def combine_pdm_om_with_anon(pdm_df: pd.DataFrame, om_combined_keyed: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine PDM data with OM keyed data including AnonLevel and Group.
    This returns a combined dataframe with added PDM features for all (AnonLevel, Group) combinations.
    """
    all_rows = []
    all_indices = []

    pdm_df.index = pdm_df.index.astype(str).str.lower()

    # Prefix PDM columns to avoid collision
    pdm_df = pdm_df.add_prefix('pdm_')

    def extract_process_id(combined_id: str) -> str:
        for level in ['_None', '_Anon1', '_Anon2']:
            for group in ['_Aalst', '_Aau', '_Prototype']:
                suffix = f"{level}{group}"
                if combined_id.endswith(suffix):
                    return combined_id[: -len(suffix)]
        raise ValueError(f"❌ Unexpected format for OM combined_id: '{combined_id}'")

    for anon_level, om_df in om_combined_keyed.items():
        for combined_id in om_df.index:
            process_id = extract_process_id(combined_id)

            if process_id not in pdm_df.index:
                raise ValueError(f"❌ PDM process_id '{process_id}' not found for OM combined_id '{combined_id}'")

            pdm_values = pdm_df.loc[process_id]
            om_values = om_df.loc[combined_id]
            combined_row = pd.concat([om_values, pdm_values])
            combined_row = combined_row[~combined_row.index.duplicated(keep='first')]  # Ensure uniqueness
            all_rows.append(combined_row)
            all_indices.append(combined_id)

    pdm_expanded_df = pd.DataFrame(all_rows, index=all_indices)
    return pdm_expanded_df


def print_pdm_om_predict_em(om: Dict[str, Dict[str, pd.DataFrame]], em_combined: Dict[str, pd.DataFrame], pdm_df: pd.DataFrame, save_base_path: str = "C:/Dev/BPMNPredictor/results/RQ5"):
    pdm_df = pdm_df.set_index("process_id")

    for anon_level in ["None", "Anon1", "Anon2"]:
           
        print(f"🔍 Running OM + PDM:{anon_level} → EM:{anon_level} prediction")
        try:
            om_df = combine_id_om_without_anon(om, anon_level=anon_level)
            em_df = combine_id_em_without_anon(em_combined, anon_level=anon_level)

            pdm_om_df = combine_pdm_om_features(om_df, pdm_df)

            result_pdf_om = pdm_om_predict_em(
                pdm_om_df, em_df)

            best_feats = set(result_pdf_om['best_model_features'])

            removed_history_filtered = [
                (feat, val) for feat, val in result_pdf_om['removed_features_history'] if feat not in best_feats
            ]
            output_lines = [
                f"\n📦 EM AnonLevel: {anon_level}\n",
                "🌟 Best Model Features:",
                *[f"  - {feat}" for feat in result_pdf_om['best_model_features']],
                "\n📉 Removed Features History:",
                *[f"  - {feat}: {val:.4f}" for feat, val in removed_history_filtered],
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
            filename = os.path.join(save_base_path, f"pdm_om_{anon_level.lower()}_predict_em_{anon_level.lower()}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_text)
        except Exception as e:
            print(f"❌ Failed OM:A0 → EM:{anon_level} prediction: {e}")

    om_combined_with_anon = create_combined_key_om(om)
    pdm_om_combined_with_anon = combine_pdm_om_with_anon(pdm_df, om_combined_with_anon)
    em_combined_with_anon = create_combined_key_em_combined(em_combined)
    em_combined_with_anon_concat = pd.concat(em_combined_with_anon.values(), axis=0, ignore_index=False)
    '''
    try:
        result_pdf_om_with_anon = pdm_om_predict_em_with_anon(
            pdm_om_combined_with_anon, em_combined_with_anon_concat)

        best_feats = set(result_pdf_om_with_anon['best_model_features'])
        print("Running OM + PDM ALL")
        removed_history_filtered_with_anon = [
            (feat, val) for feat, val in result_pdf_om_with_anon['removed_features_history'] if feat not in best_feats
        ]
        output_lines = [
            f"\n📦 PDM + OM predict EM\n",
            "🌟 Best Model Features:",
            *[f"  - {feat}" for feat in result_pdf_om_with_anon['best_model_features']],
            "\n📉 Removed Features History:",
            *[f"  - {feat}: {val:.4f}" for feat, val in removed_history_filtered_with_anon],
            "\n📊 Final Feature Weights:",
            *[f"  - {feat}: {val:.4f}" for feat, val in result_pdf_om_with_anon['final_feature_weights'].items()],
            "\n📊 R² per Target:",
            *[f"  - {target}: {score:.4f}" for target, score in result_pdf_om_with_anon['r2_per_target'].items()],
            "\n📉 RMSE per Target:",
            *[f"  - {target}: {score:.4f}" for target, score in result_pdf_om_with_anon['rmse_per_target'].items()],
            f"\n🏆 Final R²: {result_pdf_om_with_anon['final_r2']:.4f}",
            f"📉 Final RMSE: {result_pdf_om_with_anon['final_rmse']:.4f}\n"
        ]

        output_text = "\n".join(output_lines)
        filename = os.path.join(save_base_path, f"pdm_om_all_predict_em_all.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(output_text)
    except Exception as e:
        print(f"❌ Failed OM:ALL → EM:ALL prediction: {e}")
    '''