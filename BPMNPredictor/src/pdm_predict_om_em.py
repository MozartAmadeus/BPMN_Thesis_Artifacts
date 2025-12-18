import os

import pandas as pd
from src.run_experiment import \
    run_feature_elimination_multi, \
    auto_select_majority_winner_model_index_by_alpha
from typing import Dict

#RQ1 Input Characteristics
#How do process description features influence LLM output quality and expert ratings?
#-Predict Output Metrics from Process Description Metrics (PDM)
#-Predict Expert Ratings from PDM




# Is actually quite stupid as em process_id is not unique. And LLM should not be taken into account. So it just matches randomly. However was a nice try

#Must be changed to 10 runs
def pdm_predict_em(pdm: pd.DataFrame, em_combined: Dict[str, pd.DataFrame], anon_level, n_runs: int = 10) -> Dict:
    """Run feature elimination with PDM predicting EM and return best model details."""

    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    if 'process_id' not in pdm.columns:
        raise ValueError("❌ 'process_id' column not found in PDM data")
    if 'process_id' not in em_combined[anon_level].columns:
        raise ValueError(f"❌ 'process_id' column not found in EM data for anon_level '{anon_level}'")

    common_processes = pdm['process_id'].isin(em_combined[anon_level]['process_id'])
    pdm_filtered = pdm[common_processes].set_index('process_id')
    em_filtered = em_combined[anon_level].set_index('process_id')

    pdm_aligned = pdm_filtered.loc[em_filtered.index]
    features = pdm_aligned.columns.tolist()

    history = run_feature_elimination_multi(
        X_raw=pdm_aligned,
        y_raw=em_filtered[target_columns],
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

def create_averaged_em_combined(em_combined: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    averaged_combined = {}
    for anon_level, df in em_combined.items():
        averaged = df.groupby("process_id", as_index=False).mean(numeric_only=True)
        averaged_combined[anon_level] = averaged
    return averaged_combined

def print_pdm_predict_em(em_combined: Dict[str, pd.DataFrame], pdm: pd.DataFrame, save_dir: str = "C:/Dev/BPMNPredictor/results/RQ1"):
    """Run prediction for all AnonLevels and save results to separate .txt files."""
    for anon_level in ['None', 'Anon1', 'Anon2']:
        print(f"🔍 Running PDM → EM prediction for AnonLevel: {anon_level}")

        result = pdm_predict_em(pdm, em_combined, anon_level=anon_level)

        save_path = os.path.join(save_dir, f"pdm_predict_em_{anon_level}.txt")
        best_feats = set(result['best_model_features'])

        removed_history_filtered = [
            (feat, val) for feat, val in result['removed_features_history'] if feat not in best_feats
        ]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"📦 AnonLevel: {anon_level}\n")

            f.write("\n🌟 Best Model Features:\n")
            for feat in result['best_model_features']:
                f.write(f"  - {feat}\n")

            f.write("\n📉 Removed Features History:\n")
            for feat, weight in removed_history_filtered:
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📊 Final Feature Weights:\n")
            for feat, weight in result['final_feature_weights'].items():
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📈 R² per Target:\n")
            for k, v in result['r2_per_target'].items():
                f.write(f"  - {k}: {v:.4f}\n")

            f.write("\n📉 RMSE per Target:\n")
            for k, v in result['rmse_per_target'].items():
                f.write(f"  - {k}: {v:.4f}\n")

            f.write(f"\n🏆 Final R²: {result['final_r2']:.4f}\n")
            f.write(f"📉 Final RMSE: {result['final_rmse']:.4f}\n")

        em_combined_averaged = create_averaged_em_combined(em_combined)
        result_averaged = pdm_predict_em(pdm, em_combined_averaged, anon_level=anon_level)
        save_path = os.path.join(save_dir, f"pdm_predict_em_averaged_{anon_level}.txt")

        best_feats_averaged = set(result_averaged['best_model_features'])

        removed_history_filtered_averaged = [
            (feat, val) for feat, val in result_averaged['removed_features_history'] if feat not in best_feats_averaged
        ]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"📦 AnonLevel: {anon_level}\n")

            f.write("\n🌟 Best Model Features:\n")
            for feat in result_averaged['best_model_features']:
                f.write(f"  - {feat}\n")

            f.write("\n📉 Removed Features History:\n")
            for feat, weight in removed_history_filtered_averaged:
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📊 Final Feature Weights:\n")
            for feat, weight in result_averaged['final_feature_weights'].items():
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📈 R² per Target:\n")
            for k, v in result_averaged['r2_per_target'].items():
                f.write(f"  - {k}: {v:.4f}\n")

            f.write("\n📉 RMSE per Target:\n")
            for k, v in result_averaged['rmse_per_target'].items():
                f.write(f"  - {k}: {v:.4f}\n")

            f.write(f"\n🏆 Final R²: {result_averaged['final_r2']:.4f}\n")
            f.write(f"📉 Final RMSE: {result_averaged['final_rmse']:.4f}\n")







def pair_pdm_with_em(em_df: pd.DataFrame, pdm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges EM data from all 3 LLMs at anonymization level 0 with corresponding PDM rows.
    Groups EM by process_id (lowercase) and merges with PDM on lowercase process_id.
    """
    # Filter only anonymization level 0 rows from the expert data
    em_filtered = em_df[em_df['AnonLevel'] == 0].copy()

    # Normalize process_id casing to lowercase for consistent merging
    em_filtered['process_id'] = em_filtered['process_id'].str.lower()
    pdm_df['process_id'] = pdm_df['process_id'].str.lower()

    # Group expert data by process_id and average numeric expert metrics
    em_grouped = em_filtered.groupby('process_id').mean(numeric_only=True).reset_index()

    # Merge the averaged expert metrics with corresponding process descriptions
    merged = em_grouped.merge(pdm_df, on='process_id', how='inner')
    return merged


