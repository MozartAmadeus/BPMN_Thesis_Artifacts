import pandas as pd
from typing import Dict
from src.run_experiment import \
    run_feature_elimination_multi, \
    auto_select_majority_winner_model_index_by_alpha


#runs must be changed to 10
def om_predict_em(om_combined: Dict[str, pd.DataFrame], em_combined: Dict[str, pd.DataFrame], anon_level, n_runs: int = 10) -> Dict:
    """Use OM features to predict EM targets for a given AnonLevel."""

    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    if anon_level not in om_combined and anon_level not in em_combined:
        raise ValueError(f"❌ AnonLevel '{anon_level}' not found in OM or EM combined data")

    om_df = om_combined[anon_level]
    em_df = em_combined[anon_level]

    common_ids = om_df.index.intersection(em_df.index)
    if common_ids.empty:
        raise ValueError("❌ No overlapping combined keys found between OM and EM")

    om_aligned = om_df.loc[common_ids].sort_index()
    em_aligned = em_df.loc[common_ids].sort_index()

    features = [col for col in om_aligned.columns if col not in target_columns and not col.startswith('AnonLevel_') and not col.startswith('Group_') and not col.startswith('process_id')]

    history = run_feature_elimination_multi(
        X_raw=om_aligned[features],
        y_raw=em_aligned[target_columns],
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

#must be changed to 10
def om_all_predict_em_all(om_combined: Dict[str, pd.DataFrame], em_combined: Dict[str, pd.DataFrame], n_runs: int = 10) -> Dict:
    """Use all OM features to predict all EM targets across all AnonLevels combined."""
    # hot one encoded should be included (at least for anonlevel as this is important for predictiong
    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    om_df_all = pd.concat(om_combined.values(), axis=0, ignore_index=False)
    em_df_all = pd.concat(em_combined.values(), axis=0, ignore_index=False)

    common_ids = om_df_all.index.intersection(em_df_all.index)
    if common_ids.empty:
        raise ValueError("❌ No overlapping combined keys found between OM and EM across all AnonLevels")

    om_aligned = om_df_all.loc[common_ids].sort_index()
    em_aligned = em_df_all.loc[common_ids].sort_index()
#and not col.startswith('Group_')
    features = [col for col in om_aligned.columns if col not in target_columns and not col.startswith('process_id') and not col.startswith('Group_') and not col.startswith('AnonLevel_')]

    X_numeric = om_aligned[features].apply(pd.to_numeric, errors='coerce')
    if X_numeric.isnull().any().any():
        raise ValueError("❌ Non-numeric values found in features after conversion")

    y_numeric = em_aligned[target_columns].apply(pd.to_numeric, errors='coerce')
    if y_numeric.isnull().any().any():
        raise ValueError("❌ Non-numeric values found in target columns after conversion")

    history = run_feature_elimination_multi(
        X_raw=X_numeric,
        y_raw=y_numeric,
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






def print_om_predict_em(om: Dict[str, Dict[str, pd.DataFrame]], em_combined: Dict[str, pd.DataFrame], save_path: str = "C:/Dev/BPMNPredictor/results/RQ2/om_predict_em.txt"):
    from data_loader import create_combined_key_om, create_combined_key_em_combined

    om_combined = create_combined_key_om(om)
    em_combined_keyed = create_combined_key_em_combined(em_combined)

    for anon_level in ['None', 'Anon1', 'Anon2']:
        print(f"🔍 Running OM → EM prediction for AnonLevel: {anon_level}")
        try:
            result = om_predict_em(om_combined, em_combined_keyed, anon_level=anon_level)

            best_feats = set(result['best_model_features'])

            removed_history_filtered = [
                (feat, val) for feat, val in result['removed_features_history'] if feat not in best_feats
            ]

            path = save_path.replace("om_predict_em.txt", f"om_predict_em_{anon_level}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"# AnonLevel: {anon_level}\n")

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

            print(f"✅ Results saved to {path}")

        except Exception as e:
            print(f"❌ Failed for AnonLevel {anon_level}: {e}")

    try:
        all_path = save_path.replace("om_predict_em.txt", "om_all_predict_em_all.txt")
        print("🔍 Running OM → EM prediction across ALL AnonLevels")
        all_result = om_all_predict_em_all(om_combined, em_combined_keyed)

        best_feats = set(all_result['best_model_features'])

        removed_history_filtered_all = [
            (feat, val) for feat, val in all_result['removed_features_history'] if feat not in best_feats
        ]
        with open(all_path, 'w', encoding='utf-8') as f:
            f.write("# OM → EM prediction across ALL AnonLevels\n")

            f.write("\n🌟 Best Model Features:\n")
            for feat in all_result['best_model_features']:
                f.write(f"  - {feat}\n")

            f.write("\n📉 Removed Features History:\n")
            for feat, weight in removed_history_filtered_all:
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📊 Final Feature Weights:\n")
            for feat, weight in all_result['final_feature_weights'].items():
                f.write(f"  - {feat}: {weight:.4f}\n")

            f.write("\n📊 R² per Target:\n")
            for target, r2 in all_result['r2_per_target'].items():
                f.write(f"  - {target}: {r2:.4f}\n")

            f.write("\n📉 RMSE per Target:\n")
            for target, rmse in all_result['rmse_per_target'].items():
                f.write(f"  - {target}: {rmse:.4f}\n")

            f.write(f"\n🏆 Final R²: {all_result['final_r2']:.4f}\n")
            f.write(f"📉 Final RMSE: {all_result['final_rmse']:.4f}\n")

        print(f"✅ Results for ALL AnonLevels saved to {all_path}")
    except Exception as e:
        print(f"❌ Failed all-level OM → EM prediction: {e}")

