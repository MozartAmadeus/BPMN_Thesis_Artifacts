import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from .train_dl_model import (
    run_single_feature_set_runs_dl,
    run_multi_feature_set_runs_dl
)

def run_feature_elimination_single(
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        target: str,
        initial_features: List[str],
        n_runs: int = 10
) -> Dict:
    history = []
    features = initial_features.copy()

    while len(features) > 1:
        # Run full model 10x and collect scores and models
        r2_scores, rmse_scores, models = run_single_feature_set_runs_dl(X_raw, y_raw, features, target, n_runs)

        mean_r2 = sum(r2_scores) / len(r2_scores)
        mean_rmse = sum(rmse_scores) / len(rmse_scores)

        # Save best model based on r2
        best_model_index = r2_scores.index(max(r2_scores))
        best_model = models[best_model_index]

        # Compute average absolute weights for each feature
        all_weights = [abs(best_model.layers[0].get_weights()[0]) for model in models]
        summed_weights = sum(all_weights)
        avg_weights = summed_weights / len(all_weights)
        feature_weights = {feat: avg_weights[i].sum() for i, feat in enumerate(features)}

        # Identify worst feature
        worst_feature = min(feature_weights, key=feature_weights.get)

        # Record round
        history.append({
            'features': features.copy(),
            'mean_r2': mean_r2,
            'mean_rmse': mean_rmse,
            'removed_feature': worst_feature,
            'feature_weights': feature_weights
        })

        # Drop the worst feature
        features.remove(worst_feature)

    return history

def run_feature_elimination_multi(
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        target_columns: List[str],
        initial_features: List[str],
        n_runs: int = 10
) -> Dict:
    history = []
    features = initial_features.copy()

    #while len(features) > 1:
    while len(features) >= len(features):
        r2_runs, rmse_runs, r2_all_runs, rmse_all_runs, models = run_multi_feature_set_runs_dl(X_raw, y_raw, features, target_columns, n_runs)
        mean_r2_all = sum(r2_all_runs) / n_runs
        mean_rmse_all = sum(rmse_all_runs) / n_runs

        best_model_index, best_alpha, alpha_log = auto_select_majority_winner_model_index_by_alpha(
            r2_all_runs, rmse_all_runs
        )

        best_model = models[best_model_index]
        best_model_r2 = r2_all_runs[best_model_index]
        best_model_rmse = rmse_all_runs[best_model_index]
        best_model_r2_scores = r2_runs[best_model_index]
        best_model_rmse_scores = rmse_runs[best_model_index]

        all_weights = [abs(model.layers[0].get_weights()[0]) for model in models]
        summed_weights = sum(all_weights)
        avg_weights = summed_weights / len(all_weights)
        feature_weights = {feat: avg_weights[i].sum() for i, feat in enumerate(features)}

        worst_feature = min(feature_weights, key=feature_weights.get)

        history.append({
            'features': features.copy(),
            'mean_r2_all': mean_r2_all,
            'mean_rmse_all': mean_rmse_all,
            'removed_feature': worst_feature,
            'feature_weights': feature_weights,
            'best_model': best_model,
            'best_model_r2': best_model_r2,
            'best_model_rmse': best_model_rmse,
            'best_model_r2_scores': best_model_r2_scores,
            'best_model_rmse_scores': best_model_rmse_scores,
            'best_alpha_used': best_alpha,
        })

        features.remove(worst_feature)

    return history


def auto_select_majority_winner_model_index_by_alpha(
        r2_all_runs: List[float],
        rmse_all_runs: List[float],
        alpha_list: List[float] = np.linspace(0.05).tolist()
) -> Tuple[int, float, Dict[float, Dict[str, float]]]:
    score_log = {}
    win_counts = {}  # model_index -> win count
    best_overall_score = float("-inf")
    best_overall_model_index = None
    best_overall_alpha = None

    for alpha in alpha_list:
        scores = [r2 - alpha * rmse for r2, rmse in zip(r2_all_runs, rmse_all_runs)]
        max_score = max(scores)
        max_index = scores.index(max_score)

        score_log[alpha] = {
            'score': max_score,
            'r2': r2_all_runs[max_index],
            'rmse': rmse_all_runs[max_index],
            'index': max_index
        }

        # Count wins
        win_counts[max_index] = win_counts.get(max_index, 0) + 1

        # Track overall best score
        if max_score > best_overall_score:
            best_overall_score = max_score
            best_overall_model_index = max_index
            best_overall_alpha = alpha

    # Determine model with most wins
    max_wins = max(win_counts.values())
    tied_models = [model for model, count in win_counts.items() if count == max_wins]

    if len(tied_models) == 1:
        winner_index = tied_models[0]
    else:
        # Break tie by choosing the one with highest overall score
        winner_index = best_overall_model_index

    return winner_index, best_overall_alpha, score_log
