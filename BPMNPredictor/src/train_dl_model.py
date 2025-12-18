import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def build_dl_model(input_shape: int, output_shape: int = 1) -> Sequential:
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_shape)  # Multi-output if needed
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_single_dl_model(X_train: np.ndarray, y_train: np.ndarray, output_shape: int = 1) -> Sequential:
    model = build_dl_model(input_shape=X_train.shape[1], output_shape=output_shape)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
    return model


def evaluate_model_multi(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, target_columns: List[str]) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    y_pred = model.predict(X_test)
    r2_scores = {}
    rmse_scores = {}

    for i, col in enumerate(target_columns):
        r2_scores[col] = r2_score(y_test[:, i], y_pred[:, i])
        rmse_scores[col] = mean_squared_error(y_test[:, i], y_pred[:, i]) ** 0.5

    r2_all = np.mean(list(r2_scores.values()))
    rmse_all = np.sqrt(np.mean((y_test - y_pred) ** 2))
    return r2_scores, rmse_scores, r2_all, rmse_all


def run_multi_feature_set_runs_dl(
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        features: List[str],
        target_columns: List[str],
        n_runs: int = 10
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[float], List[float], List[Sequential]]:
    r2_per_run = []
    rmse_per_run = []
    r2_all_runs = []
    rmse_all_runs = []
    models = []

    for run in range(n_runs):
        # Copy selected features
        X = X_raw[features].copy()
        y = y_raw[target_columns].copy()

       # X_train, X_test, y_train, y_test = train_test_split(
       #     X, y, test_size=0.375, random_state=run
        #)
        # Split by process_id
        X_train, X_test, y_train, y_test = process_based_split(X, y, run)

        # No need to drop process_id — it's just the index
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

        model = train_single_dl_model(X_train, y_train, output_shape=len(target_columns))
        r2_scores, rmse_scores, r2_all, rmse_all = evaluate_model_multi(model, X_test, y_test, target_columns)

        r2_per_run.append(r2_scores)
        rmse_per_run.append(rmse_scores)
        r2_all_runs.append(r2_all)
        rmse_all_runs.append(rmse_all)
        models.append(model)

    return r2_per_run, rmse_per_run, r2_all_runs, rmse_all_runs, models


def process_based_split(
        X: pd.DataFrame,
        y: pd.DataFrame,
        run: int,
        train_ratio: float = 0.625
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split X and y into train/test sets based on unique process_id index.

    Ensures that all rows with the same process_id go into either train or test.
    """
    rng = np.random.RandomState(run)
    process_ids = np.array(X.index.unique())
    rng.shuffle(process_ids)

    split_idx = int(len(process_ids) * train_ratio)
    train_pids = set(process_ids[:split_idx])
    test_pids = set(process_ids[split_idx:])

    train_mask = X.index.isin(train_pids)

    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]

def run_single_feature_set_runs_dl(
        X_raw: pd.DataFrame,
        y_raw: pd.DataFrame,
        features: List[str],
        target: str,
        n_runs: int = 10
) -> Tuple[List[float], List[float], List[Sequential]]:
    r2_scores = []
    rmse_scores = []
    models = []

    for run in range(n_runs):
        X = X_raw[features].copy().values
        y = y_raw[target].copy().values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.375, random_state=run
        )

        model = train_single_dl_model(X_train, y_train)
        r2, rmse = r2_score(y_test, model.predict(X_test).flatten()), mean_squared_error(y_test, model.predict(X_test).flatten()) ** 0.5

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        models.append(model)

    return r2_scores, rmse_scores, models
