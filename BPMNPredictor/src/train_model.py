import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple

from src.data_loader import merge_pdm_with_om

def train_model(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[RandomForestRegressor, str]:
    """Train a RandomForestRegressor on given features and labels, returning the model and full result string."""
    # Drop non-numeric columns (e.g., process_id, group tags)
    X = X.select_dtypes(include=['number'])
    y = y.select_dtypes(include=['number'])

    # Align rows
    X, y = X.align(y, join="inner", axis=0)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.375, random_state=42)  # 30/18 split

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Top features as string
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)
    top_features_str = "\n".join([f"{k}: {v:.4f}" for k, v in top_features.items()])

    # Full result string
    report = f"Model: RandomForestRegressor\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nTop Features:\n{top_features_str}"

    return model, report


def train_individual_models(pdm, om, em):
    all_reports = []

    for expert in em:
        for llm in em[expert]:
            for subgroup in em[expert][llm]:
                print(f"\n📊 Training for {expert} - {llm} - {subgroup}")

                om_df = om[llm][subgroup]
                X = merge_pdm_with_om(pdm, om_df)
                X['process_id'] = X['process_id'].str.lower()
                X.set_index("process_id", inplace=True)
                X = X.fillna(0)

                y = em[expert][llm][subgroup]
                if 'process_id' in y.columns:
                    y['process_id'] = y['process_id'].str.lower()
                    y.set_index("process_id", inplace=True)
                else:
                    y.index = y.index.str.lower()
                y = y.fillna(1)

                X = X.loc[y.index]

                model, report = train_model(X, y)
                full_report = f"{expert} - {llm} - {subgroup}\n{report}\n{'='*40}"
                print(full_report)
                all_reports.append(full_report)

    return all_reports



