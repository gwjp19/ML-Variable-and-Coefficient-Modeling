import pandas as pd
import sqlite3
import xgboost as xgb
from pygam import LinearGAM, s, f
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

conn = sqlite3.connect("training2.db")
query = "SELECT * FROM team_game_stats"

training = pd.read_sql_query(query, conn)

conn.close

print(training.head())

training = training.sort_values(["team_id", "game_date"])

training["TEAM_AVG_TOV"] = (
    training.groupby("team_id")["turnovers"]
    .transform(lambda s: s.shift(1).expanding().mean())
)
training["TEAM_AVG_TOVF"] = (
    training.groupby("team_id")["turnovers_forced"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

TOV_FEATURES = ["TEAM_AVG_TOV", "TEAM_AVG_TOVF"]

BASE_XGB_PARAMS = dict(
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="mae",
    random_state=42,
    n_jobs=-1,
)

x = training[TOV_FEATURES].copy()
keep = x.notna().all(axis=1)
x = x.loc[keep].copy()
kept_index = training.index[keep.to_numpy()]
y = training.loc[kept_index, "turnovers"]

def find_best_n_estimators(X, y, max_estimators=2000, **overrides):
    split_idx = int(len(training) * 0.75)

    X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    params = dict(
        BASE_XGB_PARAMS, 
        n_estimators=max_estimators, 
        early_stopping_rounds=50
    )
    params.update(overrides)
    
    probe_model = xgb.XGBRegressor(**params)
    probe_model.fit(
        X_tr, y_tr 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    return probe_model.best_iteration + 1


def build_calibration_ready_model(X, y, **overrides):
    best_n = find_best_n_estimators(X, y, **overrides)
    params = dict(BASE_XGB_PARAMS, n_estimators=best_n)
    params.update(overrides)
    return xgb.XGBRegressor(**params), best_n

def report(y_true, probs, label):
    preds = (probs >= 0.5).astype(int)
    print(f"\n{label}")
    print("-" * len(label))
    print("Accuracy:", accuracy_score(y_true, preds))
  
X_train, training_kept_idx = build_features(train_data)
Y_train = train_data.loc[train_kept_idx, "turnovers"].astype(int)

BASE_COLUMNS = X_train.columns  

base_model, base_n_estimators = build_calibration_ready_model(X_train, Y_train)
print("Base model n_estimators (from early stopping):", base_n_estimators)


base_model.fit(X_train, Y_train)

train_data.loc[train_kept_idx, "expected_tov"] = base_model.predict(X_train)
