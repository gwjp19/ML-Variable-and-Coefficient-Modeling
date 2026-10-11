import pandas as pd
import sqlite3
import xgboost as xgb
from pygam import LinearGAM, s, f
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

tr_conn = sqlite3.connect("training2.db")
tr_query = "SELECT * FROM team_game_stats"

ts_conn = sqlite3.connect("testing.db")
ts_query = "SELECT * FROM team_game_stats"

training = pd.read_sql_query(tr_query, tr_conn)
testing = pd.read_sql_query(ts_query, ts_conn)

tr_conn.close()
ts_conn.close()

training = training.copy()
testing = testing.copy()

training["_split"] = "train"
testing["_split"] = "test"

combined = pd.concat([training, testing], ignore_index = True)

combined["game_date"] = pd.to_datetime(combined["game_date"])
combined = combined.sort_values(
    ["team_id", "game_date"], kind="stable"
).reset_index(drop=True)

combined["TEAM_AVG_TOV"] = (
    combined.groupby("team_id")["turnovers"]
    .transform(lambda s: s.shift(1).expanding().mean())
)
combined["TEAM_AVG_TOVF"] = (
    combined.groupby("team_id")["turnovers_forced"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

training = (
    combined[combined["_split"] == "train"]
    .drop(columns="_split")
    .reset_index(drop=True)
)
training = (
    combined[combined["_split"] == "test"]
    .drop(columns="_split")
    .reset_index(drop=True)
)
train_data = training
test_data = testing
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
    split_idx = int(len(X) * 0.75)

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

def report(y_true, predictions, label):
    print(f"\n{label}")
    print("-" * len(label))
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", mean_squared_error(y_test, predictions))
  
X_train, training_kept_idx = build_features(train_data)
Y_train = train_data.loc[train_kept_idx, "turnovers"].astype(int)

X_test, test_kept_idx = build_features(test_data)
y_test = test_data/loc[test_kept_idx, "turnovers"]

BASE_COLUMNS = X_train.columns  
X_test = X_test[BASE_COLUMNS]

base_model, base_n_estimators = build_calibration_ready_model(X_train, Y_train)
print("Base model n_estimators (from early stopping):", base_n_estimators)


base_model.fit(X_train, Y_train)

train_data.loc[train_kept_idx, "expected_tov"] = base_model.predict(X_train)

predictions = base_model.predict(X_test)

report(y_test, predictions, "Turnovers Model")

