import pandas as pd
import sqlite3
import xgboost as xgb
from pygam import LinearGAM, s, f

conn = sqlite3.connect("training.db")
query = "SELECT * FROM team_game_stats"

df = pd.read_sql_query(query, conn)

conn.close

print(df.head())

def build_features(df, reference_columns=None):
    
    df = df.copy()

    x = df[TOV_FEATURES].copy()

    if reference_columns is not None:
        x = x.reindex(columns=reference_columns, fill_value=0)

    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    keep = x.notna().all(axis=1)
    x = x.loc[keep].copy()
    kept_index = df.index[keep.to_numpy()]

    return x, kept_index
  
team_stats = training.grouby(team_id).agg(
  TOV=("turnovers", "mean"),
  TOVF=("turnover_forced", "mean")
)

TEAM_AVG_TOV = team_stats["TOV"]
TEAM_AVG_TOVF = team_stats["TOVF"]

TOV_FEATURES = ["TEAM_AVG_TOV", "TEAM_AVG_TOVF"]

BASE_XGB_PARAMS = dict(
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
X_train, training_kept_idx = build_features(train_data)
Y_train = train_data.loc[train_kept_idx, "outcome"].astype(int)
