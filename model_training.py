import pandas as pd
import sqlite3
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

combined = pd.concat(
    [training, testing],
    ignore_index=True
)

training["_split"] = "train"
testing["_split"] = "test"

game_keys = ["season_id", "game_id"]

combined["game_date"] = pd.to_datetime(combined["game_date"])
combined = combined.sort_values(
    ["game_date", "team_id"], kind="stable"
).reset_index(drop=True)

combined["possessions"] = (
    combined["points"] /
    combined["ppp"].replace(0, float("nan"))
)

pairings = combined[game_keys + ["team_id"]].merge(
    combined[game_keys + ["team_id"]],
    on=game_keys,
    suffixes=("", "_opp")
)

pairings = pairings[
    pairings["team_id"] != pairings["team_id_opp"]
]

if pairings.duplicated(game_keys + ["team_id"]).any():
    raise ValueError(
        "A team has multiple opponent matches for a game."
    )

pairings = pairings.rename(
    columns={"team_id_opp": "opponent_id"}
)

combined = combined.merge(
    pairings[game_keys + ["team_id", "opponent_id"]],
    on=game_keys + ["team_id"],
    how="left",
    validate="one_to_one"
)
opponent_possessions = combined[
    game_keys + ["team_id", "possessions"]
].rename(columns={
    "team_id": "opponent_id",
    "possessions": "opp_possessions"
})

combined = combined.merge(
    opponent_possessions,
    on=game_keys + ["opponent_id"],
    how="left",
    validate="many_to_one"
)

combined["TOV_RATE"] = (
    combined["turnovers"] / combined["possessions"]
)

combined["TOVF_RATE"] = (
    combined["turnovers_forced"] /
    combined["opp_possessions"]
)

combined = combined.sort_values(
    ["team_id", "game_date"], kind="stable"
).reset_index(drop=True)

combined["TEAM_AVG_POSS"] = (
    combined.groupby("team_id")["possessions"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

combined["TEAM_AVG_TOV_RATE"] = (
    combined.groupby("team_id")["TOV_RATE"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

combined["TEAM_AVG_TOVF_RATE"] = (
    combined.groupby("team_id")["TOVF_RATE"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

opponent_history = combined[
    game_keys + [
        "team_id",
        "TEAM_AVG_TOV_RATE",
        "TEAM_AVG_TOVF_RATE",
        "TEAM_AVG_POSS"
    ]
].rename(columns={
    "team_id": "opponent_id",
    "TEAM_AVG_TOV_RATE": "OPP_AVG_TOV_RATE",
    "TEAM_AVG_TOVF_RATE": "OPP_AVG_TOVF_RATE",
    "TEAM_AVG_POSS": "OPP_AVG_POSS"
})

combined = combined.merge(
    opponent_history,
    on=game_keys + ["opponent_id"],
    how="left",
    validate="many_to_one"
)

training = (
    combined[combined["_split"] == "train"]
    .drop(columns=["_split"])
    .reset_index(drop=True)
)

testing = (
    combined[combined["_split"] == "test"]
    .drop(columns=["_split"])
    .reset_index(drop=True)
)

train_data = training
test_data = testing

TOV_FEATURES = [
    "TEAM_AVG_TOV_RATE",
    "TEAM_AVG_TOVF_RATE",
    "OPP_AVG_TOV_RATE",
    "OPP_AVG_TOVF_RATE",
    "possessions"
]

model = LinearRegression(
    fit_intercept=True,
    copy_X=True,
    n_jobs=-1
)

x = training[TOV_FEATURES].copy()
keep = x.notna().all(axis=1)
x = x.loc[keep].copy()
kept_index = training.index[keep.to_numpy()]
y = training.loc[kept_index, "turnovers"]


def report(y_true, predictions, label):
    print(f"\n{label}")
    print("-" * len(label))
    print("MAE:", mean_absolute_error(y_true, predictions))
    print("RMSE:", mean_squared_error(y_true, predictions) ** 0.5)
  
X_train = train_data[TOV_FEATURES]
Y_train = train_data["turnovers"].astype(int)

X_test = test_data[TOV_FEATURES]
y_test = test_data["turnovers"].astype(int)

BASE_COLUMNS = X_train.columns  
X_test = X_test[BASE_COLUMNS]

valid_rows = X_train.notna().all(axis=1) & Y_train.notna()

X_train = X_train.iloc[valid_rows]
Y_train = Y_train.iloc[valid_rows]


valid_rows2 = X_test.notna().all(axis=1) & y_test.notna()

X_test = X_test.iloc[valid_rows2]
y_test = y_test.iloc[valid_rows2]


polynomial = PolynomialFeatures(degree=3, include_bias=False)
X_poly = polynomial.fit_transform(X_train)
x_poly = polynomial.fit_transform(X_test)
model.fit(X_poly, Y_train)

predictions = model.predict(x_poly)

test_data.loc[X_test.index, "expected_tov"] = predictions

report(y_test, predictions, "Turnovers Model")

baseline_predictions = [Y_train.mean()] * len(y_test)

report(y_test, baseline_predictions, "Base Model")
