import pandas as pd
import sqlite3
import xgboost as xgb

conn = sqlite.connect("training.db")
query = "SELECT * FROM team_game_stats"

df = pd.read_sql_query(query, conn)

conn.close

print(df.head())
