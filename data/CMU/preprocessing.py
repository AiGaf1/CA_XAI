import pandas as pd

df = pd.read_csv('DSL-StrongPasswordData.csv')
print(df.columns)
# Get metadata columns
meta_cols = ["subject", "sessionIndex", "rep"]

# Get hold and flight columns
hold_cols = [c for c in df.columns if c.startswith("H.")]
flight_cols = [c for c in df.columns if c.startswith("DD.")]

# Create empty list
rows = []

for _, row in df.iterrows():
    for h_col, f_col in zip(hold_cols, flight_cols):
        key_name = h_col.replace("H.", "")

        rows.append({
            "subject": row["subject"],
            "session": row["sessionIndex"],
            "rep": row["rep"],
            "key": key_name,
            "hold_time": row[h_col],
            "flight_time": row[f_col]
        })

# Final dataset
result = pd.DataFrame(rows)

print(result.head())