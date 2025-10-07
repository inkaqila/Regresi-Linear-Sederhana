import pandas as pd

df = pd.read_csv("caffeine_intake_tracker.csv")

print("--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Nama Kolom ---")
print(df.columns)