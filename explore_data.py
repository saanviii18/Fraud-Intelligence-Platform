import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Rows, Columns:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)