
import pandas as pd

df = pd.read_csv("data/datatraining.txt")

print("Kích thước dữ liệu:", df.shape)
print("\nTên các cột:")
print(df.columns.tolist())

print("\n5 dòng đầu:")
print(df.head())

print("\nSố lượng từng nhãn:")
print(df["Occupancy"].value_counts())
