import pandas as pd

with open("../output/archive/plots/logging.csv", "r") as csv:
    df = pd.read_csv(csv)
print(df)
random = df.loc[df["method"] == "random"]
degree = df.loc[df["method"] == "degree"]

print(random)
print(degree)