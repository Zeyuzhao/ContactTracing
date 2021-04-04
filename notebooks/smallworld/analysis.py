#%% 
import csv
import pandas as pd
from ctrace import PROJECT_ROOT
import seaborn as sns
import numpy as np
run_id = "run_jaCJw"
path = PROJECT_ROOT / "output" / run_id

with open(path / 'grader.csv', "r") as f:
    grader = pd.read_csv(f, skiprows = 1,header = None, index_col=0)

# Process each row to numpy array
grader["grader_data"] = grader.apply(lambda r: tuple(r), axis=1).apply(np.array)
grader = grader[["grader_data"]]

with open(path / 'input.csv', "r") as f:
    input = pd.read_csv(f)
grader.reset_index(inplace=True)
grader = grader.rename(columns = {0:'id'})


df = pd.merge(input, grader, how="left", on=["id", "id"])
#%%
import matplotlib.pyplot as plt

shift=400
n = 5
fig, axs = plt.subplots(n, n, figsize=(20,20))
for i in range(n):
    for j in range(n):
        id = i * n + j + shift
        ax = sns.histplot(data=list(df.iloc[i * n + j + shift]), binwidth=1, ax=axs[i, j])
        ax.set_title(f"[{id}]: ")
# %%
