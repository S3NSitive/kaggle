import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 18
plt.rcParams["patch.edgecolor"] = "k"

pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# print(train.head())

# print(train.info())

print(train.select_dtypes(np.int64).nunique().value_counts().sort_index())
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color="blue",
                                                                             figsize=(12, 6),
                                                                             edgecolor="k",
                                                                             linewidth=2)
plt.xlabel("Number of Unique values")
plt.ylabel("Count")
plt.title("Count of unique values in integer columns")
plt.show()

