import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 18
plt.rcParams["patch.edgecolor"] = "k"

pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# print(train.head())

# print(train.info())

# Integer Columns
# print(train.select_dtypes(np.int64).nunique().value_counts().sort_index())
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color="blue",
                                                                             figsize=(12, 6),
                                                                             edgecolor="k",
                                                                             linewidth=2)
plt.xlabel("Number of Unique values")
plt.ylabel("Count")
plt.title("Count of unique values in integer columns")
plt.show()

# Float Columns
plt.figure(figsize=(20, 16))
plt.style.use("fivethirtyeight")

# Color mapping
colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
poverty_mapping = OrderedDict({1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulnerable"})

print(train.select_dtypes(np.float))
# Iterate through the float columns
for i, col in enumerate(train.select_dtypes("float")):
    ax = plt.subplot(4, 2, i+1)
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train["Target"] == poverty_level, col].dropna(), ax=ax, color=color,
                    label=poverty_mapping[poverty_level])

    plt.title(f"{col.capitalize()} Distribution")
    plt.xlabel(f"{col}")
    plt.ylabel("Density")

plt.subplots_adjust(top=2)
plt.show()

# Object Columns
print(train.select_dtypes("object").head())
