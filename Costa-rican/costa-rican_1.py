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
# plt.show()

# Float Columns
plt.figure(figsize=(20, 16))
plt.style.use("fivethirtyeight")

# Color mapping
colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
poverty_mapping = OrderedDict({1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulnerable"})

# Iterate through the float columns
for i, col in enumerate(train.select_dtypes("float")):
    ax = plt.subplot(4, 2, i + 1)
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train["Target"] == poverty_level, col].dropna(), ax=ax, color=color,
                    label=poverty_mapping[poverty_level])

    plt.title(f"{col.capitalize()} Distribution")
    plt.xlabel(f"{col}")
    plt.ylabel("Density")

plt.subplots_adjust(top=2)
# plt.show()

# Object Columns
# print(train.select_dtypes("object").head())

mapping = {"yes": 1, "no": 0}

for df in [train, test]:
    df["dependency"] = df["dependency"].replace(mapping).astype(np.float64)
    df["edjefa"] = df["edjefa"].replace(mapping).astype(np.float64)
    df["edjefe"] = df["edjefe"].replace(mapping).astype(np.float64)

# print(train[["dependency", "edjefa", "edjefe"]].describe())

plt.figure(figsize=(16, 12))

# Iterate through the float columns
for i, col in enumerate(["dependency", "edjefa", "edjefe"]):
    ax = plt.subplot(3, 1, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train["Target"] == poverty_level, col].dropna(), ax=ax, color=color,
                    label=poverty_mapping[poverty_level])

    plt.title(f"{col.capitalize()} Distribution")
    plt.xlabel(f"{col}")
    plt.ylabel("Density")

plt.subplots_adjust(top=2)
# plt.show()

test["Target"] = np.nan
data = train.append(test, ignore_index=True)

# Exploring Label Distribution
heads = data.loc[data["parentesco1"] == 1].copy()

train_labels = data.loc[(data["Target"].notnull()) & (data["parentesco1"] == 1), ["Target", "idhogar"]]
label_counts = train_labels["Target"].value_counts().sort_index()

label_counts.plot.bar(figsize=(8, 6), color=colors.values(), edgecolor="k", linewidth=2)

plt.xlabel("Poverty Level")
plt.ylabel("Count")
plt.xticks([x - 1 for x in poverty_mapping.keys()], list(poverty_mapping.values()), rotation=60)
plt.title("Poverty Level Breakdown")
# plt.show()

# print(label_counts)

# Addressing Wrong Labels
# Identify Errors
all_equal = train.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]

print(f"There are {len(not_equal)} households where the family members do not all have the same target.")
# print(train[train["idhogar"] == not_equal.index[0]][["idhogar", "parentesco1", "Target"]])

# Families without Heads of Household
households_leader = train.groupby("idhogar")["parentesco1"].sum()
households_no_head = train.loc[train["idhogar"].isin(households_leader[households_leader == 0].index), :]

print(f"There are {households_no_head['idhogar'].nunique()} households without a head.")

households_no_head_equal = households_no_head.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)

print(f"{sum(households_no_head_equal == False)} Households with no head have different labels.")
