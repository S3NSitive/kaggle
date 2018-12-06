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

for household in not_equal.index:
    true_target = int(train[(train["idhogar"] == household) & (train["parentesco1"] == 1.0)]["Target"])
    train.loc[train["idhogar"] == household, "Target"] = true_target

all_equal = train.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print(f"There are {len(not_equal)} households where the family members do not all have the same target")

# Missing Variables
missing = pd.DataFrame(data.isnull().sum()).rename(columns={0: "total"})
missing["percent"] = missing["total"] / len(data)
print(missing.sort_values("percent", ascending=False).head(10).drop("Target"))


def plot_value_counts(df, col, heads_only=False):
    if heads_only:
        df = df.loc[df["parentesco1"] == 1].copy()

    plt.figure(figsize=(12, 6))
    df[col].value_counts().sort_index().plot.bar(color="blue", edgecolor="k", linewidth=2)

    plt.xlabel(f"{col}")
    plt.ylabel("Count")
    plt.title(f"{col} Value counts")
    plt.show()


plot_value_counts(data, "v18q1")

print(heads.groupby("v18q")["v18q1"].apply(lambda x: x.isnull().sum()))
data["v18q1"] = data["v18q1"].fillna(0)

own_variables = [x for x in data if x.startswith("tipo")]

data.loc[data["v2a1"].isnull(), own_variables].sum().plot.bar(figsize=(10, 8), color="green",
                                                              edgecolor="k", linewidth=2)
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
           rotation=60)
plt.title("Home Ownership Status for Households Missing Rent Payments", size=18)
plt.show()

data.loc[(data["tipovivi1"] == 1), "v2a1"] = 0
data["v2a1-missing"] = data["v2a1"].isnull()
print(data["v2a1-missing"].value_counts())

print(data.loc[data["rez_esc"].notnull()]["age"].describe())
print(data.loc[data["rez_esc"].isnull()]["age"].describe())

data.loc[((data["age"] > 19) | (data["age"] < 7)) & (data["rez_esc"].isnull()), "rez_esc"] = 0
data["rez_esc-missing"] = data["rez_esc"].isnull()
data.loc[data["rez_esc"] > 5, "rez_esc"] = 5


def plot_categoricals(x, y, data, annotate=True):
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize=False))
    raw_counts = raw_counts.rename(columns={x: "raw_count"})

    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize=True))
    counts = counts.rename(columns={x: "normalize_count"}).reset_index()
    counts["percent"] = 100 * counts["normalize_count"]
    counts["raw_count"] = list(raw_counts["raw_count"])

    plt.figure(figsize=(14, 10))
    plt.scatter(counts[x], counts[y], edgecolors="k", color="lightgreen",
                s=100 * np.sqrt(counts["raw_count"]), marker="o", alpha=0.6, linewidth=1.5)

    if annotate:
        for i, row in counts.iterrows():
            plt.annotate(xy=(row[x] - (1 / counts[x].nunique()),
                             row[y] - (0.15 / counts[y].nunique())),
                         color="navy",
                         s=f"{round(row['percent'], 1)}%")

    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())

    sqr_min = int(np.sqrt(raw_counts["raw_count"].min()))
    sqr_max = int(np.sqrt(raw_counts["raw_count"].max()))

    msizes = list(range(sqr_min, sqr_max, int((sqr_max - sqr_min) / 5)))
    markers = []

    for size in msizes:
        markers.append(plt.scatter([], [], s=100 * size,
                                   label=f"{int(round(np.square(size) / 100) * 100)}",
                                   color="lightgreen",
                                   alpha=0.6, edgecolors="k", linewidths=1.5))

    plt.legend(handles=markers, title="Counts", labelspacing=3, handletextpad=2,
               fontsize=16, loc=(1.10, 0.19))
    plt.annotate(f"* Size represents raw count while % is for a given y value.",
                 xy = (0, 1), xycoords = 'figure points', size = 10)

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()),
              counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()),
              counts[y].max() + (4 / counts[y].nunique())))
    plt.grid(None)
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"{y} vs {x}")
    plt.show()


plot_categoricals("rez_esc", "Target", data)
plot_categoricals("escolari", "Target", data, annotate=False)
plot_value_counts(data[(data["rez_esc-missing"] == 1)], "Target")
plot_value_counts(data[(data["v2a1-missing"] == 1)], "Target")
