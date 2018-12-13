import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from scipy.stats import spearmanr
from collections import Counter
from collections import OrderedDict

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.style.use("fivethirtyeight")
plt.rcParams["font.size"] = 18
plt.rcParams["patch.edgecolor"] = "k"

pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
"""
print(train.head())

print(train.info())

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
plt.show()

# Object Columns
print(train.select_dtypes("object").head())

mapping = {"yes": 1, "no": 0}

for df in [train, test]:
    df["dependency"] = df["dependency"].replace(mapping).astype(np.float64)
    df["edjefa"] = df["edjefa"].replace(mapping).astype(np.float64)
    df["edjefe"] = df["edjefe"].replace(mapping).astype(np.float64)

print(train[["dependency", "edjefa", "edjefe"]].describe())

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
plt.show()

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
plt.show()

print(label_counts)

# Addressing Wrong Labels
# Identify Errors
all_equal = train.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]

print(f"There are {len(not_equal)} households where the family members do not all have the same target.")
print(train[train["idhogar"] == not_equal.index[0]][["idhogar", "parentesco1", "Target"]])

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

id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3',
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5',
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10',
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3',
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8',
            'instlevel9', 'mobilephone', 'rez_esc-missing']
ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo',
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother',
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1',
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3',
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']
hh_ordered = ['rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2',
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']
hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe',
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']

x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_

print(f"There are no repeats: {np.all(np.array(list(Counter(x).values())) == 1)}")
print(f"We covered every variable: {len(x) == data.shape[1]}")

sns.lmplot("age", "SQBage", data=data, fit_reg=False)
plt.title("Squared Age versus Age")
plt.show()

data = data.drop(columns=sqr_)
print(data.shape)

heads = data.loc[data["parentesco1"] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
print(heads.shape)

corr_matrix = heads.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
print(to_drop)

print(corr_matrix.loc[corr_matrix["tamhog"].abs() > 0.9, corr_matrix["tamhog"].abs() > 0.9])
sns.heatmap(corr_matrix.loc[corr_matrix["tamhog"].abs() > 0.9, corr_matrix["tamhog"].abs() > 0.9],
            annot=True, cmap=plt.cm.autumn_r, fmt=".3f")
plt.show()

heads = heads.drop(columns=["tamhog", "hogar_total", "r4t3"])
sns.lmplot("tamviv", "hhsize", data, fit_reg=False, size=8)
plt.title("Household size vs number of persons living in the household")
plt.show()

heads["hhsize-diff"] = heads["tamviv"] - heads["hhsize"]
plot_categoricals("hhsize-diff", "Target", heads)

corr_matrix.loc[corr_matrix["coopele"].abs() > 0.9, corr_matrix["coopele"].abs() > 0.9]

elec = []
for i, row in heads.iterrows():
    if row["noelec"] == 1:
        elec.append(0)
    elif row["coopele"] == 1:
        elec.append(1)
    elif row["public"] == 1:
        elec.append(0)
    elif row["planpri"] == 1:
        elec.append(0)
    else:
        elec.append(np.nan)

heads["elec"] = elec
heads["elec-missing"] = heads["elec"].isnull()

plot_categoricals("elec", "Target", heads)

heads = heads.drop(columns="area2")
print(heads.groupby("area1")["Target"].value_counts(normalize=True))

heads["walls"] = np.argmax(np.array(heads[["epared1", "epared2", "epared3"]]), axis=1)
plot_categoricals("walls", "Target", heads)

heads["roof"] = np.argmax(np.array(heads[["etecho1", "etecho2", "etecho3"]]), axis=1)
heads = heads.drop(columns=["etecho1", "etecho2", "etecho3"])
heads["floor"] = np.argmax(np.array(heads[["eviv1", "eviv2", "eviv3"]]), axis=1)

# Feature Construction
heads["walls+roof+floor"] = heads["walls"] + heads["roof"] + heads["floor"]
plot_categoricals("walls+roof+floor", "Target", heads, annotate=False)

counts = pd.DataFrame(heads.groupby(["walls+roof+floor"])["Target"].value_counts(normalize=True))\
    .rename(columns={"Target": "Normalized Count"}).reset_index()
print(counts.head())

heads["warning"] = 1 * (heads["sanitario1"] + (heads["elec"] == 0) + heads["pisonotiene"]
                        + heads["abastaguano"] + (heads["cielorazo"] == 0))

plt.figure(figsize=(10, 6))
sns.violinplot(x="warning", y="Target", data=heads)
plt.title("Target vs Warning Variable")
plt.show()

plot_categoricals("warning", "Target", data=heads)

heads["bonus"] = 1 * (heads["refrig"] + heads["computer"] + (heads["v18q1"] > 0) + heads["television"])
sns.violinplot(x="bonus", y="Target", data=heads)
plt.title('Target vs Bonus Variable')
plt.show()

# Per Capita Feature
heads["phone-per-capita"] = heads["qmobilephone"] / heads["tamviv"]
heads["tablets-per-capita"] = heads["v18q1"] / heads["tamviv"]
heads["rooms-per-capita"] = heads["rooms"] / heads["tamviv"]
heads["rent-per-capita"] = heads["v2a1"] / heads["tamviv"]


def plot_corrs(x, y):
    spr = spearmanr(x, y).correlation
    pcr = np.corrcoef(x, y)[0, 1]

    data = pd.DataFrame({"x": x, "y": y})
    plt.figure(figsize=(6, 4))
    sns.regplot("x", "y", data=data, fit_reg=False)
    plt.title(f"Spearman: {round(spr, 2)}; Pearson: {round(pcr, 2)}")
    plt.show()


x = np.array(range(100))
y = x ** 2
plot_corrs(x, y)

x = np.array([1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 9])
y = np.array([1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 3, 3, 2, 4, 2, 2, 4])
plot_corrs(x, y)

x = np.array(range(-19, 20))
y = 2 * np.sin(x)
plot_corrs(x, y)

# Pearson
train_heads = heads.loc[heads["Target"].notnull(), :].copy()

pcorrs = pd.DataFrame(train_heads.corr()["Target"].sort_values()).rename(columns={"Target": "pcorr"}).reset_index()
pcorrs = pcorrs.rename(columns={"index": "feature"})

print("Most negatively correlated variables:")
print(pcorrs.head())
print("\nMost positively correlated varibales:")
print(pcorrs.dropna().tail())

# Spearman
feats = []
scorr = []
pvalues = []

for c in heads:
    if heads[c].dtype != "object":
        feats.append(c)

        scorr.append(spearmanr(train_heads[c], train_heads["Target"]).correlation)
        pvalues.append(spearmanr(train_heads[c], train_heads["Target"]).pvalue)

scorrs = pd.DataFrame({"feature": feats, "scorr": scorr, "pvalue": pvalues}).sort_values("scorr")

print('Most negative Spearman correlations:')
print(scorrs.head())
print('\nMost positive Spearman correlations:')
print(scorrs.dropna().tail())

corrs = pcorrs.merge(scorrs, on="feature")
corrs["diff"] = corrs["pcorr"] - corrs["scorr"]

print(corrs.sort_values("diff").head())
print(corrs.sort_values("diff").dropna().tail())

sns.lmplot(x="dependency", y="Target", fit_reg=True, data=train_heads, x_jitter=0.05, y_jitter=0.05)
plt.title("Target vs Dependency")
plt.show()

sns.lmplot(x="rooms-per-capita", y="Target", fit_reg=True, data=train_heads, x_jitter=0.05, y_jitter=0.05)
plt.title("Target vs rooms-per-capita")
plt.show()

variables = ["Target", "dependency", "warning", "walls+roof+floor", "meaneduc", "floor", "r4m1", "overcrowding"]
corr_mat = train_heads[variables].corr().round(2)

plt.rcParams["font.size"] = 18
plt.figure(figsize=(12, 12))
sns.heatmap(corr_mat, vmin=-0.5, vmax=0.8, center=0, cmap=plt.cm.RdYlGn_r, annot=True)
plt.show()

warnings.filterwarnings("ignore")

plot_data = train_heads[["Target", "dependency", "walls+roof+floor", "meaneduc", "overcrowding"]]

grid = sns.PairGrid(data=plot_data, size=4, diag_sharey=False, hue="Target", hue_order=[4, 3, 2, 1],
                    vars=[x for x in list(plot_data.columns) if x != "Target"])
grid.map_upper(plt.scatter, alpha=0.8, s=20)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)
grid = grid.add_legend()
plt.suptitle('Feature Plots Colored By Target', size=32, y=1.05)
plt.show()

household_feats = list(heads.columns)

# Individual Level Variables
ind = data[id_ + ind_bool + ind_ordered]
print(ind.shape)

corr_matrix = ind.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
print(to_drop)

ind = data.drop(columns="male")

print(ind[[c for c in ind if c.startswith("instl")]].head())

ind["inst"] = np.argmax(np.array(ind[[c for c in ind if c.startswith("instl")]]), axis=1)
plot_categoricals("inst", "Target", ind, annotate=False)

plt.figure(figsize=(10, 8))
sns.violinplot(x="Target", y="inst", data=ind)
plt.title('Education Distribution by Target')
plt.show()

print(ind.shape)

ind["escolari/age"] = ind["escolari"] / ind["age"]

plt.figure(figsize=(10, 8))
sns.violinplot(x="Target", y="escolari/age", data=ind)
plt.show()

ind["inst/age"] = ind["inst"] / ind["age"]
ind["tech"] = ind["v18q"] + ind["mobilephone"]
print(ind["tech"].describe())

range_ = lambda x: x.max() - x.min()
range_.__name__ = "range_"

ind_agg = ind.drop(columns="Target").groupby("idhogar").agg(["min", "max", "sum", "count", "std", range_])
print(ind_agg.head())

new_col = []
for c in ind_agg.columns.levels[0]:
    for stat in ind_agg.columns.levels[1]:
        new_col.append(f"{c}-{stat}")

ind_agg.columns = new_col
print(ind_agg.head())

ind_agg.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].head()

# Create correlation matrix
corr_matrix = ind_agg.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

print(f'There are {len(to_drop)} correlated columns to remove.')

ind_agg = ind_agg.drop(columns=to_drop)
ind_feats = list(ind_agg.columns)

final = heads.merge(ind_agg, on="idhogar", how="left")

print('Final features shape: ', final.shape)

print(final.head())

corrs = final.corr()["Target"]

print(corrs.sort_values().head())
print(corrs.sort_values().dropna().tail())

plot_categoricals('escolari-max', 'Target', final, annotate=False)

plt.figure(figsize = (10, 6))
sns.violinplot(x = 'Target', y = 'escolari-max', data = final)
plt.title('Max Schooling by Target')
plt.show()

plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Target', y = 'escolari-max', data = final)
plt.title('Max Schooling by Target')
plt.show()

plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Target', y = 'meaneduc', data = final)
plt.xticks([0, 1, 2, 3], poverty_mapping.values())
plt.title('Average Schooling by Target')
plt.show()

plt.figure(figsize = (10, 6))
sns.boxplot(x = 'Target', y = 'overcrowding', data = final)
plt.xticks([0, 1, 2, 3], poverty_mapping.values())
plt.title('Overcrowding by Target')
plt.show()

head_gender = ind.loc[ind["parentesco1"] == 1, ["idhogar", "female"]]
final = final.merge(head_gender, on="idhogar", how="left").rename(columns={"female": "female-head"})

print(final.groupby("female-head")["Target"].value_counts(normalize=True))

sns.violinplot(x = 'female-head', y = 'Target', data = final)
plt.title('Target by Female Head of Household')
plt.show()

plt.figure(figsize = (8, 8))
sns.boxplot(x = 'Target', y = 'meaneduc', hue = 'female-head', data = final)
plt.title('Average Education by Target and Female Head of Household', size = 16)
plt.show()

print(final.groupby('female-head')['meaneduc'].agg(['mean', 'count']))

joblib.dump(final, open("data/final.joblib", "wb"))
"""
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

final = joblib.load(open("data/final.joblib", "rb"))

scorer = make_scorer(f1_score, greater_is_better=True, average="macro")

train_labels = np.array(list(final[final["Target"].notnull()]["Target"].astype(np.uint8)))

train_set = final[final["Target"].notnull()].drop(columns=["Id", "idhogar", "Target"])
test_set = final[final["Target"].isnull()].drop(columns=["Id", "idhogar", "Target"])

submission_base = pd.DataFrame({'Id': test["Id"], 'idhogar': test["idhogar"]})

features = list(train_set.columns)
pipeline = Pipeline([("imputer", Imputer(strategy="median")),
                     ("scaler", MinMaxScaler())])

train_set = pipeline.fit_transform(train_set)
test_set = pipeline.fit_transform(test_set)
"""
model = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=-1)
cv_score = cross_val_score(model, train_set, train_labels, cv=10, scoring=scorer)

print(f"10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}")

model.fit(train_set, train_labels)

feature_importances = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
print(feature_importances.sort_values(by="importance", ascending=False).head())
"""

def plot_feature_importances(df, n=10, threshold=None):
    """
    Plots n most important features. Also plots the cumulative importance if
    threshold is specified and prints the number of features needed to reach threshold cumulative importance.
    Intended for use with any tree-based feature importances.

        Args:
            df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

            n (int): Number of most important features to plot. Default is 15.

            threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made.
                               Default is None.

        Returns:
            df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1)
                            and a cumulative importance column

        Note:

            * Normalization in this case means sums to 1.
            * Cumulative importance is calculated by summing features from most to least important
            * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    """
    plt.style.use("fivethirtyeight")

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_normalized"] = df["importance"] / df["importance"].sum()
    df["cumulative_importance"] = np.cumsum(df["importance_normalized"])

    plt.rcParams["font.size"] = 12

    df.loc[:n, :].plot.barh(y="importance_normalized",
                            x="feature",
                            color="darkgreen",
                            edgecolor="k",
                            figsize=(12, 8),
                            legend=False,
                            linewidth=2)

    plt.xlabel("Normalized Importance", size=18)
    plt.ylabel("")
    plt.title(f"{n} Most Important Feature", size=18)
    plt.gca().invert_yaxis()
    plt.show()

    if threshold:
        plt.figure(figsize=(8, 6))
        plt.plot(list(range(len(df))), df["cumulative_importance"], "b-")
        plt.xlabel("Number of Features", size=16)
        plt.ylabel("Cumulative importance", size=18)

        importance_index = np.min(np.where(df["cumulative_importance"] > threshold))

        plt.vlines(importance_index + 1, ymin=0, ymax=1.05, linestyles="--", colors="red")
        plt.show()

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1,
                                                                                  100 * threshold))

    return df


# norm_fi = plot_feature_importances(feature_importances, threshold=0.95)


def kde_target(df, variable):
    colors = {1: "red", 2: "orange", 3: "blue", 4: "green"}

    plt.figure(figsize=(12, 8))

    df = df[df["Target"].notnull()]

    for level in df["Target"].unique():
        subset = df[df["Target"] == level].copy()
        sns.kdeplot(subset[variable].dropna(),
                    label=f"Poverty level: {level}",
                    color=colors[int(subset["Target"].unique())])

    plt.xlabel(variable)
    plt.ylabel("Density")
    plt.title(f"{variable.capitalize()} Distribution")
    plt.show()


# kde_target(final, "meaneduc")
# kde_target(final, 'escolari/age-range_')

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

model_results = pd.DataFrame(columns=["model", "cv_mean", "cv_std"])

"""
def cv_model(train, train_labels, model, name, model_results=None):
    cv_scores = cross_val_score(model, train, train_labels, cv=10, scoring=scorer, n_jobs=1)
    print(f"{name} 10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}")

    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({"model": name,
                                                           "cv_mean": cv_scores.mean(),
                                                           "cv_std": cv_scores.std},
                                                          index=[0]),
                                             ignore_index=True)

        return model_results

model_results = cv_model(train_set, train_labels, LinearSVC(), "LSVC", model_results)
model_results = cv_model(train_set, train_labels, GaussianNB(), "GNB", model_results)
model_results = cv_model(train_set, train_labels, MLPClassifier(hidden_layer_sizes=(32, 64, 128, 64, 32)),
                         "MLP", model_results)
model_results = cv_model(train_set, train_labels, LinearDiscriminantAnalysis(), "LDA", model_results)
model_results = cv_model(train_set, train_labels, RidgeClassifierCV(), "RIDGE", model_results)

for n in [5, 10, 15]:
    model_results = cv_model(train_set, train_labels, KNeighborsClassifier(n_neighbors=n), f"KNN-{n}", model_results)

model_results = cv_model(train_set, train_labels, ExtraTreesClassifier(n_estimators=100, random_state=10),
                         "EXT", model_results)
model_results = cv_model(train_set, train_labels, RandomForestClassifier(n_estimators=100, random_state=10),
                         "RF", model_results)

model_results.set_index("model", inplace=True)
# model_results["cv_mean"].plot.bar(color="orange",
#                                   figsize=(8, 6),
#                                   yerr=list(model_results["cv_std"]),
#                                   edgecolor="k",
#                                   linewidth=2)
# plt.title("Model F1 Score Result")
# plt.ylabel("Mean F1 Score (with error bar)")
# plt.show()

model_results.reset_index(inplace=True)

test_ids = list(final.loc[final['Target'].isnull(), 'idhogar'])


def submit(model, train, train_labels, test, test_ids):
    # Train and test a model on the dataset

    # Train on the data
    model.fit(train, train_labels)
    predictions = model.predict(test)
    predictions = pd.DataFrame({'idhogar': test_ids,
                                'Target': predictions})

    # Make a submission dataframe
    submission = submission_base.merge(predictions,
                                       on='idhogar',
                                       how='left').drop(columns=['idhogar'])

    # Fill in households missing a head
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    return submission

rf = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=-1)
rf_submission = submit(RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=-1),
                       train_set, train_labels, test_set, test_ids)
rf_submission.to_csv('data/rf_submission.csv', index=False)


rf = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs=1)
rf.fit(train_set, train_labels)
predictions = rf.predict(test_set)
predictions = pd.DataFrame({'idhogar': test_ids, 'Target': predictions})

# Make a submission dataframe
submission = submission_base.merge(predictions, on='idhogar', how='left').drop(columns=['idhogar'])

# Fill in households missing a head
submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
submission.to_csv('data/rf_submission.csv', index=False)
"""

# Feature Selection
train_set = pd.DataFrame(train_set, columns=features)

corr_matrix = train_set.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1). astype(np.bool))
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
print(to_drop)

train_set = train_set.drop(columns=to_drop)
print(train_set.shape)

test_set = pd.DataFrame(test_set, columns=features)
train_set, test_set = train_set.align(test_set, axis=1, join="inner")
features = list(train_set.columns)
