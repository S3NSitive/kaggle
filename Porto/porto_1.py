"""
갓 세차를 구매한 운전자의 들뜬 마음에 찬물을 끼얹는 것은 다름 아닌 보험 청구서 내역이다.
당신이 안전한 운전자일수록, 보험 청구서로 인한 불편함은 더 크게 다가올 뿐이다. 도로에서 몇 년 동안 안전운전을 해온 당신이 그토록
많은 금액을 지불해야 한다는 것은 공평해보이지 않는다.
브라질에서 가장 큰 자동차 및 주택 보험회사인 포르토 세구로는 전적으로 동의한다. 자동차 보험 회사의 부정확한 보험 청구 예측 ㅁ델은
좋은 운전자에게 과다한 금액을 청구하고 나쁜 운전자에게 약소한 금액을 청구한다.
이번 경진대회에서, 여러분은 운전자가 내년에 자동차 보험 청구를 진행할 확률을 예측하는 모델을 개발하게 된다.
포르토 세구로 사는 지난 20년 난 기계 학습을 꾸준히 사용해 왔지만, 캐글 머신러닝 커뮤니티에서 새롭고, 더욱 강력한
기법이 발견되길 기대하고 있다. 보다 정확한 예측 모델은 운전자에게 합리적인 가격을 제공하고, 더 많은 운전자들이 자동차 보험의 혜택을
받을 수 있게 도와줄 것이다.
"""
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# print(train.shape)
# print(train.columns)
# print(train.describe())
# print(train.info())
# print(train.isnull().sum())

data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'

    # Defining the data type
    dtype = train[f].dtype

    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif dtype == 'int64':
        level = 'ordinal'
    elif dtype == 'float64':
        level = 'interval'

    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False

    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
# print(meta)

# print(meta[(meta.level == 'nominal') & (meta.keep)].index)
# print(pd.DataFrame({'count': meta.groupby(['role', 'level'])['role'].size()}).reset_index())

v = meta[(meta.level == 'interval') & meta.keep].index
# print(train[v].describe())

v = meta[(meta.level == 'ordinal') & meta.keep].index
# print(train[v].describe())

# Handling imbalanced classes
desired_apriori = 0.10

idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

undersampling_rate = ((1 - desired_apriori) * nb_1) / (nb_0 * desired_apriori)
undersampled_nb_0 = int(undersampling_rate * nb_0)
# print(f"Rate to undersample records with target=0: {undersampling_rate}")
# print(f"Number of records with target=0 after undersampling: {undersampled_nb_0}\n")

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

idx_list = list(undersampled_idx) + list(idx_1)
train = train.loc[idx_list].reset_index(drop=True)

# Data Quality Checks
vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings / train.shape[0]

        # print(f"Variable {f} has {missings} records ({missings_perc:.2%}) with missing values")

# print(f"In total, there are {len(vars_with_missing)} variables with missing values\n")

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[vars_to_drop, 'keep'] = False

mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mean_imp.fit_transform(train[['ps_car_11']]).ravel()

v = meta[(meta.level == 'nominal') & meta.keep].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    # print(f"Variable {f} has {dist_values} distinct values")


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)

    averages = temp.groupby(by=trn_series.name)[target.name].agg(['mean', 'count'])
    smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
    prior = target.mean()

    averages[target.name] = prior * (1 - smoothing) + averages['mean'] * smoothing
    averages.drop(['mean', 'count'], axis=1, inplace=True)

    ft_trn_series = pd.merge(trn_series.to_frame(trn_series.name),
                             averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                             on=trn_series.name,
                             how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_trn_series.index = trn_series.index

    ft_tst_series = pd.merge(tst_series.to_frame(tst_series.name),
                             averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
                             on=tst_series.name,
                             how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


train_encoded, test_encoded = target_encode(train['ps_car_11_cat'],
                                            test['ps_car_11_cat'],
                                            target=train.target,
                                            min_samples_leaf=100,
                                            smoothing=10,
                                            noise_level=0.01)

train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat', 'keep'] = False
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)

v = meta[(meta.level == 'nominal') & meta.keep].index

for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))
    cat_perc = train[[f, 'target']].groupby([f], as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.show()


def corr_heatmap(v):
    correlations = train[v].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f', square=True,
                linewidths=.5, annot=True, cbar_kws={'shrink': .75})
    plt.show()


v = meta[(meta.level == 'interval') & meta.keep].index
# corr_heatmap(v)

s = train.sample(frac=0.1)

# Feature engineering
v = meta[(meta.level == 'nominal') & meta.keep].index
train = pd.get_dummies(train, columns=v, drop_first=True)

v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)
train = pd.concat([train, interactions], axis=1)

# Feature selection
# Removing features with low or zero variance
selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1))

f = np.vectorize(lambda x: not x)

v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
# print('{} variables have too low variance.'.format(len(v)))
# print('These variables are {}'.format(list(v)))

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=4)
rf.fit(X_train, y_train)
importance = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importance[indices[f]]))

sfm = SelectFromModel(rf, threshold='median', prefit=True)
print(f"Number of features before selection: {X_train.shape[1]}")
n_features = sfm.transform(X_train).shape[1]
print(f"Number of features after selection: {n_features}")
selected_vars = list(feat_labels[sfm.get_support()])

train = train[selected_vars + ['target']]

scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))

print(train)
