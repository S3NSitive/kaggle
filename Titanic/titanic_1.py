import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings

plt.style.use('seaborn')
sns.set(font_scale=2.5)

warnings.filterwarnings('ignore')

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.head())

print(df_train.describe())
print(df_test.describe())

print("train data")
for col in df_train.columns:
    print(f"column: {col:>11} Percent of Nan value: "
          f"{100 * (df_train[col].isnull().sum() / df_train[col].shape[0]):.2f}")

print("\ntest data")
for col in df_test.columns:
    print(f"column: {col:>11} Percent of Nan value: "
          f"{100 * (df_test[col].isnull().sum() / df_test[col].shape[0]):.2f}")

msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.2, 0.5, 0.2))

# Pclass 분석
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - survived')
# plt.show()

print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())

pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(
    cmap='summer_r')
print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))

print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean())
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(
    by='Survived', ascending=False).plot.bar()
# plt.show()

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
# plt.show()

# Sex 분석
f, ax = plt.subplots(1, 2, figsize=(18, 8))
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).count())
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).sum())
print(pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
# plt.show()

# Sex와 Pclass 상관관계 분석
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=8, aspect=1.5)
# plt.show()

# Age 분석
print(f"제일 나이 많은 탑승객: {df_train['Age'].max():.1f}")
print(f"제일 나이 적은 탑승객: {df_train['Age'].min():.1f}")
print(f"탑승객 평균 나이: {df_train['Age'].mean():.1f}")

df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
# plt.show()

plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
# plt.show()

cummulate_survival_ratio = []
for i in range(1, 80):
    print(df_train[df_train['Age'] < i]['Survived'].sum(), len(df_train[df_train['Age'] < i]['Survived']))
    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() /
                                   len(df_train[df_train['Age'] < i]['Survived']))

plt.figure(figsize=(12, 12))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
# plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot("Pclass", "Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex", "Age", hue="Survived", data=df_train, scale="count", split=True, ax=ax[1])
ax[0].set_title('Sex and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
# plt.show()

# Embarked 분석
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(
    by='Survived', ascending=False).plot.bar()
# plt.show()

f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=df_train, ax=ax[0, 0])
ax[0, 0].set_title('(1) No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0, 1])
ax[0, 1].set_title('(2) Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1, 0])
ax[1, 0].set_title('(3) Embarked vs Survived')
sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1, 1])
ax[1, 1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

# FamilySize 분석
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

f, ax = plt.subplots(1, 3, figsize=(40, 10))
sns.countplot("FamilySize", data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)
sns.countplot("FamilySize", hue="Survived", data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(
    by="Survived", ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

# Fare
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label=f"Skewness : {df_train['Fare'].skew():.2f}", ax=ax)
g = g.legend(loc='best')
plt.show()

