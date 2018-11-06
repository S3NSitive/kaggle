import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
import missingno as msno

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set(font_scale=2.5)

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# Data Check
print(df_train.head())
print(df_train.describe(include='all'))

# Null Data Check
print(pd.isnull(df_train).sum())
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# plt.show()

# Target label Check
f, ax = plt.subplots(1, 2, figsize=(15, 8))
df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
# plt.show()

# Pclass
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())
print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(
    by='Survived', ascending=False).plot.bar()
plt.show()

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()

# Sex
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()

print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print(pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True))

sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
plt.show()

# Age
print(f"Max: {df_train['Age'].max():.1f}")
print(f"Min: {df_train['Age'].min():.1f}")
print(f"Mean: {df_train['Age'].mean():.1f}")

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

f, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()

plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()
