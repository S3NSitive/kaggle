import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing all the required ML packages
from sklearn import svm  # support vector machine
from sklearn import metrics  # accuracy measure
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.metrics import confusion_matrix  # for confusion matrix
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.model_selection import train_test_split  # training and testing data split

# Cross Validation
from sklearn.model_selection import KFold  # for K-fold cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn.model_selection import cross_val_predict  # prediction

# Ensembling
import xgboost as xg  # xgboost
from sklearn.ensemble import VotingClassifier  # voting classifier
from sklearn.ensemble import BaggingClassifier  # bagging
from sklearn.ensemble import AdaBoostClassifier  # AdaBoosting
from sklearn.ensemble import GradientBoostingClassifier  # Stochastic Gradient Boosting

warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
PassengerId = test['PassengerId']

# print(train.shape[0])
# print(test.shape[0])
print(train.columns)
print(train.isnull().sum())

# Sex
sex_mapping = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

# Embarked
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train['Embarked'] = train['Embarked'].fillna('S', inplace=True)
test['Embarked'] = test['Embarked'].fillna('S', inplace=True)
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

# FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Age
train['Initial'] = 0
test['Initial'] = 0

for i in train:
    train['Initial'] = train['Name'].str.extract('([A-Za-z]+)\.')

for i in test:
    test['Initial'] = test['Name'].str.extract('([A-Za-z]+)\.')

train['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',
                          'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
                         ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs',
                          'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)

test['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',
                         'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
                        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs',
                         'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)

dataset = pd.concat([train, test], axis=0)
print(dataset.groupby('Initial').mean())

train.loc[(train['Age'].isnull()) & (train['Initial'] == 'Master'), 'Age'] = 5
train.loc[(train['Age'].isnull()) & (train['Initial'] == 'Miss'), 'Age'] = 22
train.loc[(train['Age'].isnull()) & (train['Initial'] == 'Mr'), 'Age'] = 33
train.loc[(train['Age'].isnull()) & (train['Initial'] == 'Mrs'), 'Age'] = 37
train.loc[(train['Age'].isnull()) & (train['Initial'] == 'Other'), 'Age'] = 45

test.loc[(test['Age'].isnull()) & (test['Initial'] == 'Master'), 'Age'] = 5
test.loc[(test['Age'].isnull()) & (test['Initial'] == 'Miss'), 'Age'] = 22
test.loc[(test['Age'].isnull()) & (test['Initial'] == 'Mr'), 'Age'] = 33
test.loc[(test['Age'].isnull()) & (test['Initial'] == 'Mrs'), 'Age'] = 37
test.loc[(test['Age'].isnull()) & (test['Initial'] == 'Other'), 'Age'] = 45


def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7


train['Age_cat'] = 0
train['Age_cat'] = train['Age_cat'].apply(category_age)

Initial_mapping = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4}
train['Initial'] = train['Initial'].map(Initial_mapping)
test['Initial'] = test['Initial'].map(Initial_mapping)

# Cabin
train['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin']])
test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin']])


# Ticket
Ticket = []
for i in list(train.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])
    else:
        Ticket.append('X')




# Fare
train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)



# One-hot encoding [Cabin, Embarked, Initial, Ticket]
train = pd.get_dummies(train, columns=['Initial'], prefix='Initial')
test = pd.get_dummies(test, columns=['Initial'], prefix='Initial')

train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')
