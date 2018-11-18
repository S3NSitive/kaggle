import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing all the required ML packages
from sklearn.svm import SVC  # support vector machine
from sklearn import metrics  # accuracy measure
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.metrics import confusion_matrix  # for confusion matrix
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.model_selection import train_test_split  # training and testing data split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Cross Validation
from sklearn.model_selection import KFold  # for K-fold cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score  # score evaluation
from sklearn.model_selection import cross_val_predict  # prediction
from sklearn.model_selection import StratifiedShuffleSplit

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
# print(train.columns)
# print(train.isnull().sum())

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
# print(dataset.groupby('Initial').mean())

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
for i in range(train.shape[0]):
    if not train['Ticket'][i].isdigit():
        train['Ticket'][i] = train['Ticket'][i].replace('.', '').replace('/', '').strip().split(' ')[0]
    else:
        train['Ticket'][i] = 'X'

for i in range(test.shape[0]):
    if not test['Ticket'][i].isdigit():
        test['Ticket'][i] = test['Ticket'][i].replace('.', '').replace('/', '').strip().split(' ')[0]
    else:
        test['Ticket'][i] = 'X'

# Fare
train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

# One-hot encoding [Cabin, Embarked, Initial, Ticket, Pclass]
train = pd.get_dummies(train, columns=['Initial'], prefix='Initial')
test = pd.get_dummies(test, columns=['Initial'], prefix='Initial')

train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')

train = pd.get_dummies(train, columns=['Ticket'], prefix='Ticket')
test = pd.get_dummies(test, columns=['Ticket'], prefix='Ticket')

train = pd.get_dummies(train, columns=['Cabin'], prefix='Cabin')
test = pd.get_dummies(test, columns=['Cabin'], prefix='Cabin')

train = pd.get_dummies(train, columns=['Pclass'], prefix='Pclass')
test = pd.get_dummies(test, columns=['Pclass'], prefix='Pclass')

# drop
drop_category = ['SibSp', 'Parch', 'Name', 'PassengerId', 'Age']
train.drop(labels=drop_category, axis=1, inplace=True)
# print(train.columns)

# Modeling
y_train = train['Survived']
X_train = train.drop(labels=['Survived'], axis=1)
random_state = 2
kfold = StratifiedKFold(n_splits=5, shuffle=True)

classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                      random_state=random_state, learning_rate=0.1))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(xg.XGBClassifier(random_state=random_state))

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=y_train, scoring='accuracy', cv=kfold, n_jobs=4))

cv_means = []
cv_std = []
cv_accuracy = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    cv_accuracy.append(cv_result)

cv_res = pd.DataFrame({'CrossValMeans': cv_means, 'CrossValStd': cv_std,
                       'Algorithm': ['SVC', 'MLP', 'LogisticRegression', 'ExtraTrees', 'KNN',
                                     'DecisionTree', 'RandomForest', 'LinearDiscriminant',
                                     'AdaBoost', 'GradientBoost', 'XGBoost']})
g = sns.barplot('CrossValMeans', 'Algorithm', data=cv_res)
g.set_title('Cross Validation Scores')
g.set_xlabel('Mean Accuracy')
print(cv_res)
# plt.show()

# Hyperparameter Turning
C = [1, 50, 100, 500, 1000]
gamma = [0.001, 0.1, 0.2, 1]
subsample = [0.5, 0.7, 0.9]
colsample_bytree = [0.5, 0.7, 0.9]
min_child_weight = [1, 2, 3, 5]
max_depth = [3, 4, 8]
max_features = [2, 3, 10]
min_samples_split = [2, 3, 10]
min_samples_leaf = [2, 3, 10]
n_estimators = list(range(300, 900, 100))
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

best_estimators = []
best_scores = []

SVMC = SVC(probability=True)
SVMC_params = {'kernel': ['rbf'],
               'gamma': gamma,
               'C': C}
gsSVMC = GridSearchCV(SVMC, param_grid=SVMC_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsSVMC.fit(X_train, y_train)
SVMC_best_estimator = gsSVMC.best_estimator_
SVMC_best_score = gsSVMC.best_score_
best_estimators.append(SVMC_best_estimator)
best_scores.append(SVMC_best_score)
print('SVMC Clear')

ExtC = ExtraTreesClassifier()
ExtC_params = {'max_depth': [None],
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': [False],
               'n_estimators': n_estimators,
               'criterion': ['gini']}

gsExtC = GridSearchCV(ExtC, param_grid=ExtC_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsExtC.fit(X_train, y_train)
ExtC_best_estimator = gsExtC.best_estimator_
ExtC_best_score = gsExtC.best_score_
best_estimators.append(ExtC_best_estimator)
best_scores.append(ExtC_best_score)
print('ExtC Clear')

RFC = RandomForestClassifier()
RFC_params = {'max_depth': [None],
              'max_features': max_features,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': [False],
              'n_estimators': n_estimators,
              'criterion': ['gini']}

gsRFC = GridSearchCV(RFC, param_grid=RFC_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsRFC.fit(X_train, y_train)
RFC_best_estimator = gsRFC.best_estimator_
RFC_best_score = gsRFC.best_score_
best_estimators.append(RFC_best_estimator)
best_scores.append(RFC_best_score)
print('RFC Clear')

DTC = DecisionTreeClassifier()
AdaDTC = AdaBoostClassifier(DTC, random_state=random_state)
Ada_params = {'base_estimator__criterion': ['gini', 'entropy'],
              'base_estimator__splitter': ['best', 'random'],
              'algorithm': ['SAMME', 'SAMME.R'],
              'n_estimators': n_estimators,
              'learning_rate': learning_rate}

gsAdaDTC = GridSearchCV(AdaDTC, param_grid=Ada_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsAdaDTC.fit(X_train, y_train)
Ada_best_estimator = gsAdaDTC.best_estimator_
Ada_best_score = gsAdaDTC.best_score_
best_estimators.append(Ada_best_estimator)
best_scores.append(Ada_best_score)
print('Ada Clear')

GBC = GradientBoostingClassifier()
GBC_params = {'loss': ['deviance'],
              'max_depth': max_depth,
              'min_samples_leaf': min_samples_leaf,
              'max_features': max_features,
              'n_estimators': n_estimators,
              'learning_rate': learning_rate}

gsGBC = GridSearchCV(GBC, param_grid=GBC_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsGBC.fit(X_train, y_train)
GBC_best_estimator = gsGBC.best_estimator_
GBC_best_score = gsGBC.best_score_
best_estimators.append(GBC_best_estimator)
best_scores.append(GBC_best_score)
print('GBC Clear')

XGBC = xg.XGBClassifier(objective='binary:logistic')
XGBC_params = {'max_depth': max_depth,
               'gamma': gamma,
               # 'subsample': subsample,
               'min_child_weight': min_child_weight,
               'n_estimators': n_estimators,
               'learning_rate': learning_rate}

gsXGBC = GridSearchCV(XGBC, param_grid=XGBC_params, cv=kfold, scoring='accuracy', verbose=1, n_jobs=4)
gsXGBC.fit(X_train, y_train)
XGBC_best_estimator = gsXGBC.best_estimator_
XGBC_best_score = gsXGBC.best_score_
best_estimators.append(XGBC_best_estimator)
best_scores.append(XGBC_best_score)
print('XGBC Clear')

tuning_res = pd.DataFrame({'best_estimator': best_estimators, 'best_score': best_scores,
                            'Algorithm': ['SVC', 'ExtraTreesClassifier', 'RandomForest',
                                          'AdaBoost', 'GradientBoost', 'XGBoost']})
print(tuning_res)

votingC = VotingClassifier(estimators=[('RFC', RFC_best_estimator),
                                       ('ExtC', ExtC_best_estimator),
                                       ('SVMC', SVMC_best_estimator),
                                       ('Ada', Ada_best_estimator),
                                       ('GBC', GBC_best_estimator),
                                       ('XGBC', XGBC_best_estimator)],
                           voting='soft')
votingC = votingC.fit(X_train, y_train)

test_Survived = pd.Series(votingC.predict(test), name='Survived')
results = pd.concat([PassengerId, test_Survived], axis=1)
results.to_csv('./data/my_titanic_python_voting.csv', index=False)
