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
plt.style.use('fivethirtyeight')

data = pd.read_csv('./data/train.csv')
print(data.head())
print(data.isnull().sum()) # checking for total null values

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
# plt.show()

# Analysing The Features (Sex)
print(data.groupby(['Sex', 'Survived'])['Survived'].count())

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dear')
# plt.show()

# Analysing The Features (Pclass)
print(pd.crosstab(data.Pclass, data.Survived, margins=True))

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead')
# plt.show()

# Analysing The Features (Sex and Pclass)
print(pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True))

sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
# plt.show()

# Analysing The Features (Age)
print(f"Oldest {data['Age'].max():.1f}")
print(f"Youngest {data['Age'].min():.1f}")
print(f"Average {data['Age'].mean():.1f}")

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot('Sex', 'Age', hue='Survived', data=data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
# plt.show()

data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.') # lets extract the Salutations

print(pd.crosstab(data.Initial, data.Sex)) # Checking the Initials with the Sex

data['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',
                           'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
                          ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs',
                           'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

print(data.groupby('Initial')['Age'].mean()) # lets check the average age by Initials

# Filling Nan Ages
# Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull()) & (data.Initial == 'Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial == 'Mrs'), 'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial == 'Master'), 'Age'] = 5
data.loc[(data.Age.isnull()) & (data.Initial == 'Miss'), 'Age'] = 22
data.loc[(data.Age.isnull()) & (data.Initial == 'Other'), 'Age'] = 46

print(data.Age.isnull().any())

f, ax = plt.subplots(1, 2, figsize=(20, 10))
data[data['Survived'] == 0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')
ax[0].set_title('Survived = 0')
x1 = list(range(0, 85, 5))
ax[0].set_xticks(x1)
data[data['Survived'] == 1].Age.plot.hist(ax=ax[1], color='green', bins=20, edgecolor='black')
ax[1].set_title('Survived = 1')
x2 = list(range(0, 85, 5))
ax[1].set_xticks(x2)
# plt.show()

sns.factorplot('Pclass', 'Survived', col='Initial', data=data)
# plt.show()

# Analysing The Features (Embarked)
print(pd.crosstab([data.Embarked, data.Pclass], [data.Sex, data.Survived], margins=True))

sns.factorplot('Embarked', 'Survived', data=data)
fig=plt.gcf()
fig.set_size_inches(5, 3)
# plt.show()

f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=data, ax=ax[0, 0])
ax[0, 0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=data, ax=ax[0, 1])
ax[0, 1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked', hue='Survived', data=data, ax=ax[1, 0])
ax[1, 0].set_title('Embarked vs  Survived')
sns.countplot('Embarked', hue='Pclass', data=data, ax=ax[1, 1])
ax[1, 1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
# plt.show()

# Filling Embarked NaN
data['Embarked'].fillna('S', inplace=True)

# Analysing The Features (SibSp)
print(pd.crosstab([data.SibSp], data.Survived))

f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('SibSp', 'Survived', data=data, ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp', 'Survived', data=data, ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
# plt.show()

print(pd.crosstab(data.SibSp, data.Pclass))
print(pd.crosstab(data.Parch, data.Pclass))

f, ax = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot('Parch', 'Survived', data=data, ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch', 'Survived', data=data, ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
# plt.show()

# Analysing The Features (Fare)
print(f"Highest Fare {data['Fare'].max():.1f}")
print(f"Lowest Fare {data['Fare'].min():.1f}")
print(f"Average Fare {data['Fare'].mean():.1f}")

f, ax = plt.subplots(1, 3, figsize=(20, 8))
sns.distplot(data[data['Pclass'] == 1].Fare, ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass'] == 2].Fare, ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass'] == 3].Fare, ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
# plt.show()

# Correlation Between The Features
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2) # data.corr() --> correlations matrix
fig=plt.gcf()
fig.set_size_inches(10, 8)
# plt.show()

# Feature Engineering and Data Cleaning (Age Band)
data['Age_band'] = 0
data.loc[data['Age'] <= 16, 'Age_band'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_band'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_band'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 62), 'Age_band'] = 3
data.loc[data['Age'] > 64, 'Age_band'] = 4
print(data.head(2))

print(data['Age_band'].value_counts().to_frame())

sns.factorplot('Age_band', 'Survived', data=data, col='Pclass')
# plt.show()

# Feature Engineering and Data Cleaning (FamilySize and Alone)
data['FamilySize'] = 0
data['FamilySize'] = data['Parch'] + data['SibSp'] # family size
data['Alone'] = 0
data.loc[data.FamilySize == 0, 'Alone'] = 1

f, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.factorplot('FamilySize', 'Survived', data=data, ax=ax[0])
ax[0].set_title('FamilySize vs Survived')
sns.factorplot('Alone', 'Survived', data=data, ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
# plt.show()

sns.factorplot('Alone', 'Survived', data=data, hue='Sex', col='Pclass')
# plt.show()

# Feature Engineering and Data Cleaning (Fare_Range)
data['Fare_Range'] = pd.qcut(data['Fare'], 4)
print(data.groupby(['Fare_Range'])['Survived'].mean().to_frame())

data['Fare_cat'] = 0
data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_cat'] = 3

sns.factorplot('Fare_cat', 'Survived', data=data, hue='Sex')
# plt.show()

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)

data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range', 'PassengerId'], axis=1, inplace=True)
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size': 20})
fig = plt.gcf()
fig.set_size_inches(18, 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()

# Predictive Modeling
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data['Survived']

# Radial Support Vector Machines(rbf-SVM)
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_X, train_Y)
prediction1 = model.predict(test_X)
print(f"Accuracy for rbf SVM is {metrics.accuracy_score(prediction1, test_Y):.3f}")

# Linear Support Vector Machine(linear-SVM)
model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_X, train_Y)
prediction2 = model.predict(test_X)
print(f"Accuracy for linear SVM is {metrics.accuracy_score(prediction2, test_Y):.3f}")

# Logistic Regression
model = LogisticRegression()
model.fit(train_X, train_Y)
prediction3 = model.predict(test_X)
print(f"The accuracy of the Logistic Regression is {metrics.accuracy_score(prediction3, test_Y):.3f}")

# Decision Tree
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
prediction4 = model.predict(test_X)
print(f"The accuracy of the Decision Tree is {metrics.accuracy_score(prediction4, test_Y):.3f}")

# K-Nearest Neighbours(KNN)
model = KNeighborsClassifier()
model.fit(train_X, train_Y)
prediction5 = model.predict(test_X)
print(f"The accuracy of the KNN is {metrics.accuracy_score(prediction5, test_Y):.3f}")

a_index = list(range(1, 11))
a = pd.Series()
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in list(range(1, 11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, test_Y)))

plt.plot(a_index, a)
plt.xticks(x)
fig = plt.gcf()
fig.set_size_inches(12, 6)
# plt.show()
print(f"Accuracies for different values of n are: {a.values} with the max value as {a.values.max()}")

# Gaussian Naive Bayes
model = GaussianNB()
model.fit(train_X, train_Y)
prediction6 = model.predict(test_X)
print(f"The accuracy of the NaiveBayes is {metrics.accuracy_score(prediction6, test_Y):.3f}")

# Random Forests
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_Y)
prediction7 = model.predict(test_X)
print(f"The accuracy of the Random Forest is {metrics.accuracy_score(prediction7, test_Y):.3f}")

# Cross Validation
kfold = KFold(n_splits=10, random_state=22)  # k=10, split the data into 10 equal parts
xyz = []
accuracy = []
std = []
classifiers = ['Linear Svm', 'Radial Svm', 'Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes',
               'Random Forest']
models=[svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(), KNeighborsClassifier(n_neighbors=9),
        DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring="accuracy")
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean': xyz, 'Std': std}, index=classifiers)
print(new_models_dataframe2)

plt.subplots(figsize=(18, 6))
box = pd.DataFrame(accuracy, index=[classifiers])
box.T.boxplot()
# plt.show()

new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig = plt.gcf()
fig.set_size_inches(8, 5)
# plt.show()

# Hyper-Parameters Tuning (SVM)
C = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
kernel = ['rbf', 'linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}
gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Hyper-Parameters Tuning (Random Forest)
n_estimators = range(100, 1000, 100)
hyper = {'n_estimators': n_estimators}
gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hyper, verbose=True, n_jobs=3)
print(time)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Ensembling (Voting Classifier)
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),
                                                ('RBF', svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),
                                                ('RFor', RandomForestClassifier(n_estimators=500, random_state=0)),
                                                ('LR', LogisticRegression(C=0.05)),
                                                ('DT', DecisionTreeClassifier(random_state=0)),
                                                ('NB', GaussianNB()),
                                                ('svm', svm.SVC(kernel='linear', probability=True))],
                                    voting='soft').fit(train_X, train_Y)
print(f"The accuracy for Ensemble model is {ensemble_lin_rbf.score(test_X, test_Y)}")
cross = cross_val_score(ensemble_lin_rbf, X, Y, cv=10, scoring="accuracy")
print(f"The cross validated score is {cross.mean()}")

# Ensembling (Bagging)
# Bagged KNN
model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), random_state=0, n_estimators=700)
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print(f"The accuracy for bagged KNN is {metrics.accuracy_score(prediction, test_Y)}")
result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print(f"The cross validated score for bagged KNN is {result.mean()}")

# Bagged Decision Tree
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0, n_estimators=100)
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print(f"The accuracy for bagged Decision Tree is {metrics.accuracy_score(prediction, test_Y)}")
result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print(f"The cross validated score for bagged Decision Tree is {result.mean()}")

# AdaBoost(Adaptive Boosting)
ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)
result = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')
print(f"The cross validated score for AdaBoost is {result.mean()}")

# Stochastic Gradient Boosting
grad = GradientBoostingClassifier(n_estimators=500, random_state=0, learning_rate=0.1)
result = cross_val_score(grad, X, Y, cv=10, scoring='accuracy')
print(f"The cross validated score for Gradient Boosting is {result.mean()}")

# XGBoost
xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result = cross_val_score(xgboost, X, Y, cv=10, scoring='accuracy')
print(f"The cross validated score for XGBoost is {result.mean()}")

# Hyper-Parameter Tuning (AdaBoost)
n_estimators=list(range(100, 1000, 100))
learn_rate = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
hyper = {'n_estimators': n_estimators, 'learning_rate': learn_rate}
gd = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True, n_jobs=4)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Hyper-Parameter Tuning (XGBoost)
n_estimators=list(range(100, 1000, 100))
learn_rate = [0.05, 0.1, 0.2, 0.3, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
hyper = {'n_estimators': n_estimators, 'learning_rate': learn_rate}
gd = GridSearchCV(estimator=xg.XGBClassifier(), param_grid=hyper, verbose=True, n_jobs=4)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# Feature Importance
f, ax = plt.subplots(2, 2, figsize=(15, 12))
model = RandomForestClassifier(n_estimators=500, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0, 0])
ax[0,0].set_title('Feature Importance in Random Forests')
model = AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0, 1],
                                                                                       color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1, 0],
                                                                                       cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model = xg.XGBClassifier(n_estimators=100, learning_rate=0.9, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1, 1],
                                                                                       color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()


