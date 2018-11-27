import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

rows = train.shape[0]
columns = train.shape[1]
print(f"The train dataset contains {rows} rows and {columns} columns")

# 1. Data Quality checks
# Null or missing values check
# print(train.isnull().any().any())

train_copy = train.copy()
train_copy = train_copy.replace(-1, np.NaN)

msno.matrix(df=train_copy.iloc[:, 2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))
# plt.show()

data = [go.Bar(x=train['target'].value_counts().index.values,
               y=train['target'].value_counts().values,
               text='Distribution of target variable')]
layout = go.Layout(title='Target variable distribution')
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='./data/basic-bar')

print(Counter(train.dtypes.values))

train_float = train.select_dtypes(include=['float64'])
train_int = train.select_dtypes(include=['int64'])

colormap = plt.cm.magma
plt.figure(figsize=(16, 12))
plt.title('Pearson correlation of continuous feature', y=1.05, size=15)
sns.heatmap(train_float.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

data = [go.Heatmap(z=train_int.corr().values,
                   x=train_int.columns.values,
                   y=train_int.columns.values,
                   colorscale='Viridis',
                   reversescale=False,
                   opacity=1.0)]

layout = go.Layout(title='Pearson Correlation of Integer-type features',
                   xaxis=dict(ticks='', nticks=36),
                   yaxis=dict(ticks=''),
                   width=900,
                   height=700)

fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='./data/labelled-heatmap')

mf = mutual_info_classif(train_float.values, train.target.values, n_neighbors=3, random_state=17)

bin_col = [col for col in train.columns if '_bin' in col]
zero_list = []
one_list = []
for col in bin_col:
    zero_list.append((train[col] == 0).sum())
    one_list.append((train[col] == 1).sum())

trace1 = go.Bar(x=bin_col,
                y=zero_list,
                name='Zero count')
trace2 = go.Bar(x=bin_col,
                y=one_list,
                name='One count')
data = [trace1, trace2]
layout = go.Layout(barmode='stack',
                   title='Count of 1 and 0 in binary variables')

fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='./data/stacked-bar')

# Feature importance via Random Forest
X_train = train.drop(["id", "target"], axis=1)
y_train = train["target"]

rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4,
                            max_features=0.2, n_jobs=1, random_state=0)
rf.fit(X_train, y_train)
features = X_train.columns.values
print("----- Training Done -----")

trace = go.Scatter(y=rf.feature_importances_,
                   x=features,
                   mode="markers",
                   marker=dict(sizemode="diameter",
                               sizeref=1,
                               size=13,
                               color=rf.feature_importances_,
                               colorscale="Portland",
                               showscale=True),
                   text=features)
data = [trace]

layout = go.Layout(autosize=True,
                   title="Random Forest Feature Importance",
                   hovermode="closest",
                   xaxis=dict(ticklen=5,
                              showgrid=False,
                              zeroline=False,
                              showline=False),
                   yaxis=dict(title="Feature Importance",
                              showgrid=False,
                              zeroline=False,
                              ticklen=5,
                              gridwidth=2),
                   showlegend=False)
fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename='./data/scatter2010')

x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), reverse=False)))
trace2 = go.Bar(x=x,
                y=y,
                marker=dict(color=x,
                            colorscale="Viridis",
                            reversescale=True),
                name="Random Forest Feature importance",
                orientation="h")

layout = dict(title="Barplot of Feature importances",
              width=900,
              height=2000,
              yaxis=dict(showgrid=False,
                         showline=False,
                         showticklabels=True))

fig1 = go.Figure(data=[trace2])
fig1["layout"].update(layout)
py.plot(fig1, filename="plots")
