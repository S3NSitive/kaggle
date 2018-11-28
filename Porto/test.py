import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from subprocess import check_call
from collections import Counter
from IPython.display import Image as PImage
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

rows = train.shape[0]
columns = train.shape[1]

# print(train.isnull().any().any())

train_copy = train
train_copy = train.replace(-1, np.NaN)

msno.matrix(df=train_copy.iloc[:, 2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))
# plt.show()

data = [go.Bar(x=train["target"].value_counts().index.values,
               y=train["target"].value_counts().values,
               text="Distribution of target variable")]
layout = go.Layout(title="Target variable distribution")

fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename="data/basic-bar")

# print(Counter(train.dtypes.values))

train_float = train.select_dtypes(include=["float64"])
train_int = train.select_dtypes(include=["int64"])

colormap = plt.cm.magma
plt.figure(figsize=(16, 12))
plt.title("Pearson correlation of continuous feature", y=1.05, size=15)
sns.heatmap(train_float.corr(), linewidths=0.1, vmax=1.0, square=True,
            cmap=colormap, linecolor="white", annot=True)
# plt.show()

data = [go.Heatmap(z=train_int.corr().values,
                   x=train_int.columns.values,
                   y=train_int.columns.values,
                   colorscale="Viridis",
                   reversescale=False,
                   opacity=1.0)]

layout = go.Layout(title="Pearson Correlation of Integer-type features",
                   xaxis=dict(ticks="", nticks=36),
                   yaxis=dict(ticks=""),
                   width=900,
                   height=700)

fig = go.Figure(data=data, layout=layout)
# py.plot(fig, filename="data/labelled-heatmap")

print(train_float.columns)
mf = mutual_info_classif(train_float.values, train.target.values, n_neighbors=3, random_state=17)
print(mf)
