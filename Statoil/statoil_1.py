# https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d

import os
import pylab
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from os.path import join as opj
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from keras import initializers
from keras.models import Model
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

py.init_notebook_mode(connected=True)
plt.rcParams["figure.figsize"] = 10, 10

train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")

X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                          ((X_band_1+X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)


def plotmy3d(c, name):
    data = [go.Surface(z=c)]
    layout = go.Layout(title=name,
                       autosize=False,
                       width=700,
                       height=700,
                       margin=dict(l=65, r=50, b=64, t=90))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


plotmy3d(X_band_1[12, :, :], "iceberg")
plotmy3d(X_band_1[14, :, :], "Ship")


def getModel():
    gmodel = Sequential()
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    gmodel.add(Flatten())

    gmodel.add(Dense(512))
    gmodel.add(Activation("relu"))
    gmodel.add(Dropout(0.2))

    gmodel.add(Dense(256))
    gmodel.add(Activation("relu"))
    gmodel.add(Dropout(0.2))

    gmodel.add(Dense(1))
    gmodel.add(Activation("sigmoid"))

    mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss="binary_crossentropy", optimizer=mypotim, metrics=["accuracy"])
    gmodel.summary()

    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping("val_loss", patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]


file_path = "data/model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

target_train = train["is_iceberg"]
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)

gmodel = getModel()
gmodel.fit(X_train_cv, y_train_cv, batch_size=24, epochs=50, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks)

gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_valid, y_valid, verbose=1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis],
                         ((X_band_test_1 + X_band_test_2) / 2)[:, :, :, np.newaxis]], axis=-1)

predicted_test = gmodel.predict_proba(X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["is_iceberg"] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv("data/sub.csv", index=False)
