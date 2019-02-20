# https://www.kaggle.com/devm2024/transfer-learning-with-vgg-16-cnn-aug-lb-0-1712

import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as opj
from matplotlib import pyplot
from subprocess import check_output
from sklearn.metrics import log_loss
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from keras import initializers
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, GlobalMaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam, RMSprop, rmsprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.layers import Concatenate, LSTM, concatenate
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

plt.rcParams["figure.figsize"] = 10, 10

train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")
target_train = train["is_iceberg"]

train["inc_angle"] = pd.to_numeric(train["inc_angle"], errors="coerce")
test["inc_angle"] = pd.to_numeric(test["inc_angle"], errors="coerce")

train["inc_angle"] = train["inc_angle"].fillna(method="pad")
X_angle = train["inc_angle"]
test["inc_angle"] = pd.to_numeric(test["inc_angle"], errors="coerce")
X_test_angle = test["inc_angle"]

X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3 = (X_band_1 + X_band_2) / 2
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], X_band_3[:, :, :, np.newaxis]], axis=-1)

X_band_test_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3 = (X_band_test_1 + X_band_test_2) / 2
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis], X_band_test_3[:, :, :, np.newaxis]], axis=-1)

batch_size = 64
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=0.,
                         height_shift_range=0.,
                         channel_shift_range=0,
                         zoom_range=0.2,
                         rotation_range=10)


def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1, y, batch_size=batch_size, seed=55)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=55)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]


def get_callbacks(filepath, patience=2):
    es = EarlyStopping("val_loss", patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]


def getVggAngleModel():
    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train.shape[1:], classes=1)
    x = base_model.get_layer("block5_pool").output

    x = GlobalMaxPooling2D()(x)
    merge_one = concatenate([x, angle_layer])
    merge_one = Dense(512, activation="relu", name="fc2")(merge_one)
    merge_one = Dropout(0.3)(merge_one)
    merge_one = Dense(512, activation="relu", name="fc3")(merge_one)
    merge_one = Dropout(0.3)(merge_one)

    predictions = Dense(1, activation="sigmoid")(merge_one)

    model = Model(input=[base_model.input, input_2], output=predictions)

    sgd = SGD(lr=1e-3, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    return model


def myAngleCV(X_train, X_angle, X_test):
    K=3
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))

    y_train_pred_log = 0
    y_test_pred_log = 0
    y_valid_pred_log = 0.0 * target_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print("\n==================FOLD=", j)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        X_angle_cv = X_angle[train_idx]
        X_angle_hold = X_angle[test_idx]

        file_path = "data/%s_aug_model_weights.hdf5"%j
        callbacks = get_callbacks(filepath=file_path, patience=5)
        gen_flow = gen_flow_for_two_inputs(X_train_cv, X_angle_cv, y_train_cv)
        galaxyModel = getVggAngleModel()
        galaxyModel.fit_generator(gen_flow,
                                  steps_per_epoch=24,
                                  epochs=100,
                                  shuffle=True,
                                  verbose=1,
                                  validation_data=([X_holdout, X_angle_hold], Y_holdout),
                                  callbacks=callbacks)
        galaxyModel.load_weights(filepath=file_path)
        score = galaxyModel.evaluate([X_train_cv, X_angle_cv], y_train_cv, verbose=0)
        print("Train loss: ", score[0])
        print("Train accuracy: ", score[1])
        score = galaxyModel.evaluate([X_holdout, X_angle_hold], Y_holdout, verbose=0)
        print("Test loss: ", score[0])
        print("Test accuracy: ", score[1])

        pred_valid = galaxyModel.predict([X_holdout, X_angle_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        temp_train= galaxyModel.predict([X_test, X_test_angle])
        y_train_pred_log += temp_train.reshape(temp_train.shape[0])

    y_test_pred_log = y_test_pred_log / K
    y_train_pred_log = y_train_pred_log / K

    print("\n Train Log Loss Validation= ", log_loss(target_train, y_train_pred_log))
    print("Test Log Loss Validation= ", log_loss(target_train, y_valid_pred_log))

    return y_test_pred_log


preds = myAngleCV(X_train, X_angle, X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["is_iceberg"] = preds
submission.to_csv("data/sub2.csv", index=False)
