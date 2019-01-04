import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


class Model:
    def __init__(self, name, activation_fn=tf.nn.relu, optimizer_fn=tf.train.AdamOptimizer, learning_rate=0.001):
        with tf.name_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
            self.Y = tf.placeholder(tf.float32, [None, 10], name="Y")
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.L1 = tf.layers.conv2d(self.X, 32, [3, 3], activation=activation_fn)
            self.L1 = tf.layers.max_pooling2d(self.L1, [2, 2], [2, 2])
            self.L1 = tf.layers.dropout(self.L1, 0.7, self.is_training)

            self.L2 = tf.layers.conv2d(self.L1, 64, [3, 3], activation=activation_fn)
            self.L2 = tf.layers.max_pooling2d(self.L2, [2, 2], [2, 2])
            self.L2 = tf.layers.dropout(self.L2, 0.7, self.is_training)

            self.L3 = tf.contrib.layers.flatten(self.L2)
            self.L3 = tf.layers.dense(self.L3, 256, activation=activation_fn)
            self.L3 = tf.layers.dropout(self.L3, 0.5, self.is_training)

            self.model = tf.layers.dense(self.L3, 10, activation=None)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.Y))
            self.optimizer = optimizer_fn(learning_rate).minimize(self.cost)


def train(sess, model):
    BATCH_SIZE = 100
    total_batch = int(len(X_train) / BATCH_SIZE)

    epochs_completed = 0
    index_in_epoch = 0
    num_examples = X_train.shape[0]

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(BATCH_SIZE)
            # batch_xs = batch_xs(-1, 28, 28, 1)

            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, is_training: True})
            total_cost += cost_val

        print(f"Epoch: {epoch+1} Avg. cost = {(total_cost / total_batch):.3f}")

    print("==== End ====")


def predict(sess, model):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print(f"정확도: {sess.run(accuracy, feed_dict={X: X_val, Y: Y_val, is_training: False}):.3f}")


# serve data by batches
def next_batch(batch_size):
    global X_train
    global Y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], Y_train[start:end]


if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    Y_train = train["label"]
    X_train = train.drop(["label"], axis=1)

    del train

    X_train = X_train / 255.0
    test = test / 255.0

    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    Y_train = to_categorical(Y_train, num_classes=10)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

    bn = Model('bn')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
