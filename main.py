import h5py
from network import network
import sys

test_data = 'food_test_c101_n1000_r32x32x3.h5'
train_data = 'food_c101_n10099_r32x32x3.h5'
test = h5py.File(test_data, 'r')
data = h5py.File(train_data, 'r')

for key in data.keys():
    print(key)

images = data['images']
print(images.shape)
print(images.dtype)
print(images)

print(images[0].shape)

for i in data['category_names']:
    print(i)

import matplotlib.pyplot as plt

plt.imshow(images[0])
data["category"][0]

plt.imshow(images[10])
data["category"][10]

plt.imshow(images[100])
data["category"][100]

images[0]

import numpy as np

X_train = np.array(data["images"]) / 255
y_train = np.array([[int(i) for i in data["category"][j]] for j in range(len(data["category"]))])
X_test = np.array(test["images"]) / 255
y_test = np.array([[int(i) for i in test["category"][j]] for j in range(len(test["category"]))])

X_train = X_train[0:10000]
y_train = y_train[0:10000]
X_test = X_test[0:10000]
y_test = y_test[0:10000]

X_train.shape

y_train.shape

X_train[0]

y_train[0]


class FoodHelper():

    def __init__(self):
        self.i = 0

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = X_train
        self.training_labels = y_train

        print("Setting Up Test Images and Labels")

        self.test_images = X_test
        self.test_labels = y_test

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    #def next_batch(self, X, y, batch_size):
    #    rnd_idx = np.random.permutation(len(X))
    #    n_batches = len(X) // batch_size
    #    for batch_idx in np.array_split(rnd_idx, n_batches):
    #        X_batch, y_batch = X[batch_idx], y[batch_idx]
    #        yield X_batch, y_batch


fh = FoodHelper()
fh.set_up_images()

fh.training_images.shape

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 101])

hold_prob = tf.placeholder(tf.float32)

y_pred = network(X, hold_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def main():

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            batch = fh.next_batch(100)
            sess.run(train, feed_dict={X: batch[0], y_true: batch[1], hold_prob: 0.8})

            if i % 100 == 0:
                print('Currently on step {}'.format(i))
                print('Accuracy is:')
                # Test the Train Model
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                print(sess.run(acc, feed_dict={X: fh.test_images, y_true: fh.test_labels, hold_prob: 1.0}))
                print('\n')
        save_path = saver.save(sess, './model.ckpt')

if __name__ == "__main__":
    main()