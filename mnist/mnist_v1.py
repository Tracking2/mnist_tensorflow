# --------------------------------------------------------------------------------
# Tensorflow 1.2 + Python3.5
# Handwritten Digit Recognition Using Convolution Neural Networks
# Based on tensorflow official website
# Written by Fallon on 2018-9-23
#
# --------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Alexnet(object):

    def __init__(self, scope=None):
        self._scope = scope
        self._keep_prob = 1.0
        self._accuracy = {}
        self._learn_rate = []


    def _weight_variable(self, shape):
        weight_initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weight_initial)

    def _bias_variable(self, shape):
        bias_initial = tf.constant(0.1, shape=shape)
        return tf.Variable(bias_initial)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_net(self, input_x, input_y_):
        with tf.variable_scope(self._scope, self._scope):
            self._input_x = input_x
            self._input_y_ = input_y_
            self._x_image = tf.reshape(self._input_x, [-1, 28, 28, 1])

            self._W_conv1 = self._weight_variable([5, 5, 1, 32])
            self._b_conv1 = self._bias_variable([32])

            self._h_conv1 = tf.nn.relu(self._conv2d(self._x_image, self._W_conv1) + self._b_conv1)
            self._h_pool1 = self._max_pool_2x2(self._h_conv1)

            self._W_conv2 = self._weight_variable([5, 5, 32, 64])
            self._b_conv2 = self._bias_variable([64])

            self._h_conv2 = tf.nn.relu(self._conv2d(self._h_pool1, self._W_conv2) + self._b_conv2)
            self._h_pool2 = self._max_pool_2x2(self._h_conv2)

            self._W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
            self._b_fc1 = self._bias_variable([1024])

            self._h_pool2_flat = tf.reshape(self._h_pool2, [-1, 7 * 7 * 64])
            self._h_fc1 = tf.nn.relu(tf.matmul(self._h_pool2_flat, self._W_fc1) + self._b_fc1)

            self._keep_prob = tf.placeholder("float")
            self._h_fc1_drop = tf.nn.dropout(self._h_fc1, self._keep_prob)

            self._W_fc2 = self._weight_variable([1024, 10])
            self._b_fc2 = self._bias_variable([10])

            self._y_conv = tf.nn.softmax(tf.matmul(self._h_fc1_drop, self._W_fc2) + self._b_fc2)

            cross_entropy = -tf.reduce_sum(self._input_y_ * tf.log(self._y_conv))
            self.train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(self._y_conv, 1), tf.argmax(self._input_y_, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def compute_accuracy(self,sess, batch, name):
        with tf.name_scope(name=name):
            self.train_accuracy = sess.run(self._accuracy, feed_dict={
                                                      self._input_x: batch[0],
                                                      self._input_y_: batch[1],
                                                      self._keep_prob: 1.0})
            print('{0} Accuracy {1:6.4f}'.format(name, self.train_accuracy))

    def train(self, sess, batch):
        sess.run(self.train_step, feed_dict={
                                    self._input_x: batch[0],
                                    self._input_y_: batch[1],
                                    self._keep_prob: 0.5})


if __name__ == '__main__':

    with tf.Session() as sess:

        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, 10])
        mnist_Alx = Alexnet('test')
        mnist_Alx.build_net(x, y_)
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                mnist_Alx.compute_accuracy(sess, batch, 'TRAIN')
            mnist_Alx.train(sess, batch)

        batch_test = [mnist.test.images, mnist.test.labels]
        mnist_Alx.compute_accuracy(sess, batch_test, 'TEST')






