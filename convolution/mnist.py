import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MnistExample:
    mnist = None
    x = None
    t = None
    keep_prob = None
    x_image = None
    num_filters1 = 32
    num_filters2 = 64
    train_step = None
    accuracy = None
    loss = None

    def prepare_data(self):
        tf.reset_default_graph()
        np.random.seed(20160704)
        tf.set_random_seed(20160704)

        # load data
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])


    def define_graph(self):
        # define first layer
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, self.num_filters1], stddev=0.1))
        h_conv1 = tf.nn.conv2d(self.x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters1]))
        h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # define second layer
        W_conv2 = tf.Variable(tf.truncated_normal([5,5,self.num_filters1,self.num_filters2], stddev=0.1))
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[self.num_filters2]))
        h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * self.num_filters2])
        num_units1 = 7 * 7 * self.num_filters2
        num_units2 = 1024
        w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
        b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
        hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)
        self.keep_prob = tf.placeholder(tf.float32)
        hidden2_drop = tf.nn.dropout(hidden2, self.keep_prob)
        w0 = tf.Variable(tf.zeros([num_units2, 10]))
        b0 = tf.Variable(tf.zeros([10]))
        k = tf.matmul(hidden2_drop, w0) + b0
        p = tf.nn.softmax(k)

        # define loss (cost) function
        self.t = tf.placeholder(tf.float32, [None, 10])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k, labels=self.t))
        correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(self.t, 1))

        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def start_train(self):
        # prepare session
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # start training
        i = 0
        for i in range(801):
            batch_xs, batch_ts = self.mnist.train.next_batch(50)
            sess.run(self.train_step, feed_dict={self.x: batch_xs, self.t: batch_ts, self.keep_prob: 0.5})
            if i % 100 == 0:
                loss_vals, acc_vals = [], []
                for c in range(4):
                    start = int(len(self.mnist.test.labels) / 4 * c)
                    end = int(len(self.mnist.test.labels) / 4 * (c + 1))
                    loss_val, acc_val = sess.run([self.loss, self.accuracy],
                                                 feed_dict={self.x: self.mnist.test.images[start:end], self.t: self.mnist.test.labels[start:end],
                                                            self.keep_prob: 1.0})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                loss_val = np.sum(loss_vals)
                acc_val = np.mean(acc_vals)
                print('Step: %d, Loss: %f, Accuracy: %f'
                      % (i, loss_val, acc_val))

        saver.save(sess, 'cnn_session')
        sess.close()





