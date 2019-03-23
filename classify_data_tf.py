# coding:utf-8

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# front transparent
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# loss function
# cross_entropy = -1 * tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y), axis=[1]))
cross_entropy = -1 * tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# solve algorithm
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成数据集与标签
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# print('X', X)
print('shape X', X.shape)

# 这里是LIST，和X不同，X是array
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]


# create Session()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # train iterations
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        feed_dict1 = {x: X[start:end], y_: Y[start:end]}
        sess.run(train_step, feed_dict=feed_dict1)
        feed_dict2 = {x: X, y_: Y}
        if i % 100 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict=feed_dict2)
            print("step ", i, "   ", "loss is ", total_cross_entropy)

