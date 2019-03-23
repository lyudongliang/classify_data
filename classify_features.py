import numpy as np
import tensorflow as tf
from generate_data import generate_data, load_partial
# import tensorboard


def train():
    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='samples')
        y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='labels')

        W1 = tf.Variable(tf.random_normal(shape=(2, 32)), name='weight1')
        b1 = tf.Variable(tf.zeros(shape=(32)), name='bias1')
        W2 = tf.Variable(tf.random_normal(shape=(32, 32)), name='weight2')
        b2 = tf.Variable(tf.zeros(shape=(32)), name='bias2')
        W3 = tf.Variable(tf.random_normal(shape=(32, 32)), name='weight3')
        b3 = tf.Variable(tf.zeros(shape=(32)), name='bias3')
        W4 = tf.Variable(tf.random_normal(shape=(32, 2)), name='weight4')
        b4 = tf.Variable(tf.zeros(shape=(2)), name='bias4')

        z1 = tf.matmul(x, W1) + b1
        layer1 = tf.tanh(z1, name='layer1')
        # layer1 = tf.nn.relu(z1, name='layer1')

        z2 = tf.matmul(layer1, W2) + b2
        layer2 = tf.tanh(z2, name='layer2')
        # layer2 = tf.nn.relu(z2, name='layer2')

        z3 = tf.matmul(layer2, W3) + b3
        layer3 = tf.tanh(z3, name='layer3')
        # layer3 = tf.nn.relu(z3, name='layer3')

        out = tf.matmul(layer3, W4) + b4

        # for train
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))

        optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        predictions = tf.nn.softmax(out, name='pred')

    with tf.Session(graph=graph) as my_sess:
        # writer = tf.summary.FileWriter('board', graph)

        tf.global_variables_initializer().run()

        train_samples, train_labels = generate_data(100000)

        for partial_samples, partial_labels in load_partial(train_samples, train_labels, 100):
            # print('partial_samples, partial_labels', partial_samples, partial_labels)
            # print(len(partial_samples), np.shape(partial_samples))
            _, _loss, _train_pred = my_sess.run([optimizer, loss, predictions], feed_dict={x: partial_samples, y: partial_labels})

            # print('_loss, _train_pred\n', _loss, _train_pred)
            train_accuracy = get_accuracy(_train_pred, partial_labels)
            print('_loss, train accuracy', _loss, train_accuracy)

        # test
        test_samples, test_labels = generate_data(10000)
        _, _test_loss, _test_pred = my_sess.run([optimizer, loss, predictions], feed_dict={x: test_samples, y: test_labels})

        # print('_loss, _train_pred\n', _loss, _train_pred)
        test_accuracy = get_accuracy(_test_pred, test_labels)
        print('_test_loss test_accuracy', _test_loss, test_accuracy)

        # writer.close()


def get_accuracy(predictions, labels):
    _predictions = np.argmax(predictions, axis=1)
    # print('_predictions', _predictions)
    _labels = np.argmax(labels, axis=1)
    # print('_labels', _labels)
    accuracy = np.sum(_predictions == _labels) / np.shape(predictions)[0]
    # print('accuracy', accuracy)
    return accuracy


# for test
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#
# accurary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    train()

