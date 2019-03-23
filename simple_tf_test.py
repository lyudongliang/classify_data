import tensorflow as tf
import time


time0 = time.time()

a = tf.Variable(10)
b = tf.Variable(5)
print('a', a)
print('type a', type(a))

add_v = a + b
print('addv', add_v)

c1 = tf.constant(2)
c2 = tf.constant(3)
print('c1', c1)
print('type c1', type(c1))

add_c = c1 + c2
print('add_c', add_c)
print('type add_c', type(add_c))

sess = tf.Session()

tf.global_variables_initializer().run(session=sess)

print('addv', sess.run(add_v))

print('add_c', sess.run(add_c))

graph = tf.Graph()

with graph.as_default():
    value1 = tf.constant([1, 2])
    value2 = tf.Variable([3, 4])
    multi = value1 * value2

with tf.Session(graph=graph) as my_sess:
    tf.global_variables_initializer().run()
    print('multi', my_sess.run(multi))


def use_placeholder():
    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.placeholder(dtype=tf.float64)
        value2 = tf.Variable([3, 4], dtype=tf.float64)
        # value2 = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float64)
        mul = value1 * value2

    with tf.Session(graph=graph) as my_sess:
        tf.global_variables_initializer().run()

        value = load_from_remote()
        for partial_value in load_partial(value, 2):
            # holder_value = {value1: partial_value}
            # print('holder_value', holder_value)
            mul_result = my_sess.run(fetches=mul, feed_dict={value1: partial_value})
            print('mul_result', mul_result)


def load_from_remote():
    return range(1000)


def load_partial(value, step):
    index = 0
    while index < len(value):
        yield value[index: index + step]
        index += step
    return


use_placeholder()


print('total time', time.time() - time0)

