


import tensorflow as tf



i0 = tf.constant(0)
m0 = tf.ones([2, 2])
c = lambda i, m: i < 10
b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
it, mt = tf.while_loop(
    c, b, loop_vars=[i0, m0])
    # shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])


with tf.Session():
    print mt.eval().shape


fasdfs




i0 = tf.constant(0)
m0 = tf.ones([2, 2])
c = lambda i, m: i < 10
b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
it, mt = tf.while_loop(
    c, b, loop_vars=[i0, m0],
    shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])


with tf.Session():
    print mt.eval().shape


fasdfs


s_in_pl = tf.placeholder(tf.float32)
def foo(a):
    return tf.exp(s_in_pl)

step_count = 10

# Ignore the first return value (which will be the final value of the iteration
# counter, `i`).
_, s_final = tf.while_loop(lambda i, _: i < step_count,
                           lambda i, s_current: [i + 1, foo(s_current)],
                           [0, s_in_pl])

s_in = [1,2,3]
with tf.Session():
    print s_final.eval(feed_dict={s_in_pl: s_in})