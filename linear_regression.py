# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Create data and simulate results
x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
noise = np.random.randn(1, 2000)*0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

NUM_STEPS = 20
wb_ = []
# model structure
with tf.Graph().as_default() as g:
    x = tf.placeholder(tf.float32, shape=[None, 3], name="x")
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope1:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.add(tf.matmul(w, tf.transpose(x)), b, name="y_pred")

    with tf.name_scope('loss') as scope2:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))

    with tf.name_scope('train') as scope3:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

# model train and save
with tf.Session(graph=g) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(NUM_STEPS):
        sess.run(train, {x: x_data, y_true: y_data})
        if step % 5 == 0:
            print(step, sess.run([w, b]))
            wb_.append(sess.run([w, b]))

    print(20, sess.run([w, b]))
    saver = tf.train.Saver()
    saver.save(sess, "./model/lr.ckpt")

