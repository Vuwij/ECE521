# -*- coding: utf-8 -*-
"""
Created on Sun May 21 02:20:03 2017

@author: Givo
"""

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, name='x')
c = tf.constant(2.0, name='c')
cx_squared = c*x*x

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(cx_squared, feed_dict={x:np.array([1,2,3])}))

print(sess.run(cx_squared, feed_dict={x:np.array([[1,2],[6,7]])}))