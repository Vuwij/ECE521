# -*- coding: utf-8 -*-
"""
Created on Sun May 21 02:15:42 2017

@author: Givo
"""

import tensorflow as tf


a = tf.constant([1,2,3,4], name='a')
b = tf.constant([1,2,3,4], name='b')
c = a*b

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run([a,b,c]))
