# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf



def readMat(path):
    return np.load(path)

def writeMat(path, m):
    np.save(path, m)


def loadData(dataPath, targetPath):
    # Loading my data
    inputData = readMat(dataPath)
    target = readMat(targetPath)

    trainData = inputData[:,0:70].T
    testData = inputData[:,70:].T

    trainTarget = np.expand_dims(target[0:70], 1)
    testTarget = np.expand_dims(target[70:],1)
    return trainData, trainTarget, testData, testTarget


def buildGraph():
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[5,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 5], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    # Error definition
    meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target), 
                                                reduction_indices=1, 
                                                name='squared_error'), 
                                  name='mean_squared_error')

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss=meanSquaredError)
    return W, b, X, y_target, y_predicted, meanSquaredError, train

    
def runMult(dataPath, targetPath):

    # Build computation graph
    W, b, X, y_target, y_predicted, meanSquaredError, train = buildGraph()

    # Loading my data
    trainData, trainTarget, testData, testTarget = loadData(dataPath, targetPath)

    # Initialize session
    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)  
    initialb = sess.run(b)

    print("Initial weights: %s, initial bias: %.2f"%(initialW, initialb))
    # Training model
    wList = []
    for step in range(0,201):
        _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError, W, b, y_predicted], feed_dict={X: trainData, y_target: trainTarget})
        wList.append(currentW)
        if not (step % 50) or step < 10:        
            print("Iter: %3d, MSE-train: %4.2f, weights: %s, bias: %.2f"%(step, err, currentW.T, currentb))

    # Testing model
    errTest = sess.run(meanSquaredError, feed_dict={X: testData, y_target: testTarget})
    print("Final testing MSE: %.2f"%(errTest))      


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    runMult('./x.npy', './t2.npy')