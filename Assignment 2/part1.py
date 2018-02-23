import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt
import plotly.plotly as py

import a2_data as a2_data;

# Simple Linear Regression Test
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, [None, 1])
wls = tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(X),X)),tf.matmul(tf.transpose(X),y))

# With gradient descent
ni = tf.placeholder(tf.int32, name='n')
wk = tf.placeholder(tf.float32, [None, 1])
DeltaEn = tf.expand_dims(X[ni], axis=1) * tf.reduce_sum(tf.matmul(tf.transpose(wk), tf.expand_dims(X[ni], axis=1)) - y[ni])


def linearRegressionLS(xin, yin):
    Xin = np.concatenate((np.ones((np.shape(xin)[0], 1)), xin), axis=1)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(wls, feed_dict={
            X:Xin,
            y:yin}))

def linearRegressionSGD(xin, yin, w=None, iterations = 10000, l = 0, epsk = 0.005):
    N = (np.shape(xin)[0])
    Xin = np.concatenate((np.ones((N, 1)), xin), axis=1)
    if w is None:
        w = np.zeros((np.shape(xin)[1]+1, 1))
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for t in range(0,iterations):
        n = np.random.randint(0, N)
        en = sess.run(DeltaEn, feed_dict={
            X:Xin,
            y:yin,
            ni:n,
            wk:w})
        
        w = w - epsk * en

    return w

def linearRegressionSGDBatch(xin, yin, batchsize = 500, iterations = 20000, l = 0, epsk = 0.005):
    N = (np.shape(xin)[0])
    BatchPerEpoch = int(N / batchsize);
    EpochCount = 1;
    TotalEpochCount = int(iterations / BatchPerEpoch)
    w = np.zeros((np.shape(xin)[1]+1, 1))
    
    for iter in range(0, TotalEpochCount):
        for e in range(0, BatchPerEpoch):
            EpochCount += 1
            
            w = linearRegressionSGD(xin[e:e+batchsize], yin[e:e+batchsize], w, batchsize, 0, 0.005)
            print(EpochCount)
    
    
def linearRegressionTest():
    xin = np.array([[0],[1],[2]])
    yin = np.array(np.expand_dims(np.transpose([6,0,0]), axis=1))
    
    linearRegressionLS(xin, yin)
    linearRegressionSGD(xin, yin)
    

## Tuning the learning rate

# Load the data
trainData, trainTarget, validData, validTarget, testData, testTarget = a2_data.loadNotMNIST();

trainDataLR = np.reshape(trainData, (3500, 784))
validDataLR = np.reshape(validData, (100, 784))
testDataLR = np.reshape(testData, (145, 784))

linearRegressionSGDBatch(trainDataLR, trainTarget)