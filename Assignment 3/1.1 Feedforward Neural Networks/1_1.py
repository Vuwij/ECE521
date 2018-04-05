#!/nfs/ug/cad/cad2/ece521s/tensorflow_gpu/bin/python

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import sys

def loadData(fileName):
    with np.load(fileName) as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIdx = np.arange(len(Data))
        np.random.shuffle(randIdx)
        Data = Data[randIdx]/255.0
        Target = Target[randIdx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, trainTarget, validData, validTarget, testData, testTarget


def layerWiseBuildingBlock(X, numHiddenUnits):
    """Takes the hidden activations from the previous layer then return the weighted sum
    of the inputs for the current hidden layer"""
    # INPUT: input tensor and the number of the hidden units
    # Output: the weighted sum of the inputs for the current hidden layer

    # Xavier Initialization
    prevDim = tf.to_int32(X.get_shape()[0])
    std_dev = tf.sqrt(3.0 / tf.to_float(prevDim + numHiddenUnits))
    
    # Variable Creation
    S = tf.placeholder(tf.float32, [numHiddenUnits, 1])
    W = tf.Variable(tf.truncated_normal(shape=[prevDim,numHiddenUnits], stddev=std_dev))
    b = tf.Variable(0.0, [numHiddenUnits, 1])
    
    # Graph definition
    S = tf.matmul(W, X, transpose_a=True) + b  # dim is [None, numHiddenUnits]
    
    return S, W


def buildGraph(numLayers, numHiddenUnits, learningRate):
    """Build neural network model with ReLU activation functions"""
    
    # Variable creation
    X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
    X0 = tf.reshape(X, [28 * 28, -1])
    Ytarget = tf.placeholder(tf.float32, name='target_y')
    Yonehot = tf.one_hot(tf.to_int32(Ytarget), 10, 1.0, 0.0)
    lamda = tf.placeholder("float32", name='Lambda')
    
    # Neural Network
    # Layer 0: Input <=> Hidden
    S1, W1 = layerWiseBuildingBlock(X0, numHiddenUnits)
    X1 = tf.nn.relu(S1)
    
    # Layer 1: Hidden <=> Logits
    S2, W2 = layerWiseBuildingBlock(X1, 10)
    X2 = tf.nn.relu(S2)

    # Layer 2: Logits <=> Predicted Output
    Y = tf.nn.softmax(X2)
    
    # Losses from Cross Entropy and Weight Loss
    crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                      labels=Yonehot, logits=tf.transpose(X2)),
                                      name='mean_cross_entropy')

    weightLoss = 0.5 * lamda * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))
    loss = crossEntropyLoss + weightLoss

    # Accuracy
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(Y, 0),
                                             tf.to_int64(Ytarget))))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
    train = optimizer.minimize(loss=loss)
    
    return X, Ytarget, Y, crossEntropyLoss, train, lamda, acc


def runFullIteration(B, max_iter, wd_lambda, learningRate):
    """Run full iteration"""
    
    fileName = "notMNIST.npz"
    trainData, trainTarget, validData, validTarget, testData, testTarget = loadData(fileName)
    
    numBatches = np.floor(len(trainData)/B)

    trainLoss_list = []
    validLoss_list = []
    testLoss_list = []

    trainAcc_list = []
    validAcc_list = []
    testAcc_list = []

    numLayers=1
    numHiddenUnits = 1000

    X, y_target, y_predicted, crossEntropyError, train, Lambda, acc_h = buildGraph(numLayers, numHiddenUnits, learningRate)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        for step in range(0, max_iter+1):
            if step % numBatches == 0:

                # Sample minibatch without replacement
                randIdx = np.arange(len(trainData))
                np.random.shuffle(randIdx)
                trainData = trainData[randIdx]
                trainTarget = trainTarget[randIdx]
                i = 0  # cyclic index for mini-batch

                # storing MSE and Acc for the three datasets every epoch
                err = crossEntropyError.eval(feed_dict={X: trainData, y_target: trainTarget})
                acc = acc_h.eval(feed_dict={X: trainData, y_target: trainTarget})
                trainLoss_list.append(err)
                trainAcc_list.append(acc)

                err = crossEntropyError.eval(feed_dict={X: validData, y_target: validTarget})
                acc = acc_h.eval(feed_dict={X: validData, y_target: validTarget})
                validLoss_list.append(err)
                validAcc_list.append(acc)

                err = crossEntropyError.eval(feed_dict={X: testData, y_target: testTarget})
                acc = acc_h.eval(feed_dict={X: testData, y_target: testTarget})
                testLoss_list.append(err)
                testAcc_list.append(acc)

            # Slicing a mini-batch from the whole training dataset
            feeddict = {X: trainData[i*B:(i+1)*B], y_target: trainTarget[i*B:(i+1)*B],
                       Lambda: wd_lambda}

            # Update model parameters
            _, err, yhat = sess.run([train, crossEntropyError, y_predicted], feed_dict=feeddict)

            # storing weights every iteration
            # wList.append(currentW)
            i += 1

            # displaying training MSE error every 100 iterations
            if not (step % 100):
                print("Iter: %3d, CrossEntropyError: %4.5f, Accuracy: %4.5f" % (step, err, acc))

    data = {}
    data['trainLoss'] = trainLoss_list
    data['validLoss'] = validLoss_list
    data['testLoss'] = testLoss_list
    
    data['trainAcc'] = trainAcc_list
    data['validAcc'] = validAcc_list
    data['testAcc'] = testAcc_list
    
    return data

if __name__ == '__main__':
    # Run over 100~200 epoches
    rates = (0.005,0.0005, 0.00005)

    for rate in rates:
        rate = rates[1]
        print("Current Rate is", rate)

        data = runFullIteration(B=500, max_iter=50000, wd_lambda=0.0003, learningRate=rate)
        np.save("temp_%.7f" %(rate), data)
