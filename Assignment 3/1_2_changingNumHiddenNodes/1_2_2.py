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
    prevDim = tf.to_int32(X.get_shape()[1])
    std_dev = tf.sqrt(3.0 / tf.to_float(prevDim + numHiddenUnits))

    # Variable Creation
    S = tf.placeholder(tf.float32, [None, numHiddenUnits])
    W = tf.Variable(tf.truncated_normal(shape=[prevDim,numHiddenUnits], stddev=std_dev))
    b = tf.Variable(0.0, [1, numHiddenUnits])
    
    # Graph definition
    S = tf.matmul(X, W) + b  # dim is [None, numHiddenUnits]
    
    return S, W


def buildGraph(numLayers, numHiddenUnits, learningRate):
    """Build neural network model with ReLU activation functions"""
    
    # Variable creation
    X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
    X_flatten = tf.reshape(X, [-1, 28*28])
    y_target = tf.placeholder(tf.float32, name='target_y')
    y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis=-1)
    Lambda = tf.placeholder("float32", name='Lambda')
    
    # Graph definition
    # Input <=> Hidden
    S1, W1 = layerWiseBuildingBlock(X_flatten, numHiddenUnits)
    thetaS1 = tf.nn.relu(S1)

	S2, W2 = layerWiseBuildingBlock(thetaS1, numHiddenUnits)
	thetaS2 = tf.nn.relu(S2)
    
    # Hidden <=> Output
    S3, W3 = layerWiseBuildingBlock(thetaS2, 10)
    
    # Final output layer
    y_logit = tf.nn.relu(S3)
    y_predicted = tf.nn.softmax(y_logit)
    
    # Error and accuracy definition
    crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                      labels=y_onehot, logits=y_logit),
                                      name='mean_cross_entropy')
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted, -1),
                                             tf.to_int64(y_target))))
    weightLoss = (tf.reduce_sum(W1*W1) + tf.reduce_sum(W2*W2) + tf.reduce_sum(W3*W3)) * Lambda * 0.5
    loss = crossEntropyError + weightLoss
    
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=loss)
    
    return X, y_target, y_predicted, crossEntropyError, train, Lambda, acc


def runFullIteration(B, max_iter, wd_lambda, learningRate, numHiddenUnits):
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
                print("Iter: %3d, CrossEntropyError: %4.2f" % (step, err))

    data = {}
    data['trainLoss'] = trainLoss_list
    data['validLoss'] = validLoss_list
    data['testLoss'] = testLoss_list
    
    data['trainAcc'] = trainAcc_list
    data['validAcc'] = validAcc_list
    data['testAcc'] = testAcc_list
    
    return data


##############################################################################################
########################################## MAIN ##############################################
##############################################################################################

data = runFullIteration(B=500, max_iter=20000, wd_lambda=0.0001, learningRate=0.001, numHiddenUnits=500)
np.save("double_temp_%.7f" %(rate), data)
#data2 = np.load("temp.npy")
#print(data2.item().get('trainLoss'))

