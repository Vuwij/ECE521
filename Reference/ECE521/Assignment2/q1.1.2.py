import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


#question 1.1.2

# Load 2 class 
def load_data():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        #print("lenth before dataInx of Data %s"%(len(Data)))
        print(len(Data[-1][-1]))
        Data = Data[dataIndx]/255.
        Data = Data.reshape(-1,784)
        print(len(Data[-1]))
        #print("lenth after dataInx of Data %s"%(len(Data)))
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return  trainData, trainTarget, validData, validTarget, testData, testTarget
    
#train on Two-Class Dataset data size for 1 training example = 28*28
def build_graph():
    t = tf.placeholder(tf.float32, [None,1], name ="label")
    X = tf.placeholder(tf.float32, [None,784], name ="x") 
    regularization = 0.01

    W = tf. Variable(tf.truncated_normal(shape =[784,1], stddev = 0.5), [784,1], name = "weight")
    b = tf.Variable(tf.truncated_normal(shape =[1,], stddev = 0.5), name = "bias")
    z = tf.matmul(X,W) + b 
    t_predicted = tf.sigmoid(z) 
    
    loss_CE = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(z, t, name="loss_d"))
    loss_w = tf.reduce_sum(tf.square(W)) * regularization 
    total_loss = loss_CE + loss_w 

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(loss=total_loss) 
    
    
    result = tf.cast(tf.greater(t_predicted, 0.5),tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(t,tf.int32)),tf.float32))
    
    return train, t, X, W, b, total_loss, loss_CE, accuracy


if __name__ == '__main__':
    

    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()   
    train, t, X, W, b, total_loss, loss_CE, accuracy = build_graph()
    init = tf.initialize_all_variables()  
    randIdx = np.arange(3500)
    sess =  tf.InteractiveSession() 
    sess.run(init)     
    
    #without replacement 

    batch_size = 500
    training_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []
    
    valid_loss_list = []
    test_loss_list = []
    training_loss_list = []
    
    
    max_test_accuracy = 0
    
    sess.run(init) 
    for i in range(200):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]] 
            _,training_accuracy, training_CE = sess.run([train, accuracy, loss_CE], feed_dict= {X: input_x, t: input_y})
            
            training_loss_list.append(training_CE)
            training_accuracy_list.append(training_accuracy)
            
            valid_accuracy, valid_CE = sess.run([accuracy, loss_CE], feed_dict = {X: validData, t: validTarget})
            valid_accuracy_list.append(valid_accuracy)
            valid_loss_list.append(valid_CE)
            
            test_accuracy,test_CE = sess.run([accuracy, loss_CE], feed_dict = {X: testData, t: testTarget})
            
            if max_test_accuracy < test_accuracy:
                max_test_accuracy = test_accuracy
            test_accuracy_list.append(test_accuracy)
            test_loss_list.append(test_CE)            

            print("i: %s, j: %s"%(i,j))
         
    plt.figure(1) 
    x_plot = [i for i in range(len(valid_accuracy_list))]
    plt.plot(x_plot, training_accuracy_list, label = "training accuracy")
    plt.plot(x_plot, valid_accuracy_list, label = "valid accuracy")
    plt.plot(x_plot, test_accuracy_list, label = "test accuracy")
    plt.axis([0,200,0,1])
    plt.legend()
    
    
     #second figure for loss
    plt.figure(2) 
    plt.plot(x_plot,training_loss_list, label = "training loss")
    plt.plot(x_plot, valid_loss_list, label = "validationg loss")
    plt.plot(x_plot, test_loss_list, label = "test loss")
    plt.axis([0,200,0,2])
    plt.legend()
    
    print("max accuracy: %s"%(max_test_accuracy))