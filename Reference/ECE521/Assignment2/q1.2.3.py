import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

def load_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Data = Data.reshape(-1,784)
        Target = Target[randIndx]
        
        #convert Target to one-hot encoding 
        one_hot_target = np.zeros((len(Target), 10))
        one_hot_target[np.arange(len(Target)),Target]  = 1
        
        #split data into train, valid and test
        trainData, trainTarget = Data[:15000], one_hot_target[:15000]
        validData, validTarget = Data[15000:16000], one_hot_target[15000:16000]
        testData, testTarget = Data[16000:], one_hot_target[16000:]    
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def build_graph():
    
    #Hyperparameter
    regularization = 0.01
    n_input = 784
    n_classes = 10
    learning_rate = 0.01 
    
    X = tf.placeholder(tf.float32, [None, n_input], name ="X")
    target = tf.placeholder(tf.float32, [None, n_classes], name = "target")
    
    #xavier initialization
    dev = 3/(n_input + n_classes)
    initW = tf.truncated_normal(shape=[n_input,n_classes], stddev = dev)    
 
 
    # Weights
    weight = tf.Variable(initW, [n_input,n_classes], name="weights")
    
    #Bias (1st dim brocasted to #of training examples)
    initB = tf.zeros([1,n_classes])
    bias = tf.Variable(initB, [1, n_classes], name="bias")
    
    #output 
    z = tf.matmul(X,weight)+bias
    Y_predicted = tf.nn.softmax(z)
    
    #loss
    loss_CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=z, labels=target))
    loss_w = tf.reduce_sum(tf.square(weight)) *regularization/2
    total_loss = loss_CE + loss_w
     
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=total_loss)         
   
    
    # predictions 
    argmax = tf.reshape(tf.argmax(Y_predicted, 1), [-1,1])
    result = tf.one_hot( indices=argmax, depth=10, on_value=1., off_value=0.)
    result = tf.reshape(result, [-1,10])


    # Classification accuracy
    correction_predictions = tf.equal(tf.argmax(Y_predicted, 1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correction_predictions,tf.float32))
    
    return  train, total_loss, loss_CE, X, Y_predicted, target, regularization, accuracy, result    
    
    
if __name__ == '__main__':
   
    
    #Hyperparameter 
    batch_size = 500   
    
    #number of epochs
    epoch = 50
    
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data() 
    train, total_loss, loss_CE, X, Y_predicted, target, regularization, accuracy, result = build_graph()
    
    init = tf.initialize_all_variables()  
    randIdx = np.arange(15000)
    sess =  tf.InteractiveSession() 
    sess.run(init)     
    
    
    # record accuracy and loss for plotting
    training_acc_list = []
    test_acc_list = []
    
    train_cost_list = [] 
    test_cost_lost = []    
  
    for i in range(epoch):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]]
            
             #adam descent on training data
            _, train_acc, train_cost, t_predicted = sess.run([train, accuracy, loss_CE, result], feed_dict= {X: input_x, target: input_y})    
            
            #query test
            test_acc, test_cost = sess.run([accuracy, loss_CE], feed_dict = {X:testData, target:testTarget})
            
            if (j == 0):
                print("i: %s "%(i))
                print("training accuracy: %s"%(train_acc))
                print("test accuracy: %s"%(test_acc))   
                
            # record for plots 
            training_acc_list.append(train_acc)  
            test_acc_list.append(test_acc)
            
            train_cost_list.append(train_cost) 
            test_cost_lost.append(test_cost)             
            
    #create plots
    x_plot = [i for i in range(epoch*int(len(trainData)/batch_size))]
    plt.figure(1) 
    plt.plot(x_plot,training_acc_list, label="training acc")
    plt.plot(x_plot, test_acc_list, label = "test acc")
    plt.legend()
    
    #second figure for loss
    plt.figure(2) 
    plt.plot(x_plot,train_cost_list, label = "training loss")
    plt.plot(x_plot, test_cost_lost, label = "test loss")
    plt.legend()
    
    
    
    
    ##adam descent on training data
    #_, train_acc, train_cost, t_predicted = sess.run([train, accuracy, total_loss, result], feed_dict= {X: trainData[0:batch_size], target: trainTarget[0:batch_size]})    
 
    
    #bingo_count = 0
    #for i in range(len(t_predicted)):
        
        #if(i%10 ==0):
            #print ("i: %s"%(i))
            #print(t_predicted[i])
            #print(trainTarget[i])
        
        #match = True
        #for j in range(len(trainTarget[i])):
            #if trainTarget[i][j] != t_predicted[i][j]:
                #match = False 
        #if match:
            #bingo_count += 1
            #if(i%10==0 or i == 499):
                #print("they match!")
                #print("bingo count: %s"%(bingo_count))
                #print("total number of comparison %s"%i)
                #print("\n")
   