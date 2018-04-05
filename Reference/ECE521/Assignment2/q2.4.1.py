import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)
#2.2.1


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

def compute_weighted_sum(previous_layer, n_units):
    
    #xavier initialization
    dev = tf.sqrt(3/(n_units[0] + n_units[1]))
    initW = tf.truncated_normal(shape=[n_units[0],n_units[1]], stddev = dev )
   
    #weight matrix
    weight_matrix = tf.Variable(initW, [n_units[0], n_units[1]], name = "weight_matrix")
    
    #bias (broadcasted)
    bias = tf.Variable(tf.truncated_normal(shape=[1,n_units[1]],stddev=dev), [1,n_units[1]], name="bias")
    
    #weighted sum
    Z = tf.matmul(previous_layer,weight_matrix) + bias
    
    return Z, weight_matrix
  
def build_neural_network():
 
    # Network Hyperparameters
    n_hidden_1 = 1000 # 1st layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    regularization = 3E-4
    learning_rate = 8E-4
    keep_prob = 0.5  #dropout
    
    #weight matrices list 
    weight_matrices = [ [[]] for i in range(2) ]
    
    # weighted sum 
    Z =  [ [[]] for i in range(2) ]
    
    #inputs and targets
    X = tf.placeholder(tf.float32, [None,784], name = "reg" )
    target = tf.placeholder(tf.float32, [None,10], name = "target") 
    
    #hidden_layer_1
    Z[0], weight_matrices[0]= compute_weighted_sum(X, (n_input, n_hidden_1) )
    hidden_layer = tf.nn.relu(Z[0]) 
    
    #dropout hidden layer
    hidden_layer_dropout = tf.nn.dropout(hidden_layer,keep_prob)
    
    #output_layer
    Z[1], weight_matrices[1] = compute_weighted_sum(hidden_layer_dropout, (n_hidden_1, n_classes) )
    Y_predicted =  tf.nn.softmax(Z[1]) 
    
    # cross entropy loss
    loss_CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z[1], labels=target))
    
    # deadweight (regularization) loss
    loss_w = 0
    for i in range(len(weight_matrices)):
        loss_w += tf.reduce_sum(tf.square(weight_matrices[i]))*regularization/2
        
    # sum loss    
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
    
    error = 1 - accuracy
    return train, total_loss, X, result, target, accuracy, loss_CE,error
    


if __name__ == '__main__':
   
    
    #Hyperparameter 
    batch_size = 500 
    
    #number of epochs
    epoch = 150
    
    #display step
    display_step = 1
    
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data() 
    train, total_loss, X, result, target, accuracy, loss_CE, error= build_neural_network()
    
    init = tf.global_variables_initializer()
    randIdx = np.arange(15000)
    sess =  tf.InteractiveSession() 
    sess.run(init)         
  

   
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()      
    #saver.restore(sess, "/Users/winst/Google Drive/3S/ECE521/Assignment2/checkpoints/model.ckpt")
    #print("Model restored.")    
    
    
   
    
    # record accuracy and loss for plotting
    training_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    
    train_cost_list = [] 
    valid_cost_list = []
    test_cost_lost = []
    
    train_err_list = [] 
    valid_err_list = []
    test_err_lost = []    
    
    input_x = trainData[randIdx[0 : batch_size]]
    input_y = trainTarget[randIdx[0 :batch_size]]       
    
    before_train_acc, train_cost, t_predicted = sess.run([accuracy, loss_CE, result], feed_dict= {X: input_x, target: input_y})  
    
    
    print("before training: train acc: %s "%(before_train_acc))
    
    len_of_x_plot = 0

    
     ##train neural net
    for i in range(epoch):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]]   
            
            #adam descent on training data
            _, train_acc, train_cost, t_predicted,train_err = sess.run([train, accuracy, loss_CE, result,error], feed_dict= {X: input_x, target: input_y})    
            
        if(i%display_step==0):    
            
            len_of_x_plot += 1
            
            #query validation 
            valid_acc, valid_cost,valid_err = sess.run([accuracy, loss_CE,error], feed_dict = {X:validData, target:validTarget})

            #query test
            test_acc, test_cost,test_err = sess.run([accuracy, loss_CE,error], feed_dict = {X:testData, target:testTarget})
    
            print("i: %s "%(i))
            print("training accuracy: %s"%(train_acc))
            print("valid accuracy: %s"%(valid_acc))
            print("test accuracy: %s"%(test_acc))            
    
    
            # record for plots 
            training_acc_list.append(train_acc)  
            valid_acc_list.append(valid_acc)
            test_acc_list.append(test_acc)
            
            train_cost_list.append(train_cost) 
            valid_cost_list.append(valid_cost) 
            test_cost_lost.append(test_cost) 

            train_err_list.append(train_err) 
            valid_err_list.append(valid_err) 
            test_err_lost.append(test_err)   
   
    
    #create plots
    #x_plot = [i for i in range(epoch*int(len(trainData)/batch_size))]
    
    x_plot= [i for i in range(len_of_x_plot)]
    plt.figure(1) 
    plt.plot(x_plot,training_acc_list, label="training acc")
    plt.plot(x_plot, valid_acc_list, label = "validationg acc")
    plt.plot(x_plot, test_acc_list, label = "test acc")
    plt.legend()
    
    #second figure for loss
    plt.figure(2) 
    plt.plot(x_plot,train_cost_list, label = "training loss")
    plt.plot(x_plot, valid_cost_list, label = "validationg loss")
    plt.plot(x_plot, test_cost_lost, label = "test loss")
    plt.legend()
    

    #third figure for error
    plt.figure(3) 
    plt.plot(x_plot,train_err_list, label = "training error")
    plt.plot(x_plot, valid_err_list, label = "validationg error")
    plt.plot(x_plot, test_err_lost, label = "test error")
    plt.legend()    




  
    ##query test
    #test_acc, test_cost, t_predicted = sess.run([accuracy, total_loss,result], feed_dict = {X:testData, target:testTarget})    
    
    
    #print("FINAL test accuracy: %s"%(test_acc))       
    
    #bingo_count = 0
    #total_comp = 0
    #for i in range(len(t_predicted)):
        #total_comp += 1
        #if(i%10 ==0):
            #print ("i: %s"%(i))
            #print(t_predicted[i])
            #print(testTarget[i])
        
        #match = True
        #for j in range(len(testTarget[i])):
            #if testTarget[i][j] != t_predicted[i][j]:
                #match = False 
        #if match:
            #bingo_count += 1
            #if(i%10==0):
                #print("they match!")
                #print("bingo count: %s"%(bingo_count))
                #print("total number of comparison %s"%(total_comp))
                #print("\n")
    #print("FINAL test accuracy: %s"%(test_acc))       
    
    #print("FINAL bingo count: %s"%(bingo_count))
    #print("FINAL total number of comparison %s"%(total_comp))
    #print("Manual accuracy: %s"%(bingo_count/total_comp))
    #print("\n")    
    
    save_path = saver.save(sess, "/Users/winst/Google Drive/3S/ECE521/Assignment2/checkpoints2/model.ckpt")
    print("Model saved in file: %s" % save_path)
    