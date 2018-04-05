import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt


#question 1.1.3

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

    W = tf. Variable(tf.truncated_normal(shape =[784,1], stddev = 0.5), [784,1], name = "weight")
    b = tf.Variable(tf.truncated_normal(shape =[1,], stddev = 0.5), name = "bias")
    z = tf.matmul(X,W) + b 
    t_predicted = tf.sigmoid(z) 
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(z, t, name="ce_loss"))
    
    #optimzer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(loss=cross_entropy_loss) 
    
    #concat 1 for bias for normal X so that normal equation can be applied 
    shape = tf.shape(X)
    bias_input = tf.ones([shape[0], 1], tf.float32)
    normal_X = tf.concat(1,[bias_input, X])
    
    #weights obtained from normal equation 
    normal_W = tf.matmul(tf.matrix_inverse(tf.matmul(tf.matrix_transpose(normal_X),normal_X)),tf.matmul(tf.matrix_transpose(normal_X),t))
    
    # re-enter the weights compute above to compute l2 loss; Notice weight has dimension 785 including bias
    temp_weights = tf.placeholder(tf.float32, [785,1], name= "temp_weights")
    t_normal_predicted = tf.matmul(normal_X,temp_weights)
    
    #probe prediction, accuracy and square loss from linear regression
    linear_result =  tf.cast(tf.greater(t_normal_predicted, 0.5),tf.int32)
    linear_accuracy =  tf.reduce_mean(tf.cast(tf.equal(linear_result, tf.cast(t,tf.int32)),tf.float32))
    square_loss = tf.reduce_mean(tf.square(t_normal_predicted-t))

    #logistic results
    logistic_result = tf.cast(tf.greater(t_predicted, 0.5),tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logistic_result, tf.cast(t,tf.int32)),tf.float32))
    
    return train, t, X, W, b, cross_entropy_loss,accuracy,t_predicted,square_loss,normal_W,temp_weights, t_normal_predicted, linear_accuracy, linear_result



if __name__ == '__main__':
    
    
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data()   
    train, t, X, W, b, cross_entropy_loss,accuracy,t_predicted,square_loss,normal_W,temp_weights, t_normal_predicted, linear_accuracy,linear_result = build_graph()
    init = tf.initialize_all_variables()  
    randIdx = np.arange(3500)
    sess =  tf.InteractiveSession() 
    sess.run(init)     
    
    #without replacement 

    batch_size = 1000
    
    valid_accuracy_list = []
    test_accuracy_list = []
    valid_loss_list = []
    test_loss_list = []
    
    
   
    #train CE
    for i in range(1000):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]] 
            _,temp, CE = sess.run([train, W, cross_entropy_loss], feed_dict= {X: input_x, t: input_y})
    
    train_CE_acc = sess.run([accuracy], feed_dict= {X: trainData, t: trainTarget})
    print("training classification accuracy (CE): %s" %(train_CE_acc)) 
    
    valid_CE_acc = sess.run([accuracy], feed_dict= {X: validData, t: validTarget})
    print("validation classification accuracy (CE): %s" %(valid_CE_acc))    
    
    test_CE_acc = sess.run([accuracy], feed_dict= {X: testData, t: testTarget})
    print("test classification accuracy (CE): %s" %(test_CE_acc))       
   
    
    #finished training; now look at classification results using the trained weights; and corresponding loss if assume every test examples is from the negative class
    dummy_target = [[0] for i in range(len(testData))]
    CE_list = [] 
    prediction_list = [] 
    MSE_list = [] 
    t_normal_prediction_list = []
    
    #train square loss 
    l2_weight = sess.run([normal_W], feed_dict= {X: trainData, t: trainTarget}) 
    
    training_MSE_loss, train_linear_acc, train_linear_prediction = sess.run([square_loss, linear_accuracy, linear_result], feed_dict= {X: trainData, t: trainTarget, temp_weights: l2_weight[0]}) 
    
    print("training classification accuracy (normal): %s" %(train_linear_acc))
    
    valid_MSE_loss, valid_linear_acc, valid_linear_prediction = sess.run([square_loss, linear_accuracy, linear_result], feed_dict= {X: validData, t: validTarget, temp_weights: l2_weight[0]}) 
    
    print("validation classification accuracy (normal): %s" %(valid_linear_acc))
    
    test_MSE_loss, test_linear_acc, test_linear_prediction = sess.run([square_loss, linear_accuracy, linear_result], feed_dict= {X: testData, t: testTarget, temp_weights: l2_weight[0]}) 
        
    print("test classification accuracy (normal): %s" %(test_linear_acc))
    
    
    for i in range(len(testData)):
        CE, MSE, prediction, t_normal_prediction = sess.run([cross_entropy_loss, square_loss, t_predicted, t_normal_predicted],feed_dict = {X:[testData[i]], temp_weights: l2_weight[0], t: [dummy_target[i]]})
        CE_list.append(CE)
        MSE_list.append(MSE)
        prediction_list.append(prediction[0][0])
        t_normal_prediction_list.append(t_normal_prediction[0][0])
    
    plt.plot(prediction_list, CE_list,'ro', label = "Cross Entropy Loss")
    plt.plot(t_normal_prediction_list, MSE_list, 'bx', label = "Square Loss with Normal Equation")
    plt.xlabel('prediction y_hat in [0,1]')
    plt.ylabel('loss')
    plt.legend()
    MSE_list.sort()

