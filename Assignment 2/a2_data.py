import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt
import plotly.plotly as py

def drawImage(img):
    plt.figure(figsize = (1,1))
    img = np.reshape(img, (28, 28))
    plt.imshow(img, interpolation='nearest')
    plt.show()

def drawImageRow(imgRow):
    f, axarr = plt.subplots(1, np.shape(imgRow)[0]);
    for i in range(0, np.shape(imgRow)[0]):
        img = np.reshape(imgRow[i], (28, 28));
        p = axarr[i].imshow(img, interpolation='nearest');
        p.set_cmap('binary')
        axarr[i].set_yticklabels([])
        axarr[i].set_xticklabels([])
    plt.show()

def drawImageSet(imgSet):
    for i in range(0, int(np.shape(imgSet)[0] / 10)):
        drawImageRow(imgSet[i*10:(i+1)*10]);
    

# Load the images as data
def loadNotMNIST():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
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
    return trainData, trainTarget, validData, validTarget, testData, testTarget