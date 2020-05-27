import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import os
import random

#simple image scaling to (nR x nC) size
def scale(im, nR, nC):
  nR0 = len(im)     # source number of rows
  nC0 = len(im[0])  # source number of columns
  return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]

#defining activation functions
def softmax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))
def ReLU(x):
    return np.maximum(np.zeros(x.shape), x)
def leaky_ReLU(x):
    return np.maximum(0.01*x,x)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#defining derivative of activation functions
def der_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0
def der_leaky_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0.001
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def der_tanh(x):
    return (1-(np.power(tanh(x),2)))

# define the neural network class and functions
class NeuralNetwork:

    def __init__(self, train_data, train_label):
        self.num_of_classes = np.max(train_label) + 1
        self.num_of_input = len(train_data[0].flatten())
        self.bias = np.zeros(self.num_of_classes)
        self.weights = np.zeros((self.num_of_classes, self.num_of_input))
        self.output = np.zeros(self.num_of_classes)
        self.onehot = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

    def train(self, max_iterations):
        self.max_iterations = max_iterations
        print(self.num_of_input)
        print(self.num_of_classes)
        time.sleep(2)
        for iterations in range(self.max_iterations):
            for i in range(len(train_data)):
                for j in range(len(self.output)):
                    self.output[j] = np.dot(self.weights[j],train_data[i].flatten()) + self.bias[j]
                self.output = softmax(self.output)
                # print("output")
                # print(self.output)
                # time.sleep(1)
                loss = -(np.log(self.output[train_label[i]]+0.0000000000000001))
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_w = np.zeros(self.weights.shape)
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] = self.output[k] - self.onehot[train_label[i]][k]
                    for l in range(self.num_of_input):
                        self.d_loss_d_w[k][l] = 0.0005 * self.d_loss_d_output[k] * train_data[i].flatten()[l]
                self.weights -= self.d_loss_d_w
                print(loss)
                # print(self.output)
                print(str(i+1)+" images done!")
            print(str(iterations+1) + " iterations Done!")
            time.sleep(1)

    def evaluate(self, test_data, test_label):
        correct = 0
        for i in range(len(test_data)):
            for j in range(len(self.output)):
                self.output[j] = np.dot(self.weights[j],train_data[i].flatten()) + self.bias[j]
            self.output = softmax(self.output)
            if np.argmax(self.output) == test_label[i]:
                correct += 1
        print("The accuracy of the predictions is " + str((correct/len(test_data))*100) + "%.")

# Read the train and test data and labels
train_data = []
train_label = []
for i in range(len(os.listdir("Pics"))):
    for j in range(len(os.listdir("Pics/"+os.listdir("Pics")[i]))):
        img = imread("Pics/"+os.listdir("Pics")[i]+"/Pic (" + str(j+1) + ").jpg")
        img1 = scale(img,90,160)
        img1 = np.array(img1)
        train_data.append(img1)
        train_label.append(i)
temp = list(zip(train_data, train_label))
random.shuffle(temp)
random.shuffle(temp)
temp_train = temp[0:80]
temp_test = temp[80:]
train_data, train_label = zip(*temp_train)
test_data, test_label = zip(*temp_test)
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
plt.imshow(train_data[1])
plt.show()

#create the NeuralNetwork instance
test = NeuralNetwork(train_data, train_label)
test.train(500)
test.evaluate(train_data, train_label)
test.evalute(test_data, test_label)

