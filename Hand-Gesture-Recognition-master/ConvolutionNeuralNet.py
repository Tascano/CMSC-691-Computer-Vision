import time
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

# simple function for image scaling to (nR x nC) size
def scale(im, nR, nC):
  nR0 = len(im)     # source number of rows
  nC0 = len(im[0])  # source number of columns
  return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]

# defining the various  activation functions
def softmax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))
def ReLU(x):
    return np.maximum(np.zeros(x.shape), x)
def leaky_ReLU(x):
    return np.maximum(0.01*x, x)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# defining derivatives of the respective activation functions
def der_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0
def der_leaky_ReLU(x):
    if x >= 0:
        return 1
    else:
        return 0.01
def der_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def der_tanh(x):
    return (1-(np.power(tanh(x),2)))

'''
Neural Network Class, and its respective methods
'''

class NeuralNetwork:

    def __init__(self, x1, y1, x2, y2, n):

        # base settings
        self.debug = False
        self.train_data = x1
        self.train_label = y1
        self.test_data = x2
        self.test_label = y2
        self.max_iterations = n
        self.learning_rate = 0.005
        self.height = len(self.train_data[0])
        self.width = len(self.train_data[0][0])
        self.num_of_classes = (np.max(self.train_label)+1)
        self.num_of_input = len(self.train_data[0].flatten())

        # defining the onehot vectors
        self.onehot = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(self.num_of_classes):
            self.onehot[i][i] = 1

        # settings for first convolutional layer
        self.first_num_of_filters = 50
        self.first_filter_height = 50
        self.first_filter_width = 50
        self.first_layer_filter = np.random.normal(size=(self.first_num_of_filters, self.first_filter_height, self.first_filter_width)) * math.sqrt(2 / len(self.train_data[0].flatten()))

        # settings for second convolutional layer
        self.second_num_of_filters = 30
        self.second_filter_height = 5
        self.second_filter_width = 5
        self.second_layer_filter = np.random.normal(size=(self.second_num_of_filters, self.second_filter_height, self.second_filter_width)) * math.sqrt(2 / len(self.train_data[0].flatten()))

        # settings for first connected layer
        self.first_layer_num = 35

        self.first_layer_input = np.zeros(self.first_layer_num)
        self.first_layer_output = np.zeros(self.first_layer_input.shape)
        self.first_layer_bias = np.zeros((self.first_layer_input.shape))
        self.first_layer_weights = np.random.normal(size=(self.first_layer_num, (self.first_num_of_filters * ((self.height - self.first_filter_height + 1) // 2) * ((self.width - self.first_filter_width + 1) // 2)))) * math.sqrt(2 / (self.first_num_of_filters * ((self.height - self.first_filter_height + 1) // 2) * ((self.width - self.first_filter_width + 1) // 2)))

        # settings for final connected layer
        self.final_layer_input = np.zeros(self.num_of_classes)
        self.final_layer_output = np.zeros(self.final_layer_input.shape)
        self.final_layer_bias = np.zeros(self.final_layer_input.shape)
        self.final_layer_weights = np.random.normal(size=(self.num_of_classes, self.first_layer_num)) * math.sqrt(2 / self.first_layer_num)

    def cyclic_learning_rate(self, iterations):
        return 0.002*abs((iterations%10)-5)+0.0025

    def exponential_learning_rate(self, iterations):
        return 0.01*np.exp(-0.05*iterations)

    def exponential_cyclic_learning_rate(self, iterations):
        return (1/(30*iterations+400))*abs((iterations%10)-5)+0.0025

    def connected_layer(self, data4, weights, bias):
        input = np.zeros(bias.shape)
        for i in range(len(bias)):
            input[i] = np.dot(data4, weights[i]) + bias[i]
        return input

    def convolutional_layer_2D_input(self, data5, filters):
        conv_output = np.zeros((len(filters), self.height - self.first_filter_height + 1, self.width - self.first_filter_width + 1))
        for i in range(len(filters)):
            for j in range(self.height - self.first_filter_height + 1):
                for k in range(self.width - self.first_filter_width + 1):
                    conv_output[i][j][k] = np.sum(filters[i] * data5[j:j + self.first_filter_height, k:k + self.first_filter_width])
        return conv_output

    def conv_backprop_2D_input(self, data3, loss1):
        d_loss_d_filters = np.zeros(self.first_layer_filter.shape)
        for i in range(len(loss1)):
            for j in range(len(loss1[0])):
                for k in range(len(loss1[0][0])):
                    d_loss_d_filters[i] += loss1[i][j][k]* data3[j:j+self.first_filter_height, k:k + self.first_filter_width]
        return d_loss_d_filters

    def convolutional_layer_3D_input(self, data7, filters):
        conv_output = np.zeros((len(filters), len(data7[0]) - self.second_filter_height + 1, self.width - self.second_filter_width + 1))
        for i in range(len(filters)):
            for j in range(len(data7[0]) - self.second_filter_height + 1):
                for k in range(len(data7[0][0]) - self.second_filter_width + 1):
                    conv_output[i][j][k] = np.sum(filters[i] * data7[:, j:j + self.second_filter_height, k:k + self.second_filter_width])
        return conv_output

    def conv_backprop_3D_input(self, data3, loss1):
        d_loss_d_filters = np.zeros(self.second_layer_filter.shape)
        for i in range(len(loss1)):
            for j in range(len(loss1[0])):
                for k in range(len(loss1[0][0])):
                    d_loss_d_filters[i] += loss1[i][j][k]* data3[:, j:j+self.second_filter_height, k:k + self.second_filter_width]
        return d_loss_d_filters

    def maxpool_layer(self, data1, size):
        maxpool_output = np.zeros((len(data1), len(data1[0])//size, len(data1[0][0])//size))
        for i in range(len(data1)):
            for j in range(len(data1[0])//size):
                for k in range(len(data1[0][0])//size):
                    if j == (len(data1[0])//size)-1 and k == (len(data1[0][0])//size)-1:
                        maxpool_output[i][j][k] = np.max(data1[i][j*size:,k*size:])
                    elif j == (len(data1[0])//size)-1:
                        maxpool_output[i][j][k] = np.max(data1[i][j*size:,k*size:k*size+size])
                    elif k == (len(data1[0][0])//size)-1:
                        maxpool_output[i][j][k] = np.max(data1[i][j*size:j*size+size,k*size:])
                    else:
                        maxpool_output[i][j][k] = np.max(data1[i][j*size:j*size+size, k*size:k*size+size])
        return maxpool_output

    def maxpool_backprop(self, data2, loss, size):
        d_loss_d_maxpool_input = np.zeros((data2.shape))
        for i in range(len(loss)):
            for j in range(len(loss[0])):
                for k in range(len(loss[0][0])):
                    if j == (len(loss[0])-1) and k == (len(loss[0][0])-1):
                        position = np.unravel_index(np.argmax(data2[i][j*size:,k*size:]), data2[i][j*size:,k*size:].shape)
                    elif j == (len(loss[0])-1):
                        position = np.unravel_index(np.argmax(data2[i][j*size:,k*size:k*size+size]), data2[i][j*size:,k*size:k*size+size].shape)
                    elif k == (len(loss[0][0])-1):
                        position = np.unravel_index(np.argmax(data2[i][j*size:j*size+size,k*size:]), data2[i][j*size:j*size+size,k*size:].shape)
                    else:
                        position = np.unravel_index(np.argmax(data2[i][j*size:j*size+size, k*size:k*size+size]), data2[i][j*size:j*size+size, k*size:k*size+size].shape)
                    d_loss_d_maxpool_input[i][j * size + position[0]][k * size + position[1]] = loss[i][j][k]
        return d_loss_d_maxpool_input

    def train(self):
        for iterations in range(self.max_iterations):
            for i in range(len(self.train_data)):

                # settings for derivatives of first layer
                self.d_loss_d_first_layer_weights = np.zeros(self.first_layer_weights.shape)
                self.d_loss_d_first_layer_bias = np.zeros(self.first_layer_bias.shape)

                # settings for derivatives of final layer
                self.d_loss_d_output = np.zeros(self.num_of_classes)
                self.d_loss_d_final_layer_weights = np.zeros(self.final_layer_weights.shape)
                self.d_loss_d_final_layer_bias = np.zeros(self.final_layer_bias.shape)

                # settings for derivatives of convolution and maxpool layer
                self.d_loss_d_ReLU_input = np.zeros((self.first_num_of_filters * ((self.height - self.first_filter_height + 1) // 2) * ((self.width - self.first_filter_width + 1) // 2)))

                # print(self.train_data[i])

                # forward propogation
                self.first_conv_output = self.convolutional_layer_2D_input(self.train_data[i], self.first_layer_filter)
                if self.debug == True:
                    print("Convolution Output")
                    print(self.first_conv_output)
                    time.sleep(2)
                self.first_maxpool_output = self.maxpool_layer(self.first_conv_output, 2)
                if self.debug == True:
                    print("MaxPool Output")
                    print(self.first_maxpool_output)
                    time.sleep(2)
                self.first_ReLU_output = leaky_ReLU(self.first_maxpool_output)
                if self.debug == True:
                    print("ReLU Output")
                    print(self.first_ReLU_output)
                    time.sleep(2)

                self.second_conv_output = self.convolutional_layer_3D_input(self.first_ReLU_output, self.second_layer_filter)
                self.second_maxpool_output = self.maxpool_layer(self.second_conv_output, 2)
                self.second_ReLU_output = leaky_ReLU(self.second_maxpool_output)

                self.first_layer_input = self.connected_layer(self.second_ReLU_output.flatten(), self.first_layer_weights, self.first_layer_bias)
                self.first_layer_output = leaky_ReLU(self.first_layer_input)
                self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
                self.final_layer_output = softmax(self.final_layer_input)

                if self.debug == True:
                    print("Final Output")
                    print(self.final_layer_output)
                    print(self.onehot[train_label[i]])
                    time.sleep(2)

                if self.debug == True:
                    print("Loss")
                    loss = -(np.log(self.final_layer_output[self.train_label[i]])+0.000000000001)
                    print(loss)

                # backward propogation
                for k in range(self.num_of_classes):
                    self.d_loss_d_output[k] += self.final_layer_output[k] - self.onehot[self.train_label[i]][k]
                    self.d_loss_d_final_layer_bias[k] += self.d_loss_d_output[k]
                    for l in range(self.first_layer_num):
                        self.d_loss_d_final_layer_weights[k][l] += self.d_loss_d_output[k] * self.first_layer_output[l]
                        self.d_loss_d_first_layer_bias[l] += self.d_loss_d_output[k] * self.final_layer_weights[k][l] * der_leaky_ReLU(self.first_layer_input[l])
                        for m in range(len(self.first_ReLU_output.flatten())):
                            self.d_loss_d_first_layer_weights[l][m] += self.d_loss_d_output[k] * self.final_layer_weights[k][l] * der_leaky_ReLU(self.first_layer_input[l]) * self.first_ReLU_output.flatten()[m]
                            self.d_loss_d_ReLU_input[m] += self.d_loss_d_output[k] * self.final_layer_weights[k][l] * der_leaky_ReLU(self.first_layer_input[l]) * self.first_layer_weights[l][m] * der_leaky_ReLU(self.second_maxpool_output.flatten()[m])

                self.d_loss_d_second_maxpool_output = self.d_loss_d_ReLU_input.reshape((self.second_maxpool_output.shape))
                if self.debug == True:
                    print(self.d_loss_d_maxpool_output)
                    time.sleep(2)

                self.d_loss_d_second_maxpool_input = self.maxpool_backprop(self.first_conv_output, self.d_loss_d_second_maxpool_output, 2)
                if self.debug == True:
                    print(self.d_loss_d_maxpool_input)
                    time.sleep(2)

                self.d_loss_d_second_filters = self.conv_backprop_2D_input(self.train_data[i], self.d_loss_d_second_maxpool_input)
                if self.debug == True:
                    print(self.d_loss_d_filters)
                    time.sleep(2)



                # Updating the weights
                self.first_layer_weights -= self.exponential_cyclic_learning_rate(iterations) * self.d_loss_d_first_layer_weights
                self.first_layer_bias -= self.exponential_cyclic_learning_rate(iterations) * self.d_loss_d_first_layer_bias
                self.final_layer_weights -= self.exponential_cyclic_learning_rate(iterations) * self.d_loss_d_final_layer_weights
                self.final_layer_bias -= self.exponential_cyclic_learning_rate(iterations) * self.d_loss_d_final_layer_bias
                self.second_layer_filter -= self.exponential_cyclic_learning_rate(iterations) * self.d_loss_d_second_filters
                print(str(i + 1) + " images done!")
            print(self.evaluate_train(), self.evaluate_test())
            print(str(iterations + 1) + " epochs done!")

    # class method to evaluate the model's accuracy on the train set
    def evaluate_train(self):
        correct = 0
        for i in range(len(self.train_data)):
            self.first_conv_output = self.convolutional_layer_2D_input(self.train_data[i], self.first_layer_filter)
            self.first_maxpool_output = self.maxpool_layer(self.first_conv_output, 2)
            self.first_ReLU_output = leaky_ReLU(self.first_maxpool_output)
            self.first_layer_input = self.connected_layer(self.first_ReLU_output.flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = leaky_ReLU(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.train_label[i]:
                correct += 1
        return (correct / len(self.train_data)) * 100

    # class method to evaluate the model's accuracy on the test set
    def evaluate_test(self):
        correct = 0
        for i in range(len(self.test_data)):
            self.first_conv_output = self.convolutional_layer_2D_input(self.test_data[i], self.first_layer_filter)
            self.first_maxpool_output = self.maxpool_layer(self.first_conv_output, 2)
            self.first_ReLU_output = leaky_ReLU(self.first_maxpool_output)
            self.first_layer_input = self.connected_layer(self.first_ReLU_output.flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = leaky_ReLU(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            if np.argmax(self.final_layer_output) == self.test_label[i]:
                correct += 1
        return (correct / len(self.test_data)) * 100

    # class method to comprehensively evaluate the model's performance on the training set
    def evaluate_confusion_train(self):
        print("\nEvaluating performance on the Train Set:")
        print("\nConfusion Matrix:")
        self.confusion_matrix = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(len(self.train_data)):
            self.first_conv_output = self.convolutional_layer_2D_input(self.train_data[i], self.first_layer_filter)
            self.first_maxpool_output = self.maxpool_layer(self.first_conv_output, 2)
            self.first_ReLU_output = leaky_ReLU(self.first_maxpool_output)
            self.first_layer_input = self.connected_layer(self.first_ReLU_output.flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = leaky_ReLU(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            self.confusion_matrix[np.argmax(self.final_layer_output)][self.train_label[i]] += 1
        print(self.confusion_matrix)

        # define additional metrics
        self.true_positives = np.zeros(self.num_of_classes)
        self.false_positives = np.zeros(self.num_of_classes)
        self.false_negatives = np.zeros(self.num_of_classes)
        self.precision = np.zeros(self.num_of_classes)
        self.recall = np.zeros(self.num_of_classes)

        for j in range(self.num_of_classes):
            self.true_positives[j] = self.confusion_matrix[j][j]
            self.false_positives[j] = np.sum(self.confusion_matrix[:,j]) - self.true_positives[j]
            self.false_negatives[j] = np.sum(self.confusion_matrix[j,:]) - self.true_positives[j]
            self.recall[j] = self.true_positives[j]/(self.true_positives[j]+self.false_negatives[j])
            self.precision[j] = self.true_positives[j] / (self.true_positives[j] + self.false_positives[j])

        print("Overall Accuracy: " + str((np.sum(self.true_positives) / len(self.train_data)) * 100) + "%")

        print("\nMacro-Averaged Stats: ")
        self.macro_recall = np.sum(self.recall)/self.num_of_classes
        self.macro_precision = np.sum(self.precision)/self.num_of_classes
        self.macro_f1_score = (2*self.macro_recall*self.macro_precision)/(self.macro_recall+self.macro_precision)
        print("Macro-Averaged Precision: " + str(self.macro_precision))
        print("Macro-Averaged Recall: " + str(self.macro_recall))
        print("Macro-Averaged F1 Score: " + str(self.macro_f1_score))

        print("\nMicro-Averaged Stats: ")
        self.micro_recall = np.sum(self.true_positives)/(np.sum(self.true_positives)+np.sum(self.false_negatives))
        self.micro_precision = np.sum(self.true_positives)/(np.sum(self.true_positives)+np.sum(self.false_positives))
        self.micro_f1_score = (2*self.micro_recall*self.micro_precision)/(self.micro_recall+self.micro_precision)
        print("Micro-Averaged Precision: " + str(self.micro_precision))
        print("Micro-Averaged Recall: " + str(self.micro_recall))
        print("Micro-Averaged F1 Score: " + str(self.micro_f1_score))

    # class method to comprehensively evaluate the model's performance on the test set
    def evaluate_confusion_test(self):
        print("\nEvaluating performance on the Test Set:")
        print("\nConfusion Matrix:")
        self.confusion_matrix = np.zeros((self.num_of_classes, self.num_of_classes))
        for i in range(len(self.test_data)):
            self.first_conv_output = self.convolutional_layer_2D_input(self.test_data[i], self.first_layer_filter)
            self.first_maxpool_output = self.maxpool_layer(self.first_conv_output, 2)
            self.first_ReLU_output = leaky_ReLU(self.first_maxpool_output)
            self.first_layer_input = self.connected_layer(self.first_ReLU_output.flatten(), self.first_layer_weights, self.first_layer_bias)
            self.first_layer_output = leaky_ReLU(self.first_layer_input)
            self.final_layer_input = self.connected_layer(self.first_layer_output, self.final_layer_weights, self.final_layer_bias)
            self.final_layer_output = softmax(self.final_layer_input)
            self.confusion_matrix[np.argmax(self.final_layer_output)][self.test_label[i]] += 1
        print(self.confusion_matrix)

        # define additional metrics
        self.true_positives = np.zeros(self.num_of_classes)
        self.false_positives = np.zeros(self.num_of_classes)
        self.false_negatives = np.zeros(self.num_of_classes)
        self.precision = np.zeros(self.num_of_classes)
        self.recall = np.zeros(self.num_of_classes)

        for j in range(self.num_of_classes):
            self.true_positives[j] = self.confusion_matrix[j][j]
            self.false_positives[j] = np.sum(self.confusion_matrix[:,j]) - self.true_positives[j]
            self.false_negatives[j] = np.sum(self.confusion_matrix[j,:]) - self.true_positives[j]
            if self.true_positives[j] == 0 and self.false_negatives[j] == 0:
                self.recall[j] = 0
            else:
                self.recall[j] = self.true_positives[j]/(self.true_positives[j]+self.false_negatives[j])
            if self.true_positive[j] == 0 and self.false_positives[j] == 0:
                self.precision[j] = 0
            else:
                self.precision[j] = self.true_positives[j] / (self.true_positives[j] + self.false_positives[j])

        print("Overall Accuracy: " + str((np.sum(self.true_positives) / len(self.test_data)) * 100) + "%")

        print("\nMacro-Averaged Stats: ")
        self.macro_recall = np.sum(self.recall)/self.num_of_classes
        self.macro_precision = np.sum(self.precision)/self.num_of_classes
        self.macro_f1_score = (2*self.macro_recall*self.macro_precision)/(self.macro_recall+self.macro_precision)
        print("Precision: " + str(self.macro_precision))
        print("Recall: " + str(self.macro_recall))
        print("F1 Score: " + str(self.macro_f1_score))

        print("\nMicro-Averaged Stats: ")
        self.micro_recall = np.sum(self.true_positives)/(np.sum(self.true_positives)+np.sum(self.false_negatives))
        self.micro_precision = np.sum(self.true_positives)/(np.sum(self.true_positives)+np.sum(self.false_positives))
        self.micro_f1_score = (2*self.micro_recall*self.micro_precision)/(self.micro_recall+self.micro_precision)
        print("Precision: " + str(self.micro_precision))
        print("Recall: " + str(self.micro_recall))
        print("F1 Score: " + str(self.micro_f1_score))


'''
Main Code of the program
'''

# importing the data and applying labels to it
train_data = []
train_label = []
for i in range(len(os.listdir("Pics"))):
    print("Hand gesture "+os.listdir("Pics")[i]+" is assigned to class " + str(i)+".")
    for j in range(len(os.listdir("Pics/"+os.listdir("Pics")[i]))):
        img = imread("Pics/"+os.listdir("Pics")[i]+"/Pic (" + str(j+1) + ").jpg")
        # img1 = scale(img,36,64) # to downscale the images
        # img1 = np.array(img1)
        train_data.append(img)
        train_label.append(i)

# shuffling the data and creating the train and test split
temp = list(zip(train_data, train_label))
random.shuffle(temp)
temp_train = temp[:math.floor(0.75*len(temp))]
temp_test = temp[math.floor(0.75*len(temp)):]

train_data, train_label = zip(*temp_train)
test_data, test_label = zip(*temp_test)
train_data = np.array(train_data)/255
train_label = np.array(train_label)
test_data = np.array(test_data)/255
test_label = np.array(test_label)

# create the instance of the CNN
CNN = NeuralNetwork(train_data, train_label, test_data, test_label, 50)

# train the CNN
CNN.train()

# Evaluate the CNN
CNN.evaluate_confusion_train()
CNN.evaluate_confusion_test()
