import csv
import random
import math
import operator
import numpy as np


#data_file='C:\Users\Sudip\Documents\Vru\Machine Learning\diabetes.csv'
"""Read the CSV file and generate training and testing datasets"""
def prepare_data(file, split, train_values=[], test_values=[]):
    with open(file, 'rb') as csvfile:
        data_list = csv.reader(csvfile)
        input_dataset = list(data_list)
        #print(input_dataset)


        for x in range(len(input_dataset) - 1):
            """The range is specified as 11 since there are 11 features for the glass data """

            for y in range(11):
                input_dataset[x][y] = float(input_dataset[x][y])
            if random.random() < split:
                train_values.append(input_dataset[x])
            else:
                test_values.append(input_dataset[x])


"""Calculating the euclidean distance between data points to find out the distances"""
def calc_euclideanDistance(inst1, inst2, length):
    distance = 0
    for value in range(length):
        distance += pow((inst1[value] - inst2[value]), 2)
    return math.sqrt(distance)

"""Once the distance is calculated, the closest neighbors are selected."""
def myknnclassify(X, test, k):
    distances = []
    length = len(test) - 1
    for x in range(len(X)):
        dist = calc_euclideanDistance(test, X[x], length)
        distances.append((X[x], dist))
    distances.sort(key=operator.itemgetter(1))
    nearestneighbors = []
    #print(distances)
    for x in range(k):
        nearestneighbors.append(distances[x][0])
        #print(nearestneighbors)
    print(nearestneighbors)
    return nearestneighbors



"""Calculates the votes of the nearest neighbor in order to predict the class for each test instances."""
def getVotes(nearestneighbors):
    classVotes = {}
    for x in range(len(nearestneighbors)):
        response = nearestneighbors[x][-1]
        #print(response)
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print(sortedVotes)
    return sortedVotes[0][0]


"""Regression function that calculated the average value of the neighbors to predict the target value of the instance in the test dataset"""
def myknnregressor(nearestneighbors):
    value1=0
    for y in range(len(nearestneighbors)):
            value1= value1+nearestneighbors[y][3]
    value2=value1/4
    print("REGRESSION :> Predicted: SODIUM VALUE\t")
    print(value2)


"""Function to calculate the efficiency of the knn algo in order to measure its correctness or how many instances in the test dataset were predicted """
def cal_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


"""Main function that calls all the above functions"""
def main():
    # prepare data
    trainingSet = []
    testSet = []
    """Split ration mentioned to split the instances into training and testing values """
    split = 0.67
    #file=input("Enter the path of the CSV file")
    prepare_data('C:\Users\Sudip\Documents\Vru\Machine Learning\glass.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # generate predictions
    predictions = []
    """Value for K is hard coded as it gave the best accuracy"""
    k = 4

    for x in range(len(testSet)):
        neighbors = myknnclassify(trainingSet, testSet[x], k)
        result = getVotes(neighbors)
        result1= myknnregressor(neighbors)
        predictions.append(result)
        print('REGRESSION:> Actual :SODIUM VALUE\t' + repr(testSet[x][3]))
        print('CLASSIFICATION:> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

        print('\t')
        print('-----------------------------------------------------------------------------------------------------------------------------------------------------')
    accuracy = cal_accuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')



main()