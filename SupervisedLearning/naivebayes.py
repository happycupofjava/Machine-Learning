
from math import log
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import time
#import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np

"""Take file paths and store it in variables:"""
mushroom_instances = 'C:\Users\Sudip\Downloads\mushrooms.csv'
mushroom_features = 'C:\Users\Sudip\Documents\Vru\Machine Learning\mushrooms.names'

"""Initializing arrays to make datasets"""
features_present_in_data = []
features_absent_in_data = []

edible_mushrooms_data = []
poisonous_mushrooms_data = []

edible_train = []
poisonous_train = []

training_data = []
test_data = []

features = []
features_dictionary = {}

"""Function used to create training dataset and testing dataset """
def make_datasets():
    i=0
    j=0
    k=0
    c=0
    with open(mushroom_instances, 'r+') as dataset_file:
        dataset_lines = dataset_file.readlines()

    for line in dataset_lines:
        attr = line.split(',')

        # Get rid of newline character on last attribute
        attr[-1] = attr[-1].strip()
        """Checks the first column of the csv file since that column value is the class and then divides the data into edible or poisonous"""
        if attr[0] == 'e':
            edible_mushrooms_data.append((attr[0], attr[1:]))
            i+=1
            c+=1
            if(c<4001):

                training_data.append(edible_mushrooms_data.pop())

            else:
                test_data.append(edible_mushrooms_data.pop())

        else:
            poisonous_mushrooms_data.append((attr[0], attr[1:]))
            j+=1
            c+=1
            if(c<4001):
                training_data.append(poisonous_mushrooms_data.pop())

            else:
                test_data.append(poisonous_mushrooms_data.pop())


    k=i+j
    print(edible_mushrooms_data)
    print("Total number of mushrooms:")
    print(k)
    print("\n"+"The number of edible mushrooms:")
    print(i)
    print("\n"+"The number of poisonous mushrooms")
    print(j)
    print("\n")


    #print(test_data)
    #print(negative_dataset)

"""Function to create a feature list to store all the feature values"""
def make_feature_list():
    feature_count = 0
    val_count = 0

    for x in range(len(features)):
        features_present_in_data.append([])
        features_absent_in_data.append([])

    for x in features_present_in_data:
        for y in range(22):
            x.append(0)
    for x in features_absent_in_data:
        for y in range(22):
            x.append(0)
    """To seggregate the edible and poisonous mushroom data in the training dataset"""
    for feat in features:
        val_count = 0
        for value in features_dictionary[feat]:
            for example in training_data:
                if value == example[1][feature_count] and example[0] == 'e':
                    features_present_in_data[feature_count][val_count] += 1
            val_count += 1
        feature_count += 1
    attr_count = 0

    for feat in features:
        val_count = 0
        for value in features_dictionary[feat]:
            for example in training_data:
                if value == example[1][attr_count] and example[0] == 'p':
                    features_absent_in_data[attr_count][val_count] += 1
            val_count += 1
        attr_count += 1
    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(len(features))



"""The mapping functon that maps the .names file to the mushroom instances read from the .csv file"""
def mapping_features():
     with open(mushroom_features, 'r+') as features_file:
        # print(attributes_file)
        features_lines = features_file.readlines()
        # print(attributes_lines)
     for line in features_lines:
        #print(attributes_lines)
        pair = line.strip().split()
        features.append(pair[0])
        features_dictionary[pair[0]] = pair[1].split(',')




"""The naive bayes posterior probability calculations to train the algorithm to identify the edible and the poisonous mushroom instances"""
def naive_bayes(example, neg, pos):
    count = 0
    e_prob = 1.0
    p_prob = 1.0

    for feature in example:
        e_prob *= features_present_in_data[count][features_dictionary[features[count]].index(feature)]
        p_prob *= features_absent_in_data[count][features_dictionary[features[count]].index(feature)]

        count += 1
    if p_prob > e_prob:
        return 'p'
    else:
        return 'e'

"""Main class that calls all the functions listed above and passes the testing dataset to the naive function to make predictions"""
if __name__ == '__main__':
    start = time.time()
    make_datasets()
    mapping_features()
    make_feature_list()
    print ("dataset loaded successfully")

    num_pos = 0
    num_neg = 0


    for i in training_data:
        if i[0] == 'e':
            num_pos += 1
            edible_train.append(i[1])
        else:
            num_neg += 1
            poisonous_train.append(i[1])

    verify = 0

    for ex in test_data:
        actual = ex[0]
        calculated = naive_bayes(ex[1], num_neg, num_pos)
        print ("actual: %s classified: %s " % (actual, calculated))
        if actual == calculated:
            verify += 1

    #print(num_neg)
    #print(num_pos)

    """Calculating the accuracy of the algorithm that predicts the test instances"""
    print ("Percent correct: %f " % (float(verify * 100) / float(len(test_data))))

    print("Training dataset:")
    print(len(training_data))
    print("Test dataset:")
    print(len(test_data))

    objects = ('Training Data', 'Testing Data')
    y_pos = np.arange(len(objects))
    ctrain = len(training_data)
    ctest = len(test_data)
    performance = [ctrain, ctest]
    """ANALYTICS RUN ON THE DATA"""
    """Plot to show how much data was used for training and testing """
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('COUNT')
    plt.title('DATASETS')
    plt.show()
    plt.colormaps()
    """Plot to show how many Edible and  Poisonous mushrooms were present in the given dataset"""
    objects = ('Edible Mushrooms','Poisonous Mushrooms')
    y_pos = np.arange(len(objects))
    performance = [num_pos,num_neg]

    plt1.bar(y_pos, performance, align='center', alpha=0.5)
    plt1.xticks(y_pos, objects)
    plt1.ylabel('COUNT')
    plt1.title('MUSHROOM CLASSIFICATION')

    plt1.show()




