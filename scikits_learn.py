from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier

#import pandas as pd
import numpy as np

def make_data_set(data_file_paths, option):
    data = []
    label = []
    print("[samples]------------------------")
    print("| %4s | %4s | %4s |" % ("GG", "MB", "DG"))
    print("|", end='')
    for i in range(len(data_file_paths)):
        if option[i] != "x":
            path = data_file_paths[i]
            f = open(path, 'r')
            lines = f.readlines()
            data_temp = []
            for line in lines:
                '''
                label_temp = [0, 0, 0]
                label_temp[int(option[i])] = 1
                '''
                data_temp += [[float(value) for value in line.strip().split(",")]]

                label += [int(option[i])]
            data += data_temp
            print(" %4d |" % len(data_temp), end='')
    print()
    return data, label

def scikits_learn(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=20)
    print("[data set]-----------------------")
    print("%15s : " % "training data", len(y_train))
    print("%15s : " % "test data", len(y_test))
    print("[result]-------------------------")
    '''
    model = DecisionTreeClassifier(random_state=20)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" %("DecisionTree", accuracy_score(y_test, prediction)*100))

    model = GaussianNB()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" % ("GaussianNB", accuracy_score(y_test, prediction)*100))

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" % ("SVM", accuracy_score(y_test, prediction)*100))

    model = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 2], max_iter=5000)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" % ("MLP 10x2", accuracy_score(y_test, prediction)*100))

    model = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 4], max_iter=5000)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" % ("MLP 10x4", accuracy_score(y_test, prediction)*100))
    '''
    model = MultiOutputClassifier(KNeighborsClassifier())
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("%20s : %.2f%%" % ("MOC", accuracy_score(y_test, prediction) * 100))