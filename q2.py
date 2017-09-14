import os
import sys
import numpy as np
import math

label_map = {
    1 : 4,
    -1 : 2
}

def load_breast_cancer_train(train_file):
    train_data = np.genfromtxt(train_file, delimiter=',')
    #np.random.shuffle(train_data)
    train_labels = np.array(train_data[:, -1])
    train_vectors = np.array(train_data[:, 1:])
    train_labels[train_labels == 2] = -1
    train_labels[train_labels == 4] = 1
    train_vectors[:, -1] = 1
    return train_vectors, train_labels


def load_breast_cancer_test(test_file):
    test_data = np.genfromtxt(test_file, delimiter=',')
    test_vectors = np.array(test_data)
    appended = np.zeros((test_vectors.shape[0], test_vectors.shape[1]))
    appended[:, :-1] = test_vectors[:, 1:]
    appended[:, -1] = 1
    return appended


def precision_recall_accuracy(predicted, actual):
    assert(predicted.shape[0] == actual.shape[0])

    num = predicted.shape[0]
    total_true = len([1 for a in actual if a == 1])
    total_false = num - total_true

    true_positive = len([1 for a, p in zip(actual, predicted) if a == p and a == 1])
    false_positive = len([1 for a, p in zip(actual, predicted) if a != p and p == 1])
    true_negative = total_true - true_positive
    false_negative = total_false - false_positive

    if (true_positive + false_positive) : precision = 1.0 * true_positive / (true_positive + false_positive)
    else : precision = 0

    if (true_positive + true_negative) : recall = 1.0 * true_positive / (true_positive + true_negative)
    else : recall = 0

    if num : accuracy = 1.0 * (true_positive + false_negative) / num
    else : accuracy = 0

    return precision, recall, accuracy


def perceptron_with_relaxation(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    # weight initialization
    weight = np.zeros(dim)
    bias = 100.0
    eta = 1.0
    epoch = 0
    while epoch < 200:

        is_updated = False
        for vec, lab, i in zip(train_vectors, train_labels, range(num_train)):
            x = np.dot(vec, weight)
            if x * lab - bias <= 0:
                nor = np.dot(vec, vec)
                weight += eta * lab * ((bias - lab * x) / nor) * vec
                is_updated = True

        # validate
        correct = 0
        predicted = np.zeros(num_validation)
        for vec, lab, i in zip(validation_vectors, validation_labels, range(num_validation)):
            x = np.dot(vec, weight)
            predicted[i] = 1 if x > 0 else -1
            correct += 1 if predicted[i] == lab else 0
        #print "Accuracy : %d / %d   =    %f" % (correct, num_validation, correct * 100.0 / num_validation)
        #precision, recall, accuracy = precision_recall_accuracy(actual = validation_labels, predicted = predicted)
        #print "Precision : %f\t\tRecall : %f\t\tAccuracy : %f" % (precision, recall, accuracy)

        epoch += 1
        if not is_updated:
            break

    # test
    for vec in test_vectors:
        x = np.dot(vec, weight)
        out = label_map[1] if x > 0 else label_map[-1]
        print out
    return




def modified_perceptron(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    # weight initialization
    weight = np.zeros(dim)
    bias = 100.0
    eta = 1.0
    epoch = 0
    while epoch < 200:

        is_updated = False
        wrong = 0
        for vec, lab, i in zip(train_vectors, train_labels, range(num_train)):
            x = np.dot(vec, weight)
            if x * lab - bias <= 0:
                nor = np.dot(vec, vec)
                weight += eta * lab * ((bias - lab * x) / nor) * vec
                is_updated = True
                wrong += 1

        eta = eta * 0.95
        #eta = eta * (wrong / num_train)

        # validate
        correct = 0
        predicted = np.zeros(num_validation)
        for vec, lab, i in zip(validation_vectors, validation_labels, range(num_validation)):
            x = np.dot(vec, weight)
            predicted[i] = 1 if x > 0 else -1
            correct += 1 if predicted[i] == lab else 0
        #print "Accuracy : %d / %d   =    %f" % (correct, num_validation, correct * 100.0 / num_validation)
        #precision, recall, accuracy = precision_recall_accuracy(actual = validation_labels, predicted = predicted)
        #print "Precision : %f\t\tRecall : %f\t\tAccuracy : %f" % (precision, recall, accuracy)

        epoch += 1
        if not is_updated:
            break

    # test
    for vec in test_vectors:
        x = np.dot(vec, weight)
        out = label_map[1] if x > 0 else label_map[-1]
        print out
    return



if __name__ == '__main__':

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # load and parse data
    #print "loading train and test data"
    train_vectors, train_labels = load_breast_cancer_train(train_file)
    test_vectors = load_breast_cancer_test(test_file)

    k = int(np.floor(0.2 * train_vectors.shape[0]))
    #print k
    validation_vectors = train_vectors[:k]
    train_vectors = train_vectors[k:]

    validation_labels = train_labels[:k]
    train_labels = train_labels[k:]

    perceptron_with_relaxation(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
    modified_perceptron(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
