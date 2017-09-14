import os
import sys
import numpy as np
import math

label_map = {
    1 : 1,
    -1 : 0
}

def load_mnist_train(train_file):
    train_data = np.genfromtxt(train_file, delimiter=',')
    #np.random.shuffle(train_data)
    train_labels = np.array(train_data[:, 0])
    train_vectors = np.array(train_data)
    train_labels[train_labels == 0] = -1
    train_vectors[:, 0] = 1
    return train_vectors, train_labels

def load_mnist_test(test_file):
    test_data = np.genfromtxt(test_file, delimiter=',')
    test_vectors = np.array(test_data)
    appended = np.zeros((test_vectors.shape[0], test_vectors.shape[1] + 1))
    appended[:, 1:] = test_vectors
    appended[:, 0] = 1
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


# ------------------------
# Algorithm 1
# ------------------------
def perceptron(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    weight = np.zeros(dim)

    epoch = 0
    while epoch < 200:

        # train
        is_updated = False
        for vec, lab, i in zip(train_vectors, train_labels, range(num_train)):
            x = np.dot(vec, weight)
            if x * lab <= 0:
                weight += lab * vec
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



# ------------------------
# Algorithm 2
# ------------------------
def perceptron_with_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    weight = np.zeros(dim)
    bias = 1000

    epoch = 0
    while epoch < 200:

        # train
        is_updated = False
        for vec, lab, i in zip(train_vectors, train_labels, range(num_train)):
            x = np.dot(vec, weight)
            if x * lab - bias <= 0:
                weight += lab * vec
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


# ------------------------
# Algorithm 3
# ------------------------
def batch_perceptron_without_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    weight = np.zeros(dim)

    epoch = 0
    while epoch < 200:

        # train
        is_updated = False
        out = train_vectors.dot(weight)
        update = np.zeros(dim)

        c = 0
        for x, lab, i in zip(out, train_labels, range(num_train)):
            if x * lab <= 0 :
                update += lab * train_vectors[i]
                is_updated = True
                c += 1
        weight += update
        #print c

        # validate
        correct = 0
        predicted = np.zeros(num_validation)
        for vec, lab, i in zip(validation_vectors, validation_labels, range(num_validation)):
            x = np.dot(vec, weight)
            predicted[i] = 1 if x > 0 else -1
            correct += 1 if predicted[i] == lab else 0
        #print "Accuracy : %d / %d   =    %f" % (correct, num_validation, correct * 100.0 / num_validation)
        #precision, recall, accuracy = precision_recall_accuracy(actual = validation_labels, predicted = predicted)
        #print "Epoch : %d\t\tPrecision : %f\t\tRecall : %f\t\tAccuracy : %f\t\t%d" % (epoch, precision, recall, accuracy, c)

        if not is_updated:
            break
        epoch += 1

    # test
    for vec in test_vectors:
        x = np.dot(vec, weight)
        out = label_map[1] if x > 0 else label_map[-1]
        print out
    return


# ------------------------
# Algorithm 4
# ------------------------
def batch_perceptron_with_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors):

    num_train, dim = train_vectors.shape

    num_validation = validation_vectors.shape[0]
    assert(validation_vectors.shape[1] == dim)
    num_test = test_vectors.shape[0]
    assert(test_vectors.shape[1] == dim)

    weight = np.zeros(dim)
    bias = 1000
    epoch = 0

    while epoch < 200:

        # train
        is_updated = False
        out = train_vectors.dot(weight)
        update = np.zeros(dim)

        c = 0
        for x, lab, i in zip(out, train_labels, range(num_train)):
            if x * lab - bias <= 0 :
                update += lab * train_vectors[i]
                is_updated = True
                c += 1
        weight += update
        #print c

        # validate
        correct = 0
        predicted = np.zeros(num_validation)
        for vec, lab, i in zip(validation_vectors, validation_labels, range(num_validation)):
            x = np.dot(vec, weight)
            predicted[i] = 1 if x > 0 else -1
            correct += 1 if predicted[i] == lab else 0
        #print "Accuracy : %d / %d   =    %f" % (correct, num_validation, correct * 100.0 / num_validation)
        #precision, recall, accuracy = precision_recall_accuracy(actual = validation_labels, predicted = predicted)
        #print "Epoch : %d\t\tPrecision : %f\t\tRecall : %f\t\tAccuracy : %f\t\t%d" % (epoch, precision, recall, accuracy, c)

        if not is_updated:
            break
        epoch += 1

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
    train_vectors, train_labels = load_mnist_train(train_file)
    test_vectors = load_mnist_test(test_file)

    k = int(np.floor(0.2 * train_vectors.shape[0]))
    #print k
    validation_vectors = train_vectors[:k]
    train_vectors = train_vectors[k:]

    validation_labels = train_labels[:k]
    train_labels = train_labels[k:]

    #print "normal perceptron"
    perceptron(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
    #print "perceptron with margin"
    perceptron_with_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
    #print "batch perceptron without margin"
    batch_perceptron_without_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
    #print "batch perceptron without margin"
    batch_perceptron_with_margin(train_vectors, train_labels, validation_vectors, validation_labels, test_vectors)
