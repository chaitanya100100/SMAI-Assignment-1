import numpy as np
import os
import sys


class Attribute(object):

    def __init__(self, name, min_val, max_val, map_value):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.map_value = map_value
        return

    def disp(self):
        print self.name, self.min_val, self.max_val, self.map_value

def load_test(test_file):
    data = np.recfromcsv(test_file, names=True, dtype=None)
    to_consider = [
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company',
        'work_accident',
        'promotion_last_5years',
        'sales',
        'salary'
    ]
    non_num = [
        'sales',
        'salary'
    ]
    X_test = np.zeros([data.shape[0], len(to_consider)])
    for i, name in enumerate(to_consider):

        if name in non_num:
            unique_values, indexed_array = np.unique(data[name], return_inverse=True)
            map_value = {val : i for i, val in enumerate(unique_values)}
            X_test[:, i] = indexed_array
        else:
            X_test[:, i] = data[name]
            map_value = None

        min_val = min(X_test[:, i])
        max_val = max(X_test[:, i])
    return X_test


def load_decision_tree_data(train_file):
    data = np.recfromcsv(train_file, names=True, dtype=None)
    #np.random.shuffle(data)
    to_consider = [
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company',
        'work_accident',
        'promotion_last_5years',
        'sales',
        'salary'
    ]
    non_num = [
        'sales',
        'salary'
    ]
    label = 'left'

    Y_train = data[label]
    X_train = np.zeros([data.shape[0], len(to_consider)])
    attributes = []
    for i, name in enumerate(to_consider):

        if name in non_num:
            unique_values, indexed_array = np.unique(data[name], return_inverse=True)
            map_value = {val : i for i, val in enumerate(unique_values)}
            X_train[:, i] = indexed_array
        else:
            X_train[:, i] = data[name]
            map_value = None

        min_val = min(X_train[:, i])
        max_val = max(X_train[:, i])
        attributes.append(Attribute(name = name, min_val = min_val, max_val = max_val, map_value = map_value))
    return X_train, attributes, label, Y_train




class Node(object):
    """ represent a node in decision tree """

    def __init__(self, X_train, attributes, label, Y_train, depth = 0):
        self.X_train = X_train
        self.attributes = attributes
        self.label = label
        self.Y_train = Y_train
        self.total_true = len([0 for x in self.Y_train if x == 1])
        self.total = self.X_train.shape[0]
        self.depth = depth

        if self.should_split():
            X_train_left, X_train_right, Y_train_left, Y_train_right, attr_idx, attr_val = self.split()
            self.terminal = False
            self.attr_idx = attr_idx
            self.attr_val = attr_val

            # delete to avoid duplicates
            del self.X_train
            del self.Y_train

            # recursively build left and right node
            self.left = Node(X_train_left, self.attributes, self.label, Y_train_left, self.depth + 1)
            self.right = Node(X_train_right, self.attributes, self.label, Y_train_right, self.depth + 1)

        else :
            self.terminal = True
            return

    def should_split(self):
        """ check if current node should be terminal or it should be splitted """

        #if self.depth > 10 : return False
        if self.total < 10 : return False
        if self.total_true == self.total : return False
        if self.total_true == 0: return False
        #if float(self.total_true) / self.total > 0.97 : return False
        #if float(self.total_true) / self.total < 0.03 : return False
        return True


    def calculate_quality2(self, true, i):
        """ give quality by gini index """
        t1 = float(true)
        i1 = float(i)
        t2 = self.total_true - t1
        i2 = self.total - i1

        quality = 0.0
        if i1 : quality += (1.0 - (t1 * t1  + (i1 - t1) * (i1 - t1)) / (i1 * i1)) * i1 / self.total
        if i2 : quality += (1.0 - (t2 * t2  + (i2 - t2) * (i2 - t2)) / (i2 * i2)) * i2 / self.total

        return quality


    def calculate_quality(self, true, i):
        """ quality of split by entropy """
        t1 = float(true)
        i1 = float(i)
        t2 = self.total_true - t1
        i2 = self.total - i1
        if true and true != i:
            entropy1 = (t1 / i1) * np.log(i1 / t1) \
                    + ((i1 - t1) / i1) * np.log(i1 / (i1 - t1))
        else:
            entropy1 = 0

        if (self.total_true - true) and ((self.total_true - true) != (self.total - i)):
            entropy2 = (t2 / i2) * np.log(i2 / t2) \
                    + ((i2 - t2) / i2) * np.log(i2 / (i2 - t2))
        else:
            entropy2 = 0

        quality = (entropy1 * i + entropy2 * (self.total - i)) / self.total
        return quality


    def split(self):
        """
            iterate through possibilities and find best split
            sets split attribute, split value, groups and entropy
        """
        total = self.total
        total_true = self.total_true
        best_quality = 10.0
        best_attr_idx = -1
        best_attr_val = -1
        for idx, attr in enumerate(self.attributes):
            values = sorted(zip(self.X_train[:, idx], self.Y_train))
            unique_values = sorted(np.unique(self.X_train[:, idx]))
            true = 0
            i = 0
            for val in unique_values:
                while i < self.total and values[i][0] <= val:
                    true += values[i][1]
                    i += 1
                quality = self.calculate_quality(true, i)
                if quality < best_quality:
                    best_quality = quality
                    best_attr_idx = idx
                    best_attr_val = val

        #print best_quality, self.attributes[best_attr_idx].name, best_attr_val
        #print best_attr_idx, self.attributes[best_attr_idx].name, best_attr_val
        # divide according to the best split
        right = [i for i, vec in enumerate(self.X_train) if vec[best_attr_idx] > best_attr_val]
        left = [i for i, vec in enumerate(self.X_train) if not (vec[best_attr_idx] > best_attr_val)]
        # assert(len(left) + len(right) == self.total)

        X_train_left = self.X_train[left, :]
        X_train_right = self.X_train[right, :]
        Y_train_left = self.Y_train[left]
        Y_train_right = self.Y_train[right]
        #print self.total, X_train_left.shape[0], Y_train_left[Y_train_left == 1].shape[0]
        #print self.total, X_train_right.shape[0], Y_train_right[Y_train_right == 1].shape[0]
        #print
        return X_train_left, X_train_right, Y_train_left, Y_train_right, best_attr_idx, best_attr_val


    def predict(self, vec):
        """ given a vector, predict the class """
        if self.terminal:
            if float(self.total) / 2.0 <= self.total_true:
                return 1
            else:
                return 0
        else:
            if vec[self.attr_idx] > self.attr_val:
                return self.right.predict(vec)
            else:
                return self.left.predict(vec)



if __name__ == '__main__':

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    #print "loading data..."
    X_train, attributes, label, Y_train = load_decision_tree_data(train_file)
    X_test = load_test(test_file)

    #print "dividing into training and validation data..."
    num = X_train.shape[0]
    k = int(0.20 * num)
    X_validate = X_train[:k, :]
    X_train = X_train[k:, :]
    Y_validate = Y_train[:k]
    Y_train = Y_train[k:]


    #print "generating decision tree on training data..."
    root = Node(X_train, attributes, label, Y_train)


    #print "validating on validation data..."
    correct = 0
    for vec, lab in zip(X_validate, Y_validate):
        if root.predict(vec) == lab:
            correct += 1
    #print "Accuracy : ", correct, "/", X_validate.shape[0]

    for vec in X_test:
        print root.predict(vec)
