import sys
import os
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

ignore = [
    '<s>',
    '<\s>',
    'a',
    'an',
    'the'
]

for_ans = {}

def process_train_data(train_data_dir):

    classes = os.listdir(train_data_dir)
    classes_dict = {}
    all_words = set()

    num = 0
    label_dict = {}

    for label, cl in enumerate(classes):

        files = os.listdir(train_data_dir + cl)
        label_dict[cl] = label
        for_ans[label] = cl

        count = 0
        files_dict = {}
        for fl in files:

            #print "reading", fl, "in", cl

            f = open(train_data_dir + cl + "/" + fl,'r')
            text = f.read()
            words = text.split()
            words = [w for w in words if not w in ignore]
            unique_words, freq = np.unique(words, return_counts = True)
            files_dict[fl] = zip(unique_words, freq)
            all_words.update(unique_words)

            count += 1
            num += 1
            #if count == 3:
            #    break

        classes_dict[cl] = files_dict

    dim = len(all_words) + 1
    words_dict = {word : idx for idx, word in enumerate(all_words)}
    words_dict = defaultdict(lambda : dim - 1, words_dict)

    X_train = np.zeros((num, dim))
    Y_train = np.zeros((num,))

    i = 0
    for cl, cl_dict in classes_dict.items():
        for fl, fl_freq in cl_dict.items():
            for key, val in fl_freq:
                X_train[i, words_dict[key]] = val
            X_train[i] = X_train[i]
            Y_train[i] = label_dict[cl]
            i += 1


    return X_train, Y_train, words_dict, dim, label_dict

def process_test_data(test_data_dir, words_dict, dim, label_dict):

    num = 0
    for cl in os.listdir(test_data_dir):
        ls = os.listdir(test_data_dir + cl + '/')
        num += len(ls)

    X_test = np.zeros((num, dim))
    Y_test = np.zeros((num, ))

    i = 0
    for cl in os.listdir(test_data_dir):
        for fl in os.listdir(test_data_dir + cl + '/'):

            #print "reading", fl, "in", cl

            f = open(test_data_dir + cl + "/" + fl,'r')
            text = f.read()
            words = text.split()
            words = [w for w in words if not w in ignore]
            unique_words, freq = np.unique(words, return_counts = True)

            for word, fr in zip(unique_words, freq):
                X_test[i][words_dict[word]] = fr
            X_test[i] = X_test[i]
            Y_test[i] = label_dict[cl]

            i += 1

    return X_test, Y_test




class KNN(object):

    def __init__(self, X_train, Y_train, X_test, Y_test, K = 1, distance_type = 'cityblock'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.dim = X_train.shape[1]
        self.K = K
        self.distance_type = distance_type

        """
        self.X_train_norms = np.zeros((X_train.shape[0], ))
        for i, vec in enumerate(X_train):
            self.X_train_norms[i] = np.dot(vec, vec)
        for vec in self.X_train:
            vec = vec / norm(vec)
        for vec in self.X_test:
            vec = vec / norm(vec)
        """

    def distance(self, a, b):
        return

    def compute_distances(self, qvec):
        if self.distance_type == 'cityblock':
            sub = self.X_train - qvec
            distances = norm(sub, ord = 1, axis = 1)
        elif self.distance_type == 'euclidean':
            sub = self.X_train - qvec
            distances = norm(sub, ord = 2, axis = 1)
        elif self.distance_type == 'cosine':
            distances = np.zeros(self.X_train.shape[0])
            for i, vec in enumerate(self.X_train):
                distances[i] = 1 - np.dot(vec, qvec) / (norm(vec) * norm(qvec))

        return zip(distances, self.Y_train)


    def compute_distances2(self, qvec):
        if self.distance_type == 'cityblock':
            sub = self.X_train - qvec
            distances = norm(sub, ord = 1, axis = 1)
        elif self.distance_type == 'euclidean':
            distances = np.zeros(self.X_train.shape[0])
            for i, vec in enumerate(self.X_train):
                distances[i] = norm(vec - qvec)
        elif self.distance_type == 'cosine':
            distances = np.zeros(self.X_train.shape[0])
            for i, vec in enumerate(self.X_train):
                distances[i] = 1 - np.dot(vec, qvec) / (norm(vec) * norm(qvec))

        return zip(distances, self.Y_train)


    def classify(self):

        predict = np.zeros((self.X_test.shape[0], ))
        for i, qvec in enumerate(X_test):
            distances = self.compute_distances2(qvec)
            candidates = sorted(distances)[:self.K]
            #print "best", candidates[0]
            freq = {}
            freq = defaultdict(lambda : 0,freq)
            cand_dist = {}
            cand_dist = defaultdict(lambda : 100000000000.0, cand_dist)
            for distance, candidate in candidates:
                freq[candidate] += 1
                cand_dist[candidate] = min(cand_dist[candidate], distance)
            sorted_freq = sorted(freq.items(), key = lambda x : x[1])
            best_freq = sorted_freq[-1][1]
            ans = sorted_freq[-1][0]
            #predict[i] = ans
            #continue
            for fa, ff in reversed(sorted_freq):
                if ff == best_freq:
                    if cand_dist[fa] < cand_dist[ans]:
                        ans = fa
                        best_freq = ff
                else:
                    break
            predict[i] = ans
            print for_ans[ans]
            """
            candidates_labels = [x[1] for x in candidates]
            label, counts = np.unique(candidates_labels, return_counts = True)
            most_labels = sorted(zip(counts, label))
            predict[i] = sorted(zip(counts, label))[-1][1]
            """
        count = 0
        conf = np.zeros((len(label_dict), len(label_dict)))

        for pre, act in zip(predict, self.Y_test):
            conf[int(act)][int(pre)] += 1
            if pre == act:
                count += 1
        #print count
        #print predict
        #print self.Y_test


if __name__ == "__main__":
    train_data_dir = sys.argv[1]
    test_data_dir = sys.argv[2]

    X_train, Y_train, words_dict, dim, label_dict = process_train_data(train_data_dir)
    #print label_dict
    X_test, Y_test = process_test_data(test_data_dir, words_dict, dim, label_dict)
    #knn = KNN(X_train, Y_train, np.array(X_train), np.array(Y_train), K = 5, distance_type = 'euclidean')
    #X_test = np.copy(X_train)
    #Y_test = np.copy(Y_train)
    knn = KNN(X_train, Y_train, X_test, Y_test, K = 7, distance_type = 'cosine')
    knn.classify()
