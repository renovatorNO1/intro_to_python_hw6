import numpy as np
import scipy.spatial.distance as spd
import statistics as stats

def label_to_number(label):
    return 0 if label == b'M' else 1


def make_data(filename):
    """function that creates numpy array containing data from file"""
    arr = np.loadtxt(filename,\
                     delimiter=',',\
                     converters={1 : label_to_number})
    return np.delete(arr, 0, 1)


def split(data, p, position=0):
    split_data = np.array(np.array_split(data, p))
    test = split_data[position]
    training_positions = (split_data[i] for i in range(len(split_data)) \
                            if not i == position)
    training = np.vstack(training_positions)
    return test, training


def KNNclassifier2(training, test, k, d_formula='euclidean'):
    '''This function uses training data to provide labels for a set of test
    data.'''
    distances = spd.cdist(test, training[:, 1:], d_formula)
    sorted_distancess = np.argsort(distances)
    labels = []
    for i in range(len(test)):
        k_closest = sorted_distancess[i][:k]
        all_labels = [training[j,0] for j in k_closest]
        labels.append(stats.median(all_labels))

    return labels


def KNNclassifier(training, test, k, d_formula='euclidean'):
    '''This function uses training data to provide labels for a set of test
    data.'''
    return [stats.median((training[row, 0] \
        for row in np.argsort(distance_arr)[:k])) \
            for distance_arr in spd.cdist(test, training[:, 1:], d_formula)]
                

def NNclassifier2(training, test, d_formula='euclidean'):
    '''This function uses training data to provide labels for a set of test
    data.'''
    distances = spd.cdist(test, training[:, 1:], d_formula)
    minimum_rows = np.argmin(distances, axis=1)
    
    labels = []
    for row_number in minimum_rows:
        labels.append(training[row_number][0])

    return labels


def NNclassifier(training, test, d_formula='euclidean'):
    '''This function uses training data to provide labels for a set of test
    data.'''
    return [training[row][0] \
    for row in np.argmin(spd.cdist(test, training[:, 1:], d_formula), axis=1)]

def n_validator(data, p, classifier, *args):
    '''This function estimates the performance of a classifier in a particular
    setting.'''
    correct = 0
    for i in range(p):
        test, training = split(data, p, i)
        classifier_labels = classifier(training, np.delete(test, 0, 1), *args)

        labels = test[:, 0]        
        correct += sum((labels[i] == classifier_labels[i]\
                        for i in range(len(labels))))
    
    return correct/len(data)


def main():
    arr = make_data('wdbc-tiny.data')
    print(n_validator(arr, 10, KNNclassifier, 1))

    arr = make_data('wdbc-small.data')
    print(n_validator(arr, 50, KNNclassifier, 1))

    arr = make_data('wdbc-full.data')
    accuracies = []
    for i in range(1, 16, 2):
        accuracies.append(((n_validator(arr, 5, KNNclassifier, i)), i))
    accuracies.sort()
    print(accuracies[-1][1])


if __name__ == '__main__':
    main()