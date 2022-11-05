import io
import math
import urllib.request

import pandas as pd
from numpy import add
from scipy.io import arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import heapq

# KNN algorithm
# For each point calculate its distance to all other points
    # Methods to calculate distance:
        # Euclidean distance - sqrt(sum(ai-bi)^2)
            # For nominal values just a 0 or a 1 if they match or don't
# Save the top k instances in order

# To predict a point's output - classification
    # No weight - each of the n nearest neighbors get 1 vote, votes summed by output class, most votes wins
    # Inverse Weighted square distances - each of the n nearest neighbors gets a vote weighted by 1/d^2, votes summed, most votes wins

# To predict a point's output - regression
    # Output is the weighted mean of the n nearest neighbors
        # For each n nearest neighbor
            # Get its weight by 1/d^2, multiply by the output
        # Sum those
        # Divide by the sum of the weights

def load_data(url: str):
    ftp_stream = urllib.request.urlopen(url)
    data, meta = arff.loadarff(io.StringIO(ftp_stream.read().decode('utf-8')))
    data_frame = pd.DataFrame(data)
    return data_frame

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, columntype=[], weight_type='inverse_distance', regression=False):  ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype  # Note This won't be needed until part 5
        self.weight_type = weight_type
        self.X = None
        self.y = None
        self.regression = regression

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X, k_array):
        k_dictionary = {}
        for row in X:
            new_x, distances, new_y = get_distances(self.X, self.y, row)
            for k in k_array:
                new_x_copy = get_top_k(new_x, k)
                new_y_copy = get_top_k(new_y, k)
                distances_copy = get_top_k(distances, k)
                if self.regression:
                    if self.weight_type == 'inverse_distance':
                        inv_distances = get_inv_dist_squared(distances_copy)
                        numerator = dot_col_vecs(inv_distances, new_y_copy)
                        denominator = inv_distances.sum()
                        guess = float(numerator)/denominator
                        self.add_k_guess(k_dictionary, guess, k)
                    else:
                        guess = new_y_copy.sum()/float(get_num_elements(new_y_copy))
                        self.add_k_guess(k_dictionary, guess, k)
                else:
                    if self.weight_type == 'inverse_distance':
                        inv_distances = get_inv_dist_squared(distances_copy)
                        aggregates = get_sum_aggregates(np.append(inv_distances[..., None], new_y_copy[..., None], axis=1), 1, 0)
                        guess = aggregates[0, 0]
                        self.add_k_guess(k_dictionary, guess, k)
                    else:
                        guess = get_mode(new_y_copy)
                        self.add_k_guess(k_dictionary, guess, k)

        return k_dictionary

    # Returns the Mean score given input data and labels
    def score(self, X, y, k_array):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        k_dictionary = self.predict(X, k_array)
        finalScores = []
        for k in k_array:
            results = k_dictionary[k]
            if self.regression:
                mse = ((y - results)**2).sum() / float(len(y))
                finalScores.append(mse)
            else:
                correct = 0
                for i in range(0, len(results)):
                    if results[i] == y[i]:
                        correct += 1
                finalScores.append(float(correct) / len(results))
        return finalScores

    def get_prediction_by_count(self, y):
        return get_mode(y)

    def add_k_guess(self, k_dictionary, guess, k):
        if k not in k_dictionary:
            k_dictionary[k] = []
        k_dictionary[k].append(guess)

# Gets the euclidean distance between new point and all X rows for continuous data
def real_dist(X, newPoint, ax=1):
    x = X - newPoint
    s = (x.conj() * x).real
    sqrtFunction = np.vectorize(sqrt)
    return sqrtFunction(add.reduce(s, axis=ax, keepdims=False))

def sqrt(value):
    return math.sqrt(value)


# Gets the euclidean distance between new point and all X rows for nominal data with a 0/1 distance metric
def cat_dist(X, newPoint, ax=1):
    return np.linalg.norm(X != newPoint, axis=ax)

def get_inv_dist_squared(distances):
    return 1.0 / (distances**2+0.0000001)

# Returns the X and y values sorted by distance from the new point along with their corresponding distances
def get_distances(X, y, newPoint):
    distances = real_dist(X, newPoint)
    augmented_with_y = np.append(X, np.column_stack([y]), axis=1)
    augmented_with_dist = np.append(augmented_with_y, np.column_stack([distances]), axis=1)
    _, num_cols = augmented_with_dist.shape
    sorted_augmented = augmented_with_dist[augmented_with_dist[:, num_cols - 1].argsort()]
    new_x = sorted_augmented[:,:-2]
    new_y = sorted_augmented[:,-1]
    dist = sorted_augmented[:,-2]
    return new_x, new_y, dist

# Gets the top k rows from a matrix
def get_top_k(matrix, k):
    if len(matrix.shape) == 1:
        return matrix[:k]
    else:
        return matrix[:k, :]

# Gets a list of unique values for a 1d array
def get_unique(array):
    return np.unique(array)

# Gets the most frequent number from an array
def get_mode(array):
    vals, counts = np.unique(array, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))
    return vals[mode_value].flatten()[0]

# Aggregates a 2d matrix by the group column, summing the agg column
def get_sum_aggregates(array, group_col_index, agg_col_index):
    unique_vals = get_unique(array[:, group_col_index])
    my_aggregation = np.array([[unique_val, array[array[:,group_col_index]==unique_val,agg_col_index].sum()] for unique_val in unique_vals], dtype='O')
    my_sorted_aggregation = my_aggregation[(-my_aggregation[:,-1]).argsort()]
    return my_sorted_aggregation

def dot_col_vecs(x1, x2):
    return np.dot(x1, x2)

def get_num_elements(array):
    product = 1
    for elem in array.shape:
        product *= elem
    return product

def decodeBytes(value):
    if type(value) == 'bytes':
        return value.decode()
    return value


def do_debug():
    vectorizedDecoder = np.vectorize(decodeBytes)

    training_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/glass_train.arff"))
    train_x = training_data[:,:-1]
    train_y = vectorizedDecoder(training_data[:,-1])

    testing_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/glass_test.arff"))
    test_x = testing_data[:,:-1]
    test_y = vectorizedDecoder(testing_data[:,-1])

    knn = KNNClassifier(weight_type="")
    knn.fit(train_x, train_y)
    print(knn.score(test_x, test_y, [3]))

    knn2 = KNNClassifier()
    knn2.fit(train_x, train_y)
    print(knn2.score(test_x, test_y, [3]))

def do_eval():
    vectorizedDecoder = np.vectorize(decodeBytes)

    training_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/diabetes_train.arff"))
    train_x = training_data[:,:-1]
    train_y = vectorizedDecoder(training_data[:,-1])

    testing_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/diabetes_test.arff"))
    test_x = testing_data[:,:-1]
    test_y = vectorizedDecoder(testing_data[:,-1])

    knn = KNNClassifier(weight_type="")
    knn.fit(train_x, train_y)
    print(knn.score(test_x, test_y, [3]))

    knn2 = KNNClassifier()
    knn2.fit(train_x, train_y)
    print(knn2.score(test_x, test_y, [3]))

def do_magic():
    vectorizedDecoder = np.vectorize(decodeBytes)

    training_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/magic_telescope_train.arff"))
    train_x = training_data[:,:-1]
    train_y = vectorizedDecoder(training_data[:,-1])

    testing_data = np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/magic_telescope_test.arff"))
    test_x = testing_data[:,:-1]
    test_y = vectorizedDecoder(testing_data[:,-1])

    knn = KNNClassifier(weight_type="")
    knn.fit(train_x, train_y)
    # print(knn.score(test_x, test_y, [1,3,5,7,9,11,13,15]))

    combined_data = np.concatenate((train_x,test_x))
    combined_normalized = normalize_data(combined_data)
    num_train_x_rows, _ = train_x.shape

    normalized_train_x = combined_normalized[:num_train_x_rows, :]
    normalized_test_x = combined_normalized[num_train_x_rows:, :]

    knn2 = KNNClassifier(weight_type="")
    knn2.fit(normalized_train_x, train_y)
    print(knn2.score(normalized_test_x, test_y, [1,3,5,7,9,11,13,15]))


def normalize_data(array):
    newarray = array.copy()
    _, num_cols = array.shape
    for i in range(0, num_cols):
        curCol = array[:,i]
        curColMax = np.max(curCol)
        curColMin = np.min(curCol)
        def normalize(value):
            return (value - curColMin) / float(curColMax - curColMin)
        normalizeVectorized = np.vectorize(normalize)
        newarray[:,i] = normalizeVectorized(curCol)
    return newarray

def do_housing():
    vectorizedDecoder = np.vectorize(decodeBytes)
    training_data = vectorizedDecoder(np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/housing_train.arff")))
    train_x = training_data[:,:-1]
    train_y = training_data[:,-1]

    testing_data = vectorizedDecoder(np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/housing_test.arff")))
    test_x = testing_data[:,:-1]
    test_y = testing_data[:,-1]

    combined_data = np.concatenate((train_x, test_x))
    combined_normalized = normalize_data(combined_data)
    num_train_x_rows, _ = train_x.shape

    normalized_train_x = combined_normalized[:num_train_x_rows, :]
    normalized_test_x = combined_normalized[num_train_x_rows:, :]

    knn = KNNClassifier(weight_type='',regression=True)
    knn.fit(normalized_train_x, train_y)
    k_array = [1,3,5,7,9,11,13,15]
    mses = knn.score(normalized_test_x, test_y, k_array)

    plt.plot(k_array, mses)
    plt.title("Housing MSE by k-value")
    plt.xlabel("K-value")
    plt.ylabel("MSE")
    plt.show()


def do_magic_distance_weighting():
    vectorizedDecoder = np.vectorize(decodeBytes)

    training_data = np.array(
        load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/magic_telescope_train.arff"))
    train_x = training_data[:, :-1]
    train_y = vectorizedDecoder(training_data[:, -1])

    testing_data = np.array(
        load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/magic_telescope_test.arff"))
    test_x = testing_data[:, :-1]
    test_y = vectorizedDecoder(testing_data[:, -1])

    combined_data = np.concatenate((train_x, test_x))
    combined_normalized = normalize_data(combined_data)
    num_train_x_rows, _ = train_x.shape

    normalized_train_x = combined_normalized[:num_train_x_rows, :]
    normalized_test_x = combined_normalized[num_train_x_rows:, :]

    knn = KNNClassifier()
    knn.fit(normalized_train_x, train_y)
    print(knn.score(normalized_test_x, test_y, [3]))


def do_housing_distance_weighting():
    vectorizedDecoder = np.vectorize(decodeBytes)
    training_data = vectorizedDecoder(np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/housing_train.arff")))
    train_x = training_data[:,:-1]
    train_y = training_data[:,-1]

    testing_data = vectorizedDecoder(np.array(load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/housing_test.arff")))
    test_x = testing_data[:,:-1]
    test_y = testing_data[:,-1]

    combined_data = np.concatenate((train_x, test_x))
    combined_normalized = normalize_data(combined_data)
    num_train_x_rows, _ = train_x.shape

    normalized_train_x = combined_normalized[:num_train_x_rows, :]
    normalized_test_x = combined_normalized[num_train_x_rows:, :]

    knn = KNNClassifier(regression=True)
    knn.fit(normalized_train_x, train_y)
    k_array = [1,3,5,7,9,11,13,15]
    mses = knn.score(normalized_test_x, test_y, k_array)

    plt.plot(k_array, mses)
    plt.title("Housing MSE by k-value")
    plt.xlabel("K-value")
    plt.ylabel("MSE")
    plt.show()


def do_credit():

if __name__ == "__main__":
    # do_debug()
    # do_eval()
    # do_magic()
    # do_housing()
    # do_magic_distance_weighting()
    do_housing_distance_weighting()