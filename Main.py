import io
import urllib.request

import pandas as pd
from scipy.io import arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

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
            new_x, new_y, distances = get_distances(self.X, self.y, row)
            for i, k in enumerate(k_array):
                new_x = get_top_k(new_x, k)
                new_y = get_top_k(new_y, k)
                distances = get_top_k(distances, k)
                if self.regression:
                    if self.weight_type == 'inverse_distance':
                        inv_distances = get_inv_dist_squared(distances)
                        numerator = dot_col_vecs(inv_distances, new_y)
                        denominator = inv_distances.sum()
                        guess = float(numerator)/denominator
                        self.add_k_guess(k_dictionary, guess, k)
                    else:
                        guess = new_y.sum()/float(get_num_elements(new_y))
                        self.add_k_guess(k_dictionary, guess, k)
                else:
                    if self.weight_type == 'inverse_distance':
                        inv_distances = get_inv_dist_squared(distances)
                        aggregates = get_sum_aggregates(np.append(new_y[..., None], inv_distances[..., None], axis=1), 1, 0)
                        guess = aggregates[0, 0]
                        self.add_k_guess(k_dictionary, guess, k)
                    else:
                        guess = get_mode(new_y)
                        self.add_k_guess(k_dictionary, guess, k)

        return k_dictionary

    # Returns the Mean score given input data and labels
    def score(self, X, y, k):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        return 0

    def get_prediction_by_count(self, y):
        return get_mode(y)

    def add_k_guess(self, k_dictionary, guess, k):
        if k not in k_dictionary:
            k_dictionary[k] = []
        k_dictionary[k].append(guess)

# Gets the euclidean distance between new point and all X rows for continuous data
def real_dist(X, newPoint, ax=1):
    return np.linalg.norm(X - newPoint, axis=ax)

# Gets the euclidean distance between new point and all X rows for nominal data with a 0/1 distance metric
def cat_dist(X, newPoint, ax=1):
    return np.linalg.norm(X != newPoint, axis=ax)

def get_inv_dist_squared(distances):
    return 1.0 / (distances**2)

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
    return np.bincount(array).argmax()

# Aggregates a 2d matrix by the group column, summing the agg column
def get_sum_aggregates(array, group_col_index, agg_col_index):
    unique_vals = get_unique(array[:, group_col_index])
    my_aggregation = np.array([[unique_val, array[array[:,group_col_index]==unique_val,agg_col_index].sum()] for unique_val in unique_vals])
    my_sorted_aggregation = my_aggregation[(-my_aggregation[:,-1]).argsort()]
    return my_sorted_aggregation

def dot_col_vecs(x1, x2):
    return np.dot(x1, x2)

def get_num_elements(array):
    product = 1
    for elem in array.shape:
        product *= elem
    return product

def get_debug_data():
    data = load_data(r"https://raw.githubusercontent.com/cs472ta/CS472/master/datasets/glass_train.arff")
    return np.array(data)

def challenge_question():
    data = np.array([
        [1,5,100],
        [0,8,101],
        [9,9,101],
        [10,10,100]
    ])
    new_point = np.array([[2,6]])
    knnClassifier = KNNClassifier()
    knnClassifier.fit(data[:,:-1],data[:,-1])
    print(knnClassifier.predict(new_point, 3))

if __name__ == "__main__":
    get_debug_data()