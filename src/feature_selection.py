import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans


class KMeansFeatures(TransformerMixin, BaseEstimator):
    """
    Integrate a pipeline step that fits KMeans on the dataset and transforms it by appending both the cluster assignment 
    and the distances to the cluster centroids as new features.
    """

    def __init__(self, n_clusters, label_feature=True, distance_feature=True):
        self.n_clusters = n_clusters
        self.label_feature = label_feature
        self.distance_feature = distance_feature
        self.kmeans = None


    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.kmeans.fit(X)
        return self
    
    
    def transform(self, X):
        X_new = X.copy()
        if self.label_feature:
            labels = self.kmeans.fit_predict(X)
            X_new['cluster label'] = labels
        if self.distance_feature:
            distances = self.kmeans.fit_transform(X)
            for i in range(distances.shape[1]):
                X_new[f'cluster {i+1} distance'] = distances[:,i]
        return X_new
