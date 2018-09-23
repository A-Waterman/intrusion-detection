import pandas as pd
import numpy as np
from src.metrics import matthews_correlation_coefficent, false_positives
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from abc import abstractmethod


class base_detector():
    """ Base detector class, all detectors inherit methods
    """
    @abstractmethod
    def log_liklihood(self, X):
        return
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        """Predicts if the system is under attack for each sample  

        Parameters
        ----------
            X : array, sample features to use
        
        Returns
        -------
            predictions : list, prediction if the system is under attack for each time-step
        """
        return self.clf.predict(self.log_likelihood(X)) 
    
    def predict_safe_probability(self, X):
        """Predicts the probability that the system is not under attack

        Parameters
        ----------
            X : array, sample features to use
        
        Returns
        -------
            predictions : list, prediction probability that the system is not under attack
        """
        return self.clf.predict_proba(self.log_likelihood(X))[:,0]
    
    def predict_attack_probability(self, X):
        """Predicts the probability that the system is under attack

        Parameters
        ----------
            X : array, sample features to use
        
        Returns
        -------
            predictions : list, prediction probability that the system is under attack
        """
        return self.clf.predict_proba(self.log_likelihood(X))[:,1]
    
    def score(self, X, y):
        return matthews_correlation_coefficent(y, self.predict(X))
    
    def false_positives(self, X, y):
        return false_positives(y, self.predict(X))

    
class intrusion_detector(base_detector):
    def __init__(self, X, gmm_components=36, pca_components=28, neighbors=10, state=None):
        self.safe_gmm = GaussianMixture(n_components=gmm_components, 
                                        covariance_type='full', random_state=state)
        self.train_pca = PCA(n_components=pca_components, whiten=True, random_state=state)
        self.clf = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform',
                                        algorithm='auto', metric='minkowski')
        self.safe_gmm.fit(X)

    def log_likelihood(self, X):
        ll_gmm = self.safe_gmm.score_samples(X).reshape(-1, 1)
        ll_pca = self.train_pca.score_samples(X).reshape(-1, 1)
        return np.concatenate((ll_gmm, ll_pca), axis=1)
        
    def fit(self, X, y):
        self.train_pca.fit(X)
        self.clf.fit(self.log_likelihood(X), y)


class dual_cov_detector(base_detector):
    def __init__(self, first_cov, second_cov, neighbors=10, weights='uniform'):
        self.first_cov = first_cov
        self.second_cov = second_cov
        self.clf = KNeighborsClassifier(n_neighbors=neighbors, weights=weights,
                                        algorithm='auto', metric='minkowski')
        
    def log_likelihood(self, X):
        ll_first = self.first_cov.score_samples(X).reshape(-1, 1)
        ll_second = self.second_cov.score_samples(X).reshape(-1, 1)
        return np.concatenate((ll_first, ll_second), axis=1)
    
    def fit_cov(self, X_first, X_second):
        self.first_cov.fit(X_first)
        self.second_cov.fit(X_second)
        
    def fit(self, X, y):
        self.clf.fit(self.log_likelihood(X), y)
        

class cov_detector(base_detector):
    def __init__(self, cov_estimator, neighbors=10, weights='uniform'):
        self.cov = cov_estimator
        self.clf = KNeighborsClassifier(n_neighbors=neighbors, weights=weights,
                                        algorithm='auto', metric='minkowski')
    
    def log_likelihood(self, X):
        return self.cov.score_samples(X).reshape(-1, 1)
    
    def fit_cov(self, X):
        self.cov.fit(X)
        
    def fit(self, X, y):
        self.clf.fit(self.log_likelihood(X), y)
