import pandas as pd
import numpy as np
from src.util import transform_datasets
from src.preprocessing import full_data, split_data
from sklearn.model_selection import StratifiedKFold
from src.model import cov_detector, dual_cov_detector
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis, PCA


def train_valid_split(X, y, shuffle=True, random_state=0):
    skf = StratifiedKFold(n_splits=2, shuffle=shuffle, random_state=random_state)
    train, valid = skf.split(X, y)
    train_index = train[0]
    valid_index = valid[0]
    return X[train_index], y[train_index], X[valid_index], y[valid_index]


def experiment(exp):
    np.random.seed(0)

    safe_dataset, train_dataset, test_dataset = transform_datasets(full_data)

    X_safe, y_safe = split_data(safe_dataset)
    X_train, y_train = split_data(train_dataset)
    X_test, y_test = split_data(test_dataset)

    X_train, y_train, X_valid, y_valid = train_valid_split(X_train, y_train)

    exp.fit_cov(X_safe)
    exp.fit(X_train, y_train)
    
    print("Safe fp: {}".format(exp.false_positives(X_safe, y_safe)))
    print("Valid MCC: {:.3f}".format(exp.score(X_valid, y_valid)))
    print("Valid fp: {}".format(exp.false_positives(X_valid, y_valid)))


def GMM_experiment(n_components, covariance_type='full', random_state=None, neighbors=10, weights='uniform'):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm_exp = cov_detector(gmm, neighbors=neighbors, weights=weights)

    experiment(gmm_exp)

def FA_experiment(n_components, neighbors=10, weights='uniform'):
    fa = FactorAnalysis(n_components=43)
    fa_exp = cov_detector(fa, neighbors=neighbors, weights=weights)

    experiment(fa_exp)


def main():
    FA_experiment(n_components=43)


if __name__ == '__main__':
    main()
