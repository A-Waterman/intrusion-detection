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


def experiment(exp, seed=0):
    np.random.seed(seed)

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


def dual_experiment(exp, seed=0):
    np.random.seed(seed)

    safe_dataset, train_dataset, test_dataset = transform_datasets(full_data)

    X_safe, y_safe = split_data(safe_dataset)
    X_train, y_train = split_data(train_dataset)
    X_test, y_test = split_data(test_dataset)

    X_train, y_train, X_valid, y_valid = train_valid_split(X_train, y_train)

    exp.fit_cov(X_safe, X_train)
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


def PCA_experiment(n_components, whiten=False, random_state=None, neighbors=10, weights='uniform'):
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    pca_exp = cov_detector(pca, neighbors=neighbors, weights=weights)
    experiment(pca_exp)


def dual_GMM_experiment(n_components_first, n_components_second, covariance_type='full', random_state=None, neighbors=10, weights='uniform'):
    gmm_0 = GaussianMixture(n_components=n_components_first, covariance_type=covariance_type, random_state=random_state)
    gmm_1 = GaussianMixture(n_components=n_components_second, covariance_type=covariance_type, random_state=random_state)
    gmm_exp = dual_cov_detector(gmm_0, gmm_1, neighbors=neighbors, weights=weights)
    dual_experiment(gmm_exp)


def dual_FA_experiment(n_components_first, n_components_second, random_state=None, neighbors=10, weights='uniform'):
    fa_0 = FactorAnalysis(n_components=n_components_first, random_state=random_state)
    fa_1 = FactorAnalysis(n_components=n_components_second, random_state=random_state)
    fa_exp = dual_cov_detector(fa_0, fa_1, neighbors=neighbors, weights=weights)
    dual_experiment(fa_exp)


def dual_PCA_experiment(n_components_first, n_components_second, random_state=None, neighbors=10, weights='uniform'):
    pca_0 = PCA(n_components=n_components_first, whiten=True, random_state=random_state)
    pca_1 = PCA(n_components=n_components_second, whiten=True, random_state=random_state)
    pca_exp = dual_cov_detector(pca_0, pca_1, neighbors=neighbors, weights=weights)
    dual_experiment(pca_exp)


def combined_GMM_FA_experiment(n_components_gmm, n_components_fa, covariance_type='full', random_state=None, neighbors=10, weights='uniform'):
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type=covariance_type, random_state=random_state)
    fa = FactorAnalysis(n_components=n_components_fa, random_state=random_state)
    combined_exp = dual_cov_detector(gmm, fa, neighbors=neighbors, weights=weights)
    dual_experiment(combined_exp)


def main():
    combined_GMM_FA_experiment(n_components_gmm=36, n_components_fa=43, random_state=0)


if __name__ == '__main__':
    main()
