import pandas as pd
import numpy as np
from src.util import transform_datasets
from src.preprocessing import full_data, split_data
from sklearn.model_selection import StratifiedKFold
from src.model import cov_detector, dual_cov_detector
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis, PCA
import argparse
import sys


def train_valid_split(X, y, shuffle=True, random_state=0):
    skf = StratifiedKFold(n_splits=2, shuffle=shuffle, random_state=random_state)
    train, valid = skf.split(X, y)
    train_index = train[0]
    valid_index = valid[0]
    return X[train_index], y[train_index], X[valid_index], y[valid_index]


def experiment(exp, seed):
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


def dual_experiment(exp, seed):
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


def GMM_experiment(n_components, covariance_type='full', neighbors=10, weights='uniform', seed=0):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=None)
    gmm_exp = cov_detector(gmm, neighbors=neighbors, weights=weights)

    experiment(gmm_exp, seed)


def FA_experiment(n_components, neighbors=10, weights='uniform', seed=0):
    fa = FactorAnalysis(n_components=43)
    fa_exp = cov_detector(fa, neighbors=neighbors, weights=weights)

    experiment(fa_exp, seed)


def PCA_experiment(n_components, whiten=False, neighbors=10, weights='uniform', seed=0):
    pca = PCA(n_components=n_components, whiten=whiten, random_state=None)
    pca_exp = cov_detector(pca, neighbors=neighbors, weights=weights)
    experiment(pca_exp, seed)


def dual_GMM_experiment(n_components_first, n_components_second, covariance_type='full', neighbors=10, weights='uniform', seed=0):
    gmm_0 = GaussianMixture(n_components=n_components_first, covariance_type=covariance_type, random_state=None)
    gmm_1 = GaussianMixture(n_components=n_components_second, covariance_type=covariance_type, random_state=None)
    gmm_exp = dual_cov_detector(gmm_0, gmm_1, neighbors=neighbors, weights=weights)
    dual_experiment(gmm_exp, seed)


def dual_FA_experiment(n_components_first, n_components_second, neighbors=10, weights='uniform', seed=0):
    fa_0 = FactorAnalysis(n_components=n_components_first, random_state=None)
    fa_1 = FactorAnalysis(n_components=n_components_second, random_state=None)
    fa_exp = dual_cov_detector(fa_0, fa_1, neighbors=neighbors, weights=weights)
    dual_experiment(fa_exp, seed)


def dual_PCA_experiment(n_components_first, n_components_second, neighbors=10, weights='uniform', seed=0):
    pca_0 = PCA(n_components=n_components_first, whiten=True, random_state=None)
    pca_1 = PCA(n_components=n_components_second, whiten=True, random_state=None)
    pca_exp = dual_cov_detector(pca_0, pca_1, neighbors=neighbors, weights=weights)
    dual_experiment(pca_exp, seed)


def combined_GMM_FA_experiment(n_components_gmm, n_components_fa, covariance_type='full', neighbors=10, weights='uniform', seed=0):
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type=covariance_type, random_state=None)
    fa = FactorAnalysis(n_components=n_components_fa, random_state=None)
    combined_exp = dual_cov_detector(gmm, fa, neighbors=neighbors, weights=weights)
    dual_experiment(combined_exp, seed)


def combined_GMM_PCA_experiment(n_components_gmm, n_components_pca, covariance_type='full', neighbors=10, weights='uniform', seed=0):
    gmm = GaussianMixture(n_components=n_components_gmm, covariance_type=covariance_type, random_state=None)
    pca = PCA(n_components=n_components_pca, whiten=True, random_state=None)
    combined_exp = dual_cov_detector(gmm, pca, neighbors=neighbors, weights=weights)
    dual_experiment(combined_exp, seed)


def main(args):
    if args.estimator == "GMM":
        GMM_experiment(n_components=args.components[0], neighbors=args.neighbors, seed=args.seed)
    elif args.estimator == "FA":
        FA_experiment(n_components=args.components[0], neighbors=args.neighbors, seed=args.seed)
    elif args.estimator == "PCA":
        PCA_experiment(n_components=args.components[0], neighbors=args.neighbors, seed=args.seed)
    else:
        print("Invalid estimator:", args.estimator)
        print("Supported estimators: 'GMM', 'FA', or 'PCA'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--estimator", help="the covariance matrix estimator")
    parser.add_argument("-c", "--components", help="the number of components", type=int, nargs='+')
    parser.add_argument("-n", "--neighbors", help="the number of neighbors", type=int, nargs='?', default=10)
    parser.add_argument("--seed", help="initial seed", type=int, nargs='?', default=0)

    args = parser.parse_args()
    if args.estimator is None:
        print("Error: missing an estimator. Valid options are: 'GMM', 'FA', or 'PCA'.")
        parser.print_usage()
        sys.exit(-1)

    main(args=args)
