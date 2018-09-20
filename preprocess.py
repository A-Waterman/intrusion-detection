import pandas as pd
import numpy as np
from src.util import transform_datasets
from src.preprocessing import full_data, split_data
from src.model import intrusion_detector


def main():
    np.random.seed(0)

    safe_dataset, train_dataset, test_dataset = transform_datasets(full_data)

    X_safe, y_safe = split_data(safe_dataset)
    X_train, y_train = split_data(train_dataset)
    X_test, y_test = split_data(test_dataset)

    best = intrusion_detector(X_safe, gmm_components=36, pca_components=28, neighbors=10)
    best.fit(X_train, y_train)

    print("Safe fp: {}".format(best.false_positives(X_safe, y_safe)))
    print("Train MCC: {:.3f}".format(best.score(X_train, y_train)))
    print("Test MCC: {:.3f}".format(best.score(X_test, y_test)))
    print("Test fp: {}".format(best.false_positives(X_test, y_test)))


if __name__ == '__main__':
    main()
