import pandas as pd
import numpy as np
from src.util import transform_datasets
from src.preprocessing import full_data, split_data, full_components
from src.model import intrusion_detector


def main():
    np.random.seed(0)

    safe_dataset, train_dataset, test_dataset = transform_datasets(full_data)

    X_safe, y_safe = split_data(safe_dataset)
    X_train, y_train = split_data(train_dataset)
    X_test, y_test = split_data(test_dataset)

    best = intrusion_detector(X_safe, gmm_components=36, pca_components=28, neighbors=10)
    best.fit(X_train, y_train)

    preprocessed_safe = best.predict_attack_probability(X_safe)
    preprocessed_train = best.predict_attack_probability(X_train)
    preprocessed_test = best.predict_attack_probability(X_test)

    safe_components, train_components, test_components = transform_datasets(full_components)


if __name__ == '__main__':
    main()
