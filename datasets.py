import pandas as pd
import numpy as np


def normalize(base, dataset):
    for index in range(len(base.columns)):
        dataset.rename(columns={dataset.columns[index]: base.columns[index]}, inplace=True)
    return dataset


def main():
    training_0 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset03.csv")
    training_1 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset04.csv")

    training_1 = normalize(training_0, training_1)


if __name__ == '__main__':
    main()
