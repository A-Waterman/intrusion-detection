import pandas as pd
import numpy as np


def main():
    training_0 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset03.csv")
    training_1 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset04.csv")

    # normalize column names (e.g. ' ATT_FLAG' to 'ATT_FLAG')
    for index in range(len(training_0.columns)):
        training_1.rename(columns={training_1.columns[index]: training_0.columns[index]}, inplace=True)


if __name__ == '__main__':
    main()
