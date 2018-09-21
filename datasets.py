import pandas as pd
import numpy as np
from pathlib import Path
from src.util import label_attack


def normalize(base, dataset):
    for index in range(len(base.columns)):
        dataset.rename(columns={dataset.columns[index]: base.columns[index]}, inplace=True)
    return dataset


def label_attacks(dataset, type=""):
    Attacks = ['ATT_FLAG',
                'ATT_T1', 'ATT_T2', 'ATT_T3', 'ATT_T4', 'ATT_T5', 'ATT_T6', 'ATT_T7',
                'ATT_PU1', 'ATT_PU2', 'ATT_PU3', 'ATT_PU4', 'ATT_PU5', 'ATT_PU6', 'ATT_PU7',
                'ATT_PU8', 'ATT_PU9', 'ATT_PU10', 'ATT_PU11', 'ATT_V2']

    for key in Attacks:
        dataset[key] = 0

    if type == "safe":
        return dataset
    else:
        print("Error: type must be one of 'safe', 'train', or 'test'")
        return None


def main():
    if Path("datasets/train_0.csv").exists():
        training_0 = pd.read_csv("datasets/train_0.csv")
    else:
        training_0 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset03.csv")
        training_0 = label_attacks(training_0, "safe")
        training_0.to_csv('datasets/train_0.csv', index=False)


if __name__ == '__main__':
    main()
