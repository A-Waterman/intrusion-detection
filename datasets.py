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

    if type == "train":
        label_attack(dataset, 1727, 1776, ['ATT_T7'])
        label_attack(dataset, 2027, 2050, ['ATT_T7', 'ATT_PU10', 'ATT_PU11'])
        label_attack(dataset, 2337, 2396, ['ATT_T1'])
        label_attack(dataset, 2827, 2920, ['ATT_T1', 'ATT_PU1', 'ATT_PU2']) # J269
        label_attack(dataset, 3497, 3556, ['ATT_PU7'])
        label_attack(dataset, 3727, 3820, ['ATT_T4', 'ATT_PU7'])
        label_attack(dataset, 3927, 4036, ['ATT_T1', 'ATT_T4', 'ATT_PU1', 'ATT_PU2', 'ATT_PU7'])
    
    return dataset


def main():
    if Path("datasets/train_0.csv").exists() == False:
        training_0 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset03.csv")
        training_0 = label_attacks(training_0, "safe")
        training_0.to_csv('datasets/train_0.csv', index=False)
    else:
        training_0 = pd.read_csv("datasets/train_0.csv")

    if Path("datasets/train_1.csv").exists() == False:
        training_1 = pd.read_csv("https://www.batadal.net/data/BATADAL_dataset04.csv")
        training_1 = normalize(training_0, training_1)
        training_1 = label_attacks(training_1, "train")
        training_1.to_csv('datasets/train_1.csv', index=False)


if __name__ == '__main__':
    main()
