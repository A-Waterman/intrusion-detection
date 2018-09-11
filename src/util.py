import pandas as pd
import numpy as np
from math import log
from sklearn.utils.extmath import fast_logdet

# code adapted from 
def samplewise_log_likelihood(X, mean, precision):
    """Return the log-likelihood of each sample.
        See - http://www.miketipping.com/papers/met-mppca.pdf
        code adapted from https://github.com/scikit-learn/scikit-learn/blob/ed5e127b/sklearn/decomposition/pca.py#L516
        
        Parameters
        ----------
            X : array, shape(n_samples, n_features), the sample data
            mean: float, the mean of the current model 
            precision: array, shape(n_features, n_features), precision matrix of the current model
        
        Returns
        -------
            ll : array, shape (n_samples,) : Log-likelihood of each sample under the current model
    """
    Xr = X - mean
    n_features = X.shape[1]
    log_like = np.zeros(X.shape[0])
    log_like = -.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
    log_like -= .5 * (n_features * log(2. * np.pi) - fast_logdet(precision))
    return log_like.reshape(-1, 1)

def load_datasets():
    safe = pd.read_csv('datasets/train_0.csv')
    train = pd.read_csv('datasets/train_1.csv')
    test = pd.read_csv('datasets/test.csv')
    
    return safe, train, test

def transform_datasets(func):
    safe_dataset, train_dataset, test_dataset = load_datasets()
    
    safe = func(safe_dataset)
    train = func(train_dataset)
    test = func(test_dataset)
    
    return safe, train, test

def label_attack(dataset, index_start, index_end, components):
    dataset.loc[index_start:index_end, 'ATT_FLAG'] = 1
    for component in components:
        dataset.loc[index_start:index_end, component] = 1

def label_save_datasets():
    # load the downloaded datasets
    training_01 = pd.read_csv('datasets/BATADAL_dataset03.csv')
    training_02 = pd.read_csv('datasets/BATADAL_dataset04.csv')
    test_dataset = pd.read_csv('datasets/BATADAL_test_dataset.csv')
    
    # test_dataset missing attack flag
    test_dataset['ATT_FLAG'] = 0
    
    # normalize column names (e.g. ' ATT_FLAG' to 'ATT_FLAG')
    for index in range(len(training_01.columns)):
        training_02.rename(columns={training_02.columns[index]: training_01.columns[index]},
                           inplace=True)

    for index in range(len(training_01.columns)):
        test_dataset.rename(columns={test_dataset.columns[index]: training_01.columns[index]},
                            inplace=True)
    
    Attacks = ['ATT_FLAG',
               'ATT_T1', 'ATT_T2', 'ATT_T3', 'ATT_T4', 'ATT_T5', 'ATT_T6', 'ATT_T7',
               'ATT_PU1', 'ATT_PU2', 'ATT_PU3', 'ATT_PU4', 'ATT_PU5', 'ATT_PU6', 'ATT_PU7',
               'ATT_PU8', 'ATT_PU9', 'ATT_PU10', 'ATT_PU11', 'ATT_V2']
    
    # initalize
    for key in Attacks:
        training_01[key] = 0
        training_02[key] = 0
        test_dataset[key] = 0
        
    # label second training dataset with attacks 
    label_attack(training_02, 1727, 1776, ['ATT_T7'])
    label_attack(training_02, 2027, 2050, ['ATT_T7', 'ATT_PU10', 'ATT_PU11'])
    label_attack(training_02, 2337, 2396, ['ATT_T1'])
    label_attack(training_02, 2827, 2920, ['ATT_T1', 'ATT_PU1', 'ATT_PU2']) # J269
    label_attack(training_02, 3497, 3556, ['ATT_PU7'])
    label_attack(training_02, 3727, 3820, ['ATT_T4', 'ATT_PU7'])
    label_attack(training_02, 3927, 4036, ['ATT_T1', 'ATT_T4', 'ATT_PU1', 'ATT_PU2', 'ATT_PU7'])
    
    # label test dataset with attacks
    label_attack(test_dataset, 297, 366, ['ATT_T3', 'ATT_PU4', 'ATT_PU5'])
    label_attack(test_dataset, 632, 696, ['ATT_T2', 'ATT_V2'])
    label_attack(test_dataset, 867, 897, ['ATT_PU3'])
    label_attack(test_dataset, 939, 969, ['ATT_PU3'])
    label_attack(test_dataset, 1229, 1328, ['ATT_T2', 'ATT_V2']) # J14, J422
    label_attack(test_dataset, 1574, 1653, ['ATT_T7', 'ATT_PU10', 'ATT_PU11']) # J14, J422
    label_attack(test_dataset, 1940, 1969, ['ATT_T4'])
    
    # save datasets
    training_01.to_csv('datasets/train_0.csv', index=False)
    training_02.to_csv('datasets/train_1.csv', index=False)
    test_dataset.to_csv('datasets/test.csv', index=False)
