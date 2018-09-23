import pandas as pd
import numpy as np

def full_data(dataframe):
    return dataframe.drop(['DATETIME', 'ATT_T1', 'ATT_T2', 'ATT_T3', 'ATT_T4', 'ATT_T5', 
                           'ATT_T6', 'ATT_T7','ATT_PU1', 'ATT_PU2', 'ATT_PU3', 'ATT_PU4', 
                           'ATT_PU5','ATT_PU6', 'ATT_PU7', 'ATT_PU8', 'ATT_PU9', 'ATT_PU10',
                           'ATT_PU11', 'ATT_V2'], axis = 1)

def full_components(dataframe):
  return dataframe[['ATT_T1', 'ATT_T2', 'ATT_T3', 'ATT_T4', 'ATT_T5', 
                    'ATT_T6', 'ATT_T7','ATT_PU1', 'ATT_PU2', 'ATT_PU3', 'ATT_PU4', 
                    'ATT_PU5','ATT_PU6', 'ATT_PU7', 'ATT_PU8', 'ATT_PU9', 'ATT_PU10',
                    'ATT_PU11', 'ATT_V2']]

def continous_data(dataframe):
    return dataframe[['L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
                      'F_PU1', 'F_PU2', 'F_PU3', 'F_PU4', 'F_PU5', 'F_PU6', 
                      'F_PU7', 'F_PU8', 'F_PU9', 'F_PU10', 'F_PU11', 'F_V2',
                      'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
                      'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422',
                      'ATT_FLAG']]

def discrete_data(dataframe):
    return dataframe[['S_PU1', 'S_PU2', 'S_PU3', 'S_PU4', 'S_PU5', 'S_PU6', 
                      'S_PU7', 'S_PU8', 'S_PU9', 'S_PU10', 'S_PU11', 'S_V2',
                      'ATT_FLAG']]

def dma_1_data(dataframe):
    return dataframe[['P_J280', 'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3', 
                      'P_J269', 'L_T1', 'P_J302', 'P_J307',
                      'ATT_FLAG']]

def dma_2_data(dataframe):
    return dataframe[['L_T2', 'P_J14', 'F_V2', 'S_V2', 'P_J422', 'P_J289', 'F_PU6', 'S_PU6', 
                      'F_PU7', 'S_PU7', 'P_J415', 'L_T4', 'P_J300',
                      'ATT_FLAG']]

def dma_3_data(dataframe):
    return dataframe[['P_J300', 'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'P_J256', 'L_T3', 
                      'ATT_FLAG']]

def dma_4_data(dataframe):
    return dataframe[['P_J307', 'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11','P_J317', 'L_T6', 'L_T7', 
                      'ATT_FLAG']]

def dma_5_data(dataframe):
    return dataframe[['P_J302', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9', 'P_J306', 'L_T5', 
                      'ATT_FLAG']]

def split_data(dataframe):
    label = dataframe[['ATT_FLAG']].values.ravel()
    features = dataframe.drop(['ATT_FLAG'], axis = 1).values
    return features, label

# Code to normalize data based on hour, resulted in poorer performance

def split_datetime(dataframe):
    """ Split the date-time feature of a dataset into date and hour

        Parameters
        ----------
            dataframe : dataframe, shape(n_samples, 64), the full dataset

        Returns
        -------
            dataframe : the dataframe with the date-time feature split into date and hour
    """
    labels = dataframe[['ATT_FLAG', 'ATT_T1', 'ATT_T2', 'ATT_T3', 'ATT_T4', 'ATT_T5', 
                        'ATT_T6', 'ATT_T7','ATT_PU1', 'ATT_PU2', 'ATT_PU3', 'ATT_PU4', 
                        'ATT_PU5', 'ATT_PU6', 'ATT_PU7', 'ATT_PU8', 'ATT_PU9', 'ATT_PU10', 
                        'ATT_PU11', 'ATT_V2']]
    features = dataframe.drop(labels, axis = 1)
    
    date_time = features['DATETIME'].str.split(' ', 1, expand=True)
    features = features.join(date_time.rename(columns={0:'DATE', 1:'HOUR'}))
    
    return features.join(labels)

HOURS = ['00', '01', '02', '03', '04', '05', '06', '07', 
         '08', '09', '10', '11', '12', '13', '14', '15', 
         '16', '17', '18', '19', '20', '21', '22', '23']

def extract_hourly_mean_std(dataframe):
    """ Get the hourly mean and standard deviation for each hour in the dataset

        Parameters
        ----------
            dataframe : the dataset

        Returns
        -------
            list : the hourly mean and standard deviation for all 24 hours
    """
    return [dataframe.loc[dataframe['HOUR'] == hour].describe()[1:3] for hour in HOURS]

def normalize_by_hour(dataframe, stats):
    df = dataframe.copy()
    for hour in range(24):
        df.loc[df['HOUR'] == HOURS[hour], 'HOUR'] = hour
    hours = df['HOUR'].values
    vals = df.drop(['DATETIME', 'DATE', 'HOUR'], axis = 1).values
    
    # extract mean and std from stats
    stats_mean = [stats[index].values[0] for index in range(len(stats))]
    stats_std = [stats[index].values[1] for index in range(len(stats))]
    
    # ensure no divison by 0
    for index in range(len(stats_std)):
        stats_std[index][stats_std[index] == 0] = 1.0
    
    # normalize
    for index in range(len(vals)):
        hour = index % len(stats)
        vals[index] = (vals[index] - stats_mean[hour]) / stats_std[hour]

    return vals
