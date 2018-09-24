import numpy as np

"""
import keras.backend as K

## Code obtained from: https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras/40289696#40289696
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
"""

# numpy version of above code
def matthews_correlation_coefficent(y_true, y_pred, epsilon=1e-07):
     """Compute the Matthews correlation coefficient (MCC)
        Adapted from: https://stackoverflow.com/questions/39895742/matthews-correlation-coefficient-with-keras/40289696#40289696

    Parameters
    ----------
       y_true : array, true labels
       y_pred : array, predicted labels
       epsilon : small number to prevent divide by zero errors
        
    Returns
    -------
        MCC : int, the Matthews correlation coefficient (0.0 is random guessing, 1.0 is perfect correlation)
     """
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    
    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / (denominator + epsilon)

def false_positives(y_true, y_pred):
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_neg = 1 - np.round(np.clip(y_true, 0, 1))

    return np.sum(y_neg * y_pred_pos)
