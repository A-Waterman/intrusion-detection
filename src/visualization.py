import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import split_data

def plot(df, title="", x="", y="", xlabel="", ylabel=""):
    plt.plot(x, y, data=df[df["LABEL"] == 0], 
             linestyle='', marker='.', markersize=8, color="blue", alpha=0.5, label='Safe')
    
    plt.plot(x, y, data=df[df["LABEL"] == 1], 
             linestyle='', marker='.', markersize=8, color="red", alpha=0.5, label='Attack')
    
    plt.title(title)
    plt.legend(markerscale=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def plot_predictions(y, predictions, title="Predictions", ylabel="Attack Probability"):
    df = prepare_prediction_plot(y, predictions)
    plot(df, title, x="Hour", y="Predicted", xlabel="Time (Hours)", ylabel=ylabel)
    
def prepare_prediction_plot(y, prediction):
    time = np.array(range(len(prediction))).reshape(-1, 1)
    features = np.concatenate((time, prediction.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(features, columns=["Hour", "Predicted"])
    return df.join(pd.DataFrame(y, columns=["LABEL"]))

def plot_confusion_matrix(y, prediction, title='Confusion matrix'):
    plt.figure()
    _plot_confusion_matrix(cm=confusion_matrix(y, prediction), 
                           classes=['safe', 'attack'], title=title)
    plt.show()

# Source obtained from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def _plot_confusion_matrix(cm, classes, title,
                           normalize=False,
                           cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class log_likihood_visualizer(): 
    def __init__(self, cov, min_difference):
        self.cov = cov
        self.scaler = MinMaxScaler((0, 1))
        self.min_diff = min_difference

    def fit(self, dataframe):
        features, label = split_data(dataframe)
        self.cov.fit(features)
        self.clip_max = np.max(self.log_likihood(features))
        self.clip_min = self.clip_max - self.min_diff
    
    def log_likihood(self, features):
        return self.cov.score_samples(features).reshape(-1, 1)
    
    def transform(self, dataframe):
        features, label = split_data(dataframe)
        
        time = np.array(range(len(dataframe))).reshape(-1, 1)
        clipped_liklihood = np.clip(self.log_likihood(features), self.clip_min, self.clip_max)
        scaled_liklihood = self.scaler.fit_transform(clipped_liklihood)

        dim_features = np.concatenate((time, scaled_liklihood), axis=1)
        df = pd.DataFrame(dim_features, columns=["Time", "Liklihood"])
        return df.join(pd.DataFrame(label, columns=["LABEL"]))
    
    def plot(self, dataframe, title=""):
        plot(self.transform(dataframe), title, x="Time", y="Liklihood", xlabel="Time (Hours)", ylabel="Liklihood (Cliped & Scaled)")
