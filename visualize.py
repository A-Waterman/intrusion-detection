import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.util import transform_datasets
from src.preprocessing import split_data
from src.preprocessing import full_data, continous_data, discrete_data
from src.preprocessing import dma_1_data, dma_2_data, dma_3_data, dma_4_data, dma_5_data
from sklearn.covariance import ShrunkCovariance
from src.util import samplewise_log_likelihood
from src.visualization import log_likihood_visualizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.visualization import plot

class conv_estimator():
    def __init__(self, cov=ShrunkCovariance(shrinkage=0)):
        self.cov = cov
        self.mean = None
        
    def fit(self, X):
        self.cov.fit(X)
        self.mean = np.mean(X, axis=0)

    def score_samples(self, X):
        return samplewise_log_likelihood(X, self.mean, self.cov.get_precision())

class tsne_visualizer(): 
    def __init__(self):
        self.tsne = TSNE(n_components=2, random_state=0)

    def transform(self, dataframe):
        features, label = split_data(dataframe)
        dim_features = self.tsne.fit_transform(features)
        df = pd.DataFrame(dim_features, columns=["DIM_1", "DIM_2"])
        return df.join(pd.DataFrame(label, columns=["LABEL"]))
    
    def plot(self, dataframe, title=""):
        plot(self.transform(dataframe), title, x="DIM_1", y="DIM_2", xlabel="t-SNE DIM 1", ylabel="t-SNE DIM 2")


def ll_visualizer_conv(safe, target):
    vs = log_likihood_visualizer(cov=conv_estimator(ShrunkCovariance(shrinkage=0)), 
                                                    min_difference=200)
    vs.fit(safe)
    vs.plot(target, title="Safe Dataset Likelihood")


def ll_visualizer_pca(safe, target):
    pca = PCA(n_components=28, whiten=True, random_state=0)
    vs = log_likihood_visualizer(pca, 2000)
    vs.fit(safe)
    vs.plot(target)


def ll_visualizer_gmm(safe, target):
    gmm = GaussianMixture(n_components=40, random_state=0)
    vs = log_likihood_visualizer(gmm, 300)
    vs.fit(safe)
    vs.plot(target)


def plot_tsne_visualizer(target, title):
    vs = tsne_visualizer()
    vs.plot(target, title=title)


def main():
    pass

if __name__ == '__main__':
    main()
