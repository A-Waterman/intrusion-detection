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

class conv_estimator():
    def __init__(self, cov=ShrunkCovariance(shrinkage=0)):
        self.cov = cov
        self.mean = None
        
    def fit(self, X):
        self.cov.fit(X)
        self.mean = np.mean(X, axis=0)

    def score_samples(self, X):
        return samplewise_log_likelihood(X, self.mean, self.cov.get_precision())


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


def main():
    pass

if __name__ == '__main__':
    main()
