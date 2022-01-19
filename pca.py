import numpy as np
from numpy.linalg import svd
from numpy import linalg as la
from sklearn.decomposition import PCA

class pca:
    # Class methods accept scaled data and output principal components
    def __init__(self,input_data, num_samples, num_features, num_components):
        #initialize variables here
        self.principal_components = 0
        self.principal_axis = 0
        self.features = num_features
        self.samples = num_samples
        self.n_components = num_components
        self.input_data = input_data
        self.transformed_data = 0
        self.variance_explained = 0
        
    def pca_scipy(self):
        # PCA using scipy routines
        
        # fit and create manifold
        pca = PCA(n_components=self.n_components)
        self.principal_components = pca.fit_transform(self.input_data)
        self.transformed_data = pca.inverse_transform(self.principal_components)
        self.principal_axis = pca.components_
    

