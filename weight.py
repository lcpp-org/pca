import numpy as np

class weighter:
    def __init__(self,data, weight_array, features, samples):
        #initialize variables
        self.data = data
        self.weight_array = weight_array
        self.weight_mag_array = np.ones(len(self.weight_array))
        self.features = features
        self.samples = samples
        if len(self.weight_array) != features:
            print("ERROR: WEIGHT ARRAY DOES NOT HAVE LEN = FEATURES")
        if np.max(self.weight_array)>1 or np.min(self.weight_array)<0 or \
           np.sum(np.floor(self.weight_array)) != np.sum(self.weight_array):
            print("ERROR: WEIGHT ARRAY NOT COMPOSED OF INTEGER 1 OR 0")
        
    def set_wght_mgntd_norm(self, percent):
        # set weight magnitudes by percent normalization
        self.weighted = percent
        self.unweighted = 1-percent
        self.weight_mag_array = self.weight_array
        
    def set_wght_mgntd_mag(self, multiplier):
        # set weight magnitudes by multiplying target species
        self.weight_mag_array = np.multiply(self.weight_array, multiplier)

    def weight_features(self):
        # weight certain features by preferrentially scaling magnitudes
        for i in range(len(self.weight_array)):
            self.data[:,i] = np.multiply(self.data[:,i], self.weight_mag_array[i])
            
    def unweight_features(self):
        # unweight certain features; not sure if needed
        for i in range(len(self.weight_array)):
            self.data[:,i] = np.divide(self.data[:,i], self.weight_mag_array[i])