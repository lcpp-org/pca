import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class nnregressor:
    def __init__(self, density, principal_components, rates, D,\
                 principal_axes, num_samples, num_features, n_components):
        # variable initialization
        self.density = density
        self.principal_components = principal_components
        self.S = rates
        self.Spc = np.zeros(self.S.shape)
        self.D = D
        self.A = principal_axes
        self.features = num_features
        self.samples = num_samples
        self.n_components = n_components
        
        self.predictor_array = []
        self.model = []
        
    def create_or_load_model(self,filename="no file specified"):
        if filename == "no file specified":
            # create model
            self.create_model()
        else:
            self.load_model(filename)
    
    def create_model(self):
        # creates a neural network model (currently testing)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        for i in range(self.n_components):
        	# create model

            new_model = Sequential()
            new_model.add(normalizer)
            new_model.add(Dense(self.n_components*2, input_dim=self.n_components,\
                                 kernel_initializer='normal', activation='relu'))
            new_model.add(Dense(self.n_components*5,\
                                 kernel_initializer='normal', activation='relu'))
            #new_model.add(Dense(self.n_components*5,\
            #                     kernel_initializer='normal', activation='relu'))
            new_model.add(Dense(1, kernel_initializer='normal'))
            # compile model
            new_model.compile(loss='mean_squared_error', optimizer='adam')
            self.model.append(new_model)
        return
    
    def load_model(self):
        print("currenly empty")
        exit(0)
        return
        
    def find_target_vector(self):
        # sets the target vector
        dinv = np.linalg.inv(self.D)
        self.Spc = np.divide(self.S,self.density).dot(dinv).dot(self.A)

    def train_NN(self):
        # train neural network regression model
        self.find_target_vector()
        
        for i in range(self.n_components):
            print('nn '+str(i+1)+'/'+str(self.n_components))
            self.model[i].fit( self.principal_components, self.Spc[:,i],
                               validation_split=0.2, verbose=1, epochs=100)
        return
    
    def run_regression(self, input_array):
        # INPUTS: array of state variables
        # OUTPUTS: array of predicted values
        #print(input_array.size)
        predicted_values = np.zeros(input_array.size)
        for i in range(self.n_components):
            predicted_values[i] = self.model[i].predict(input_array)#.flatten()
        return predicted_values



if __name__ == "__main__":
    exit()