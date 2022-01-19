import numpy as np
from sklearn.preprocessing import StandardScaler

class scaler:
    def __init__(self, input_array, samples, features,do_weights, w_mag, weight_array):
        #initialize variables
        self.feature_index = 1
        self.sample_index = 0
        self.features = features
        self.samples = samples
        self.rates = 0
        # Copy data and correct array shape
        if input_array.shape[self.feature_index] == features:
            self.data = input_array
            self.original_data = input_array
        else:
            self.data = input_array.transpose
            self.original_data = input_array.transpose
        # Scale Matrix
        self.D = np.zeros((features,features))
        self.feature_mean = 0
        # Weight matrix
        self.do_weights = do_weights
        self.weight_array = weight_array
        self.W = np.identity(self.features)
        for i in range(self.features):
            if self.weight_array[i] == 1:
                self.W[i,i] *= w_mag
        self.W = np.linalg.inv(self.W)
        #print(self.W)

############################################
############################################ Construction of scaling array
############################################

    def scale_scipy_std(self):
        # Standard Deviation Scaling (scipy)
        self.sc = StandardScaler(with_mean=False) #,with_std=False)
        self.data = self.sc.fit_transform(self.data)
        
    def scale_std(self):
        # Standard Deviation Scaling
        for i in range(self.D.shape[1]):
            self.D[i,i] = np.std(self.data[:,i])
        if self.do_weights == 1:
            self.D = self.D.dot(self.W)
            
    def scale_pareto(self):
        # Pareto Scaling
        for i in range(self.D.shape[self.feature_index]):
            self.D[i,i] = np.sqrt(np.std(self.data[i]))
        if self.do_weights == 1:
            self.D = self.D.dot(self.W)
        
    def scale_vast(self):
        # Vast Scaling
        for i in range(self.D.shape[self.feature_index]):
            self.mean_of_feature = np.mean(self.data[self.feature_index])
            self.D[i,i] = np.power( np.std(self.data[i]),2.0)/self.mean_of_feature
        if self.do_weights == 1:
            self.D = self.D.dot(self.W)
        
    def scale_range(self):
        # Range scaling
        for i in range(self.D.shape[self.feature_index]):
            self.max_range = np.max( self.data[i] - np.mean(self.data[i]) )
            self.min_range = np.min( self.data[i] - np.mean(self.data[i]) )
            self.D[i,i] = self.max_range - self.min_range
        if self.do_weights == 1:
            self.D = self.D.dot(self.W)
    
    def scale_level(self):
        # Level scaling
        for i in range(self.D.shape[self.feature_index]):
            self.D[i,i] = np.mean(self.data[i])
        if self.do_weights == 1:
            self.D = self.D.dot(self.W)
            
############################################
############################################ Log Scaling methods
############################################

    def scale_bisymlog(self):
        # Bisymmetric Log Transfer Scaling
        C = 1.0 #Shaping constant
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i,j] = np.sign(self.data[i,j])* \
                                 np.log10(1.+np.abs(self.data[i,j]/C))
    
    def unscale_bisymlog(self):
        # Bisymmetric Log Transfer Unscaling
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i,j] = np.sign(self.data[i,j])* \
                                 (-1.0+np.power(10.,np.abs(self.data[i,j])))
                                 
    def scale_log(self):
        # apply logarithmic scaling
        self.data = np.log(self.data)
        
    def unscale_log(self):
        # unapply logarithmic scaling
        self.data = np.exp(self.data)
        
############################################
############################################ Apply/Unapply Scaling
############################################
    
    def apply_scaling(self):
        # applies scaling matrix D
        if np.sum(self.D)==0:
            print("ERROR: EMPTY SCALING MATRIX")
 #       if self.do_weights == 1:
 #           self.set_wght_mgntd_mag()
 #           temp = np.linalg.inv(self.D).dot(self.W)
 #       else:
        temp = np.linalg.inv(self.D)
        self.data = np.dot(self.data, temp)
        
    def unapply_scaling(self):
        #print(self.data.shape, self.D.shape)
#        if self.do_weights == 1:
#            self.set_wght_mgntd_mag()
#            temp = self.D.dot(np.linalg.inverse(self.W))
#        else:
        temp = self.D
        self.data = np.dot(self.data, temp)

############################################
############################################ WEIGHTS
############################################

    def set_wght_mgntd_mag(self):
        # set weight magnitudes by multiplying target species
        for i in range(self.features):
            if self.weight_array == 1:
                self.W[i,i] *= self.multiplier
                
#    def weight_features(self):
#        # weight certain features by preferrentially scaling magnitudes
#        # modifies scale matrix
#        self.D = self.D.dot(self.W)
            
#    def unweight_features(self):
#        # unweight certain features; not sure if needed
#        self.D = self.D.dot(np.linalg.inverse(self.W))

        
############################################
############################################ MISC
############################################

    def load_data(self, newdata):
        # loads new data to be scaled or unscaled;
        # USE WITH CAUTION
        self.data = newdata
        
    def reset_data(self):
        # resets data to pre-scaling state; mainly for testing
        self.data = self.original_data
        
    def find_mean(self):
        # finds the mean of a matrix dimension
        for i in range(self.data.shape[1]):
            self.feature_mean[i] = np.mean(self.data[:,i])
        self.feature_mean.reshape(1,-1)
    
    def center(self):
        # subtracts the mean from the matrix
        if type(self.feature_mean)==type(0):
            self.feature_mean = np.zeros(self.features+1)
            print("Finding Mean (should only happen once)")
            self.find_mean()
        #print(self.data.shape, self.feature_mean.shape)
        for i in range(self.data.shape[1]):
                self.data[:,i] = np.subtract(self.data[:,i],self.feature_mean[i])
                
    def center_rates(self,rates):
        # subtracts species mean from rate expression
        self.rates = rates
        for i in range(self.rates.shape[1]):
                self.rates[:,i] = np.subtract(self.rates[:,i],self.feature_mean[i])
                
    def uncenter(self):
        for i in range(self.data.shape[1]):
                self.data[:,i] = np.add(self.data[:,i],self.feature_mean[i])
                
#    def center_easy(self):
#        self.mean_ = np.mean(self.data, axis=0)
#        self.data -= self.mean_