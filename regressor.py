import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import Product
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR
from pyearth import Earth

class regressor:
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
        
    def find_target_vector(self):
        # sets the target vector
        dinv = np.linalg.inv(self.D)
        self.Spc = np.divide(self.S,self.density).dot(dinv).dot(self.A)

    def train_GPR(self):
        # train gaussian process regressor model
        self.find_target_vector()
        
        kernel = RationalQuadratic(1.0,length_scale_bounds="fixed")*RBF(1.0,length_scale_bounds="fixed")+\
                 ConstantKernel(1.0,constant_value_bounds="fixed")*RBF(1.0,length_scale_bounds="fixed")

        for i in range(self.n_components):
            print('gpr '+str(i+1)+'/'+str(self.n_components))
            gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).\
                fit(self.principal_components, self.Spc[:,i])
            self.predictor_array.append(gpr)
            
    def train_SVR(self):
        # train support vector regression model  
        self.find_target_vector()
          
        for i in range(self.n_components):
            print('svr '+str(i+1)+'/'+str(self.n_components))
            svr = SVR(kernel='rbf',degree=2).\
                fit(self.principal_components, self.Spc[:,i])
            self.predictor_array.append(svr)
            
    def train_MARS(self):
        # train multivariate adaptive regression spline model
        self.find_target_vector()
        
        for i in range(self.n_components):
            print('mars '+str(i+1)+'/'+str(self.n_components))
            mars = Earth().\
                fit(self.principal_components, self.Spc[:,i])
            self.predictor_array.append(mars)  
            
    def train_GPR_real(self):
        # train gaussian process regressor model
        # PROBABLY USELESS

        # find source vector 
        #dinv = np.linalg.inv(self.D)
        #self.Spc = np.dot(np.dot(np.divide(self.S,self.density),dinv),self.A)
        self.Spc = self.S
        for i in range(self.n_components):
            print('gpr '+str(i+1)+'/'+str(self.n_components))
            kernel = RBF()
            gpr = GaussianProcessRegressor(kernel=kernel,
                    random_state=0).fit(self.density, self.Spc[:,i])
            self.predictor_array.append(gpr)
            
    def train_GPR_onefeat(self):
        # train gaussian process regressor model
        self.find_target_vector()
        
        kernel = ConstantKernel(1.0,constant_value_bounds="fixed")* \
                 RBF(1.0,length_scale_bounds="fixed")+\
                 ConstantKernel(1.0,constant_value_bounds="fixed")* \
                 RBF(1.0,length_scale_bounds="fixed")
                     
        for i in range(self.n_components):
            print('gpr '+str(i+1)+'/'+str(self.n_components))
            gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).\
                fit(self.principal_components[:,i].reshape(-1,1), self.Spc[:,i])
            self.predictor_array.append(gpr)
            
    def train_MARS_onefeat(self):
        # train multivariate adaptive regression spline model
        self.find_target_vector()
        
        for i in range(self.n_components):
            print('mars '+str(i+1)+'/'+str(self.n_components))
            mars = Earth().\
                fit(self.principal_components[:,i].reshape(-1,1), self.Spc[:,i])
            self.predictor_array.append(mars)  
            
    def run_regression(self, input_array):
        # Gaussian process regressor
        # INPUTS: array of state variables
        # OUTPUTS: array of predicted values
        #print(input_array.size)
        predicted_values = np.zeros(input_array.size)
        for i in range(self.n_components):
            predicted_values[i] = self.predictor_array[i].predict(input_array)
            
        return predicted_values
    
    def run_regression_onefeat(self, input_array):
        # Gaussian process regressor
        # INPUTS: array of state variables
        # OUTPUTS: array of predicted values
        #print(input_array.size)
        predicted_values = np.zeros(input_array.size)
        print(input_array.shape)
       #exit()
        for i in range(self.n_components):
            predicted_values[i] = self.predictor_array[i].predict(input_array[:,i].reshape(-1,1))
            
        return predicted_values