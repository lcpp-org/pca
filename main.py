import numpy as np
import matplotlib.pyplot as plt
from pca import pca
from scale import scaler
from integrator import integrator
from regressor import regressor
from weight import weighter
from inputs import inputs
from tests import *
from routines import *

###########
########### Main function which calls routines, currently used for testing
###########

### random number initilization, if desired
np.random.seed(43)

    

### Each of these routines is a test to verify some aspect of the code
#test_func_scaler()
#test_func_pca()
#test_func_io()
#test_func_holistic(10)
#test_func_regressor(31)



### This is an example of a full run of the program:
# - First, we generate a number of datasets with a random distribution
#   of starting points but which use the same reaction rate
number_of_datasets = 100
input_data = test_func_create_dataset_simple_2(number_of_datasets)
# - Next, we select the number of principle components we wish to retain
#   (currently, must be equal to number of species, i.e. no reduction)
n_comp = 3
# - Finally, we run the regression example
test_func_regressor_nn_example(input_data,number_of_datasets,n_comp)

