import numpy as np
import matplotlib.pyplot as plt
from pca import pca
from scale import scaler
from integrator import integrator
from regressor import regressor
from neuralnet import nnregressor
from weight import weighter
from inputs import inputs

###############
############### functions to create test datasets
############### all 3 are identical except for target functions (should combine)

def test_func_create_dataset(number_of_datasets):
    ### create dataset for regression test
    max_dev = 0.1

    integ1 = integrator(10,0,0,0,0)
    for i in range(number_of_datasets):
        test_points = [0.5+max_dev*np.random.rand(), 0.2+max_dev*np.random.rand(),\
                       0.1+max_dev*np.random.rand(), 0.0+max_dev*np.random.rand()]
        result = integ1.integrate_ODEs_test(test_points)
        if i ==0:
            array_shape = result.y.shape
            ppa = integ1.t.size
            data_array = np.zeros((array_shape[0], ppa*number_of_datasets))
            rate_array = np.zeros(data_array.shape)
        data_array[:,i*ppa:(i+1)*ppa] = result.y
        result_rates = np.gradient(result.y,result.t,axis=1)
        rate_array[:,i*ppa:(i+1)*ppa] = result_rates
    return data_array, rate_array

def test_func_create_dataset_simple(number_of_datasets):
    ### create dataset for regression test
    max_dev = 0.1

    integ1 = integrator(10,0,0,0,0)
    for i in range(number_of_datasets):
        test_points = [0.5+max_dev*np.random.rand(), 0.1+max_dev*np.random.rand()]
        result = integ1.integrate_ODEs_test_simple(test_points)
        if i ==0:
            array_shape = result.y.shape
            ppa = integ1.t.size
            data_array = np.zeros((array_shape[0], ppa*number_of_datasets))
            rate_array = np.zeros(data_array.shape)
        data_array[:,i*ppa:(i+1)*ppa] = result.y
        result_rates = np.gradient(result.y,result.t,axis=1)
        rate_array[:,i*ppa:(i+1)*ppa] = result_rates
    return data_array, rate_array

def test_func_create_dataset_simple_2(number_of_datasets):
    ### create dataset for regression test
    max_dev = 0.01

    integ1 = integrator(10,0,0,0,0)
    for i in range(number_of_datasets):
        test_points = [0.5+max_dev*np.random.rand(), 0.1+max_dev*np.random.rand(), \
                       0.2+max_dev*np.random.rand()]
        result = integ1.integrate_ODEs_test_simple_2(test_points)
        if i ==0:
            array_shape = result.y.shape
            ppa = integ1.t.size
            data_array = np.zeros((array_shape[0], ppa*number_of_datasets))
            rate_array = np.zeros(data_array.shape)
        data_array[:,i*ppa:(i+1)*ppa] = result.y
        result_rates = np.gradient(result.y,result.t,axis=1)
        rate_array[:,i*ppa:(i+1)*ppa] = result_rates
    return data_array, rate_array


###############
############### functions to run larger tests of multiple functions / classes
############### 


def test_func_scaler():
    # runs a test of std scaling to compare against scipy standard scaler
    test_mat = np.array([[1,2,4],[4,5,6],[7,8,9],[10,11,12]])
    
    num_features = 3
    num_samples = 4
    
    sc = scaler(test_mat,num_samples,num_features)
    
    
    sc.scale_std()
    print(sc.data)
    sc.apply_scaling()
    print(sc.data)
    plt.figure(1)
    plt.plot(sc.data)
    
    
    print("\n\n\n")
    sc.reset_data()
    
    print(sc.data)
    sc.scale_scipy_std()
    print(sc.data)
    plt.figure(2)
    plt.plot(sc.data)
    
    plt.show()


def test_func_pca():
    print("obsolete test, exiting")
    exit()
    # runs a test of PCA methods to compare against scipy
    #test_mat = np.array([[1,2,4],[4,5,6],[7,8,9],[10,11,12]])
    
    #features = 3
    #samples = 4
    
    #test_mat = np.random.rand(samples,features)
    
    #pcac = pca(test_mat, samples, features, 3)
    #pcac.pca_scipy()
    #print("\n\n\n")
    #print("Principal Axis A = \n",pcac.principal_axis, \
    #      '\n\n',\
    #      'Principle Components Z =\n',pcac.principal_components)
    #print("\n\n\n")
    #pcac.pca_EVD()

def test_func_io():
    # tests IO class
    dirname = "data"
    ioc = inputs()
    ioc.read_dens_file(dirname)
    ioc.read_rates_file(dirname)

    #ioc.permute_data_in_time()
    print(ioc.bulkdata.shape)
    print(ioc.bulkrates.shape)

def test_func_holistic(n_components):
    # runs a test on multiple modules, in sequence
    # - read inputs
    # - scale and weight
    # - apply pca
    # - unapply pca
    # - print results
    
    ### LOAD INPUT DATA
    print("Reading Inputs")
    dirname = "data"
    ioc = inputs()
    ioc.read_dens_file(dirname)
    #ioc.read_rates_file(dirname)
    ioc.salt_arrays_for_log()
    ### INITIALIZE SCALER
    print("Initializing Scaler")
    num_samples, num_features = ioc.bulkdata.shape
    w_array = [1,0,0,0]
    w_mag = 1.
    sc = scaler(ioc.bulkdata, num_samples, num_features,1,w_mag,w_array)
    
    # SET SCALER TO STD SCALING AND APPLY
    # (USE LOG AS WELL)
    print("Scaling")
    sc.scale_log()
    sc.scale_std()
    sc.apply_scaling()
    # SET PCA AND APPLY
    print("Applying PCA")
    pcac = pca(sc.data, num_samples, num_features, n_components)
    pcac.pca_scipy()
    # LOAD TRANSOFMRED DATA INTO SCALER AND UNSCALE
    print("Loading New Data")
    sc.load_data(pcac.transformed_data)
    print("Unscaling")
    sc.unapply_scaling()
    sc.unscale_log()

    print("Results")
    plt.figure(1)
    plt.semilogy(sc.original_data)
    plt.ylim([10e-1,10e20])
    plt.figure(3)
    plt.semilogy(sc.data)
    plt.ylim([10e-1,10e20])

    plt.show()
    
def test_func_regressor(n_comp):
    # runs a test on multiple modules, in sequence
    # - read inputs
    # - scale and weight
    # - apply pca
    # - train GPR (or other) regressor
    # - integrate ODE in PC space for set time
    # - unapply pca
    # - print results
    
    ### LOAD INPUT DATA
    print("Reading Inputs")
    dirname = "data"
    ioc = inputs()
    ioc.read_dens_file(dirname)
    ioc.read_rates_file(dirname)
    ioc.salt_arrays_for_log()
    ioc.permute_data_in_time()
    ### INITIALIZE SCALER
    print("Initializing Scaler")
    num_samples, num_features = ioc.data_rand.shape
    sc = scaler(ioc.data_rand, num_samples, num_features)
    #print(ioc.bulkdata)
    
    # SET SCALER TO STD SCALING AND APPLY
    # (USE LOG AS WELL)
    print("Scaling")
    sc.scale_log()
    sc.scale_std()
    sc.apply_scaling()
    # SET PCA AND APPLY
    print("Applying PCA")
    pcac = pca(sc.data, num_samples, num_features, n_comp)
    pcac.pca_scipy()
    # LOAD TRANSOFMRED DATA INTO SCALER AND UNSCALE
    print("Loading New Data")
    sc.load_data(pcac.transformed_data)
    print("Unscaling")
    sc.unapply_scaling()
    sc.unscale_log()
    
    print("Initializing Regression")
    reg = regressor(ioc.data_rand, pcac.principal_components, ioc.rates_rand, sc.D, \
                    pcac.principal_axis, num_samples, num_features, n_comp)
    print("Training regression model")
    reg.train_GPR()
    
    print("Initializing integrator")
    integ = integrator(1, pcac.principal_axis, pcac.principal_components, \
                       ioc.data_rand, reg)
    
    print("Integrating Principal Component Expressions")
    #result = integ.integrate_ODEs()
    
    #print("Converting Principal components back to real state variables")
    #computed_densities = result.y.T.dot(pcac.principal_axis)
    
    #plt.figure(1)
    #plt.semilogy(result.t,result.y.T)
    plt.figure(2)
    plt.plot(ioc.data_rand)
    plt.figure(3)
    plt.plot(ioc.rates_rand)
    #plt.figure(2)
    #plt.semilogy(range(reg.Spc.shape[0]),reg.Spc[:,0], range(reg.Spc.shape[0]),-reg.Spc[:,0])
    

    plt.show()
    
def test_func_regressor_example(input_data, number_of_datasets, n_comp):
    # regression example using data from test_func_create_dataset()
    # runs a test on multiple modules, in sequence
    # - read inputs
    # - scale and weight
    # - apply pca
    # - train GPR, SVR, or other regression 
    # - integrate ODE in PC space for set time
    # - unapply pca
    # - print results
    
    final_time = 10.0
    data_array = input_data[0].T
    rate_array = input_data[1].T
    
    #############################
    ############################# INPUT DATA
    ############################# 
    ### LOAD INPUT DATA
    print("Reading Inputs")
    ioc = inputs()
    ioc.bulkdata = data_array
    ioc.bulkrates = rate_array
    ioc.salt_arrays_for_log()
    ioc.permute_data_in_time()
    ### INITIALIZE SCALER
    print("Initializing Scaler")
    num_samples, num_features = ioc.data_rand.shape
    w_array = [1,0,0,0]
    w_mag = 1.
    sc = scaler(ioc.data_rand, num_samples, num_features,1,w_mag,w_array)
    
    #############################
    ############################# SCALE DATA AND RUN PCA
    #############################
    
    # SET SCALER TO STD SCALING AND APPLY
    # (USE LOG AS WELL)
    print("Scaling")
    sc.scale_std()
    sc.scale_log()
    sc.center()
    sc.apply_scaling()
    
    # SET PCA AND APPLY
    print("Applying PCA")
    pcac = pca(sc.data, num_samples, num_features, n_comp)
    pcac.pca_scipy()
    plt.figure(1)
    plt.plot(pcac.principal_components)

    #############################
    ############################# CREATE REGRESSOR AND INTEGRATE
    ############################# 
    
    print("Initializing Regression")
    reg = regressor(ioc.data_rand, pcac.principal_components, ioc.rates_rand, sc.D, \
                    pcac.principal_axis, num_samples, num_features, n_comp)
    print("Training regression model")
    reg.train_SVR()
    
    print("Initializing integrator")
    integ = integrator(final_time, pcac.principal_axis, pcac.principal_components, \
                       ioc.data_rand, reg)
    
    print("Integrating Principal Component Expressions")
    # compute initial points by the following procedure
    #  1. get initial points
    y0 = ioc.bulkdata[0,:]
    
    #  2. scale initial points
    sc.load_data(y0)
    sc.data = sc.data.reshape(1,-1)
    sc.scale_log()
    sc.center()
    sc.apply_scaling()
    
    #  3. convert to principal components
    y0 = sc.data.dot(pcac.principal_axis)
    result = integ.integrate_ODEs(y0.flatten())
                                 
    print("Converting Principal components back to real state variables")
    computed_densities = (result.y.T).dot(pcac.principal_axis.T)

    # LOAD TRANSOFMRED DATA INTO SCALER AND UNSCALE
    print("Loading New Data")
    sc.load_data(computed_densities)
    print("Unscaling")
    sc.unapply_scaling()
    sc.uncenter()
    sc.unscale_log()
    computed_densities = computed_densities.dot(pcac.principal_axis.T)

    #plt.figure(5)
    #plt.plot(result.t,result.y.T)
    plt.figure(6)
    plt.plot(sc.data)
    plt.ylim([0,1.0])
    plt.figure(7)
    plt.plot(ioc.bulkdata[0:50,:])
    plt.ylim([0,1.0])
    #print(ioc.bulkdata[0,:])
    #print(y0)
    
def test_func_regressor_nn_example(input_data, number_of_datasets, n_comp):
    # Should be identical to regular nonlinear regression code except with 
    # NN functions replacing GPR, SVR, etc
    #
    # regression example using data from test_func_create_dataset()
    # runs a test on multiple modules, in sequence
    # - read inputs
    # - scale and weight
    # - apply pca
    # - train NN regressor
    # - integrate ODE in PC space for set time
    # - unapply pca
    # - print results
    
    final_time = 10.0
    data_array = input_data[0].T
    rate_array = input_data[1].T
    
    #############################
    ############################# INPUT DATA
    #############################
    ### LOAD INPUT DATA
    print("Reading Inputs")
    ioc = inputs()
    ioc.bulkdata = data_array
    ioc.bulkrates = rate_array
    ioc.salt_arrays_for_log()
    ioc.permute_data_in_time()
    ### INITIALIZE SCALER
    print("Initializing Scaler")
    num_samples, num_features = ioc.data_rand.shape
    w_array = [1,0,0,0]
    w_mag = 1.
    sc = scaler(ioc.data_rand, num_samples, num_features,1,w_mag,w_array)

    
    #############################
    ############################# SCALE DATA AND RUN PCA
    #############################
    
    # SET SCALER TO STD SCALING AND APPLY
    # (USE LOG AS WELL)
    print("Scaling")
    sc.scale_std()
    sc.scale_log()
    sc.center()
    sc.apply_scaling()
    # species weighting
    # SET PCA AND APPLY
    print("Applying PCA")
    pcac = pca(sc.data, num_samples, num_features, n_comp)
    pcac.pca_scipy()
    plt.figure(1)
    plt.plot(pcac.principal_components)

    #############################
    ############################# CREATE REGRESSOR AND INTEGRATE
    ############################# 
    
    print("Initializing Regression")
    reg = nnregressor(ioc.data_rand, pcac.principal_components, ioc.rates_rand, sc.D, \
                    pcac.principal_axis, num_samples, num_features, n_comp)
    reg.create_or_load_model()
    
    print("Training regression model")
    reg.train_NN()
    
    print("Initializing integrator")
    integ = integrator(final_time, pcac.principal_axis, pcac.principal_components, \
                       ioc.data_rand, reg)
    
    print("Integrating Principal Component Expressions")
    # compute initial points by the following procedure
    #  1. get initial points
    y0 = ioc.bulkdata[0,:]
    #  2. scale initial points
    sc.load_data(y0)
    sc.data = sc.data.reshape(1,-1)
    sc.scale_log()
    sc.center()
    sc.apply_scaling()
    #  3. convert to principal components
    y0 = sc.data.dot(pcac.principal_axis)
    result = integ.integrate_ODEs(y0.flatten())
                                 
    print("Converting Principal components back to real state variables")
    computed_densities = (result.y.T).dot(pcac.principal_axis.T)

    # LOAD TRANSOFMRED DATA INTO SCALER AND UNSCALE
    print("Loading New Data")
    sc.load_data(computed_densities)
    print("Unscaling")
    sc.unapply_scaling()
    sc.uncenter()
    sc.unscale_log()
    computed_densities = computed_densities.dot(pcac.principal_axis.T)

    #plt.figure(5)
    #plt.plot(result.t,result.y.T)
    plt.figure(6)
    plt.plot(sc.data)
    plt.ylim([0,1.0])
    plt.figure(7)
    plt.plot(ioc.bulkdata[0:50,:])
    plt.ylim([0,1.0])
    #print(ioc.bulkdata[0,:])
    #print(y0)