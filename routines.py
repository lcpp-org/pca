# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pca import pca
from scale import scaler
from integrator import integrator
from regressor import regressor
from weight import weighter
from inputs import inputs

def test_func_regressor_production():
    # regression example using data from test_func_create_dataset()
    n_comp = 31
    w_species = 0
    final_time = 10e-5
    #############################
    ############################# INPUT DATA
    ############################# 
    ### LOAD INPUT DATA
    print("Reading Inputs")
    ioc = inputs()
    dirname = 'data/uox_multiple/qtfiles/'
    powers = ['050','100','150','200','250','300']
    lengths = ['5','6','7','8','9','10']
    for i in range(1):#len(powers)):
        for j in range(len(lengths)-3):
            simname = powers[i]+'_'+lengths[j]
            if i != 3 and j != 3:
                ioc.read_dens_file(dirname+simname)
                ioc.read_rates_file(dirname+simname)
    ioc.salt_arrays_for_log()
    ioc.permute_data_in_time()

    ### INITIALIZE SCALER
    print("Initializing Scaler")
    num_samples, num_features = ioc.data_rand.shape
    w_array = np.zeros(num_features)
    w_array[w_species] = 0
    w_mag = 10.
    #sc = scaler(ioc.data_rand, num_samples, num_features, 1,w_mag,w_array)
    sc = scaler(ioc.data_rand, num_samples, num_features, 1,w_mag,w_array)
    
    #############################
    ############################# SCALE DATA AND RUN PCA
    #############################
    
    # SET SCALER TO STD SCALING AND APPLY
    # (USE LOG AS WELL)
    print("Scaling")
    sc.scale_log()
    sc.scale_std()
    sc.center()
    sc.apply_scaling()
    
    # SET PCA AND APPLY
    print("Applying PCA")
    pcac = pca(sc.data, num_samples, num_features, n_comp)
    pcac.pca_scipy()

    #############################
    ############################# CREATE REGRESSOR AND INTEGRATE
    ############################# 
    
    print("Initializing Regression")
    reg = regressor(ioc.data_rand, pcac.principal_components, ioc.rates_rand, sc.D, \
                    pcac.principal_axis, num_samples, num_features, n_comp)
    print("Training regression model")
    reg.train_GPR()
    
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

   #print(computed_densities.shape)
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
    plt.semilogy(sc.data[:,:])
    plt.ylim([10e11,10e19])
    
    plt.figure(7)
    plt.semilogy(ioc.bulkdata[:,:])
    plt.ylim([10e11,10e19])
    print(ioc.bulkdata[0,:])
    print(y0)