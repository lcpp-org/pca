import numpy as np
from scipy.integrate import solve_ivp

####### Functions to integrate various ODE Types:
    # ODE_function: General purpose integrator of dn/dt = k
    # ODE_function_onefeat: General purpose integrator of dn/dt = k
    #                       but with different regression function 
    # ODE_function_test: Integrates a test 4-species ODE
    # ODE_function_test_simple: Integrates a test 2-species ODE
    # ODE_function_test_simple_2: implements a test 3-species ODE
    
    
def ODE_function(t, state, regressor_class):
    # the function dy/dt = f(t,y) = S_n
    
    # first find source terms via regressor
    source_terms = regressor_class.run_regression(state.reshape(1,-1))

    # then return next state
    dndt = source_terms
    
    return dndt

def ODE_function_onefeat(t, state, regressor_class):
    # the function dy/dt = f(t,y) = S_n
    
    # first find source terms via regressor
    source_terms = regressor_class.run_regression_onefeat(state.reshape(1,-1))

    # then return next state
    dndt = source_terms
    
    return dndt

def ODE_function_test(t,state, k12, k13, k23, k24, k34 ):
    # test ode fucntion, system of 4 species
    dn1dt = -k12*state[0] + k13*state[2]
    dn2dt =  k12*state[0] - k23*state[1] - k24*state[1]
    dn3dt =  k23*state[1] - k13*state[2] - k34*state[2]
    dn4dt =  k24*state[1] + k34*state[2]
    return np.array([dn1dt, dn2dt, dn3dt, dn4dt])

def ODE_function_test_simple(t,state,):
    # test ode fucntion, system of 2 species
    dn1dt = -0.5*state[0]
    dn2dt = 0.5*state[0]
    return np.array([dn1dt, dn2dt])

def ODE_function_test_simple_2(t,state,):
    # test ode fucntion, system of 3 species
    dn1dt = -0.5*state[0]
    dn2dt = 0.5*state[0] - 0.5*state[1]
    dn3dt = 0.5*state[1]
    return np.array([dn1dt, dn2dt, dn3dt])


##########################
##########################
##########################
####### Class holding integration functions as well as relevant information
    # Each integration routine calls the function which matches its name, as
    # listed above
    
class integrator:
    def __init__(self,final_time, principal_axes, principal_components, \
                 density, regressor_class):
        # variable initialization
        self.t_span = (0.0, final_time)
        self.t = np.arange(0.0,final_time, final_time/100)
        self.regressor_class = regressor_class
        self.A = principal_axes
        self.Z = principal_components
        self.density = density
        
        
    def integrate_ODEs(self,y0):
        # this integrates the ODEs described in ODE_function
        # use initial values from PC matrix, or from real densities
        # real densities requires multiplying the row vector by the
        # principal axes matrix, like so
        # y0_components = y0_real.dot(self.A)

        p = (self.regressor_class,)
        result = solve_ivp(ODE_function, self.t_span, y0, args=p, \
                           method='LSODA',t_eval=self.t)
        return result
    
    def integrate_ODEs_onefeat(self,y0):
        # this integrates the ODEs described in ODE_function
        # use initial values from PC matrix, or from real densities
        # real densities requires multiplying the row vector by the
        # principal axes matrix, like so
        # y0_components = y0_real.dot(self.A)
        
        #y0 = self.Z[0,:]
        p = (self.regressor_class,)
        result = solve_ivp(ODE_function_onefeat, self.t_span, y0, args=p, \
                           method='LSODA',t_eval=self.t)
        return result
    
    def integrate_ODEs_test(self, initial_state):
        # TEST ODE INTEGRATION FUNCTION
        # use initial values from PC matrix, or from real densities
        # real densities requires multiplying the row vector by the
        # principal axes matrix, like so
        # y0_components = y0_real.dot(self.A)
        
        self.t_span = (0.0, 10.0)
        self.t = np.arange(0.0,10.0,10.0/50)
        
        y0 = initial_state
        p = (0.5,1.5,1,0.2,0.2)
        result = solve_ivp(ODE_function_test, self.t_span, y0, args=p, \
                           method='LSODA',t_eval=self.t)
        return result
    
    def integrate_ODEs_test_simple(self, initial_state):
        # TEST ODE INTEGRATION FUNCTION
        # use initial values from PC matrix, or from real densities
        # real densities requires multiplying the row vector by the
        # principal axes matrix, like so
        # y0_components = y0_real.dot(self.A)
        
        self.t_span = (0.0, 10.0)
        self.t = np.arange(0.0,10.0,10.0/50)
        
        y0 = initial_state
        result = solve_ivp(ODE_function_test_simple, self.t_span, y0, \
                           method='LSODA',t_eval=self.t)

        return result
    
    def integrate_ODEs_test_simple_2(self, initial_state):
        # TEST ODE INTEGRATION FUNCTION
        # use initial values from PC matrix, or from real densities
        # real densities requires multiplying the row vector by the
        # principal axes matrix, like so
        # y0_components = y0_real.dot(self.A)
        
        self.t_span = (0.0, 10.0)
        self.t = np.arange(0.0,10.0,10.0/50)
        
        y0 = initial_state
        result = solve_ivp(ODE_function_test_simple_2, self.t_span, y0, \
                           method='LSODA',t_eval=self.t)
        return result