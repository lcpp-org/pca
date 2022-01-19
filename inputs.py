import numpy as np
import matplotlib.pyplot as plt


def speciesTag(species):
  # returns species name for species index
  dictionary = {
     0: "Time",
     1: "electrons",        
     2: "U",        
     3: "UO",       
     4: "UO2",      
     5: "UO3",      
     6: "U2O2",     
     7: "U2O3",     
     8: "U^+",      
     9: "UO^+",     
    10: "UO2^+",    
    11: "UO2^-",    
    12: "UO3^-",    
    13: "O",        
    14: "O(1D)",    
    15: "O(1S)",    
    16: "O2(V1)",   
    17: "O2(V2)",   
    18: "O2(V3)",   
    19: "O2(V4)",   
    20: "O2(A1)",   
    21: "O2(B1)",   
    22: "O2(4.5EV)",
    23: "O3",       
    24: "O^+",      
    25: "O2^+",     
    26: "O4^+",    
    27: "O^-",      
    28: "O2^-",     
    29: "O3^-",    
    30: "O4^-",     
    31: "O2"  
    }
  return dictionary.get(species)

class inputs:
    def __init__(self):
        # variable initialization
        self.data = 0
        self.bulkdata = 0
        self.time = 0
        self.power = 0
        self.rawrates = 0
        self.qtmatrix = 0
        self.rates = 0
        self.bulkrates = 0

        self.data_rand = 0
        self.rates_rand = 0
    
    def read_dens_file(self, dirname):
        # reads a qt_densities file and retrieves the raw data
        filename = dirname+"/qt_densities.txt"
        print("reading: "+filename)
        values = np.genfromtxt(filename, delimiter='')
        #header = values[0]
        self.time = values[1::,0]
        self.data = values[1::,1::]
        #conditions = self.read_conditions_file(dirname)

        # append if necessary
        if type(self.bulkdata) == type(self.data):
            self.bulkdata = np.append(self.bulkdata, self.data[1600:,:], 0)
        if type(self.bulkdata) != type(self.data):
            self.bulkdata = self.data[1600:,:]

    
    def read_rates_file(self,dirname):
        # reads a qt_rates file and retrieves the raw data
        filename = dirname+"/qt_rates.txt"
        print("reading: "+filename)
        values = np.genfromtxt(filename, delimiter='')
        #header = values[0]
        self.time = values[1::,0]
        self.rawrates = values[1::,1::]
        # process rates
        self.read_qtmat_file(dirname)
        self.find_source_terms()
        # append if necessary
        if type(self.bulkrates) == type(self.rates):
            #print(type(self.bulkrates), type(self.rates))
            self.bulkrates = np.append(self.bulkrates, self.rates[1600:,:], 0)
        if type(self.bulkrates) != type(self.rates):
            self.bulkrates = self.rates[1600:,:]
            
    def read_qtmat_file(self,dirname):
        # reads a qt_matrix file, which is a species <-> reactions
        # rate convertor
        filename = dirname+"/qt_matrix.txt"
        print("reading: "+filename)
        values = np.genfromtxt(filename, delimiter='')
        self.qtmatrix = values
        
    def read_conditions_file(self,dirname):
        filename = dirname+"/qt_conditions.txt"
        print("reading: ",filename)
        values = np.genfromtxt(filename, delimiter='')
        self.time = values[1::,0]
        self.power = values[1::,1]
        #plt.figure(1)
        #plt.plot(self.power)
        #plt.show()
                        
    def find_source_terms(self):
        #assemble the per-species rates matrix
        self.rates = np.zeros(self.data.shape)
        for i in range(self.rates.shape[0]):
          for j in range(self.rates.shape[1]):
            for k in range(self.rawrates.shape[1]):
              self.rates[i,j]+=self.rawrates[i,k]*self.qtmatrix[j,k]

    def permute_data_in_time(self):
        ### randomly choose observations from time, voltage, and pressure spaces
        #np.random.seed(43)
        shuffler = np.random.permutation(self.bulkrates.shape[0])
        self.data_rand = self.bulkdata[shuffler]
        self.rates_rand = self.bulkrates[shuffler]
        
    def salt_arrays_for_log(self):
        # this adds a very small value to densities to allow for easy usage of np.log()
        for i in range(self.bulkdata.shape[0]):
            for j in range(self.bulkdata.shape[1]):
                if self.bulkdata[i,j] <= 0.:
                    self.bulkdata[i,j] = 1e-10