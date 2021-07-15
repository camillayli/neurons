# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:52:34 2021

@author: Camilla
"""


import numpy as np

#For this, we are sticking to this simple case, but I wrote out f(x) in case 
#we want to use another function relating the activity of the presynaptic neuron 
#to the input to the postsynaptic neuron
def f(x):
    """
    Equation 2 from the reference
    """
    if (x <= 0):
        output = 0
        #in this case, 
        #change in activity of the neuron only depends on itself
    else: 
        output = x
    return output

# Function to perform the "talking" of the neurons

def talking(function, weights, initials, t_size, dt, t_initial, theta, mean=0, sigma=0, num=2, ext = 0):
    """
    Implementing dynamics described in Equation 1 (from the reference)
    
    Parameters:
        function: firing rate function relating firing rate to current
        
        weights: 2D array 
            each element in weights describes the strength/type of the synapse 
            from presynaptic neuron (column) to postsynaptic neuron (row)

            elements in the main diagonal must be set to zero, 
            since each postsynaptic neuron cannot get a presynaptic signal from itself
        
        initials: 1D array 
            initial conditions for each neuron

        t_size: integer
            number of iterations in the time array

        dt: float
            size of the time steps

        mean, sigma = of the noise that we want, generated for the system

        num: integer
            number of neurons in the system.

        ext: external input. We're setting this to 0 for now
    """

    values = np.zeros((t_size, num))
    # column indicates neuron, row indicates time, with row index 0 being the initial t = 0
    # num = number of rows in this array
    # t_size = number of columns in this array
    
    values[0] = initials
    # puts the initial values for all neurons in the row t = 0

    for t in range(1, t_size):
        # loops through time
        # Lower limit: 1, because we will not be updating the initial values in the t = 0 row
        # Upper limit: t_size
        # t = row index

        for n in range(0, num):
            # loops through all the different neurons
            # n = column index
            
            dot = np.dot(weights[n], values[t - 1])
            #dot product of vector containing weights and vector containing previous states of neurons
            
            #weights[n]: row containing strength/type of the synapses 
            #from presynaptic neurons to a particular postsynaptic neuron n 
            #values[t - 1]: row containing states of neurons in the previous time 
            
            #euler integration of equation 1:
            values[t][n] = values[t - 1][n] * (1 - dt/tao) + (dt/tao)*function(dot + theta + ext)
            #theta: a constant indicating bias
            #ext: external input, set to 0 for now
                 
    return values


# Parameters
theta = 2
initials = [10, 10, 8]
tao = 5
weights = [[0, -3, 1],
           [2, 0, -1],
           [2, 1, 0]]
#each element in weights describes the strength/type of the synapse 
#from presynaptic neuron (column) to postsynaptic neuron (row)

#elements in the main diagonal are set to zero, 
#since each postsynaptic neuron cannot get a presynaptic signal from itself

t_length = 100
dt_size = 0.1
tarray_size = int(t_length / dt_size)

sim = talking(f, weights, initials, tarray_size, dt_size, 0, theta, num = 3)
print(sim)
