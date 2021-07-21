# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:02:07 2021

@author: Camilla
"""

import numpy as np

#For this, we are sticking to this simple case, but I wrote out f(x) in case 
#we want to use another function relating the activity of the presynaptic neuron 
#to the input to the postsynaptic neuron
def f(x):
    """
    Equation 2 from the reference
    Input: 1D array
    Output: the same 1D array, but with negative values set to zero
    """
    output = x.clip(min = 0)
    return output

# Function to perform the "talking" of the neurons

def talking(firing, weights, initials, t_size, dt, t_initial, theta, tau, mean=0, sigma=0, num=2, ext = 0):
    """
    Implementing dynamics described in Equation 1 (from the reference)
    
    Parameters:
        firing: firing rate function relating firing rate to current
        
        weights: 2D array 
            each element in weights describes the strength/type of the synapse 
            from presynaptic neuron (row) to postsynaptic neuron (column)
            elements in the main diagonal must be set to zero, 
            since each postsynaptic neuron cannot get a presynaptic signal from itself
        
        initials: 1D array 
            initial conditions for each neuron
            
        t_size: integer
            number of iterations in the time array
            
        dt: float
            size of the time steps
            
        theta: constant
            a bias
        
        tau: 1D numpy array
            time scales for each neuron
        
        mean, sigma: of the noise that we want, generated for the system
        
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

        dot = np.dot(values[t - 1], weights)
        #matrix product of vector containing previous states of neurons and
        #the matrix containing weights (column: postsynaptic neurons, row: presynaptic neurons)
       
        #values[t - 1]: row containing states of neurons in the previous time      
        
        values[t] = values[t - 1] * (1 - dt/tau) + (dt/tau)*firing(dot + theta + ext)
        #euler integration of equation 1, applied to all neurons for this time t
        #theta: a constant indicating bias
        #ext: external input, set to 0 for now
        #these operations with 1D arrays are element by element
                 
    return values


# Parameters
theta = 2
initials = [10, 10, 8]
tau = np.array([5, 2, 3])
weights = [[0, -3, 1],
           [2, 0, -1],
           [2, 1, 0]]

#each element in weights describes the strength/type of the synapse 
#from presynaptic neuron (row) to postsynaptic neuron (column)

#elements in the main diagonal are set to zero, 
#since each postsynaptic neuron cannot get a presynaptic signal from itself

t_length = 100
dt_size = 0.1
tarray_size = int(t_length / dt_size)

sim = talking(f, weights, initials, tarray_size, dt_size, 0, theta, tau, num = 3)
print(sim)