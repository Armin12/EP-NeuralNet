# Author: Armin Najarpour Foroushani -- <armin.najarpour@gmail.com>

import numpy as np
import time


def hard_sigmoid(u):
    """Hard sigmoid function
    
    For a given value u as input, it gives hard sigmoid of u as output.
    
    Parameters
    ----------
    u : float
    
    Returns
    ----------
    float
    
    """
    if u > 1:
        ro = 1
    elif u < 0:
        ro = 0
    else:
        ro = u
    return ro


def hard_sigmoid_array(u):
    """Hard sigmoid function for an array of values
    
    For a given array u as input, it gives hard sigmoid of u as output.
    
    Parameters
    ----------
    u : numpy.ndarray
    
    Returns
    ----------
    numpy.ndarray
    
    """
    ro = np.zeros(len(u))
    for i in np.arange(len(u)):
        ro[i] = hard_sigmoid(u[i])
    return ro


def hard_sigmoid_derivative(u):
    """Derivative of hard sigmoid function
    
    For a given value u as input, it gives hard sigmoid derivative of u as output.
    
    Parameters
    ----------
    u : float
    
    Returns
    ----------
    float
    
    """
    if u > 1:
        ro = 0
    elif u < 0:
        ro = 0
    else:
        ro = 1
    return ro


def free_phase(w, b, v, n_iter, eps, fan_in, fan_hidden, fan_out):
    """Free phase function
    
    This function determines states dynamics and their final values, and the 
    convergence time as output according to the free phase.
    
    Parameters
    ----------
    w : numpy.ndarray
       Weights
    b : float
       bias
    v : numpy.ndarray
       inputs
    n_iter : int
       number of iterations to update state values
    eps : float
        step size
    fan_in : int
       number of input states
    fan_hidden : int
       number of hidden states
    fan_out : int
       number of output states
    
    Returns
    ----------
    numpy.ndarray
    numpy.ndarray
    float
    """
    start_time = time.time()
    s = np.asarray(
        np.random.RandomState().uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)),
                                        size=(fan_hidden + fan_out,)))
    u = np.hstack((v, s))
    S = np.zeros((fan_hidden + fan_out, n_iter))
    for j in np.arange(n_iter):
        for i in np.arange(len(s)):
            s[i] = hard_sigmoid(s[i] + eps * (
                        hard_sigmoid_derivative(s[i]) * (np.inner(hard_sigmoid_array(u), w[i + fan_in]) + b) - s[i]))
        u = np.hstack((v, s))
        S[:, j] = s
    run_time = time.time() - start_time
    return s, S, run_time


def weaklyClamped_phase(w, b, v, y_onehot, epoch, beta, n_iter, eps, fan_in, fan_hidden, fan_out):
    """Weakly clamped phase function
    
    This function determines states dynamics and their final values, and the 
    convergence time as output according to the weakly clamped phase.
    
    Parameters
    ----------
    w : numpy.ndarray
       Weights
    b : float
       bias
    v : numpy.ndarray
       inputs
    y_onehot : numpy.ndarray
       onehot coding of labels
    epoch : int
       epoch number
    beta : float
       beta value in EP
    n_iter : int
       number of iterations to update state values
    eps : float
        step size
    fan_in : int
       number of input states
    fan_hidden : int
       number of hidden states
    fan_out : int
       number of output states
    
    Returns
    ----------
    numpy.ndarray
    numpy.ndarray
    float
    """
    start_time = time.time()
    s = np.asarray(
        np.random.RandomState().uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)),
                                        size=(fan_hidden + fan_out,)))
    u = np.hstack((v, s))
    S = np.zeros((fan_hidden + fan_out, n_iter))
    for j in np.arange(n_iter):
        for i in np.arange(len(s)):
            if i < fan_hidden:
                s[i] = hard_sigmoid(s[i] + eps * (
                            hard_sigmoid_derivative(s[i]) * (np.inner(hard_sigmoid_array(u), w[i + fan_in]) + b) - s[
                        i]))
            else:
                s[i] = hard_sigmoid(s[i] + eps * (
                            hard_sigmoid_derivative(s[i]) * (np.inner(hard_sigmoid_array(u), w[i + fan_in]) + b) - s[
                        i] + beta * (y_onehot[epoch, i - fan_hidden] - s[i])))
        u = np.hstack((v, s))
        S[:, j] = s
    run_time = time.time() - start_time
    return s, S, run_time


def weight_update(u_ff, u_wc, alpha, beta, w, fan_all):
    """Weight update function
    
    This function updates the weights.
    
    Parameters
    ----------
    u_ff : numpy.ndarray
       free phase updated units
    u_wc : numpy.ndarray
       weakly clamped phase updated units
    alpha : float
       
    beta : float
       beta value in EP
    w : numpy.ndarray
       Weights
    fan_all : int
       number of all units
    
    Returns
    ----------
    numpy.ndarray
    """
    mult_wc = np.matmul(np.reshape(hard_sigmoid_array(u_wc), (fan_all, 1)),
                        np.reshape(hard_sigmoid_array(u_wc), (1, fan_all)))
    mult_ff = np.matmul(np.reshape(hard_sigmoid_array(u_ff), (fan_all, 1)),
                        np.reshape(hard_sigmoid_array(u_ff), (1, fan_all)))
    delta_w = alpha * (1 / beta) * (mult_wc - mult_ff)
    delta_w[np.diag_indices(fan_all)] = 0
    w = w + delta_w
    return w
