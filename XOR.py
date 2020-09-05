# XOR function

import numpy as np
from EP_functions import free_phase, weaklyClamped_phase, weight_update
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'sans-serif','weight' : 'bold','size'   : '20'}
rc('font', **font)  # pass in the font dict as kwargs


# Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
y_onehot=np.array([[0],[1],[1],[0]])


# Hyperparameters
eps = 0.97
beta = 1
alpha = 0.15
n_iter = 100
n_epoches = 100

fan_in = 2
fan_hidden = 2
fan_out = 1
fan_all = fan_in + fan_hidden + fan_out


# Initialization
w = np.asarray(np.random.RandomState().uniform(low=-np.sqrt(6. / (fan_in+fan_out)),high=np.sqrt(6. / (fan_in+fan_out)),size=(fan_all, fan_all)))
w = (w + w.T)/2
w[np.diag_indices(fan_all)] = 0

b = 0

# Training
for e_epoch in np.arange(n_epoches):
    order = np.argsort(np.random.random(y.shape))
    X = X[order]
    y = y[order]
    y_onehot = y_onehot[order]
    for epoch in np.arange(4):
        v = X[epoch]
        # free phase
        s, S_ff, run_time = free_phase(w, b, v, n_iter, eps, fan_in, fan_hidden, 
                                      fan_out)
        u_ff = np.hstack((v, s))
        
        # weakly clamped phase
        s, S_wc, run_time = weaklyClamped_phase(w, b, v, y_onehot, epoch, beta,
                                                n_iter, eps, fan_in, fan_hidden,
                                                fan_out)
        u_wc = np.hstack((v, s))
        
        # Weight update
        w = weight_update(u_ff, u_wc, alpha, beta, w, fan_all)
        

# Prediction
order = np.argsort(np.random.random(y.shape))
X = X[order]
y = y[order]
y_onehot = y_onehot[order]

n_iter = 30
RGB_color_array = np.array([[0.92,1,0.41], [0.04,0.6,0.95], [0.98,0.82,0.02]])
for epp in np.arange(4):
    v = X[epp]
    
    # free phase
    s, S_ff, run_time = free_phase(w, b, v, n_iter, eps, fan_in, fan_hidden, 
                                   fan_out)
    print(X[epp], y[epp], s[fan_hidden:fan_hidden + fan_out])
    print('Run time=', run_time)
    
    # Plot ff prediction
    fig = plt.figure(figsize=(12, 8))
    plt.plot(S_ff[0,:], linewidth=8.0, color='red', linestyle='--')
    plt.plot(S_ff[1,:], linewidth=8.0, color='red', linestyle=':')
    plt.plot(S_ff[2,:], linewidth=8.0, color=RGB_color_array[2])
    plt.ylabel('State values', fontsize=30)
    plt.axis('tight')
    plt.tight_layout()
    plt.ylim(0, 1.1)
    plt.xlabel('Time step', fontsize=30)
    plt.legend(('h3', 'h4', 'y5'), loc='center right', fontsize=30)
    plt.show()
    fig.savefig('ffstate_dynamics{}.png' .format(epp), dpi=100)
