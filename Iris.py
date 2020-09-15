# Iris classification

import numpy as np
from EP_functions import free_phase, weaklyClamped_phase, weight_update
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn import datasets

font = {'family': 'sans-serif', 'weight': 'bold', 'size': '20'}
rc('font', **font)  # pass in the font dict as kwargs

# Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
y_onehot = np.zeros((150, 3), dtype=int)
y_onehot[np.arange(150), y] = 1

# Hyperparameters
eps = 0.9
beta = 1
alpha = 0.15
n_iter = 30

fan_in = 4
fan_hidden = 5
fan_out = 3
fan_all = fan_in + fan_hidden + fan_out

# Initialization
w = np.asarray(np.random.RandomState().uniform(low=-np.sqrt(6. / (fan_in + fan_out)),
                                               high=np.sqrt(6. / (fan_in + fan_out)),
                                               size=(fan_all, fan_all)))
w = (w + w.T) / 2
w[np.diag_indices(fan_all)] = 0

b = 0

# Training
ff_run_time = np.zeros(120)
wc_run_time = np.zeros(120)

for epoch in np.arange(120):
    v = X[epoch]

    # free phase
    s, S_ff, run_time = free_phase(w, b, v, n_iter, eps, fan_in, fan_hidden,
                                   fan_out)
    u_ff = np.hstack((v, s))
    ff_run_time[epoch] = run_time

    # weakly clamped phase
    s, S_wc, run_time = weaklyClamped_phase(w, b, v, y_onehot, epoch, beta,
                                            n_iter, eps, fan_in, fan_hidden,
                                            fan_out)
    u_wc = np.hstack((v, s))
    wc_run_time[epoch] = run_time

    # Weight update
    w = weight_update(u_ff, u_wc, alpha, beta, w, fan_all)

# Plot free phase state dynamics (training)
fig = plt.figure(figsize=(9, 12))
plt.plot(S_ff.T, linewidth=3.0)
plt.ylabel('State Values', fontsize=30)
plt.axis('tight')
plt.tight_layout()
plt.ylim(0, 1.2)
plt.xlabel('Number of iterations', fontsize=30)
plt.legend(('h1', 'h2', 'h3', 'h4', 'h5', 'y1', 'y2', 'y3'),
           loc='upper right')
plt.show()

# Prediction
acc = 0
for epoch in np.arange(120, 150):

    v = X[epoch]

    # free phase
    s, S_ff, run_time = free_phase(w, b, v, n_iter, eps, fan_in, fan_hidden,
                                   fan_out)
    if np.array_equal(y[epoch], np.argmax(np.array(s[fan_hidden:fan_hidden +
                                                                fan_out]))):
        acc = acc + 1

acc = acc * 100 / 30
print('Accuracy percentage', acc)
