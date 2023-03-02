'''
Generate plots of activation function for the thesis.
'''

from visualisation_parameter_setup import * 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

def sig(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def step(x):
    return np.array(x >= 0, dtype=np.int)

x = np.arange(-5, 5, step = 0.001)

# perceptron step function
# plt.plot(x, step(x))
# plt.grid(True, "both")
# plt.show()

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize = (9, 3))

#https://matplotlib.org/stable/tutorials/text/usetex.html
ax1.plot(x, sig(x))
ax1.grid(True, "both")
ax1.set_title("sigmoid")
# ax1.annotate(r"$\sum_{i=0}^\infty x_i$", xy=(0.05, 0.95), xycoords='axes fraction', font = font_prop)

ax2.plot(x, relu(x))
# ax2.annotate("f(x) = max(0, x)", xy=(0.05, 0.95), xycoords='axes fraction')
ax2.grid(True, "both")
ax2.set_title("ReLU")

ax3.plot(x, tanh(x))
# ax3.annotate("f(x) = max(0, x)", xy=(0.05, 0.95), xycoords='axes fraction')
ax3.grid(True, "both")
ax3.set_title("tanh")

fig.tight_layout()


# plt.plot(x, sig(x))
# plt.grid(True, "both")
# plt.set_title("sigmoid")

plt.show()
print("")
