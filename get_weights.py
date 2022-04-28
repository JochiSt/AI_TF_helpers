import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


def get_weights(model):
    #for layer in model.layers:
    #    weights = layer.get_weights() # list of numpy arrays
    return model.get_weights()

def plot_weights(model):
    flat_weights = np.array( [] )

    weights = get_weights(model)
    for weight in weights:
        flat_weights = np.append(flat_weights, weight.flatten())

    h, b = np.histogram(flat_weights, bins=100)
    plt.figure( figsize=(7,7) )
    plt.bar( b[:-1], h, width=b[1]-b[0] )
    plt.semilogy()

    # percentate of zeros in the weights
    rel_zeros = np.sum(flat_weights==0) / np.size(flat_weights)
    print("% of zeros = {}".format( rel_zeros*100. ) )

    ax = plt.gca()
    plt.text(0.1, 0.9, '%% of zeros = %5.3f'%(rel_zeros*100),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes )


    plt.savefig(model.name+"_weights.png")
    plt.show()

