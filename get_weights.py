import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


def get_weights(model):
    #for layer in model.layers:
    #    weights = layer.get_weights() # list of numpy arrays
    return model.get_weights()

def plot_weights( models):

    plt.figure( figsize=(7,7) )
    plt.semilogy()

    bins = -1* np.flip( np.logspace(-3,1, num=250) )
    bins = np.append(bins, np.linspace(-1e-3, 1e-3, 10))
    bins = np.append(bins, np.logspace(-3,1, num=250))

    for i,model in enumerate(models):
        flat_weights = np.array( [] )

        weights = get_weights(model)
        for weight in weights:
            flat_weights = np.append(flat_weights, weight.flatten())

        # percentate of zeros in the weights
        rel_zeros = np.sum(flat_weights==0) / np.size(flat_weights)

        #h, b = np.histogram(flat_weights, bins=1000)
        #plt.bar( b[:-1], h, width=b[1]-b[0], label="%s %5.3f %% zeros"%(model.name, rel_zeros*100) )

        plt.hist( flat_weights, bins=bins, label="%s %5.3f %% zeros"%(model.name, rel_zeros*100) )

    model_names = ""
    for model in models:
        model_names += "_" + model.name

    plt.xscale('symlog')
    plt.xlabel('weights')
    plt.ylabel('counts')

    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/weights"+model_names+".png")
    plt.show()

