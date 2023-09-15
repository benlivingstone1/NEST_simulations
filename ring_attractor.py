from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn

import nest


# Creates Sombrero or Ricker wavelet function
# sig1 controls how wide the peak is
# sig2 controls how fast the negative lobes asymptote to 0 

def sombrero(i, j, num_neurons, sigma1 = 4, sigma2 = 15):
    # Calculate the shortest distance between neurons i and j, taking into account periodic boundary conditions
    distance = min(abs(i - j), num_neurons - abs(i - j))
    return 5 * (1 - (distance ** 2) / sigma1 ** 2) * np.exp(-(distance ** 2) / (2 * sigma2 ** 2))


if __name__ == "__main__": 
    nest.ResetKernel()
    
    num_neurons = 100
    parameter_dict = {"I_e": 175.0, "tau_m": 20.0, "V_m": -77.0, "V_th": -55.0}
    # parameter_dict = {"tau_m": 20.0, "V_m": -77.0}
    neurons = nest.Create("iaf_psc_alpha", num_neurons, params=parameter_dict)

    # Connect neurons with all-to-all connectivity and sombrero weights 
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:  # no self-connections
                nest.Connect(neurons[i], neurons[j], syn_spec={"weight": sombrero(j, i, num_neurons)})

    # # Create DC generator
    # generator = nest.Create("dc_generator")
    # # Set the amplitude of the DC generator
    # nest.SetStatus(generator, {"amplitude": 188.0})

    # Create a Poisson generator 
    generator = nest.Create("poisson_generator")
    # Set the rate of the Poisson generator
    nest.SetStatus(generator, {"rate": 7000.0})

    # Connect generator to first 10 neurons 
    nest.Connect(generator, neurons)

    spikerecorder = nest.Create("spike_recorder")
    nest.Connect(neurons, spikerecorder)

    nest.Simulate(1000.0)

    nest.raster_plot.from_device(spikerecorder)
    plt.show()

    # # ***********************************
    # # INSPECT WEIGHTS OF SINGLE NEURON
    # # ***********************************
    # neur_inspect = neurons[1]
    # connections = nest.GetConnections(source=neur_inspect)

    # weights = nest.GetStatus(connections, "weight")
    # # Create a histogram of the weights
    # plt.hist(weights, bins=20)
    # plt.xlabel('Weight')
    # plt.ylabel('Number of connections')
    # # plt.plot(weights)
    # plt.show()

    # # ***********************
    # # CHECK WEIGHTS FUNCTION
    # # ***********************
    # x = np.linspace(1, 100, 1000)
    # output = []
    # for i in x:
    #     out = sombrero(i, 50)
    #     output.append(out)

    # plt.figure(1)
    # plt.plot(x, output)
    # plt.show()


