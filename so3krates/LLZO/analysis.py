import wandb
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (12, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)
import numpy as np
from scipy.stats import gaussian_kde
import argparse

def plot_energy_and_force_distributions(data_file=None,output_file=None):
    data = np.load(data_file)
    x = np.arange(min(data['E']) - 2, max(data['E']) + 2, 0.1)
    print(min(data['E']))
    #kernel = gaussian_kde(data['E'])
    plt.subplot(1, 2, 1)
    #plt.plot(x, kernel(x), '-', color='#cb0000', lw=3)
    plt.hist(data['E'], bins=100, color='#cb0000')
    print(len(data['E']))
    plt.xlabel('Energy (eV)')
    plt.ylabel('Frequency')

    F = np.ndarray.flatten(data['F'])
    F = np.absolute(F)
    #x = np.arange(min(F) - 1, max(F) + 1, 0.1)
    #kernel = gaussian_kde(F)

    plt.subplot(1, 2, 2)
    #plt.plot(x, kernel(x), '-', color='#cb0000', lw=3)
    plt.hist(F, bins=300, color='#cb0000')
    print(len(F))
    #plt.xlim([-0.3,0.3])
    plt.xscale('log')
    plt.xlabel('Force (eV/atom)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='workflow control for analysing the data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plot_data_distribution", action='store_true',
                        help="plot the disribution of the energies and forces in the database for model training")
    parser.add_argument("-d",  "--data_file", type=str, help="data file containing the energies and forces for model training")
    parser.add_argument("-o", "--output_file", type=str, help="name of the output file")
    args = parser.parse_args()

    if args.plot_data_distribution:
        plot_energy_and_force_distributions(data_file=args.data_file, output_file=args.output_file)
