import h5py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '11',
          'figure.figsize': (7, 4),
          'axes.labelsize': 16,
          'axes.titlesize': 16,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13}
pylab.rcParams.update(params)

time_step=1.0
traj = h5py.File('trajectory_300K.h5')
potential_energies=traj['potential_energy'][100:]
plt.plot([time_step*i/1000.0 for i in range(len(potential_energies))],potential_energies,'-',c='#FFB55A',label='300 K')

traj = h5py.File('trajectory_1500K.h5')
potential_energies=traj['potential_energy'][100:]
plt.plot([time_step*i/1000.0 for i in range(len(potential_energies))],potential_energies,'-',c='#F694BD',label='1500 K')


plt.xlabel('Time (ps)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.tight_layout()
plt.savefig('energy_trajectory.pdf')