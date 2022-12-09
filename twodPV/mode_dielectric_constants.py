import re
import numpy as np

from pymatgen.io.vasp.outputs import *

from core.dao.vasp import VaspReader

from matplotlib import rc, patches
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

rc('text', usetex=True)
params = {'legend.fontsize': '12',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)


class PhononMode(object):

    def __init__(self, index=0, freq=None, eigenvec=None):
        self.index = index
        self.freq = freq  # in THz
        self.eigenvec = eigenvec

    @property
    def is_imaginary(self):
        return np.imag(self.freq) != 0.0


class PhononSepctrum(object):

    def __init__(self):
        self.modes = []

    def get_mode_by_freq(self, freq):
        for m in self.modes:
            if m.freq == freq * TETRA * 2 * math.pi:
                return m


def get_phonon_spectrum():
    global spectrum, freq, mode, read_eigenvectors, mode
    f = open('OUTCAR', 'r')
    all_lines = f.readlines()
    spectrum = PhononSepctrum()
    for line in all_lines:
        ls = line.split()
        # Read in mode eigenfrequencies
        if re.match(r"(\s+)(\d+)(\s+)f(\D+)=(\s+)(\d+).(\d+)(\s+)THz", line):
            if 'f/i' not in line:
                freq = float(ls[3])
            else:
                freq = float(ls[2]) * 1j

            freq = freq * TETRA * 2 * math.pi
            mode = PhononMode(index=int(ls[0]) - 1, freq=freq)

            read_eigenvectors = True
            this_eigenvector = []

        # Read in mode eigenvectors
        if read_eigenvectors:
            if re.match(r"(\s+)(\S+).(\d+)" * 6, line):
                this_atom_mode = np.array([float(k) for k in line.strip().split()[-3:]])
                this_eigenvector.append(this_atom_mode)

            elif ('X ' not in line) and ('THz' not in line):
                read_eigenvectors = False
                mode.eigenvec = np.array(this_eigenvector)
                spectrum.modes.append(mode)
    return spectrum


TETRA = 1e12
ELECTRON_CHARGE = 1.602176487e-19  # Coulomb
A_IN_METRE = 1e-10  # coverts angstorm to meter
AMU = 1.66e-27  # kg
EPSILON_0 = 8.854E-12  # C2N−1m−2

apply_2D_correction = False
read_eigenvectors = False

spectrum = get_phonon_spectrum()

outcar = Outcar('./OUTCAR')
born_charges = outcar.born

crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
print(crystal.lattice.volume)

if apply_2D_correction:
    all_z_positions = np.array([a.scaled_position.z for a in crystal.asymmetric_unit[0].atoms])
    all_z_positions = all_z_positions - np.round(all_z_positions)
    all_z_positions = [z * crystal.lattice.c for z in all_z_positions]
    slab_thick = max(all_z_positions) - min(all_z_positions)

from core.models.element import atomic_mass_dict

mass_list = [math.sqrt(1.0 / atomic_mass_dict[a.label.upper()]) for a in
             crystal.all_atoms(sort=False)]  # Here mass is given in the atomic mass unit (a.m.u.)

Z_tensor = [[0.0 for _ in [0, 1, 2]] for m in range(len(spectrum.modes))]
for m in range(len(spectrum.modes)):  # loop around the mode index
    for alpha in [0, 1, 2]:  # loop around the x,y,z direction
        Z_m_alpha = 0

        for i in range(len(mass_list)):  # loop around the number of atoms in the system
            for gamma in [0, 1, 2]:  # loop around the x,y,z direction
                # print(m,i,gamma,np.shape(spectrum.modes[m].eigenvec))
                Z_m_alpha += born_charges[i][alpha][gamma] * mass_list[i] * spectrum.modes[m].eigenvec[i][gamma]

        Z_tensor[m][alpha] = Z_m_alpha
    print('Mode','\t',m,'\t','spatial averaged Born charge ','\t',sum(Z_tensor[m][:]))

print("""======Mode-Dependent Dielectric Constants======""")

volume = crystal.lattice.volume * A_IN_METRE ** 3
sum_e_xy = 0
sum_e_xy_raw = 0

N = len([m for m in spectrum.modes if np.imag(m.freq) == 0.0])

frequencies = []
mode_dielectrics = []

for m in range(len(spectrum.modes)):

    if np.imag(spectrum.modes[m].freq) == 0:
        e_xx = (Z_tensor[m][0] * ELECTRON_CHARGE) ** 2 / ((spectrum.modes[m].freq) ** 2 * volume) / EPSILON_0 / AMU
        e_yy = (Z_tensor[m][1] * ELECTRON_CHARGE) ** 2 / ((spectrum.modes[m].freq) ** 2 * volume) / EPSILON_0 / AMU
        av_raw = (e_xx + e_yy) / 2.0

        if apply_2D_correction:
            av = (crystal.lattice.c / slab_thick) * (av_raw + (1 / N) * (1 - slab_thick / crystal.lattice.c))
            sum_e_xy += av
            print('mode', '\t', m, '\t', '{:.5f}'.format(spectrum.modes[m].freq / TETRA / (2 * math.pi)), '\t', 'THz',
                  '\t', '\t', '{:.5f}'.format(av_raw), '\t', '{:.5f}'.format(av))
            mode_dielectrics.append(av)
        else:
            print('mode', '\t', m, '\t', '{:.5f}'.format(spectrum.modes[m].freq / TETRA / (2 * math.pi)), '\t', 'THz',
                  '\t', '\t', '{:.5f}'.format(av_raw))
            mode_dielectrics.append(av_raw)

        frequencies.append(spectrum.modes[m].freq / TETRA / (2 * math.pi))

        sum_e_xy_raw += av_raw

print("Summed across modes raw:", sum_e_xy_raw)
print("Summed across modes corrected:", sum_e_xy)

plt.bar(frequencies, mode_dielectrics, align='center',  width=0.2, color='#FEE715FF')
#plt.yscale('log')
plt.xlabel('Phonon mode frequency ($\\omega_{m}$, THz)')
plt.ylabel('$\\varepsilon_{2D}^{\\parallel}(\\omega_{m})$')
plt.tight_layout()
plt.savefig('mode_dielectrics.pdf')
