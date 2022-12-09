from ase.io.vasp import read_vasp_xml

from core.dao.vasp import VaspReader, VaspWriter
from core.models import Crystal, cVector3D
import numpy as np
import os

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

todo = 'analysis'
#todo = 'setup'

x =np.arange(0,0.104,0.004)
y =np.arange(0,0.104,0.004)

if todo == 'setup':
    cwd = os.getcwd()
    from mode_dielectric_constants import get_phonon_spectrum

    spectrum = get_phonon_spectrum()

    Q_x = spectrum.get_mode_by_freq(2.471918).eigenvec
    Q_y = spectrum.get_mode_by_freq(2.471752).eigenvec

    crystal = VaspReader(input_location='./POSCAR').read_POSCAR()

    a_x = crystal.lattice.a
    a_y = crystal.lattice.b
    print(a_x,a_y)

    if not os.path.exists('scan'):
        os.makedirs('scan')
    os.chdir('scan')

    for delta_x in x:
        for delta_y in y:

            disp_x = Q_x * delta_x * a_x
            disp_y = Q_y * delta_y * a_y

            #print('delta_x='+'{:.5f}'.format(delta_x), 'delta_y='+'{:.5f}'.format(delta_y))
            #print('{:.5f}'.format(np.linalg.norm(disp_x)),'{:.5f}'.format(np.linalg.norm(disp_y)))
            crystal = VaspReader(input_location='../POSCAR').read_POSCAR()
            new_crystal = Crystal(lattice=crystal.lattice, asymmetric_unit=crystal.asymmetric_unit,
                                  space_group=crystal.space_group)
            print('old',new_crystal.asymmetric_unit[0].atoms[0].position)
            for i, atom in enumerate(new_crystal.asymmetric_unit[0].atoms):
                atom.position += cVector3D(disp_x[i][0], disp_x[i][1], disp_x[i][2])
            #print('new+x',new_crystal.asymmetric_unit[0].atoms[0].position)
            for i, atom in enumerate(new_crystal.asymmetric_unit[0].atoms):
                atom.position += cVector3D(disp_y[i][0], disp_y[i][1], disp_y[i][2])
            #print('new+y',new_crystal.asymmetric_unit[0].atoms[0].position)


            print('{:.5f}'.format(delta_x), '{:.5f}'.format(delta_y))#, '{:.5f}'.format(np.linalg.norm(disp_x)), '{:.5f}'.format(np.linalg.norm(disp_y)))

            this_folder = 'scan_Qx_'+'{:.5f}'.format(delta_x).replace('.','_')+'_Qy_'+'{:.5f}'.format(delta_y).replace('.','_')
            if not os.path.exists(this_folder):
               os.makedirs(this_folder)
            os.chdir(this_folder)
            VaspWriter().write_structure(new_crystal,'POSCAR')
            os.chdir('..')

    os.chdir(cwd)

if todo == 'analysis':
    cwd = os.getcwd()
    x_count = 0
    y_count = 0


    X, Y = np.meshgrid(x, y)
    print(np.shape(X))
    energy_grid = np.zeros(np.shape(X))

    delta_x = 0.0000
    delta_y = 0.0000
    this_folder = 'scan_Qx_' + '{:.5f}'.format(delta_x).replace('.', '_') + '_Qy_' + '{:.5f}'.format(
        delta_y).replace('.', '_')
    this_folder = 'scan_Qx_0_00000_Qy_0_00000/'
    os.chdir(cwd + '/' + this_folder)
    atoms = [a for a in read_vasp_xml(index=-1)][-1]

    ref = atoms.get_calculator().get_potential_energy() / 7.0

    os.chdir(cwd)

    for delta_y in y:
        for delta_x in x:
            this_folder = 'scan_Qx_' + '{:.5f}'.format(delta_x).replace('.', '_') + '_Qy_' + '{:.5f}'.format(
                delta_y).replace('.', '_')
            os.chdir(cwd + '/' + this_folder)
            atoms = [a for a in read_vasp_xml(index=-1)][-1]

            total_energy = atoms.get_calculator().get_potential_energy() / 7.0

            os.chdir(cwd)
            #if (x_count == 0) and (y_count == 0):
            #    ref = total_energy
            energy_grid[y_count][x_count] = total_energy - ref
            print(this_folder, '\t', total_energy - ref, ref)
            x_count += 1

        y_count += 1
        x_count = 0

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, energy_grid, [0.001,0.002,0.004,0.006,0.008,0.012,0.016,0.024,0.032,0.040,0.048,0.056])
    # CS = ax.matshow(energy_grid)
    ax.clabel(CS, inline=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("energy_contour.pdf")
