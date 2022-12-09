import matplotlib.pyplot as plt

from similarity_analysis import *

db = connect('2dpv.db')
lattice_constants = []
for i in range(len(A_site_list)):
    for a in A_site_list[i]:
        for b in B_site_list[i]:
            for c in C_site_list[i]:

                pm3m_formation_e = None
                # get the corresponding bulk structures
                system_name = a + b + c
                uid = system_name + '3_pm3m'

                row = None
                row = db.get(selection=[('uid', '=', uid)])
                if row is not None:
                    try:
                        pm3m_formation_e = row.key_value_pairs['formation_energy']
                    except KeyError:
                        continue
                if pm3m_formation_e is None:
                    continue

                bulk_atoms = row.toatoms()

                lattice_const = bulk_atoms.cell[0][0]
                lattice_constants.append(lattice_const)

min_l=min(lattice_constants)
max_l=max(lattice_constants)
gap = max_l-min_l
x_grid = np.linspace(min_l-0.1*gap,max_l+0.1*gap,500)
pdf = kde_scipy(np.array(lattice_constants), x_grid, bandwidth=0.05)
plt.plot(x_grid, pdf, '-', lw=1.5)
plt.tight_layout()
plt.savefig('lattice_const.pdf')