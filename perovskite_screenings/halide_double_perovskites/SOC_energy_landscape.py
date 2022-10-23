import matplotlib.pyplot as plt
import numpy as np

from analysis import *

def energy_diff_by_elements(db,uids):
    energy_diff_dict={}
    for uid in uids:
        row = None
        formation_energy = None
        formation_energy_SOC = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            continue
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)

            chemistry, octahedral_factor, octahedral_mismatch, generalised_tolerance_factor = geometric_fingerprint(crystal)

            if chemistry['A_cation'] not in energy_diff_dict.keys():
                energy_diff_dict[chemistry['A_cation']] = []
            if chemistry['X_anion'] not in energy_diff_dict.keys():
                energy_diff_dict[chemistry['X_anion']] = []
            if chemistry['M_cation_mono'] not in energy_diff_dict.keys():
                energy_diff_dict[chemistry['M_cation_mono']] = []
            if chemistry['M_cation_tri'] not in energy_diff_dict.keys():
                energy_diff_dict[chemistry['M_cation_tri']] = []

            try:
                formation_energy = row.key_value_pairs['formation_energy']
                print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom')
            except KeyError:
                continue

            try:
                formation_energy_SOC = row.key_value_pairs['formation_energy_soc']
            except KeyError:
                continue

        if (formation_energy != None) and (formation_energy_SOC != None):
            energy_diff=formation_energy-formation_energy_SOC
            energy_diff_dict[chemistry['A_cation']].append(energy_diff)
            energy_diff_dict[chemistry['X_anion']].append(energy_diff)
            energy_diff_dict[chemistry['M_cation_tri']].append(energy_diff)
            energy_diff_dict[chemistry['M_cation_mono']].append(energy_diff)

    f=open('soc_data_diff.dat','w')
    for k in energy_diff_dict.keys():
        f.write(k+','+str(np.mean(energy_diff_dict[k]))+'\n')


def formation_energy_landscape_correlations(db, uids):
    all_data_dict = {x: {'formation_energies': [], 'formation_energies_SOC': []} for x in ['F','Cl','Br','I']}
    color_dict = {'F': '#061283', 'Cl': '#FD3C3C', 'Br': '#FFB74C', 'I': '#138D90'}


    min_energy = 100000
    max_energy = -100000

    min_energy_SOC = 100000
    max_energy_SOC = -100000

    for uid in uids:
        row = None
        formation_energy = None
        formation_energy_SOC = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            continue
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)

            chemistry, octahedral_factor, octahedral_mismatch, generalised_tolerance_factor = geometric_fingerprint(crystal)

            try:
                formation_energy = row.key_value_pairs['formation_energy']
                print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom')
            except KeyError:
                continue

            try:
                formation_energy_SOC = row.key_value_pairs['formation_energy_soc']
            except KeyError:
                continue

        if (formation_energy != None) and (formation_energy_SOC != None):
            all_data_dict[chemistry['X_anion']]['formation_energies'].append(formation_energy)
            all_data_dict[chemistry['X_anion']]['formation_energies_SOC'].append(formation_energy_SOC)
            print(chemistry['X_anion'], len(all_data_dict[chemistry['X_anion']]['formation_energies']))

            if formation_energy < min_energy:
                min_energy = formation_energy
            if formation_energy > max_energy:
                max_energy = formation_energy

            if formation_energy_SOC < min_energy_SOC:
                min_energy_SOC = formation_energy_SOC
            if formation_energy_SOC > max_energy_SOC:
                max_energy_SOC = formation_energy_SOC

    for k in color_dict.keys():
        print(k,len(all_data_dict[k]['formation_energies']))
        plt.scatter(all_data_dict[k]['formation_energies'], all_data_dict[k]['formation_energies_SOC'], alpha=0.6,
                    marker='o', s=25, edgecolor=None, c=color_dict[k])

    legend_elements = [Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F'),
                       Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl'),
                       Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br'),
                       Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
    plt.legend(handles=legend_elements, loc=2, fontsize=12, ncol=1)

    plt.plot([min_energy-0.1,max_energy+0.1],[min_energy-0.1,max_energy+0.1],'k--')

    plt.xlabel('$E_{f}$ (eV/atom)')
    plt.ylabel('$E_{f}^{SOC}$ (eV/atom)')
    plt.tight_layout()
    plt.savefig('EF_SOC_corr.pdf')

if __name__ == "__main__":

    dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')

    # ====================================================================
    # this is a hack to get all the uids from the database
    all_uids = []
    _db = sqlite3.connect(dbname)
    cur = _db.cursor()
    cur.execute("SELECT * FROM systems")
    rows = cur.fetchall()

    for row in rows:
        for i in row:
            if 'uid' in str(i):
                this_dict = json.loads(str(i))
                this_uid = this_dict['uid']
                if 'dpv' in this_uid:
                    all_uids.append(this_uid)
    # ====================================================================

    # use the ASE db interface
    db = connect(dbname)


    formation_energy_landscape_correlations(db,all_uids)

    #energy_diff_by_elements(db,all_uids)
    #import ptable_plotter
    #ptable_plotter('soc_data_diff.dat',output_filename='soc_ptable.pdf',show=True)