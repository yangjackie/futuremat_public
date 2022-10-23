from analysis import *

params = {'legend.fontsize': '14',
          'figure.figsize': (3,3),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)


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

    octahedral_factors = []
    tolerance_factors = []

    this_octa = []
    this_t = []



    for uid in all_uids :
        row = None
        formation_energy = None
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            continue
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)

            chemistry, octahedral_factor, octahedral_mismatch, generalised_tolerance_factor = geometric_fingerprint(
                crystal)
            print(octahedral_factor, octahedral_mismatch, generalised_tolerance_factor)

            octahedral_factors.append(octahedral_factor)
            tolerance_factors.append(generalised_tolerance_factor)

            if (chemistry['X_anion']=='F') and (chemistry['A_cation']=='Li'):
                this_octa.append(octahedral_factor)
                this_t.append(generalised_tolerance_factor)

    plt.scatter(octahedral_factors, tolerance_factors, marker='o', c='#BCBABE',
                edgecolor=None, alpha=0.45, s=25)

    plt.scatter(this_octa, this_t, marker='o', c='#BCBABE',
                edgecolor='#2D4262', alpha=0.9, s=25)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('energy_landscape_highlight_stable.pdf')