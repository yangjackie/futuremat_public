from collections import namedtuple

e = Exception("Value undefined")

Element = namedtuple('Element', ['symbol', 'atomic_number', 'vdw_radius', 'pp_choice'])

vdw_radii = {"H": 1.09, "He": 1.40, "Li": 1.82, "Be": 2.00, "B": 2.00,
             "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
             "Na": 2.27, "Mg": 1.73, "Al": 2.00, "Si": 2.10, "P": 1.80,
             "S": 1.80, "Cl": 2.0, "Ar": 1.88, 'K': e, 'Ca': e, 'Sc': e,
             'Ti': e, 'V': e, 'Cr': e, 'Mn': e, 'Fe': e, 'Co': e, 'Ni': e, 'Cu': e, 'Zn': e, 'Ga': e,
             'Ge': e, 'As': e, 'Se': e, 'Br': e, 'Kr': e, 'Rb': e, 'Sr': e, 'Y': e, 'Zr': e, 'Nb': e,
             'Mo': e, 'Tc': e, 'Ru': e, 'Rh': e, 'Pd': e, 'Ag': e, 'Cd': e, 'In': e, 'Sn': e, 'Sb': e,
             'Te': e, 'I': 1.98, 'Xe': 2.16, 'Cs': e, 'Ba': e, 'La': e, 'Ce': e, 'Pr': e, 'Nd': e, 'Pm': e,
             'Sm': e, 'Eu': e, 'Gd': e, 'Tb': e, 'Dy': e, 'Ho': e, 'Er': e, 'Tm': e, 'Yb': e, 'Lu': e,
             'Hf': e, 'Ta': e, 'W': e, 'Re': e, 'Os': e, 'Ir': e, 'Pt': e, 'Au': e, 'Hg': e, 'Tl': e,
             'Pb': e, 'Bi': e, 'Po': e, 'At': e, 'Rn': e, 'Fr': e, 'Ra': e, 'Ac': e, 'Th': e, 'Pa': e,
             'U': e, 'Np': e, 'Pu': e, 'Am': e, 'Cm': e, 'Bk': e, 'Cf': e, 'Es': e, 'Fm': e, 'Md': e,
             'No': e, 'Lr': e, 'Rf': e, 'Db': e, 'Sg': e, 'Bh': e, 'Hs': e, 'Mt': e, 'Ds': e,
             'Rg': e, 'Cn': e, 'Nh': e, 'Fl': e, 'Mc': e, 'Lv': e, 'Ts': e, 'Og': e}

atomic_numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
                  'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21,
                  'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
                  'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
                  'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
                  'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
                  'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                  'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
                  'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
                  'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                  'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                  'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}

U_corrections = {'Ti': {'d': 5.0}, 'Mn': {'d': 3.0}, 'La': {'f': 9.0}, 'Zr': {'d': 5.79}}

orbital_index = {'d': 2, 'f': 3}

high_spin_states = {'Mn': 4}


"""
Default choices of pseudopotentials for PBE calculations in VASP as recommended by the Materials Project Team.
see https://materialsproject.org/wiki/index.php/Pseudopotentials_Choice for more information
"""
pbe_pp_choices = {'Ru': 'Ru', 'Re': 'Re_pv', 'Rf': None, 'Rg': None, 'Ra': 'Ra_sv', 'Rb': 'Rb_sv', 'Rn': 'Rn',
                  'Rh': 'Rh_pv', 'Be': 'Be_sv', 'Ba': 'Ba_sv', 'Bh': None, 'Bi': 'Bi', 'Bk': None, 'Br': 'Br',
                  'Og': None, 'H': 'H', 'P': 'P', 'Os': 'Os_pv', 'Es': None, 'Ge': 'Ge_d', 'Gd': 'Gd', 'Ga': 'Ga_d',
                  'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'C': 'C', 'Pb': 'Pb_d', 'Pa': 'Pa', 'Pd': None, 'Xe': 'Xe',
                  'Po': 'Po', 'Pm': 'Pm_3', 'Hs': None, 'Ho': 'Ho_3', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'He': 'He', 'Md': None,
                  'Mg': 'Mg_pv', 'Mc': None, 'K': 'K_sv', 'Mn': 'Mn_pv', 'O': 'O', 'Mt': None, 'S': 'S', 'W': 'W',
                  'Zn': 'Zn', 'Eu': 'Eu', 'Zr': 'Zr_sv', 'Er': 'Er_3', 'Nh': None, 'Ni': 'Ni_pv', 'No': None,
                  'Na': 'Na_pv', 'Nb': 'Nb_pv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Np': 'Np', 'Fr': 'Fr_sv', 'Fe': 'Fe_pv',
                  'Fl': None, 'Fm': None, 'B': 'B', 'F': 'F', 'Sr': 'Sr_sv', 'N': 'N', 'Kr': 'Kr', 'Si': 'Si',
                  'Sn': 'Sn_d', 'Sm': 'Sm_3', 'V': 'V_sv', 'Sc': 'Sc_sv', 'Sb': 'Sb', 'Sg': None, 'Se': 'Se',
                  'Co': 'Co', 'Cn': None, 'Cm': 'Cm', 'Cl': 'Cl', 'Ca': 'Ca_sv', 'Cf': None, 'Ce': 'Ce', 'Cd': 'Cd',
                  'Tm': 'Tm_3', 'Cs': 'Cs_sv', 'Cr': 'Cr_pv', 'Cu': 'Cu_pv', 'La': 'La', 'Ts': None, 'Li': 'Li_sv',
                  'Lv': None, 'Tl': 'Tl_d', 'Lu': 'Lu_3', 'Lr': None, 'Th': 'Th', 'Ti': 'Ti_pv', 'Te': 'Te',
                  'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Ta': 'Ta_pv', 'Yb': 'Yb', 'Db': None, 'Dy': 'Dy_3', 'Ds': None,
                  'I': 'I', 'U': 'U', 'Y': 'Y_sv', 'Ac': 'Ac', 'Ag': 'Ag', 'Ir': 'Ir', 'Am': 'Am', 'Al': 'Al',
                  'As': 'As', 'Ar': 'Ar', 'Au': 'Au', 'At': 'At', 'In': 'In_d', 'Mo': 'Mo_pv'}

transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                     'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

rare_earth_metals = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac',
                     'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

element_name_dict = {value: key for key, value in atomic_numbers.items()}

element_dict = {key: Element(symbol=key,
                             atomic_number=atomic_numbers[key],
                             vdw_radius=vdw_radii[key],
                             pp_choice=pbe_pp_choices[key]) for key, v in atomic_numbers.items()}

__max_covalent_by_symbol = {("C", "H"): 1.28
    , ("C", "C"): 1.65
    , ("C", "N"): 1.55
    , ("C", "O"): 1.55
    , ("C", "F"): 1.45
    , ("C", "S"): 1.90
    , ("C", "Cl"): 1.85
    , ("C", "Br"): 1.95
    , ("N", "H"): 1.20
    , ("N", "N"): 1.55
    , ("N", "O"): 1.55
    , ("N", "S"): 1.70
    , ("N", "Hg"): 2.80
    , ("O", "H"): 1.3
    , ("O", "O"): 1.70
    , ("O", "S"): 1.50
    , ("B", "F"): 1.45
    , ("B", "C"): 1.65
    , ("I", "Hg"): 2.80
    , ("Br", "Hg"): 2.50
    , ("S", "S"): 2.5
    , ("H", "H"): 0.85
    , ("C", "I"): 2.2
    , ("S", "Cl"): 1.7
    , ("S", "F"): 1.6
    , ("Cl", "H"): 1.35
    , ("H", "F"): 1.00
    , ("O", "F"): 1.50
    , ("N", "F"): 1.45
    , ("F", "F"): 1.50
                            }
_max_covalent_by_symbol = {}

for k, v in __max_covalent_by_symbol.items():
    _max_covalent_by_symbol[(k[1], k[0])] = v
    _max_covalent_by_symbol[(k[0], k[1])] = v

# Store the ionic radii for elements in different charge states. May not be the most correct value, guide only.
# (Note pymatgen does store ionic radii, but might not have them for every charge state).
ionic_radii = {'Li': {1: 0.76},
               'Na': {1: 1.02},
               'K': {1: 1.38},
               'Rb': {1: 1.52},
               'Cs': {1: 1.67},
               'Mg': {2: 0.721},
               'Ca': {2: 1},
               'Sr': {2: 1.18},
               'Ba': {2: 1.35},
               'F': {-1: 1.33},
               'Cl': {-1: 1.81},
               'Br': {-1: 1.96},
               'I': {-1: 2.2},
               'O': {-2: 1.4},
               'S': {-2: 1.84},
               'Se': {-2: 1.98},
               'Te': {-2: 2.21},
               'Pb': {2: 1.19},
               'Ge': {2: 0.73},
               'Sn': {2: (1.19 + 0.73) / 2},
               'V': {2: 0.79, 5: 0.54},
               'Ta': {2: 0.72, 5: 0.64},
               'Nb': {2: 0.72, 5: 0.64},
               'Ti': {4: 0.605},
               'Zr': {4: 0.72}}
