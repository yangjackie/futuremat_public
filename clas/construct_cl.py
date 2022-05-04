from pymatgen.analysis.reaction_calculator import ComputedReaction
from clas.pull_mp_data import PullEntriesMP
from clas.get_strucutre import get_ele
from pymatgen.entries.computed_entries import GibbsComputedStructureEntry
from pymatgen.entries.entry_tools import EntrySet
from monty.json import MontyDecoder
from pymatgen.core.composition import Composition
from copy import deepcopy
import json


api_key = "" # Please enter your Materials Project API here. More details see: https://next-gen.materialsproject.org/api


def chemical_looping_entries(cl_list, family):
    pull_mp = PullEntriesMP(api_key)
    print("Current database version: ", pull_mp.database_version)
    redox_materials = []
    gases = []
    for i in cl_list:
        redox_materials.append(i["reactant_entries"])
        redox_materials.append(i["product_entries"])
        gases = gases + i["gaseous_reactants"]+i["gaseous_products"]+i["possible_yields"]
    total_redox_materials = list(set(redox_materials))
    total_gases = list(set(gases))
    total_redox_dict = {m: pull_mp.get_entries_from_json(family, m+"_"+family+"_0k.json")
                        for m in total_redox_materials} # {"**.json": [CompotedEntry]}
    total_gases_dict = pull_mp.get_entries_for_gases(total_gases)
    cl_entry_list = []
    for i in cl_list:
        cl_entry_list.append({"reactant_entries": total_redox_dict[i["reactant_entries"]],
                              "product_entries": total_redox_dict[i["product_entries"]],
                              "gaseous_reactants": [total_gases_dict[j] for j in i["gaseous_reactants"]],
                              "gaseous_products": [total_gases_dict[j] for j in i["gaseous_products"]],
                              "possible_yields": [total_gases_dict[j] for j in i["possible_yields"]]
                              })
    return cl_entry_list


class ChemicalReaction:
    """
    reactants (solids) + gaseous_reactants -> products (solids) + gaseous_products + possible_yields
    """
    def __init__(self, reactant_entries, product_entries,
                 gaseous_reactants, gaseous_products, possible_yields):
        self.reactant_entries = reactant_entries
        self.product_entries = product_entries
        self.gaseous_reactants = gaseous_reactants
        self.gaseous_products = gaseous_products
        self.possible_yields = possible_yields

    @staticmethod
    def get_material_id(json_file):
        with open(json_file) as f:
            data = json.load(f)
        ids=[material["material_id"] for material in list(data.values())[0]]
        return ids

    @staticmethod
    def chemical_reaction(reactant_entries, product_entries, possible_yield_entries, mode="strict"):
        try:
            rxn = ComputedReaction(reactant_entries, product_entries)
        except:
            try:
                rxn = ComputedReaction(reactant_entries, product_entries+possible_yield_entries)
            except:
                return None
        if rxn.reactants[0] != rxn.products[0]: # avoid self reactions
            if mode=="strict":
                reactant_comp = [entry.composition.reduced_composition for entry in reactant_entries]
                product_comp = [entry.composition.reduced_composition for entry in product_entries+possible_yield_entries]
                if set(rxn.reactants).issubset(set(reactant_comp)) and set(rxn.products).issubset(set(product_comp)):
                    return rxn
            else:
                return rxn

    def get_total_entries(self, reactant_entry, product_entry, gaseous_entries):
        total_reactants = reactant_entry + [v for k, v in gaseous_entries.items() if k in self.gaseous_reactants]
        total_products = product_entry + [v for k, v in gaseous_entries.items() if k in self.gaseous_products]
        possible_yields = [v for k, v in gaseous_entries.items() if k in self.possible_yields]
        return total_reactants, total_products, possible_yields


    def search_reactions(self, balance_equation="strict"):
        """
        :param balance_equation: if allow reactants and products change side
        :return: Dict(cation1: [rxn11, rxn12, ...], cation2: [rxn21, rxn22, ...])
        """
        rxns_dict = {}   # group rxns by the cations involved in the redox process to accelerate the paring process
        if len(self.reactant_entries[0].composition) == 1: # "ele_0k.json", M1 + M2 + N2 -> M1M2N
            for i in self.product_entries:
                product_elements = [e.symbol for e in i.composition.elements]
                product_cations = [e for e in product_elements if e not in ["O", "N", "H"]]
                cation_str = ''.join(sorted(product_cations))
                reactant_entry = [entry for entry in self.reactant_entries
                                  if entry.composition.elements[0].symbol in product_cations]
                rxn = self.chemical_reaction(reactant_entry+self.gaseous_reactants,
                                             [i] + self.gaseous_products,
                                             self.possible_yields,
                                             balance_equation)
                if rxn:
                    if cation_str in rxns_dict:
                        rxns_dict[cation_str].append(rxn)
                    else:
                        rxns_dict[cation_str] = [rxn]
        elif len(self.product_entries[0].composition) == 1: # "ele_0k.json", M1M2O + H2 -> M1 + M2 + H2O
            for i in self.reactant_entries:
                reactant_elements = [e.symbol for e in i.composition.elements]
                reactant_cations = [e for e in reactant_elements if e not in ["O", "N", "H"]]
                cation_str = ''.join(sorted(reactant_cations))
                product_entry = [entry for entry in self.product_entries
                                 if entry.composition.elements[0].symbol in reactant_cations]
                rxn = self.chemical_reaction([i]+self.gaseous_reactants,
                                             product_entry + self.gaseous_products,
                                             self.possible_yields,
                                             balance_equation)
                if rxn:
                    if cation_str in rxns_dict:
                        rxns_dict[cation_str].append(rxn)
                    else:
                        rxns_dict[cation_str] = [rxn]
        else:
            for i in self.reactant_entries:
                elements_i = [e.symbol for e in i.composition.elements]
                cations_i = [e for e in elements_i if e not in ["O", "N", "H"]]
                for j in self.product_entries:
                    elements_j = [e.symbol for e in j.composition.elements]
                    cations_j = [e for e in elements_j if e not in ["O", "N", "H"]]
                    if cations_j == cations_i:
                        cation_str = ''.join(sorted(cations_i))
                        rxn = self.chemical_reaction([i] + self.gaseous_reactants,
                                                     [j] + self.gaseous_products,
                                                     self.possible_yields,
                                                     balance_equation)
                        if rxn:
                            if cation_str in rxns_dict:
                                rxns_dict[cation_str].append(rxn)
                            else:
                                rxns_dict[cation_str] = [rxn]
        return rxns_dict


class ChemicalReactionLoop:
    """
    Construct chemical loops. **Reactions are not normalised.**
    """
    def __init__(self, cl_inputs):
        self.cl_inputs = cl_inputs
        self._num_steps = len(cl_inputs)
        self._all_cl_rxns = self.get_loop()  # 2-step cl: [(rxn1, rxn2), (rxn1, rxn2), ..., (rxn1, rxn2)]

    def get_all_reactions(self):
        return [ChemicalReaction(**i) for i in self.cl_inputs]



    def get_loop(self):
        cl_list = []
        all_reaction_list = self.get_all_reactions()
        all_reaction_list = [r.search_reactions() for r in all_reaction_list]
        if self._num_steps>3:
            raise ValueError("Number of reactions in the chemical looping process should be not larger than 3.")
        if self._num_steps == 2:
            for cation, rxns in all_reaction_list[0].items():
                if cation in all_reaction_list[1]:
                    for i in rxns:
                        for j in all_reaction_list[1][cation]:
                            total_reactants = set(i.reactants + j.reactants)
                            total_products = set(i.products + j.products)
                            reduced_reactants = total_reactants.difference(total_products)
                            reduced_products = total_products.difference(total_reactants)
                            if {r.reduced_formula for r in reduced_reactants}.issubset({"N2", "H2"}) and {r.reduced_formula for r in reduced_products} == {"H3N"}:
                                cl_list.append(deepcopy((i, j)))
                                # print("rxn 1: {}\nrxn 2: {}".format(i,j))
                                # print("=" * 50)

        elif self._num_steps == 3:
            for cation, rxns in all_reaction_list[0].items():
                if cation in all_reaction_list[1] and cation in all_reaction_list[2]:
                    for i in rxns:
                        for j in all_reaction_list[1][cation]:
                            for k in all_reaction_list[2][cation]:
                                total_reactants = set(i.reactants + j.reactants + k.reactants)
                                total_products = set(i.products + j.products + k.products)
                                reduced_reactants = total_reactants.difference(total_products)
                                reduced_products = total_products.difference(total_reactants)
                                if {r.reduced_formula for r in reduced_reactants}.issubset({"N2", "H2"}) and {r.reduced_formula for r in reduced_products} == {"H3N"}:
                                    cl_list.append(deepcopy((i, j, k)))
                                    # print("rxn 1: {}\nrxn 2: {}\nrxn 3: {}".format(i, j, k))
                                    # print("=" * 50)

        return cl_list

    def get_ammonia_yield_step(self):
        try:
            for index, rxn_dict in enumerate(self.cl_inputs):
                if "H3N" in [gas_entry.composition.reduced_formula for gas_entry in rxn_dict["gaseous_products"]]:
                    ammonia_yield_step = index
                    return ammonia_yield_step
        except:
            raise ValueError("No NH3 is listed in the products")

    def get_redox_pairs(self, cl):
        ammonia_yield_rxn = cl[self.get_ammonia_yield_step()]
        gas_comp = [Composition(k) for k in ["H2", "H2O", "N2", "NH3"]]
        if Composition("H2O") in ammonia_yield_rxn.reactants:
            m_oxidised_comp = [product for product in ammonia_yield_rxn.products if product not in gas_comp] # e.g. oxides
            m_reduced_comp = [reactant for reactant in ammonia_yield_rxn.reactants if reactant not in gas_comp] # e.g. nitrides
        elif Composition("H2") in ammonia_yield_rxn.reactants:
            m_oxidised_comp = [reactant for reactant in ammonia_yield_rxn.reactants if reactant not in gas_comp]
            m_reduced_comp = [product for product in ammonia_yield_rxn.products if product not in gas_comp]
        else:
            raise ValueError("Must be a new reaction route.")
        return m_oxidised_comp[0], m_reduced_comp[0]

    def normalise_to_NH3(self):
        ammonia_yield_step = self.get_ammonia_yield_step()
        for cl in self._all_cl_rxns:
            ammonia_yield_rxn = cl[ammonia_yield_step]
            ammonia_yield_rxn.normalize_to(Composition("NH3"))
            m_oxidised_comp, m_reduced_comp = self.get_redox_pairs(cl)
            m_oxidised_coeff = ammonia_yield_rxn.get_coeff(m_oxidised_comp)
            m_reduced_coeff = ammonia_yield_rxn.get_coeff(m_reduced_comp)
            other_rxns = [rxn for index, rxn in enumerate(cl) if index != ammonia_yield_step]
            for rxn in other_rxns:
                try:
                    rxn.normalize_to(m_oxidised_comp, factor=m_oxidised_coeff)
                except:
                    try:
                        rxn.normalize_to(m_reduced_comp, factor=m_reduced_coeff)
                    except ValueError as ex:
                        print("Can't normalise this chemical loop!", ex)

    def reaction_energies(self):
        """
        :return: Reaction energies of all chemical loops. List(Tuple[energy1, energy2...])
        """
        energies_list = []
        for cl in self._all_cl_rxns:
            energies = tuple(map(lambda rxn: round(rxn.calculated_reaction_energy, 4), cl))
            energies_list.append(energies)
        return energies_list

    def limiting_energies(self):
        """
        :return: Limiting energies of all chemical loops. List(float)
        """
        return [max(energy) for energy in self.reaction_energies()]

    def limiting_step(self):
        """
        :return: Limiting indices of all chemical loops. List(int)
        """
        return [energy.index(max(energy)) for energy in self.reaction_energies()]

    def num_cl(self):
        return len(self.reaction_energies())


class GibbsChemicalReactionLoop(ChemicalReactionLoop):
    def __init__(self, input_cl_list, temp):
        super().__init__(input_cl_list)
        self.temp = temp
        self._all_cl_rxns = self.gibb_reaction_cl()

    @staticmethod
    def get_element_entries(entries):
        elements = EntrySet(entries).chemsys
        try:
            with open("data/ele_computed_structure_entries.json", "r") as f:
                ele_structure_entries = json.load(f, cls=MontyDecoder)
        except FileNotFoundError as ex:
            print("{}\n Download from Materials Project".format(ex))
            ele_structure_entries = get_ele(entry=True, family=None)
        ele_structure_entries = [ele for ele in ele_structure_entries if
                                 ele.composition.elements[0].symbol in elements]
        return ele_structure_entries

    def get_gibbs_computed_structure_entries(self, entries):
        entries_comp = [entry.composition for entry in entries]
        ele_entries = self.get_element_entries(entries)
        gibbs_computed_structure_entries = GibbsComputedStructureEntry.from_entries(list(set(entries + ele_entries)),
                                                                                    temp=self.temp)
        return [entry for entry in gibbs_computed_structure_entries if entry.composition in entries_comp]

    def get_gibbs_rxn(self, rxn):
        reactant_gibbs_entries = self.get_gibbs_computed_structure_entries(rxn._reactant_entries)
        product_gibbs_entries = self.get_gibbs_computed_structure_entries(rxn._product_entries)
        return ComputedReaction(reactant_gibbs_entries, product_gibbs_entries)

    def gibb_reaction_cl(self):
        cl_gibbs_list = []
        for cl in self._all_cl_rxns:
            cl_gibbs = tuple(map(self.get_gibbs_rxn, cl))
            cl_gibbs_list.append(cl_gibbs)
        return cl_gibbs_list

    def get_pair_formation_energies(self):
        energy_list = []
        for cl in self._all_cl_rxns:
            energy_dict = {}
            m_oxidised_comp, m_reduced_comp = self.get_redox_pairs(cl)
            ammonia_yield_rxn = cl[self.get_ammonia_yield_step()]
            for entry in ammonia_yield_rxn.all_entries:
                comp = entry.composition
                if comp.reduced_formula == m_oxidised_comp.reduced_formula:
                    energy_dict["m_oxidised"] = {"material": m_oxidised_comp.reduced_formula,
                                                 "energy_0k": round(entry.formation_enthalpy_per_atom, 4),
                                                 "energy_gf_sisso": round(entry.gf_sisso()/comp.num_atoms, 4)}
                elif comp.reduced_formula == m_reduced_comp.reduced_formula:
                    energy_dict["m_reduced"] = {"material": m_reduced_comp.reduced_formula,
                                                "energy_0k": round(entry.formation_enthalpy_per_atom, 4),
                                                "energy_gf_sisso": round(entry.gf_sisso()/comp.num_atoms, 4)}
            energy_list.append(energy_dict.copy())
        return energy_list

    def parse_to_dict(self):
        data = {}
        redox_pair_dict = {"chemical_loop": [], "rxn_energy": None, "gf_sisso": None, "gf_0k": None}
        for cl, formation, reaction  in zip(self._all_cl_rxns, self.get_pair_formation_energies(), self.reaction_energies()):
            redox_pair = formation["m_oxidised"]["material"]+'/'+formation["m_reduced"]["material"]
            gf_0k = [formation["m_oxidised"]["energy_0k"], formation["m_reduced"]["energy_0k"]]
            gf_sisso = [formation["m_oxidised"]["energy_gf_sisso"], formation["m_reduced"]["energy_gf_sisso"]]
            if redox_pair not in data.keys():
                data[redox_pair] = deepcopy(redox_pair_dict)
                data[redox_pair]["gf_0k"] = gf_0k
                data[redox_pair]["chemical_loop"]= [str(step) for step in cl]
            data[redox_pair]["gf_sisso"]= gf_sisso
            data[redox_pair]["rxn_energy"]= list(reaction)
        return data
