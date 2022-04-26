from mp_api import MPRester
from pathlib import Path
import json

class PullEntriesMP:
    def __init__(self, api_key):
        self.api_key = api_key
        self.mpr = MPRester(self.api_key)

    @property
    def database_version(self):
        return self.mpr.get_database_version()

    def get_entries_from_ids(self, mp_ids):
        if not mp_ids:
            return list()
        else:
            entries_list = []
            for d in self.mpr.thermo.search_thermo_docs(material_ids=mp_ids):
                try:
                    entries_list.append(d.entries["GGA"])
                except KeyError:
                    try:
                        entries_list.append(d.entries["GGA+U"])
                    except KeyError:
                        raise Exception("Can't find entries for: {} {}".format(d.material_id, d.composition))
            return entries_list

    @staticmethod
    def split_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_entries_from_json(self, family, json_file):
        chunk = 1000
        json_path = Path("data") / family / json_file
        with open(json_path) as f:
            data = json.load(f)
        material_ids = [material["material_id"] for material in list(data.values())[0]]
        material_ids_split = list(self.split_list(material_ids, chunk))
        docs_list = []
        for c in material_ids_split:
            docs_list+=self.get_entries_from_ids(c)
        return docs_list

    def get_entries_for_gases(self, gases):
        # collection of materials in gas phase. Only used for balancing the chemical equations
        gas_mp_id = {"H2O": "mp-697111",
                     "N2": "mp-25",
                     "H2": "mp-24504",
                     "NH3": "mp-29145"}
        return {g : self.get_entries_from_ids([gas_mp_id[g]])[0] for g in gases}