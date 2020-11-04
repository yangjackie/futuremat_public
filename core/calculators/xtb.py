import os
import random
import shutil
import string
import subprocess
import glob
import time

from core.calculators.abstract_calculator import Calculator
from ase.atoms import Atoms
from ase import io
from ase.units import Hartree, mol, kcal

run_types = ['scc', 'grad', 'vip', 'vea', 'vipea', 'vomega', 'vfukui', 'esp', 'stm', 'opt', 'metaopt', 'modef', 'ohess',
             'omd', 'metadyn', 'siman']
run_type_implemented = ['opt', 'vipea']

optimization_levels = ['crude', 'sloppy', 'loose', 'lax', 'normal', 'tight', 'vtight', 'extreme']


class XTB(Calculator):

    def __init__(self, atoms: Atoms, scratch=None, run_type='opt', optimization_level=None, optimization_cycles=200):
        self.atoms = atoms
        self.cwd = os.getcwd()
        self.set_scratch_directory(scratch)
        self.set_run_type(run_type)
        self.set_optimization_level(optimization_level)
        self.optimization_cycles = 200
        self.result = {}

    def set_scratch_directory(self, scratch):
        if scratch is not None:
            self.scratch_dir = scratch + '/' + ''.join([random.choice(string.ascii_letters) for _ in range(10)])
        else:
            self.scratch_dir = os.getcwd() + '/' + ''.join([random.choice(string.ascii_letters) for _ in range(10)])
        pass

        try:
            os.makedirs(self.scratch_dir, exist_ok=False)
        except OSError:
            pass

        os.chdir(self.scratch_dir)

    def set_run_type(self, run_type):
        try:
            assert run_type in run_type_implemented
            self.run_type = run_type
        except AssertionError:
            print("Run type " + str(run_type) + " has not been implemented in this wrapper!")

    def set_optimization_level(self, optimization_level):
        if optimization_level is None:
            self.optimization_level = 'normal'
        else:
            try:
                assert self.optimization_level in optimization_levels
            except AssertionError:
                self.optimization_level = 'normal'

    def setup_inputs(self):
        io.write('input.xyz', self.atoms, format='xyz')

    def run(self):
        cmd = ['/opt/intel/intelpython3/bin/xtb', 'input.xyz', '--' + self.run_type]
        if self.run_type == 'vipea':
            cmd += ['--vparam',
                    '/Users/jack_yang/PycharmProjects/futuremat/core/calculators/xtb_data/param_ipea-xtb.txt']

        with open('xtb.log', "w") as self.outfile:
            subprocess.run(cmd, stdout=self.outfile, stderr=subprocess.PIPE)

    def parse_output(self):
        if self.run_type == 'opt':
            optimized_str, total_energy = self.__get_optimised_molecule_and_energy_from_xyz(
                open('xtbopt.xyz', 'r'))
            homo_lumo_gap = self.__get_homo_lumo_gap(open('xtb.log', 'r'))
            self.result['optimized_structure'] = optimized_str
            self.result['total_energy'] = total_energy
            self.result['homo_lumo_gap'] = homo_lumo_gap
            self.result['success'] = True
        if self.run_type == 'vipea':
            ip, ea = self.__get_vipea(open('xtb.log', 'r'))
            self.result['vip'] = ip
            self.result['vea'] = ea
            if (self.result['vip']  is not None) and (self.result['vea'] is not None):
                self.result['vipea_success'] = True
            else:
                self.result['vipea_success'] = False
                # except:
            #    self.result['vipea_success'] = False

    def tear_down(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.scratch_dir)

    def execute(self):
        start_time = time.time()
        self.setup_inputs()
        self.run()

        if self.run_type == 'opt':
            try:
                self.parse_output()
                print("Final energy:\t" + str(self.result['total_energy']) + '\t kcal/mol, execution time: ' + str(
                    time.time() - start_time) + ' secs')
            except Exception as e:
                self.result['success'] = False
                print(e)
                print("Output parsing failed, could be a failed xtb optimisation, pass and clean scratch")
        elif self.run_type == 'vipea':
            try:
                self.parse_output()
                print("VIP: " + str(self.result['vip']) + ' eV.  VEA: ' + str(
                    self.result['vea']) + ' eV. execution time: ' + str(time.time() - start_time) + ' secs')
            except Exception as e:
                self.result['vipea_success'] = False
                print(e)
                print("Output parsing failed, could be a failed xtb run, pass and clean scratch")

        self.tear_down()
        return self.result


    def __get_optimised_molecule_and_energy_from_xyz(self, fileobj):
        lines = fileobj.readlines()
        natoms = int(lines[0])
        total_energy_kcal_mol = float(lines[1].split()[1]) * Hartree * mol / kcal  # store energy in kcal/mol
        symbols = []
        positions = []
        for line in lines[2:2 + natoms]:
            symbol, x, y, z = line.split()[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        return Atoms(symbols=symbols, positions=positions), total_energy_kcal_mol


    def __get_vipea(self, fileobj):
        lines = fileobj.readlines()
        ip = None
        ea = None
        for l in lines:
            if 'delta SCC IP (eV)' in l:
                ip = float(l.split()[-1])
            if 'delta SCC EA (eV)' in l:
                ea = float(l.split()[-1])
        return ip, ea


    def __get_homo_lumo_gap(self, fileobj):
        _holder = []
        for line in fileobj.readlines():
            if 'HOMO-LUMO GAP ' in line:
                _holder.append(float(line.split()[-3]))
        return _holder[-1]


if __name__ == "__main__":
    xyzs = glob.glob('*xyz')
    atoms = io.read(xyzs[-1])
    XTB(atoms=atoms).execute()
