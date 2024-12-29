"""
Modules containing methods  to run molecular dynamic simulations
"""
import jax.numpy as jnp
import numpy as np
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, ZeroRotation, Stationary)
from ase import Atoms
from ase.units import fs, kB
from ase.build.supercells import make_supercell
from ase.io import write

from mlff import mdx
from mlff.mdx import MLFFPotential
import os, shutil
import random
import typing
import pickle

from so3krates.LLZO.phonon_dos import prepare_ml_traj
from core.external.vasp.analysis import velocity_autocorrelation_function, get_phonon_dos, vibrational_free_energies
from core.utils.loggings import setup_logger


def run_molecular_dynamics(atoms: Atoms,
                           ckpt_dir: str = '',
                           potential: MLFFPotential = None,
                           pbc: bool = True,
                           temperature: float = 1500,
                           thermo_stat: str = 'velocityverlet',
                           dtype: jnp.dtype = jnp.float64,
                           timestep: float = 1.0,
                           Nsteps: int = 10000,
                           save_frequency: int = 1,
                           run_interval: int = 1,
                           opt_init: bool = True,
                           supercell: list = [1, 1, 1]):
    if pbc:
        # check that the correct periodic boundary condition is set
        assert atoms.get_cell() is not None
        assert atoms.get_pbc() is not None

    #make the supercell for running the simulation
    print("Creating a "+str(supercell)+" supercell.")
    P=np.diag(np.array(supercell))
    atoms = make_supercell(atoms,P)
    write('POSCAR_start',atoms,format='vasp')
    print("Running molecular dynamics for atoms with "+(str(atoms.get_number_of_atoms()))+" atoms in the supercell.")

    T0 = temperature * kB
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0 / kB)

    # create the potential from checkpoint file
    if potential is None:
        assert ckpt_dir is not None
        assert os.path.exists(ckpt_dir)
        potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)

    calc = mdx.CalculatorX.create(potential)

    # setup thermostat:
    print("Running MD with " + thermo_stat + " thermostat")
    if thermo_stat.lower() == 'nosehoover':
        integratorx = mdx.NoseHooverX.create(timestep=timestep * fs, temperature=T0, ttime=1, calculator=calc)
    elif thermo_stat.lower() == 'langevin':
        integratorx = mdx.LangevinX.create(timestep=timestep * fs, temperature=T0, friction=0.5, calculator=calc,
                                           fixcm=False)  # this is the stable one with friction=0.5
    elif thermo_stat.lower() == 'velocityverlet':
        integratorx = mdx.VelocityVerletX.create(timestep=timestep * fs, calculator=calc)
    else:
        raise NotImplementedError('thermo_stat not implemented')

    simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()), save_frequency=save_frequency,
                               run_interval=run_interval)

    atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
    atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff, skin=0.5)

    if opt_init:
        # this default seems work after many testing
        optimizer = mdx.LBFGS.create(atoms=atomsx, potential=potential)
        atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=50000, tol=0.08)
    else:
        atomsx_opt = atomsx

    atomsx_opt = mdx.scale_momenta(atomsx_opt, T0=T0)
    atomsx_opt = mdx.zero_rotation(mdx.zero_translation(atomsx_opt))

    simulator.run(integratorx, atomsx_opt, steps=Nsteps)

    return


def run_molecular_dynamics_batch(data,
                                 ckpt_dir_opt: str = '',
                                 potential_opt: MLFFPotential = None,
                                 ckpt_dir_md: str = '',
                                 potential_md: MLFFPotential = None,
                                 pbc: bool = True,
                                 temperature: list[float] = [1500.0],
                                 thermo_stat: str = 'velocityverlet',
                                 dtype: jnp.dtype = jnp.float64,
                                 timestep: typing.Optional[float] = 1.0,
                                 Nsteps: typing.Optional[int] = 10000,
                                 save_frequency: typing.Optional[int] = 1,
                                 run_interval: typing.Optional[int] = 1,
                                 opt_init: typing.Optional[bool] = True,
                                 part: typing.Optional[int] = 0,
                                 batch_size: typing.Optional[int] = 0,
                                 save_trajectory: typing.Optional[bool] = False,
                                 calculate_free_energies: typing.Optional[bool] = True,
                                 save_from_i: typing.Optional[int] = 0,
                                 save_every_i: typing.Optional[int] = 1,
                                 save_up_to: typing.Optional[int] = None
                                 ):
    if potential_opt is None:
        assert ckpt_dir_opt is not None
        assert os.path.exists(ckpt_dir_opt)
        potential_opt = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir_opt, add_shift=True, dtype=dtype)
    if potential_md is None:
        assert ckpt_dir_md is not None
        assert os.path.exists(ckpt_dir_md)
        potential_md = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir_md, add_shift=True, dtype=dtype)

    cwd = os.getcwd()
    logger = setup_logger(output_filename=cwd + '/sampler_' + str(part) + '.log')
    working_directory = cwd + '/temp_' + str(random.getrandbits(32))
    os.makedirs(working_directory, exist_ok=True)
    os.chdir(working_directory)

    results = []

    if batch_size * part > len(data['R']):
        return

    if batch_size * (part + 1) < len(data['R']):
        endpoint = batch_size * (part + 1)
    else:
        endpoint = len(data['R'])

    for idx in range(batch_size * part, endpoint, 1):

        result_dict = {'system_name': None,
                       'optimised_structure': None,
                       'local_minimum_energy': None,
                       'initial_optimisation_succeed': False,
                       'energy_unit': 'eV',
                       'molecular_dynamics': {k: None for k in temperature}}

        if 'system_name' in data.keys():
            system_name = data['system_name'][idx]
        else:
            system_name = None
        result_dict['system_name'] = system_name
        logger.info('Structure id: {}'.format(idx))

        atoms = Atoms(positions=data['R'][idx], numbers=data['z'][idx], cell=data['unit_cell'][idx],
                      pbc=[True, True, True])
        atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)

        ####################################################
        #
        # Structure optimisation stage
        #
        ####################################################

        initial_optimisation_succeed = False
        if opt_init:
            atomsx = atomsx.init_spatial_partitioning(cutoff=potential_opt.cutoff, skin=0.5)
            try:
                # optimizer = mdx.GradientDescent.create(potential=potential_opt, learning_rate=5e-4)
                optimizer = mdx.LBFGS.create(atoms=atomsx, potential=potential_opt)
                atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=50000, tol=0.08)
                initial_optimisation_succeed = True
                energy_of_optimised_structure = potential_opt(atomsx_opt.to_graph()).sum()
                logger.info("Structure\t" + str(idx) + "\t Energy of optimised structure: " + str(
                    energy_of_optimised_structure))

                optimised_atoms = Atoms(positions=atomsx_opt.get_positions(),
                                        numbers=atomsx_opt.get_atomic_numbers(),
                                        cell=atomsx_opt.get_cell(),
                                        pbc=[True, True, True])

                result_dict['optimised_structure'] = optimised_atoms  # note this is an atomx object!
                result_dict['local_minimum_energy'] = energy_of_optimised_structure
            except:
                logger.info("Structure\t" + str(idx) + "\t Optimisation failed to converge! Will not proceed to MD")
        else:
            atomsx_opt = atomsx

        result_dict['initial_optimisation_succeed'] = initial_optimisation_succeed

        ####################################################
        #
        # Molecular dynamics stage
        #
        ####################################################

        # initial_optimisation_succeed = False
        md_run_completed = False
        if (not opt_init) or (opt_init and initial_optimisation_succeed):
            for i, temp in enumerate(temperature):
                result_dict['molecular_dynamics'][temp] = {'md_succeed': md_run_completed,
                                                           'phonon_energies': None,
                                                           'phonon_dos': None,
                                                           'fvib': None,
                                                           'internal_energy': None}

                simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()), save_frequency=1, run_interval=1)

                MaxwellBoltzmannDistribution(optimised_atoms, temperature_K=temp)
                atomsx = mdx.AtomsX.create(atoms=optimised_atoms, dtype=dtype)
                atomsx = atomsx.init_spatial_partitioning(cutoff=potential_md.cutoff, skin=0.5)
                atomsx = mdx.scale_momenta(atomsx, T0=temp * kB)
                atomsx = mdx.zero_rotation(mdx.zero_translation(atomsx))

                logger.info("Creating integrators for temperature {}".format(temp))
                calc = mdx.CalculatorX.create(potential_md)
                integratorx = get_thermostat(temp * kB, calc, thermo_stat, timestep)

                try:
                    simulator.run(integratorx, atomsx, steps=Nsteps)
                    md_run_completed = True
                    result_dict['molecular_dynamics'][temp]['md_succeed'] = md_run_completed
                except RuntimeError:
                    logger.info("Structure\t" + str(idx) + "\t molecular dynamic simulations failed!")
                    pass

                if calculate_free_energies and md_run_completed:
                    all_ml_frames = prepare_ml_traj('trajectory.h5', save_from_i=save_from_i, save_every_i=save_every_i,
                                                    save_up_to=save_up_to, cell=atoms.cell)
                    omega, ph_dos = get_phonon_dos(all_ml_frames, potim=timestep, nblock=1, unit='meV')
                    #omega = [_omega * 1e3 for _omega in omega] #this is wrong!
                    fvib = vibrational_free_energies(all_ml_frames, temp=temp, potim=timestep, nblock=1)
                    u_0 = np.mean(np.array([c.potential_energy for c in all_ml_frames]))
                    logger.info("Vibrational free energy " + str(fvib * 1000) + " meV/atom")
                    logger.info("Internal energy:\t" + str(u_0 * 1000) + '\t meV/atom')

                    result_dict['molecular_dynamics'][temp]['phonon_energies'] = omega
                    result_dict['molecular_dynamics'][temp]['phonon_dos'] = ph_dos
                    result_dict['molecular_dynamics'][temp]['fvib'] = fvib
                    result_dict['molecular_dynamics'][temp]['internal_energy'] = u_0
                    # saving the trajectory is too costly
                if not save_trajectory:
                    os.remove('trajectory.h5')
                else:
                    # need to rename the trajectory file before saving it
                    pass
        results.append(result_dict)

    os.chdir(cwd)

    output_name = 'sampling_run_' + str(part) + '.bp'
    # with open(output_name,'wb') as output_file:
    output_file = open(output_name, 'wb')
    pickle.dump(results, output_file)

    if not save_trajectory:
        shutil.rmtree(working_directory)
    return


def get_thermostat(temperature, calc, thermo_stat, timestep):
    if thermo_stat.lower() == 'nosehoover':
        integratorx = mdx.NoseHooverX.create(timestep=timestep * fs, temperature=temperature, ttime=1, calculator=calc)
    elif thermo_stat.lower() == 'langevin':
        integratorx = mdx.LangevinX.create(timestep=timestep * fs, temperature=temperature, friction=0.5,
                                           calculator=calc,
                                           fixcm=False)  # this is the stable one with friction=0.5
    elif thermo_stat.lower() == 'velocityverlet':
        integratorx = mdx.VelocityVerletX.create(timestep=timestep * fs, calculator=calc)
    else:
        raise NotImplementedError('thermo_stat not implemented')
    return integratorx


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='script for running MD with NN potentials',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # block to control the molecular dynamic simulations
    parser.add_argument('-data', '--data', type=str, help='data file containing the starting structure')
    parser.add_argument('-temp', '--temp', type=float, nargs='+', help='temperature at which to run the simulation')
    parser.add_argument('-steps', '--steps', type=int, default=6000, help='number of steps to run the simulations')

    # block to control the creation of MLFF potential
    parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory. Defaults to the current directory.')
    parser.add_argument('--ckpt_dir_opt', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory to create potential for optimisation. Defaults to the current directory.')
    parser.add_argument('--ckpt_dir_md', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory to create potential for MD. Defaults to the current directory.')

    # block to control the batching of calculations
    parser.add_argument('-br', '--batch_run', action='store_true', help='whether this is a batch run.')
    parser.add_argument('-part', '--part', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=50)

    # block to control the post-analysis on the fly
    parser.add_argument('-fe', '--calculate_free_energy', action='store_true',
                        help='whether to calculate the vibrational free energies after the MD simulations.')

    parser.add_argument('-save_trajectory', '--save_trajectory', action='store_true',
                        help='whether to save the MD trajectory after MD run. Caution: for batch run, this can take up a lot of disk space!')
    parser.add_argument('--save_every_i', type=int, required=False, default=1)
    parser.add_argument('--save_from_i', type=int, required=False, default=0)
    parser.add_argument('--save_up_to', type=int, required=False, default=-1)

    parser.add_argument('-thermostat', '--thermostat', type=str, default='velocityverlet',
                        choices=['velocityverlet', 'langevin', 'nosehoover'],
                        help='which thermostat to use in the calculation')

    parser.add_argument('-supercell', '--supercell', type=float, nargs=3, required=False, default=[1, 1, 1],
                        help='size of the supercell to be use in the calculation')

    args = parser.parse_args()

    data = np.load(args.data)

    if not args.batch_run:
        atoms = Atoms(positions=data['R'][99], numbers=data['z'][99], cell=data['unit_cell'][0], pbc=[True, True, True])
        run_molecular_dynamics(atoms=atoms,
                               ckpt_dir=args.ckpt_dir,
                               temperature=args.temp[0],
                               Nsteps=args.steps,
                               thermo_stat=args.thermostat,
                               supercell=args.supercell)
    else:
        part = args.part - 1

        print(args.temp)
        run_molecular_dynamics_batch(data=data,
                                     ckpt_dir_opt=args.ckpt_dir_opt,
                                     ckpt_dir_md=args.ckpt_dir_md,
                                     opt_init=True,
                                     temperature=args.temp,
                                     Nsteps=args.steps,
                                     part=part,
                                     batch_size=args.batch_size,
                                     calculate_free_energies=True,
                                     save_from_i=args.save_from_i,
                                     save_up_to=args.save_up_to,
                                     save_every_i=args.save_every_i,
                                     thermo_stat=args.thermostat)
