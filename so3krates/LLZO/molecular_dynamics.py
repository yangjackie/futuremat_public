import jax.numpy as jnp

import numpy as np
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, ZeroRotation, Stationary)
from ase import Atoms
from ase.units import fs, kB

from mlff import mdx
import os

ps = 1000*fs

data = np.load("/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing_warmup/MD_1500K_NH/MD_1500K.npz")
#atoms = Atoms(positions=data['R'][250], numbers=data['z'][250], cell=data['unit_cell'][0], pbc=[True,True,True])

#data = np.load("/scratch/dy3/jy8620/LLZO/llzo_12_44/opt_data_cLLZO_12_44_first_frame.npz")
atoms = Atoms(positions=data['R'][0], numbers=data['z'][0], cell=data['unit_cell'][0], pbc=[True,True,True])

T0 = 1500 * kB

MaxwellBoltzmannDistribution(atoms, temperature_K=T0/kB)

vel = atoms.get_velocities()
T = atoms.get_temperature()

# when it comes to running dynamics switch to mdx
ckpt_dir = os.getcwd()
dtype = jnp.float64

potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)

calc = mdx.CalculatorX.create(potential)
#integratorx = mdx.NoseHooverX.create(timestep=1.0*fs, temperature=T0, ttime=1.0, calculator=calc)
integratorx = mdx.LangevinX.create(timestep=1.0*fs, temperature=T0, friction=0.9, calculator=calc,fixcm=True) #this is the stable one with friction=0.5
#integratorx = mdx.VelocityVerletX.create(timestep=1*fs, calculator=calc)

simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()), save_frequency=1, run_interval=1)

atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,skin=0.5)

optimizer = mdx.GradientDescent.create(potential=potential,learning_rate=5e-4)
atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=50000, tol=0.08)
#atomsx_opt = atomsx
atomsx_opt = mdx.scale_momenta(atomsx_opt, T0=T0)
atomsx_opt = mdx.zero_rotation(mdx.zero_translation(atomsx_opt))

Nsteps=5000

simulator.run(integratorx, atomsx_opt, steps=Nsteps)
