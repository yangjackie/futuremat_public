# twodPV 
## High-throughput screening of two-dimenional perovskite materials

This package contains code and data from high-throughput screening of two-dimensional materials.
The construction of the infrastructures take many inspirations from the  Computational 2D Materials Database (C2DB)
(https://cmr.fysik.dtu.dk/c2db/c2db.html) with modifications that suits our own needs, as well as the available 
computational infrastructures. 

The main computational driver is the VASP code, and our computational infrastructures
are based on the setup of Raijin/Gadi @ National Computing Infrastructure Australia 
(see https://nci.org.au/).

### Key dependencies

1. Python3
2. The Materials Genome Project and the Pymatgen code (https://pymatgen.org/, version '2019.7.2'+)
3. The Atomic Simulation Environment (https://wiki.fysik.dtu.dk/ase/, version '3.18.1'+)
4. MyQueue (https://myqueue.readthedocs.io/en/latest/index.html)

### Note on the MyQueue package

Some modifications are needed in order to make `myqueue` package work properly to adapt our own queue structures.
In this case, the `pbs.py` module in the original `myqueue` package is redirected to the modified `futuremat.utils.pbs` 
module. A key modification in this module is that a file called `node_info` is created upon
submitting the job to the queue with 

`mq submit folder/`

command. This writes the name of the node where this job has been submitted to in the submission directory.
Upon start of a TASK, this file will be read in order to get the number of cores per node information
and the VASP INCAR file will be updated accordingly. This is important to make sure VASP runs efficiently.
See (./geometry_optimisation.py).

### Key steps in the workflow

***Building the bulk perovskite library and distorted structures***

The `./bulk_library.py` script sets up calculation folders for bulk perovskites. Furthermore, for each
perovskite strutures, 10 randomly distorted structures will be made. This allows one to assess the 
energy landscape of perovskites given its chemical compositions ABC3.

***Energies of the consitituting elements***

The `./elements.py` script sets up calculation folder to perform geometry optimisations on the consitutent elements,
from which the energy (chemical potentials) can be extracted for the calculations of the formation energies of different
perovskite structures in both the bulk and 2D phases.

***Building the 2D PV library***

The `./2D_library.py`'` script will build 2D slabs with different orientations, surface terminations and thicknesses. 
Corresponding folders will be set up to run the calculations.

***Geometry optimisations***

The `./geometry_optimisation.py` module contains codes that ochestrate the geometry optimisation workflow adopted in this work for
for all structures. Basically all structures will be optimised with full spin polarisation (`ISPIN=2` in VASP). 
Occasionally this may encounter problems in converging the self-consistent field. In this case, the code will
automatically re-start with a spin non-polarised structure optimisation for a few steps, and a WAVECAR will be generated
for a spin polarisation  structure optimisation to proceed. We found that this generally solved the problem.
Otherwise, use should inspect the problematic case manually and further rectify the problem, if necessary.

The module contains methods that are specific to optimising different structures (bulk/2D) under different constraints
(full relax/symmetry perserving.) To perform a structural optimisation in the folder `SrTiO_3` (which is a two-D material
made of SrTiO3 with 3 atomic layer), excute the following in the folder above:

`mq submit twodPV.geometry_optimisation@default_two_d_optimisation -R 32:normalsl:1h -n SrTiO_3 SrTiO_3/ -T`

which requested a `normalsl` node with 32 cores for 1 h walltime. (You need to make sure the python code is 
in your `PYTHONPATH`, of course.)

***Job resubmission due to error***

The `myqueue` package contains useful command line tools that can help you to efficiently resubmit failed VASP jobs,
particuarly those failed due to time out. Our implemented method automatically start a geometry optimisation 
from a CONTCAR file if exists in the folder, so restart mechanism is by defult built in. To restart all timed-out calculations
in folder containing all two-D material calculations, for examples, in 

`./slab_100_AO/SrTiO_3/`,
`./slab_100_AO/SrTiO_5/`,
etc

simply run at the folder above `slab_100_AO`:

`mq resubmit -r slab_100_AO/ -s T`
