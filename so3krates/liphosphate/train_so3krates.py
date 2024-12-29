"""
Module adapted from the MLFF package (https://github.com/thorben-frank/mlff/blob/main/examples/train_so3krates.py)
to control and perform the training of the SO3krates neural network forcefield, that is to be tested on the
garnet-like (e.g. LLZO) solid electrolyte dataset.
"""

import argparse
from pathlib import Path

import portpicker
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import os
import numpy as np

from mlff.io.io import create_directory, bundle_dicts, save_dict
from mlff.io import load_params_from_ckpt_dir
from mlff.data import DataTuple, DataSet
from mlff.properties.property_names import *
from mlff.nn import So3krates
from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.cAPI.process_argparse import StoreDictKeyPair
import mlff.properties.property_names as pn

"""
Monitoring the training outcomes with different choices of the hyperparameters using the weight and loss
web application. Note that if this is to be run on a compute node in the HPC, which does not have an internet
connection to the outside world, the option
       wandb offline
must be specified in the job submission script. The results can be regularly uploaded to the portal by using
       wandb sync --sync-all
"""
import wandb
import random

parser = argparse.ArgumentParser(description='Controls for running the SO3 neural networks for the LLZO data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-dp', '--data_path', type=str, help='the .npz file that stores the training and test data')
parser.add_argument('-cp', '--checkpoint_path', type=str,
                    help='path to save the checkpoint output from training the n.n.', default='ckpt_dir')

# arguments controlling model trainings
parser.add_argument('-n_train', '--n_train', type=int, help='number of training data to use', default=200)
parser.add_argument('-n_valid', '--n_valid', type=int, help='number of valid data to use', default=200)
parser.add_argument('-n_batch_train', '--n_batch_train', type=int, help='batch size for training', default=20)
parser.add_argument('-n_batch_valid', '--n_batch_valid', type=int, help='batch size for validation', default=20)
parser.add_argument('-w_en', '--energy_weight', type=float, help='weight for the energy component in the loss function',
                    default=0.95)
parser.add_argument('-w_f', '--force_weight', type=float, help='weight for the force component in the loss function',
                    default=0.05)
parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs to train', default=500)

# arguments controlling the network structures
parser.add_argument('-F', '--F', type=int, default=32)
parser.add_argument('-n_layer', '--n_layer', type=int, default=3)
parser.add_argument('-rcut', '--rcut', type=float, help='Cutoff radius to find neighboring atoms', default=5)
parser.add_argument('--H', type=int, required=False, default=2, help='Number of heads.')
# parser.add_argument('--degrees',  type=int, required=False, default=3,
#                        help='Degrees for the spherical harmonic coordinates.')
# wandb arguments
parser.add_argument('-p', '--project', type=str, help='name of the wandb project', default=None)
parser.add_argument('--lr', type=float, required=False, default=1e-3)
parser.add_argument('--lr_decay_plateau', action=StoreDictKeyPair, required=False, default=None)

lr_decay_exp_default = {'transition_steps': 10_000, 'decay_factor': 0.9}
parser.add_argument('--lr_decay_exp', action=StoreDictKeyPair, required=False, default=lr_decay_exp_default)
parser.add_argument('--lr_decay_exp_df', type=float, default=0.9)

parser.add_argument('--lr_warmup', action=StoreDictKeyPair, required=False, default=None)

parser.add_argument('--restart_from_ckpt_dir', type=str, required=False, default=None,
                    help='Path to a checkpoint directory from which to load model parameters and start the '
                         'training.')

parser.add_argument('-degrees', '--degrees', type=int, nargs='+',
                    help='angular momentums for the spherical harmonics ')

args = parser.parse_args()

# =================== Main part of the script =========================
port = portpicker.pick_unused_port()
print(port)
# this wont work for parallelisations on CPUs, only for GPU and TPU


jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

prop_keys = {
    energy: 'E',
    force: 'F',
    atomic_position: 'R',
    atomic_type: 'z',
    unit_cell: 'unit_cell',
    pbc: 'pbc',
    idx_i: 'idx_i',
    idx_j: 'idx_j',
    node_mask: 'node_mask',
    cell_offset: 'cell_offset',
}

inputs = [pn.atomic_type, pn.atomic_position, pn.idx_i, pn.idx_j, pn.node_mask]
inputs += [pn.unit_cell]
inputs += [pn.cell_offset]

# Set up the filesystem for this calculation
args.checkpoint_path = os.getcwd() + '/' + args.checkpoint_path
ckpt_dir = os.path.join(args.checkpoint_path, 'module')
ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

# Loading the data for training
if not os.path.isfile(args.data_path):
    raise Exception(f"Data file {args.data_path} does not exist!")
data = dict(np.load(args.data_path))

# set up the property keys for this set

# set up the dataset

data_set = DataSet(data=data, prop_keys=prop_keys)
data_set.random_split(n_train=args.n_train,
                      n_valid=args.n_valid,
                      n_test=None,
                      mic=True,
                      r_cut=args.rcut,
                      training=True,
                      seed=random.randint(1, 1000))

data_set.shift_x_by_mean_x(x=energy)
# persisting this particular set (good idea for back tracking what's going on!
data_set.save_splits_to_file(ckpt_dir, 'splits.json')
data_set.save_scales(ckpt_dir, 'scales.json')

d = data_set.get_data_split()  # not sure

# setting up the model, just take the existing example from mlff
net = So3krates(F=args.F,
                n_layer=args.n_layer,
                prop_keys=prop_keys,
                geometry_embed_kwargs={'degrees': args.degrees,
                                       'r_cut': args.rcut,
                                       'mic': True
                                       },
                so3krates_layer_kwargs={'n_heads': args.H,
                                        'degrees': args.degrees})

obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

opt = Optimizer()
tx = opt.get(learning_rate=args.lr)

coach = Coach(inputs=inputs,
              targets=[energy, force],
              epochs=args.epochs,
              training_batch_size=args.n_batch_train,
              validation_batch_size=args.n_batch_valid,
              loss_weights={energy: args.energy_weight, force: args.force_weight},
              ckpt_dir=ckpt_dir,
              data_path=args.data_path,
              net_seed=0,
              training_seed=0)
loss_fn = get_loss_fn(obs_fn=obs_fn,
                      weights=coach.loss_weights,
                      prop_keys=prop_keys)

data_tuple = DataTuple(inputs=coach.inputs,
                       targets=coach.targets,
                       prop_keys=prop_keys)

train_ds = data_tuple(d['train'])
valid_ds = data_tuple(d['valid'])

inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])

restart_from_ckpt_dir = None
if args.restart_from_ckpt_dir is not None:
    restart_from_ckpt_dir = (Path(args.restart_from_ckpt_dir).absolute().resolve()).as_posix()
    assert restart_from_ckpt_dir != ckpt_dir

if restart_from_ckpt_dir is None:
    params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
else:
    print(f"Restarting training from {restart_from_ckpt_dir}.")
    params = load_params_from_ckpt_dir(restart_from_ckpt_dir)
    print("parameters loaded successfully " + str(params))

train_state, h_train_state = create_train_state(net,
                                                params,
                                                tx,
                                                polyak_step_size=None,
                                                plateau_lr_decay={'patience': 50,
                                                                  'decay_factor': 1.0
                                                                  },
                                                scheduled_lr_decay={'exponential': {'transition_steps': 100000,
                                                                                    'decay_factor': args.lr_decay_exp_df}
                                                                    # default setting is 0.9
                                                                    }
                                                )

h_net = net.__dict_repr__()
h_opt = opt.__dict_repr__()
h_coach = coach.__dict_repr__()
h_dataset = data_set.__dict_repr__()
h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

if args.project is None:
    raise Exception("Please specify the project name for wandb")

wandb.init(config=h, project=args.project)
coach.run(train_state=train_state,
          train_ds=train_ds,
          valid_ds=valid_ds,
          loss_fn=loss_fn,
          log_every_t=1,
          restart_by_nan=True,
          use_wandb=True)
