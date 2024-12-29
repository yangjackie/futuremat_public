import argparse
import portpicker
import jax
import jax.numpy as jnp

import os
import numpy as np

from mlff.io.io import create_directory, bundle_dicts, save_dict, read_json
from mlff.data import DataTuple, DataSet
from mlff.properties.property_names import *
from mlff.nn import So3krates
from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
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
parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs to train', default=10)

# arguments controlling the network structures
parser.add_argument('-F', '--F', type=int, default=32)
parser.add_argument('-n_layer', '--n_layer', type=int, default=3)
parser.add_argument('-rcut', '--rcut', type=float, help='Cutoff radius to find neighboring atoms', default=5)
parser.add_argument('--H', type=int, required=False, default=2, help='Number of heads.')
# parser.add_argument('--degrees',  type=int, required=False, default=3,
#                        help='Degrees for the spherical harmonic coordinates.')
# wandb arguments
parser.add_argument('-p', '--project', type=str, help='name of the wandb project', default=None)



args = parser.parse_args()

port = portpicker.pick_unused_port()
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
    node_mask: 'node_mask'
}

args.checkpoint_path = os.getcwd() + '/' + args.checkpoint_path
ckpt_dir = os.path.join(args.checkpoint_path, 'module')
ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

temperatures = [200, 300]
data_paths = ['disp_' + str(t) + 'K.npz' for t in temperatures]
datas = [dict(np.load(p)) for p in data_paths]
data_sets = [DataSet(data=data, prop_keys=prop_keys) for data in datas]
for t, data_set in enumerate(data_sets):
    data_set.random_split(n_train=args.n_train,
                          n_valid=args.n_valid,
                          n_test=None,
                          mic=True,
                          r_cut=args.rcut,
                          training=True,
                          seed=random.randint(1, 1000))
    data_set.shift_x_by_mean_x(x=energy)
    data_set.save_splits_to_file(ckpt_dir, 'splits_' + str(t) + 'K.json')
    data_set.save_scales(ckpt_dir, 'scales_' + str(t) + 'K.json')

splited_data = [data_set.get_data_split() for data_set in data_sets]

# setting up the model, just take the existing example from mlff
net = So3krates(F=args.F,
                n_layer=args.n_layer,
                prop_keys=prop_keys,
                geometry_embed_kwargs={'degrees': [1, 2],
                                       'r_cut': args.rcut,
                                       },
                so3krates_layer_kwargs={'n_heads': args.H,
                                        'degrees': [1, 2]})

obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

opt = Optimizer()
tx = opt.get(learning_rate=1e-3)

coach = Coach(inputs=[atomic_position, atomic_type, idx_i, idx_j, node_mask],
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

train_state = None

steps_per_epoch = args.n_train // args.n_batch_train
steps_in_epoch = steps_per_epoch * args.epochs
num_epochs_ran = 0
for iterations in range(100):
    for d_idx, data in enumerate(splited_data):
        train_ds = data_tuple(data['train'])
        valid_ds = data_tuple(data['valid'])
        inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])

        if (iterations == 0) & (d_idx == 0):
            params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
            train_state, h_train_state = create_train_state(net, params, tx,
                                                            polyak_step_size=None,
                                                            plateau_lr_decay={'patience': 50, 'decay_factor': 1.},
                                                            scheduled_lr_decay={
                                                                'exponential': {'transition_steps': 10_000,
                                                                                'decay_factor': 0.9}})
            h_net = net.__dict_repr__()
            h_opt = opt.__dict_repr__()
            h_coach = coach.__dict_repr__()
            h_dataset = data_sets[d_idx].__dict_repr__()
            h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
            save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)
            wandb.init(config=h, project=args.project)

        coach.run(train_state=train_state,
                  train_ds=train_ds,
                  valid_ds=valid_ds,
                  loss_fn=loss_fn,
                  log_every_t=1,
                  restart_by_nan=True,
                  use_wandb=True,
                  start_idx=1 + steps_in_epoch * num_epochs_ran)
        num_epochs_ran += 1
