import argparse
import os
from pathlib import Path
import json
import jax
import numpy as np
from functools import partial
from pprint import pprint
from typing import Dict
import flax

from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.io import read_json, load_params_from_ckpt_dir
from mlff.nn.stacknet import (
    get_energy_force_stress_fn,
    get_obs_and_force_fn,
    get_observable_fn,
    init_stack_net,
)
from mlff.training import Coach
from mlff.properties import property_names as pn
from mlff.data import DataSet, DataTuple
from mlff.inference.evaluation import evaluate_model, mae_metric, rmse_metric, r2_metric


def scale(_k, _v):
    return scales[prop_keys_inv[_k]]['scale'] * _v


def shift(_k, _v, _z):
    shifts = np.array(scales[prop_keys_inv[_k]]['per_atom_shift'], np.float64)[_z.astype(int)].sum(
        axis=-1)  # shape: (B)
    return _v + np.expand_dims(shifts, [i for i in range(1, _v.ndim)])


def scale_and_shift_fn(_x: Dict, _z: np.ndarray):
    return {_k: shift(_k, scale(_k, _v), _z) for (_k, _v) in _x.items()}


def test_obs_fn(params, inputs):
    z_key = prop_keys[pn.atomic_type]
    nn_out = jax.tree_map(lambda _x: np.array(_x, dtype=np.float64), _test_obs_fn(params, inputs))
    z = inputs[z_key]  # shape: (B,n)

    if pn.stress in targets:
        stress_key = prop_keys[pn.stress]
        cell_key = prop_keys[pn.unit_cell]

        cell = inputs[cell_key]  # shape: (B,3,3)
        cell_volumes = np.abs(np.linalg.det(cell))  # shape: (B)
        scaled_stress = nn_out[stress_key]  # shape: (B,3,3)
        stress = scaled_stress / cell_volumes[:, None, None]  # shape: (B,3,3)
        nn_out[stress_key] = stress

    return scale_and_shift_fn(nn_out, z)

parser = argparse.ArgumentParser(description='Evaluate the so3krates NN model.')
parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                    help='Path to the checkpoint directory. Defaults to the current directory.')
parser.add_argument("--prop_keys", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                    help='Property keys of the data set. Needs only to be specified, if e.g. the keys of the '
                         'properties in the data set that the model is applied to differ from the keys the model'
                         'has been trained on.')
parser.add_argument('--batch_size', type=int, required=False, default=10,
                        help="Batch size of the inference passes. Default=10")
parser.add_argument('--targets', nargs='+', required=False, default=None)
parser.add_argument('--jax_dtype', type=str, required=False, default='x64',
                    help='Set JAX default dtype. Default is jax.numpy.float32')
parser.add_argument('--apply_to', type=str, required=False, default=None,
                    help='Path to data file that the model should be applied to. '
                         'Defaults to the training data file.')
parser.add_argument('--output', type=str, required=False, default='results',
                    help='Name of the output file to save the prediction outcome (for making correlation plots etc)')
args = parser.parse_args()

# Read arguments
ckpt_dir = args.ckpt_dir
prop_keys = args.prop_keys
_targets = args.targets
apply_to = args.apply_to
batch_size = args.batch_size

jax_dtype = args.jax_dtype
if jax_dtype == 'x64':
    jax.config.update("jax_enable_x64", True)

ckpt_dir = (Path(args.ckpt_dir).absolute().resolve()).as_posix()
print("Loading parameters from ", ckpt_dir)
h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))  # assume this only loads the network structures?
scales = read_json(os.path.join(ckpt_dir, 'scales.json'))

coach = Coach(**h['coach'])
targets = _targets if _targets is not None else coach.targets

print("Initialising the neural network")
test_net = init_stack_net(h)
print("SO3krates initialised!\n")

# some stuff about property keys
_prop_keys = test_net.prop_keys
if prop_keys is not None:
    _prop_keys.update(prop_keys)
    test_net.reset_prop_keys(prop_keys=_prop_keys)
prop_keys = test_net.prop_keys
# print("The following property keys have been initialised: ")
# for k in prop_keys:
#    print(k)

r_cut = [x[list(x.keys())[0]]['r_cut'] for x in h['stack_net']['geometry_embeddings'] if
         list(x.keys())[0] == 'geometry_embed'][0]
# mic = [x[list(x.keys())[0]]['mic'] for x in h['stack_net']['geometry_embeddings'] if
#             list(x.keys())[0] == 'geometry_embed'][0]
# not sure why mic is picked up wrongly, hard-coded it to be true
mic = True

network_params = load_params_from_ckpt_dir(ckpt_dir)  # these are the actual learnt parameters for the network?
print("Learnt network parameters loaded!")

print("Prediction targets are:", targets)
if pn.force in targets:
    if pn.stress in targets:
        _test_obs_fn = jax.jit(jax.vmap(get_energy_force_stress_fn(test_net), in_axes=(None, 0)))
    else:
        _test_obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(test_net), in_axes=(None, 0)))
else:
    _test_obs_fn = jax.jit(jax.vmap(get_observable_fn(test_net), in_axes=(None, 0)))

prop_keys_inv = {v: k for (k, v) in prop_keys.items()}

# ==================================================================
# Data loading block
# ==================================================================
# for the moment only applying to the npz file generated previously
if Path(apply_to).suffix == '.npz':
    test_data = dict(np.load(apply_to))
else:
    raise NotImplementedError

print("start loading the data from:",apply_to)
test_data_set = DataSet(prop_keys=prop_keys, data=test_data)
test_data_set.random_split(n_train=0,
                           n_valid=0,
                           n_test=len(test_data['E']), #we want to test them on all data, including those being used for training
                           mic=mic,
                           training=False,
                           r_cut=r_cut)
d_test = test_data_set.get_data_split()['test']
print("test data loaded!")

test_data_tuple = DataTuple(inputs=coach.inputs,
                                targets=targets,
                                prop_keys=prop_keys)
test_input, test_obs = test_data_tuple(d_test)

test_metrics, test_obs_pred = evaluate_model(params=network_params,
                                                 obs_fn=test_obs_fn,
                                                 data=(test_input, test_obs),
                                                 batch_size=batch_size,
                                                 metric_fn={'mae': partial(mae_metric, pad_value=0),
                                                            'rmse': partial(rmse_metric, pad_value=0),
                                                            'R2': partial(r2_metric, pad_value=0)}
                                                 )
print(f'Metrics on the testing data: ')
pprint(test_metrics)
np.savez(args.output,predictions=test_obs_pred['predictions'],targets=test_obs_pred['targets'],ckpt_dir=ckpt_dir,apply_to=apply_to)

print(test_obs_pred.keys())
#flattened_dict=flax.traverse_util.flatten_dict(test_obs_pred['intermediate'],sep='/')
