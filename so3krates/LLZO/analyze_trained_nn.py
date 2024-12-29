from mlff.io import load_params_from_ckpt_dir
from mlff.nn.stacknet import StackNet
from mlff.nn import So3krates
from mlff.nn.stacknet import get_obs_and_force_fn, get_observable_fn, get_energy_force_stress_fn
from pathlib import Path

from mlff.training import Optimizer, create_train_state
import numpy as np

CKPT_DIR="/scratch/dy3/jy8620/LLZO/llzo_12_44/training_more_complex_nn_in_equi_variant_new/ckpt_dir/learning_curve_4000_longer_train_2/module/"
CKPT_DIR = (Path(CKPT_DIR).absolute().resolve()).as_posix()

net=StackNet.create_from_ckpt_dir(CKPT_DIR)
params = load_params_from_ckpt_dir(CKPT_DIR)

opt = Optimizer()
tx = opt.get(learning_rate=1e-3)
train_state, h_train_state = create_train_state(net,params,tx)
print(np.shape(params['params']['layers_1']['FeatureBlock_0']['attention_fn']['VmapConvAttentionCoefficients_0']['Dense_0']['kernel']))
print(np.shape(params['params']['layers_1']['FeatureBlock_0']['attention_fn']['VmapConvAttentionCoefficients_0']['Dense_1']['kernel']))
print('aggregated',np.shape(params['params']['layers_1']['FeatureBlock_0']['attention_fn']['VmapAttentionAggregation_0']['Dense_0']['kernel']))

print(params['params']['layers_1']['FeatureBlock_0']['attention_fn']['VmapAttentionAggregation_0']['Dense_0'].keys())
print(params['params']['layers_1']['FeatureBlock_0']['attention_fn']['VmapAttentionAggregation_0'].keys())
print(params['params']['layers_1']['FeatureBlock_0']['attention_fn'].keys())
print(params['params']['layers_1']['FeatureBlock_0'].keys())
print(params['params']['layers_1'].keys())
print(params['params'].keys())