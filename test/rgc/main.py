import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import binned_statistic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time
import argparse

from poglm import utils, inference, distributions, model


## arguments
parser = argparse.ArgumentParser()
parser.add_argument('idx', type=int)
args = parser.parse_args()

vari_list = ['F', 'FS', 'FB']
distribution_list = ['Poisson', 'categorical', 'GS-score', 'GS-pathwise', 'exponential', 'Rayleigh', 'half-normal']
n_hid_neurons_list = [1, 2, 3]
seed_list = np.arange(10)

arg_index = np.unravel_index(args.idx, (len(vari_list), len(distribution_list), len(n_hid_neurons_list), len(seed_list)))
vari, distribution, n_hid_neurons, seed = vari_list[arg_index[0]], distribution_list[arg_index[1]], n_hid_neurons_list[arg_index[2]], seed_list[arg_index[3]]
print(f'args.idx: {args.idx}')
print(f'parameters: {vari}_{distribution}_{n_hid_neurons}_{seed}')


def load_data():
    temp = loadmat(f'rgcData_Nature08/SpTimesRGC.mat', squeeze_me=False, struct_as_record=False)['SpTimes'][0]
    n_time_bins = 20 * 60 * 120 # 20 min * 119.9820 Hz
    time_bins = np.linspace(1, n_time_bins, n_time_bins)
    n_neurons = 27
    spikes = np.zeros((n_time_bins, n_neurons))
    for i in range(n_neurons):
        spikes[:, i] = binned_statistic(temp[i][:, 0], None, bins=np.hstack(([0], time_bins)), statistic='count')[0].T
    return spikes

spikes = load_data()

## hyper-parameters
decay = 0.25
window_size = 5
n_vis_neurons = spikes.shape[1]
n_neurons = n_vis_neurons + n_hid_neurons
basis = utils.exp_basis(decay, window_size, window_size)


vis_spikes_list_train, vis_spikes_list_test = torch.from_numpy(spikes[:96000].reshape(960, 100, -1)).to(torch.float32), torch.from_numpy(spikes[96000:].reshape(480, 100, -1)).to(torch.float32)
convolved_vis_spikes_list_train = utils.convolve_spikes_with_basis(vis_spikes_list_train, basis, direction='forward')
convolved_vis_spikes_list_test = utils.convolve_spikes_with_basis(vis_spikes_list_test, basis, direction='forward')
rev_convolved_vis_spikes_list_train = utils.convolve_spikes_with_basis(vis_spikes_list_train, basis, 'backward')
rev_convolved_vis_spikes_list_test = utils.convolve_spikes_with_basis(vis_spikes_list_test, basis, 'backward')
train_dataset = TensorDataset(vis_spikes_list_train, convolved_vis_spikes_list_train, rev_convolved_vis_spikes_list_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)


n_epochs = 20
print_freq = 1
n_monte_carlo = 5

## model initialization
torch.manual_seed(seed)
if vari == 'F':
    vari_model = model.Forward
elif vari == 'FS':
    vari_model = model.ForwardSelf
elif vari == 'FB':
    vari_model = model.ForwardBackward
if distribution == 'Poisson':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.Poisson())
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.Poisson())
    grad = 'score'
elif distribution == 'categorical':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.Categorical(n_categories=5))
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.Categorical(n_categories=5))
    grad = 'score'
elif distribution == 'GS-score':
    tau = 1.0
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.GumbelSoftmax(n_categories=5, tau=tau))
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.GumbelSoftmax(n_categories=5, tau=tau))
    grad = 'score'
elif distribution == 'GS-pathwise':
    tau = 0.5
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.GumbelSoftmax(n_categories=5, tau=tau))
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.GumbelSoftmax(n_categories=5, tau=tau))
    grad = 'pathwise'
elif distribution == 'exponential':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.Exponential())
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.Exponential())
    grad = 'pathwise'
elif distribution == 'Rayleigh':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.Rayleigh())
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.Rayleigh())
    grad = 'pathwise'
elif distribution == 'half-normal':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.HalfNormal())
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.HalfNormal())
    grad = 'pathwise'

inf_optimizer = torch.optim.Adam(inf_model.parameters(), lr=0.02)
vari_optimizer = torch.optim.Adam(vari_model.parameters(), lr=0.02)

## training
start = time.time()
__ = inference.variational_inference(inf_model, vari_model, inf_optimizer, vari_optimizer, train_dataloader, n_epochs, n_monte_carlo, grad, print_freq)
end = time.time()

torch.save(inf_model.state_dict(), f'model/{vari}_{distribution}_{n_hid_neurons}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/{vari}_{distribution}_{n_hid_neurons}_{seed}_vari.pt')

## evaluate
inf_model.distribution = distributions.Poisson()
vari_model.distribution = distributions.Poisson()
df = inference.evaluate(inf_model, vari_model, vis_spikes_list_test, convolved_vis_spikes_list_test, rev_convolved_vis_spikes_list_test, n_monte_carlo=n_monte_carlo).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/{vari}_{distribution}_{n_hid_neurons}_{seed}.csv', index=False)
