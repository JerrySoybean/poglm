import numpy as np
import pandas as pd

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
distribution_list = ['true', 'Poisson', 'categorical', 'GS-score', 'GS-pathwise', 'exponential', 'Rayleigh', 'half-normal']
trial_list = np.arange(10)
seed_list = np.arange(0, 10)

arg_index = np.unravel_index(args.idx, (len(vari_list), len(distribution_list), len(trial_list), len(seed_list)))
vari, distribution, trial, seed = vari_list[arg_index[0]], distribution_list[arg_index[1]], trial_list[arg_index[2]], seed_list[arg_index[3]]
print(f'args.idx: {args.idx}', flush=True)
print(f'parameters: {vari}_{distribution}_{trial}_{seed}', flush=True)

## hyper-parameters
decay = 0.25
window_size = 5
n_neurons = 5
n_vis_neurons = 3
basis = utils.exp_basis(decay, window_size, window_size)

n_epochs = 20
print_freq = 1
n_monte_carlo = 5

## training data
df = pd.read_pickle('data.pkl')

spikes_list_train = df.at[trial, 'spikes_list_train']
convolved_spikes_list_train = df.at[trial, 'convolved_spikes_list_train']
rev_convolved_spikes_list_train = utils.convolve_spikes_with_basis(spikes_list_train, basis, 'backward')

train_dataset = TensorDataset(spikes_list_train[:, :, :n_vis_neurons], convolved_spikes_list_train[:, :, :n_vis_neurons], rev_convolved_spikes_list_train[:, :, :n_vis_neurons])
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False)

## model initialization
torch.manual_seed(seed)
if vari == 'F':
    vari_model = model.Forward
elif vari == 'FS':
    vari_model = model.ForwardSelf
elif vari == 'FB':
    vari_model = model.ForwardBackward
if distribution == 'true':
    inf_model = model.POGLM(n_neurons, n_vis_neurons, basis, distributions.Poisson())
    vari_model = vari_model(n_neurons, n_vis_neurons, basis, distributions.Poisson())
    grad = 'score'
elif distribution == 'Poisson':
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
with torch.no_grad():
    inf_model.linear.weight.data = torch.zeros((n_neurons, n_neurons))
    inf_model.linear.bias.data = torch.zeros((n_neurons, ))
if distribution == 'true':
    with torch.no_grad():
        inf_model.load_state_dict(df.at[trial, 'gen_model'])
        inf_model.linear.weight.requires_grad = False
        inf_model.linear.bias.requires_grad = False

inf_optimizer = torch.optim.Adam(inf_model.parameters(), lr=0.05)
vari_optimizer = torch.optim.Adam(vari_model.parameters(), lr=0.1)

if df.at[trial, 'gen_model']['linear.bias'].sum() == 0:
    inf_model.linear.bias.requires_grad = False
    if vari == 'FB':
        vari_model.linear_forward.bias.requires_grad = False
    else:
        vari_model.linear.bias.requires_grad = False

## training
start = time.time()
epoch_loss_list = inference.variational_inference(inf_model, vari_model, inf_optimizer, vari_optimizer, train_dataloader, n_epochs, n_monte_carlo, grad, print_freq, gen_model_state_dict=df.at[trial, 'gen_model'])
end = time.time()

true_to_learned = utils.match_hidden_neurons_according_weight(df.at[trial, 'gen_model'], inf_model)
inf_model.permute_hidden_neurons(true_to_learned)
vari_model.permute_hidden_neurons(true_to_learned)


## save model
torch.save(inf_model.state_dict(), f'model/{vari}_{distribution}_{trial}_{seed}_inf.pt')
torch.save(vari_model.state_dict(), f'model/{vari}_{distribution}_{trial}_{seed}_vari.pt')
with open(f'npy/{vari}_{distribution}_{trial}_{seed}.npy', 'wb') as f:
    np.save(f, epoch_loss_list)


## test data
spikes_list_test = df.at[trial, 'spikes_list_test']
vis_spikes_list_test = spikes_list_test[:, :, :inf_model.n_vis_neurons]
hid_spikes_list_test = spikes_list_test[:, :, inf_model.n_vis_neurons:]
convolved_spikes_list_test = df.at[trial, 'convolved_spikes_list_test']
convolved_vis_spikes_list_test = convolved_spikes_list_test[:, :, :inf_model.n_vis_neurons]
convolved_hid_spikes_list_test = convolved_spikes_list_test[:, :, inf_model.n_vis_neurons:]
rev_convolved_vis_spikes_list_test = utils.convolve_spikes_with_basis(vis_spikes_list_test, basis, 'backward')

## test
inf_model.distribution = distributions.Poisson()
vari_model.distribution = distributions.Poisson()
df = inference.evaluate(inf_model, vari_model, vis_spikes_list_test, convolved_vis_spikes_list_test, rev_convolved_vis_spikes_list_test, hid_spikes_list_test, convolved_hid_spikes_list_test, n_monte_carlo=2000).mean().to_frame().T
df['time'] = end - start
df.to_csv(f'csv/{vari}_{distribution}_{trial}_{seed}.csv', index=False)
