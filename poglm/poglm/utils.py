import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations


def exp_basis(decay: float, window_size: int, time_span: float):
    """Exponential decay basis.
    
    \phi(t) = \beta exp(-\beta t)

    Parameters
    ----------
    decay : float
        Decay parameter.
    window_size : int
        Number of time bins descretized.
    time_span : float
        Max influence time span.

    Returns
    -------
    basis : ndarray of shape (window_size,)
        Descretized basis.
    """

    basis = torch.zeros(window_size)
    dt = time_span / window_size
    t = torch.linspace(dt, time_span, window_size)
    basis = torch.exp(-decay * t)
    basis /= (dt * basis.sum(axis=0)) # normalization
    return basis
    

def poisson_to_categorical(mean: torch.FloatTensor, n_categories: int, eps: float = 1e-8) -> torch.FloatTensor:
    """Poisson parameter to log of the categorical parameters.
    
    Parameters
    ----------
    mean : torch.FloatTensor of shape (*,)
        Mean or the parameter of the Poisson distribution.
    n_categories : int
        Number of categories.
    eps: float
        Default, 1e-8.
    
    Returns
    -------
    ln_p : torch.FloatTensor of shape (*, n_categories)
        Log of the categorical parameters.
    """
    
    ln_pi = torch.zeros(list(mean.shape) + [n_categories])
    for k in range(1, n_categories):
        ln_pi[..., k] = k * (mean + eps).log() - mean - torch.lgamma(torch.tensor(k+1))
    ln_pi[..., 0] = (1 - ln_pi[..., 1:].exp().sum(dim=-1)).log()
    return ln_pi


def convolve_spikes_with_basis(spikes_list: torch.FloatTensor, basis: torch.FloatTensor, direction: str) -> torch.FloatTensor:
    """Convolve soft spike train soft_spikes_list[:, :, j] with a single basis.

    Parameters
    ----------
    spikes_list : torch.FloatTensor of shape (n_seq, n_time_bins, n_neurons) or (n_seq, n_time_bins, n_neurons, max_n_spikes)
        Spike train. The values can be continuous that are from soft spike train.
    basis : torch.FloatTensor of shape (window_size,)
        Descretized basis.
    direction : str in ['forward' | 'backward']

    Returns
    -------
    convolved_spikes_list : torch.FloatTensor of shape (n_time_bins, n_neurons)
        Convolved spike train.
    """
    
    window_size = len(basis)
    if len(spikes_list.shape) == 4:
        spikes_list = spikes_list @ torch.arange(spikes_list.shape[-1], dtype=torch.float32)
    n_seq, n_time_bins, n_neurons = spikes_list.shape

    if direction == 'forward':
        convolved_spikes_list = torch.zeros_like(spikes_list)
        padded_spikes_list = torch.cat((torch.zeros((n_seq, window_size, n_neurons)), spikes_list), dim=-2)
        for i in range(window_size):
            convolved_spikes_list = convolved_spikes_list + basis[-(i+1)] * padded_spikes_list[:, i:n_time_bins+i, :]
        return convolved_spikes_list
    elif direction == 'backward':
        rev_convolved_spikes_list = torch.zeros_like(spikes_list)
        padded_spikes_list = torch.cat((spikes_list, torch.zeros((n_seq, window_size, n_neurons))), dim=-2)
        for i in range(window_size):
            rev_convolved_spikes_list = rev_convolved_spikes_list + basis[i] * padded_spikes_list[:, i+1:n_time_bins+i+1, :]
        return rev_convolved_spikes_list


def params_abs_error(gen_model: dict, inf_model: dict) -> tuple:
    with torch.no_grad():
        weight_error = (gen_model['linear.weight'] - inf_model['linear.weight']).abs()
        bias_error = (gen_model['linear.bias'] - inf_model['linear.bias']).abs()
    return weight_error, bias_error


def match_hidden_neurons_according_weight(gen_model: dict, inf_model: nn.Module) -> torch.LongTensor:
    all_possible_permutations = torch.tensor(list(permutations(range(inf_model.n_vis_neurons, inf_model.n_neurons))))
    n_possible_permutations = len(all_possible_permutations)
    all_possible_permutations = torch.cat((torch.arange(inf_model.n_vis_neurons).expand((n_possible_permutations, inf_model.n_vis_neurons)), all_possible_permutations), dim=1)
    
    error_list = torch.zeros(n_possible_permutations)
    with torch.no_grad():
        for permutation in range(n_possible_permutations):
            error_list[permutation] = (gen_model['linear.weight'] - inf_model.linear.weight.data[all_possible_permutations[permutation], :][:, all_possible_permutations[permutation]]).abs().mean().item()
        true_to_learned = all_possible_permutations[error_list.argmin()]
    return true_to_learned


def continuous_to_discrete(timestamps_list: list, dt: float, T: float) -> np.ndarray:
    """Convert timestamps spike data to discretized spike count data.
    
    Parameters
    ----------
    timestamps_list : list of shape (n_neurons,)
        Spiking time for each neuron.
    dt : float
        Width of time bins.
    T : float
        Final time.

    Returns
    -------
    spikes : ndarray of shape (n_time_bins, n_neurons)
        Discretized spike count.
    """
    n_time_bins = int(T / dt)
    time_bins = np.linspace(0, T, n_time_bins+1)
    n_neurons = len(timestamps_list)
    spikes = np.zeros((n_time_bins, n_neurons))

    for neuron in range(n_neurons):
        spikes[:, neuron] = np.histogram(timestamps_list[neuron], bins=time_bins)[0]
    return spikes


def visualize_linear(state_dict: dict, n_neurons: int, n_vis_neurons: int, ax, v=1):
    w = torch.zeros((n_neurons, n_neurons+1))
    if 'linear.bias' in state_dict.keys():
        if state_dict['linear.bias'].shape[0] == n_neurons:
            # POGLM
            w[:, 0] = state_dict['linear.bias']
            w[:, 1:] = state_dict['linear.weight']
        elif state_dict['linear.weight'].shape[1] == n_vis_neurons:
            # forward
            w[n_vis_neurons:, 0] = state_dict['linear.bias']
            w[n_vis_neurons:, 1:n_vis_neurons+1] = state_dict['linear.weight']
        else:
            # forward-self
            w[n_vis_neurons:, 0] = state_dict['linear.bias']
            w[n_vis_neurons:, 1:] = state_dict['linear.weight']
    else:
        # forward-backward
        w[n_vis_neurons:, 0] = state_dict['linear_forward.bias']
        w[n_vis_neurons:, 1:n_vis_neurons+1] = state_dict['linear_forward.weight']
        w[:n_vis_neurons, 1+n_vis_neurons:] = state_dict['linear_backward.weight'].T
    im = ax.matshow(w, cmap='seismic', vmin=-v, vmax=v)
    ax.tick_params(left=False, top=False, bottom=False, labelleft=False, labeltop=False)
    ax.hlines(y=[n_vis_neurons-0.5], xmin=[0.5], xmax=[n_neurons+0.5], colors='y')
    ax.vlines(x=[0.5, n_vis_neurons+0.5], ymin=[-0.5, -0.5], ymax=[n_neurons-0.5, n_neurons-0.5], colors='y')
    return im


def visualize_weight(w: torch.FloatTensor, n_neurons: int, n_vis_neurons: int, ax, v=1):
    im = ax.matshow(w, cmap='seismic', vmin=-v, vmax=v)
    ax.tick_params(left=False, top=False, bottom=False, labelleft=False, labeltop=False)
    ax.hlines(y=[n_vis_neurons-0.5], xmin=[-0.5], xmax=[n_neurons-0.5], colors='y')
    ax.vlines(x=[n_vis_neurons-0.5], ymin=[-0.5], ymax=[n_neurons-0.5], colors='y')
    return im


# def visualize_spikes(spikes, firing_rates_pred, firing_rates=None, fig=None, ax=None, n_neurons_plot=None, n_time_bins_plot=None) -> None:
#     n_time_bins, n_neurons = spikes.shape
#     if n_neurons_plot is None:
#         n_neurons_plot = n_neurons
#     if n_time_bins_plot is None:
#         n_time_bins_plot = n_time_bins
#     if fig is None:
#         fig, axs = plt.subplots(n_neurons_plot, 1, figsize=(10, 3*n_neurons_plot), sharex=True)
#     n_time_bins = spikes.shape[0]
#     for neuron in range(n_neurons_plot):
#         axs[neuron].plot(spikes[:n_time_bins_plot, neuron], label='spikes')
#         axs[neuron].plot(firing_rates_pred[:n_time_bins_plot, neuron], label='predicted firing rates')
#         if firing_rates is not None:
#             axs[neuron].plot(firing_rates[:n_time_bins_plot, neuron], label='firing rates')
#         axs[neuron].set_ylabel(f"neuron {neuron}")
#     plt.xlabel("$t$")
#     axs[0].legend()
#     # plt.suptitle("Conditional intensity (firing rates / dt) and spike trains for all neurons")
#     # return fig


# def half_of_max(continuous_pred: torch.FloatTensor):
#     warnings.warn('Please use `utils.predict_type` instead.', DeprecationWarning)
#     return torch.bucketize(continuous_pred, torch.tensor([-torch.inf, continuous_pred.min()/100, continuous_pred.max()/100, torch.inf])) - 2
