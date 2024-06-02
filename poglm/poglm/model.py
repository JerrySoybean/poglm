import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

from poglm import utils, distributions


class POGLM(nn.Module):
    def __init__(
            self,
            n_neurons: int,
            n_vis_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson()
        ) -> None:
        super().__init__()

        self.n_neurons = n_neurons
        self.n_vis_neurons = n_vis_neurons
        self.n_hid_neurons = n_neurons - n_vis_neurons
        
        self.basis = basis
        self.flipped_basis = torch.flip(self.basis, (0,))
        self.window_size = len(self.basis)
        
        self.distribution = distribution

        self.linear = nn.Linear(n_neurons, n_neurons)
    
    def permute_hidden_neurons(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self.linear.bias.data = self.linear.bias.data[true_to_learned]
            self.linear.weight.data = self.linear.weight.data[true_to_learned, :][:, true_to_learned]

    def forward(self, convolved_spikes_list: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear(convolved_spikes_list))

    def complete_log_likelihood(
            self,
            hid_spikes_list: torch.FloatTensor,
            convolved_hid_spikes_list: torch.FloatTensor,
            vis_spikes: torch.FloatTensor,
            convolved_vis_spikes: torch.FloatTensor
        ) -> torch.FloatTensor:
        n_samples = hid_spikes_list.shape[0]
        convolved_spikes_list = torch.cat((convolved_vis_spikes.expand((n_samples, -1, -1)), convolved_hid_spikes_list), dim=2)
        firing_rates_list = self.forward(convolved_spikes_list)
        vis_part = distributions.poisson_log_likelihood(vis_spikes.expand((n_samples, -1, -1)), firing_rates_list[:, :, :self.n_vis_neurons]).sum(dim=(-2, -1))
        hid_part = self.distribution.log_likelihood(hid_spikes_list, self.distribution.mean_to_params(firing_rates_list[:, :, self.n_vis_neurons:])).sum(dim=(-2, -1))
        return vis_part + hid_part
    
    def sample(self, n_time_bins: int, n_samples: int = 1) -> torch.FloatTensor:
        with torch.no_grad():
            spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_neurons))
            convolved_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            firing_rates_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))

            for t in range(n_time_bins):
                convolved_spikes_list[:, t, :] = self.flipped_basis @ spikes_list[:, t:t+self.window_size, :]
                firing_rates_list[:, t, :] = self.forward(convolved_spikes_list[:, t, :])
                spikes_list[:, t+self.window_size, :] = torch.poisson(firing_rates_list[:, t, :])
            spikes_list = spikes_list[:, self.window_size:, :]
            return spikes_list, convolved_spikes_list, firing_rates_list
    
    def sample_exact_distribution(self, n_time_bins: int, n_samples: int = 1) -> torch.FloatTensor:
        with torch.no_grad():
            vis_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_vis_neurons))
            if self.distribution.name == 'Gumbel-Softmax':
                hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons, self.distribution.n_categories))
                tmp = torch.arange(self.distribution.n_categories, dtype=torch.float32)
            else:
                hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons))
            convolved_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            firing_rates_list = torch.zeros((n_samples, n_time_bins, self.n_neurons))
            for t in range(n_time_bins):
                convolved_spikes_list[:, t, :self.n_vis_neurons] = self.flipped_basis @ vis_spikes_list[:, t:t+self.window_size, :]
                if self.distribution.name == 'Gumbel-Softmax':
                    convolved_spikes_list[:, t, self.n_vis_neurons:] = self.flipped_basis @ (hid_spikes_list[:, t:t+self.window_size, :, :] @ tmp)
                else:
                    convolved_spikes_list[:, t, self.n_vis_neurons:] = self.flipped_basis @ hid_spikes_list[:, t:t+self.window_size, :]
                firing_rates_list[:, t, :] = self.forward(convolved_spikes_list[:, t, :])
                vis_spikes_list[:, t+self.window_size, :] = torch.poisson(firing_rates_list[:, t, :self.n_vis_neurons])
                hid_spikes_list[:, t+self.window_size, :] = self.distribution.sample(self.distribution.mean_to_params(firing_rates_list[:, t, self.n_vis_neurons:]))
            vis_spikes_list = vis_spikes_list[:, self.window_size:, :]
            hid_spikes_list = hid_spikes_list[:, self.window_size:, :]
            return vis_spikes_list, hid_spikes_list, convolved_spikes_list, firing_rates_list

class VariationalDistribution(nn.Module):
    def __init__(
            self,
            n_neurons: int,
            n_vis_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson()
        ) -> None:
        super().__init__()
    
        self.n_neurons = n_neurons
        self.n_vis_neurons = n_vis_neurons
        self.n_hid_neurons = n_neurons - n_vis_neurons
        
        self.basis = basis
        self.flipped_basis = torch.flip(self.basis, (0,))
        self.window_size = len(self.basis)

        self.distribution = distribution
    
    @abstractmethod
    def init_params(self) -> None:
        pass

    @abstractmethod
    def permute_hidden_neurons(self) -> None:
        pass

    @abstractmethod
    def forward(self) -> torch.FloatTensor:
        pass
    
    @abstractmethod
    def sample_and_log_likelihood(
            self,
            convolved_vis_spikes: torch.FloatTensor,
            rev_convolved_vis_spikes: torch.FloatTensor = None,
            n_samples: int = 1,
            grad: str = 'score'
        ) -> tuple:
        pass


class Forward(VariationalDistribution):
    def __init__(
            self,
            n_neurons: int,
            n_vis_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson()
        ) -> None:
        super().__init__(n_neurons, n_vis_neurons, basis, distribution)

        self.linear = nn.Linear(n_vis_neurons, self.n_hid_neurons)
        # self.init_params()
    
    def init_params(self):
        self.linear.weight.data = torch.zeros((self.n_hid_neurons, self.n_vis_neurons))
        self.linear.bias.data = torch.zeros((self.n_hid_neurons,))
    
    def permute_hidden_neurons(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self.linear.bias.data = self.linear.bias.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons]
            self.linear.weight.data = self.linear.weight.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons, :]

    def forward(self, convolved_vis_spikes: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear(convolved_vis_spikes))

    def sample_and_log_likelihood(
            self,
            convolved_vis_spikes: torch.FloatTensor,
            rev_convolved_vis_spikes: torch.FloatTensor = None,
            n_samples: int = 1,
            grad: str = 'score'
        ) -> tuple:
        hid_firing_rates = self.forward(convolved_vis_spikes).expand((n_samples, -1, -1))
        hid_params_list = self.distribution.mean_to_params(hid_firing_rates)
        if grad == 'score':
            with torch.no_grad():
                hid_spikes_list = self.distribution.sample(hid_params_list)
        elif grad == 'pathwise':
            hid_spikes_list = self.distribution.sample(hid_params_list)
        convolved_hid_spikes_list = utils.convolve_spikes_with_basis(hid_spikes_list, self.basis, direction='forward')
        hid_log_likelihood_list = self.distribution.log_likelihood(hid_spikes_list, hid_params_list).sum(dim=(-2, -1))
        return hid_spikes_list, convolved_hid_spikes_list, hid_log_likelihood_list


class ForwardSelf(VariationalDistribution):
    def __init__(
            self,
            n_neurons: int,
            n_vis_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson()
        ) -> None:
        super().__init__(n_neurons, n_vis_neurons, basis, distribution)

        self.linear = nn.Linear(n_neurons, self.n_hid_neurons)
        # self.init_params()
    
    def init_params(self):
        self.linear.weight.data = torch.zeros((self.n_hid_neurons, self.n_neurons))
        self.linear.bias.data = torch.zeros((self.n_hid_neurons,))
    
    def permute_hidden_neurons(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self.linear.bias.data = self.linear.bias.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons]
            self.linear.weight.data = self.linear.weight.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons, :][:, true_to_learned]

    def forward(self, convolved_spikes: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear(convolved_spikes))

    def sample_and_log_likelihood(
            self, convolved_vis_spikes: torch.FloatTensor,
            rev_convolved_vis_spikes: torch.FloatTensor = None,
            n_samples: int = 1,
            grad: str = 'score'
        ) -> tuple:
        n_time_bins = convolved_vis_spikes.shape[0]
        if self.distribution.name == 'categorical':
            hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons))
            hid_params_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons, self.distribution.n_categories))
        elif self.distribution.name == 'Gumbel-Softmax':
            hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons, self.distribution.n_categories))
            hid_params_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons, self.distribution.n_categories))
            tmp = torch.arange(self.distribution.n_categories, dtype=torch.float32)
        else:
            hid_spikes_list = torch.zeros((n_samples, n_time_bins + self.window_size, self.n_hid_neurons))
            hid_params_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons))
        convolved_hid_spikes_list = torch.zeros((n_samples, n_time_bins, self.n_hid_neurons))
        
        for t in range(n_time_bins):
            if self.distribution.name == 'Gumbel-Softmax':
                convolved_hid_spikes_list[:, t, :] = self.flipped_basis @ (hid_spikes_list[:, t:t+self.window_size, :, :] @ tmp)
            else:
                convolved_hid_spikes_list[:, t, :] = self.flipped_basis @ hid_spikes_list[:, t:t+self.window_size, :]
            hid_params_list[:, t] = self.distribution.mean_to_params(self.forward(torch.cat((
                    convolved_vis_spikes[t, :].expand((n_samples, -1)),
                    convolved_hid_spikes_list[:, t, :]
            ), dim=-1)))
            if grad == 'score':
                with torch.no_grad():
                    hid_spikes_list[:, t+self.window_size] = self.distribution.sample(hid_params_list[:, t])
            else:
                hid_spikes_list[:, t+self.window_size] = self.distribution.sample(hid_params_list[:, t])
        hid_spikes_list = hid_spikes_list[:, self.window_size:, :]
        hid_log_likelihood_list = self.distribution.log_likelihood(hid_spikes_list, hid_params_list).sum(dim=(-2, -1))
        return hid_spikes_list, convolved_hid_spikes_list, hid_log_likelihood_list


class ForwardBackward(VariationalDistribution):
    def __init__(
            self,
            n_neurons: int,
            n_vis_neurons: int,
            basis: torch.FloatTensor,
            distribution: distributions.Distribution = distributions.Poisson()
        ) -> None:
        super().__init__(n_neurons, n_vis_neurons, basis, distribution)

        self.linear_forward = nn.Linear(n_vis_neurons, self.n_hid_neurons)
        self.linear_backward = nn.Linear(n_vis_neurons, self.n_hid_neurons, bias=False)
        # self.init_params()
    
    def init_params(self):
        self.linear_forward.weight.data = torch.zeros((self.n_hid_neurons, self.n_vis_neurons))
        self.linear_forward.bias.data = torch.zeros((self.n_hid_neurons, ))
        self.linear_backward.weight.data = torch.zeros((self.n_hid_neurons, self.n_vis_neurons))
    
    def permute_hidden_neurons(self, true_to_learned: torch.LongTensor) -> None:
        with torch.no_grad():
            self.linear_forward.bias.data = self.linear_forward.bias.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons]
            self.linear_forward.weight.data = self.linear_forward.weight.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons, :]
            self.linear_backward.weight.data = self.linear_backward.weight.data[true_to_learned[self.n_vis_neurons:] - self.n_vis_neurons, :]
    
    def forward(self, convolved_vis_spikes: torch.FloatTensor, rev_convolved_vis_spikes: torch.FloatTensor) -> torch.FloatTensor:
        return torch.sigmoid(self.linear_forward(convolved_vis_spikes) + self.linear_backward(rev_convolved_vis_spikes))
    
    def sample_and_log_likelihood(
            self, convolved_vis_spikes: torch.FloatTensor,
            rev_convolved_vis_spikes: torch.FloatTensor = None,
            n_samples: int = 1,
            grad: str = 'score'
        ) -> tuple:
        hid_firing_rates = self.forward(convolved_vis_spikes, rev_convolved_vis_spikes).expand((n_samples, -1, -1))
        hid_params_list = self.distribution.mean_to_params(hid_firing_rates)
        if grad == 'score':
            with torch.no_grad():
                hid_spikes_list = self.distribution.sample(hid_params_list)
        elif grad == 'pathwise':
            hid_spikes_list = self.distribution.sample(hid_params_list)
        convolved_hid_spikes_list = utils.convolve_spikes_with_basis(hid_spikes_list, self.basis, direction='forward')
        hid_log_likelihood_list = self.distribution.log_likelihood(hid_spikes_list, hid_params_list).sum(dim=(-2, -1))
        return hid_spikes_list, convolved_hid_spikes_list, hid_log_likelihood_list





# class ForwardBackwardLMM(DPP):
#     def __init__(self,
#             n_neurons: int,
#             n_vis_neurons: int,
#             dt: float,
#             basis: torch.FloatTensor,
#             distribution=distributions.GumbelSoftmax(n_categories=5, tau=0.5),
#             device=torch.device("cpu")) -> None:
        
#         super().__init__(n_neurons, dt, device)

#         self.n_vis_neurons = n_vis_neurons # number of visible neurons
#         self.n_hid_neurons = self.n_neurons - n_vis_neurons # number of hidden neurons
#         self.basis = basis
#         self.flipped_basis = torch.flip(self.basis, (0,))
#         self.window_size = len(self.basis)
#         self.distribution = distribution

#         self.forward_linear = nn.Linear(in_features=self.n_vis_neurons, out_features=self.n_hid_neurons*distribution.n_categories)
#         self.backward_linear = nn.Linear(in_features=self.n_vis_neurons, out_features=self.n_hid_neurons*distribution.n_categories)
    
#     def forward(self, convolved_vis_spikes: torch.FloatTensor, rev_convolved_vis_spikes: torch.FloatTensor) -> torch.FloatTensor:
#         n_time_bins = convolved_vis_spikes.shape[0]
#         logits = self.forward_linear(convolved_vis_spikes).reshape(n_time_bins, self.n_hid_neurons, self.distribution.n_categories) + self.backward_linear(rev_convolved_vis_spikes).reshape(n_time_bins, self.n_hid_neurons, self.distribution.n_categories)
#         p = F.softmax(logits, dim=-1)
#         return p

# class LinearMaxProcess(DPP):
#     def __init__(self, n_neurons: int, dt: float, basis: torch.FloatTensor, max_n_spikes: int = 5, device=torch.device("cpu")) -> None:
#         super().__init__(n_neurons, dt, device)

#         self.basis = basis
#         self.flipped_basis = torch.flip(self.basis, (0,))
#         self.window_size = len(self.basis)
#         self.max_n_spikes = max_n_spikes

#         self.linear = nn.Linear(in_features=n_neurons, out_features=n_neurons*max_n_spikes)
    
#     @property
#     def weight(self):
#         return self.linear.weight.reshape(self.n_neurons, self.max_n_spikes).permute(0, 2, 1)
    
#     @property
#     def bias(self):
#         return self.linear.bias.reshape(self.n_neurons, self.max_n_spikes)
    
#     def forward(self, convolved_spikes: torch.FloatTensor) -> torch.FloatTensor:
#         n_time_bins = convolved_spikes.shape[0]
#         logits = self.linear(convolved_spikes).reshape(n_time_bins, self.n_neurons, self.max_n_spikes)
#         probs = F.softmax(logits, dim=-1)
#         return probs
    
#     def log_likelihood(self, spikes: torch.FloatTensor, probs: torch.FloatTensor) -> torch.FloatTensor:
#         suppressed_spikes = spikes.clone()
#         suppressed_spikes[suppressed_spikes >= self.max_n_spikes] = self.max_n_spikes - 1
#         return torch.sum((probs+1e-16).log() * F.one_hot(suppressed_spikes.to(torch.int64), self.max_n_spikes), dim=(-3, -2, -1))
        
#     def firing_rates(self, probs):
#         return probs @ torch.arange(self.max_n_spikes)
    