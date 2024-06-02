import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from poglm import utils, distributions, model


def elbo(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor, grad: str) -> torch.FloatTensor:
    """ELBO.
    """
    
    if grad == 'score':
        ln_p_list_values = ln_p_list.detach()
        ln_q_list_values = ln_q_list.detach()
        elbo_values = ln_p_list_values - ln_q_list_values
        return torch.mean(ln_p_list - ln_p_list_values + elbo_values * (ln_q_list - ln_q_list_values) + elbo_values)
    elif grad == 'pathwise':
        return (ln_p_list - ln_q_list).mean()
    else:
        raise ValueError('grad parameter is not in [score | pathwise].')


def marginal_log_likelihood(ln_p_list: torch.FloatTensor, ln_q_list: torch.FloatTensor) -> torch.FloatTensor:
    return torch.logsumexp(ln_p_list - ln_q_list, dim=0) - np.log(ln_q_list.shape[0])


def variational_inference(
        inf_model: model.POGLM,
        vari_model: model.VariationalDistribution,
        inf_optimizer: torch.optim.Optimizer,
        vari_optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        n_epochs: int = 1000,
        n_monte_carlo: int = 1000,
        grad: str = 'score',
        print_freq: int = 100,
        gen_model_state_dict: dict = None) -> np.array:
        
    epoch_loss_list = np.zeros(n_epochs)
    print('epoch', 'loss', 'weight error', 'bias error')
    
    for epoch in range(n_epochs):
        for vis_spikes_list, convolved_vis_spikes_list, rev_convolved_vis_spikes_list in dataloader:
            batch_size = vis_spikes_list.shape[0]
            loss = 0

            for sample in range(batch_size):
                vis_spikes = vis_spikes_list[sample]
                convolved_vis_spikes = convolved_vis_spikes_list[sample]
                rev_convolved_vis_spikes = rev_convolved_vis_spikes_list[sample]

                hid_spikes_list, convolved_hid_spikes_list, ln_q_list = vari_model.sample_and_log_likelihood(convolved_vis_spikes, rev_convolved_vis_spikes=rev_convolved_vis_spikes, n_samples=n_monte_carlo, grad=grad)
                ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)
                loss -= elbo(ln_p_list, ln_q_list, grad)

            loss /= batch_size
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()
            loss.backward()
            inf_optimizer.step()
            vari_optimizer.step()
            
            epoch_loss_list[epoch] += loss.item()
            
        epoch_loss_list[epoch] /= len(dataloader)
        if epoch % print_freq == 0:
            weight_error, bias_error = None, None
            if gen_model_state_dict is not None:
                weight_error = (gen_model_state_dict['linear.weight'] - inf_model.linear.weight.detach()).abs().mean().item()
                bias_error = (gen_model_state_dict['linear.bias'] - inf_model.linear.bias.detach()).abs().mean().item()
            print(epoch, epoch_loss_list[epoch], weight_error, bias_error, flush=True)
    return epoch_loss_list


def vis(
        inf_model: model.POGLM,
        vari_model: model.VariationalDistribution,
        inf_optimizer: torch.optim.Optimizer,
        vari_optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        n_epochs: int = 1000,
        n_monte_carlo: int = 1000,
        print_freq: int = 100,
        gen_model_state_dict: dict = None) -> torch.FloatTensor:
        
    epoch_loss_list = torch.zeros(n_epochs)
    print('epoch', 'loss', 'weight error', 'bias error')
    
    for epoch in range(n_epochs):
        for vis_spikes_list, convolved_vis_spikes_list, rev_convolved_vis_spikes_list in dataloader:
            batch_size = vis_spikes_list.shape[0]
            loss = 0

            for sample in range(batch_size):
                vis_spikes = vis_spikes_list[sample]
                convolved_vis_spikes = convolved_vis_spikes_list[sample]
                rev_convolved_vis_spikes = rev_convolved_vis_spikes_list[sample]

                hid_spikes_list, convolved_hid_spikes_list, ln_q_list = vari_model.sample_and_log_likelihood(convolved_vis_spikes, rev_convolved_vis_spikes=rev_convolved_vis_spikes, n_samples=n_monte_carlo, grad='score')
                ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list, convolved_hid_spikes_list, vis_spikes, convolved_vis_spikes)
                ln_V = torch.logsumexp(2*(ln_p_list.detach() - ln_q_list), dim=0) - np.log(ln_q_list.shape[0])
                loss += -marginal_log_likelihood(ln_p_list, ln_q_list.detach()) + 0.5 * (ln_V - ln_V.detach())

            loss /= batch_size
            inf_optimizer.zero_grad()
            vari_optimizer.zero_grad()
            loss.backward()
            inf_optimizer.step()
            vari_optimizer.step()
            
            epoch_loss_list[epoch] += loss.item()
            
        epoch_loss_list[epoch] /= len(dataloader)
        if epoch % print_freq == 0:
            weight_error, bias_error = None, None
            if gen_model_state_dict is not None:
                weight_error = (gen_model_state_dict['linear.weight'] - inf_model.linear.weight.detach()).abs().mean().item()
                bias_error = (gen_model_state_dict['linear.bias'] - inf_model.linear.bias.detach()).abs().mean().item()
            print(epoch, epoch_loss_list[epoch].item(), weight_error, bias_error, flush=True)
    return epoch_loss_list


def evaluate(
        inf_model: model.POGLM,
        vari_model: model.VariationalDistribution,
        vis_spikes_list: torch.FloatTensor,
        convolved_vis_spikes_list: torch.FloatTensor,
        rev_convolved_vis_spikes_list: torch.FloatTensor,
        hid_spikes_list: torch.FloatTensor = None,
        convolved_hid_spikes_list: torch.FloatTensor = None,
        n_monte_carlo: int = 1000,
        seed: int = 0
    ) -> pd.DataFrame:
    n_samples = vis_spikes_list.shape[0]
    
    if hid_spikes_list is None:
        df = pd.DataFrame(index=np.arange(n_samples), columns=['marginal log-likelihood', 'ELBO'])
    else:
        df = pd.DataFrame(index=np.arange(n_samples), columns=['pred complete log-likelihood', 'hid log-likelihood', 'marginal log-likelihood', 'ELBO'])
    
    torch.manual_seed(seed)

    with torch.no_grad():
        for sample in range(n_samples):
            vis_spikes = vis_spikes_list[sample]
            convolved_vis_spikes = convolved_vis_spikes_list[sample]
            rev_convolved_vis_spikes = rev_convolved_vis_spikes_list[sample]

            if hid_spikes_list is not None:
                hid_spikes = hid_spikes_list[sample]
                convolved_hid_spikes = convolved_hid_spikes_list[sample]
                df.at[sample, 'pred complete log-likelihood'] = inf_model.complete_log_likelihood(hid_spikes[None, ...], convolved_hid_spikes[None, ...], vis_spikes, convolved_vis_spikes)[0].item()
                if isinstance(vari_model, model.Forward):
                    hid_params = vari_model.forward(convolved_vis_spikes)
                elif isinstance(vari_model, model.ForwardSelf):
                    hid_params = vari_model.forward(torch.cat((convolved_vis_spikes, convolved_hid_spikes), dim=-1))
                elif isinstance(vari_model, model.ForwardBackward):
                    hid_params = vari_model.forward(convolved_vis_spikes, rev_convolved_vis_spikes)
                df.at[sample, 'hid log-likelihood'] = distributions.poisson_log_likelihood(hid_spikes, hid_params).sum(dim=(-2, -1)).item()
            
            hid_spikes_list_sampled, convolved_hid_spikes_list_sampled, ln_q_list = vari_model.sample_and_log_likelihood(convolved_vis_spikes, rev_convolved_vis_spikes, n_samples=n_monte_carlo)
            ln_p_list = inf_model.complete_log_likelihood(hid_spikes_list_sampled, convolved_hid_spikes_list_sampled, vis_spikes, convolved_vis_spikes)
            df.at[sample, 'marginal log-likelihood'] = marginal_log_likelihood(ln_p_list, ln_q_list).item()
            df.at[sample, 'ELBO'] = (ln_p_list - ln_q_list).mean().item()
    return df
