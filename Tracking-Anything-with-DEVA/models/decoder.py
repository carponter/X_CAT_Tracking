import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import FlattenMlp
import models.pytorch_utils as ptu
import mbrl.models as models
import mbrl
import mbrl.util.math

def unreduced_loss(mlp: models.GaussianMLP, model_in, target):
    assert not mlp.deterministic
    assert model_in.ndim == target.ndim
    if model_in.ndim == 2:  # 添加ensemble维度
        model_in = model_in.unsqueeze(0)
        target = target.unsqueeze(0)
    pred_mean, pred_logvar = mlp.forward(model_in, use_propagation=False)
    if target.shape[0] != mlp.num_members:
        target = target.repeat(mlp.num_members, 1, 1)
    nll = mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False).squeeze(0).mean(dim=-1)
    return nll

class FOCALDecoder(nn.Module):
    def __init__(self, 
                 obs_size,
                 action_size,
                 task_embedding_size,
                 task_embedd_is_deterministic,
                 device, 
                 num_layers, 
                 ensemble_size, 
                 hidden_size, 
                 ) -> None:
        super(FOCALDecoder, self).__init__()
        input_size = obs_size + action_size + task_embedding_size
        if task_embedd_is_deterministic:
            input_size += task_embedding_size
            
        output_dynamic_size = obs_size
        output_reward_size = 1
        self.ensemble_size = ensemble_size
        self.device = device
        
        self.dynamic_decoder = models.GaussianMLP(
            in_size=input_size,
            out_size=output_dynamic_size,
            device=device,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hidden_size,
            learn_logvar_bounds=True,
            deterministic=False
        ).requires_grad_(True)        
        self.reward_decoder = models.GaussianMLP(
            in_size=input_size,
            out_size=output_reward_size,
            device=device,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hidden_size,
            learn_logvar_bounds=True,
            deterministic=False
        ).requires_grad_(True)

    def forward(self, task_embedding, state, action):
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
    
        mean_state, logvar_state = self.dynamic_decoder(input_tensor)
        mean_reward, logvar_reward = self.reward_decoder(input_tensor)

        mean_state = mean_state.squeeze(0)
        logvar_state = logvar_state.squeeze(0)
        mean_reward = mean_reward.squeeze(0)
        logvar_reward = logvar_reward.squeeze(0)
        
        return mean_state, logvar_state, mean_reward, logvar_reward

    def loss(self, task_embedding, state, action, target_reward, target_state):
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
        
        state_target = (target_state - state).repeat(self.ensemble_size, 1, 1)
        reward_target = target_reward.repeat(self.ensemble_size, 1, 1)

        state_loss, _ = self.dynamic_decoder.loss(input_tensor, state_target)
        reward_loss, _ = self.reward_decoder.loss(input_tensor, reward_target)
        
        return state_loss + reward_loss

    def unreduced_loss(self, task_embedding, state, action, target_reward, target_state):
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
        
        state_target = target_state - state
        
        state_loss = unreduced_loss(self.dynamic_decoder, input_tensor, state_target.repeat(self.ensemble_size, 1, 1))
        reward_loss = unreduced_loss(self.reward_decoder, input_tensor, target_reward.repeat(self.ensemble_size, 1, 1))
        
        return state_loss + reward_loss
