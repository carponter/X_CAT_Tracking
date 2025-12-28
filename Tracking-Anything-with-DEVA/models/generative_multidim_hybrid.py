import torch
import torch.nn as nn
import numpy as np
from models.networks import FlattenMlp
from models.decoder import FOCALDecoder
import models.pytorch_utils as ptu


def reparameterize(mean, logvar):
    """Reparameterization trick for VAE"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


class VelocityEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=4):
        super().__init__()
        self.mlp = FlattenMlp(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=[hidden_size],
        )
    
    def forward(self, x):

        return self.mlp(x)

class Decoder(nn.Module):
    """Decoder for CVAE - unchanged from original"""
    def __init__(self, z_dim: int, state_size: int, action_size: int, reward_size: int,
                 hidden_size: int, num_hidden_layers: int, output_variance: str, 
                 predict_state_difference: bool, merge_reward_next_state: bool, 
                 logvar_min: float, logvar_max: float):
        super(Decoder, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.reward_size = reward_size
        self.z_dim = z_dim

        assert output_variance in ['zero', 'parameter', 'output', 'output_raw', 'reference']
        self.variance_mode = output_variance
        if output_variance == 'parameter':
            self.logvar = nn.Parameter(torch.zeros(1, state_size + reward_size))
        if output_variance not in ['output', 'output_raw']:
            self.decoder = FlattenMlp(input_size=z_dim + state_size + action_size,
                                      output_size=state_size + reward_size,
                                      hidden_sizes=[hidden_size for i in range(num_hidden_layers)])
        elif output_variance == 'output_raw':
            self.decoder = nn.ModuleList([
                FlattenMlp(input_size=z_dim + state_size + action_size,
                           output_size=state_size * 2,
                           hidden_sizes=[hidden_size] * num_hidden_layers),
                FlattenMlp(input_size=z_dim + state_size + action_size,
                           output_size=reward_size * 2,
                           hidden_sizes=[hidden_size] * num_hidden_layers)
            ])
        # only use output -> focal-decoder
        else:
            self.decoder = FOCALDecoder(
                state_size, action_size, z_dim, False,
                ptu.device if ptu.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                num_hidden_layers, 1, hidden_size
            )
        if output_variance == 'reference':
            self.reference_decoder = FlattenMlp(
                input_size=state_size + action_size,
                output_size=(state_size + reward_size) * 2,
                hidden_sizes=[hidden_size] * num_hidden_layers,
            )
        self.predict_state_difference = predict_state_difference
        self.merge_reward_next_state = merge_reward_next_state
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward_decoder(self, obs, action, z=None):
        if z is None:
            z = torch.randn(obs.shape[0], self.z_dim, device=obs.device)
        if self.variance_mode in ['output', 'output_raw']:
            if self.variance_mode == 'output':
                mean_s_, logvar_s_, mean_r, logvar_r = self.decoder(z, obs, action) ## this way
            else:
                mean_s_, logvar_s_ = self.decoder[0](torch.cat([z, obs, action], dim=-1)).chunk(2, dim=-1)
                mean_r, logvar_r = self.decoder[1](torch.cat([z, obs, action], dim=-1)).chunk(2, dim=-1)
            if self.predict_state_difference:
                mean_s_ = mean_s_ + obs
            return mean_s_, logvar_s_, mean_r, logvar_r
        out = self.decoder(torch.cat([obs, action, z], dim=-1))
        s_, r = out[:, 0:self.state_size], out[:, self.state_size:]
        assert s_.shape == obs.shape
        if self.predict_state_difference:
            s_ = obs + s_
        return s_, r

    def _losses(self, obs, action, reward, next_obs, z):
        ref_loss_s_ = ref_loss_r = None
        if self.variance_mode in ['output', 'output_raw']:
            mean_s_, logvar_s_, mean_r, logvar_r = self.forward_decoder(obs, action, z)
            logvar_s_ = torch.clamp(logvar_s_, self.logvar_min, self.logvar_max)
            logvar_r = torch.clamp(logvar_r, self.logvar_min, self.logvar_max)
        else:
            mean_s_, mean_r = self.forward_decoder(obs, action, z)
            if self.variance_mode == 'zero':
                logvar_s_ = torch.zeros_like(mean_s_, device=mean_s_.device)
                logvar_r = torch.zeros_like(mean_r, device=mean_r.device)
            elif self.variance_mode == 'parameter':
                logvar_s_ = self.logvar[:, :self.state_size].expand_as(mean_s_)
                logvar_r = self.logvar[:, self.state_size:].expand_as(mean_r)
                logvar_s_ = torch.clamp(logvar_s_, self.logvar_min, self.logvar_max)
                logvar_r = torch.clamp(logvar_r, self.logvar_min, self.logvar_max)
            elif self.variance_mode == 'reference':
                # Use an unconditional reference model to produce the variance across environments
                ref_mean, logvar = self.reference_decoder(torch.cat([obs, action], dim=-1)).chunk(2, dim=-1)
                ref_mean_s_, ref_mean_r = ref_mean[:, :self.state_size], ref_mean[:, self.state_size:]
                if self.predict_state_difference:
                    ref_mean_s_ = ref_mean_s_ + obs
                logvar_s_, logvar_r = logvar[:, :self.state_size], logvar[:, self.state_size:]
                logvar_s_ = torch.clamp(logvar_s_, self.logvar_min, self.logvar_max)
                logvar_r = torch.clamp(logvar_r, self.logvar_min, self.logvar_max)
                if self.merge_reward_next_state:
                    ref_loss_s_ = (torch.cat([ref_mean_s_, ref_mean_r], dim=-1)
                                   - torch.cat([next_obs, reward], dim=-1)).pow(2) / logvar.exp() + logvar
                    ref_loss_r = torch.zeros_like(ref_mean_r)
                else:
                    ref_loss_s_ = (ref_mean_s_ - next_obs).pow(2) / logvar_s_.exp() + logvar_s_
                    ref_loss_r = (ref_mean_r - reward).pow(2) / logvar_r.exp() + logvar_r
                # Reference variances are not trained by VAE losses
                logvar_s_ = logvar_s_.detach()
                logvar_r = logvar_r.detach()
            else:
                raise ValueError(f"Invalid variance mode: {self.variance_mode}")
            
        if self.merge_reward_next_state:
            unscaled_loss_s_ = (torch.cat([mean_s_, mean_r], dim=-1) - torch.cat([next_obs, reward], dim=-1)).pow(2)
            unscaled_loss_r = torch.zeros_like(mean_r)
            logvar = torch.cat([logvar_s_, logvar_r], dim=-1)
            loss_s_ = unscaled_loss_s_ / logvar.exp() + logvar
            loss_r = torch.zeros_like(mean_r)
        else:
            unscaled_loss_s_ = (mean_s_ - next_obs).pow(2)
            unscaled_loss_r = (mean_r - reward).pow(2)
            loss_s_ = unscaled_loss_s_ / logvar_s_.exp() + logvar_s_
            loss_r = unscaled_loss_r / logvar_r.exp() + logvar_r

        return loss_s_, loss_r, unscaled_loss_s_, unscaled_loss_r, ref_loss_s_, ref_loss_r

    def losses(self, obs, action, reward, next_obs, z):
        """Returns generator that applies .mean() to each loss (matches generative_multidim.py)"""
        return (loss.mean() if loss is not None else None for loss in self._losses(obs, action, reward, next_obs, z))


class CVAE_4D_Hybrid(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_hidden_layers: int,
                 z_dim: int,
                 action_size: int,
                 state_size: int,
                 reward_size: int = 1,
                 num_trackers: int = 3,
                 num_targets: int = 4,
                 linear_v_range: tuple = (100.0, 400.0),  # (velo_range)
                 angular_v_range: tuple = (15.0, 90.0),   # (angle_range)
                 mode: str = 'cat_all',  # 'full_encoding', 'cat_all', 'avgpool_fusion'
                 predict_state_difference: bool = False,
                 merge_reward_next_state: bool = False,
                 output_variance: str = 'output',
                 logvar_min: float = -15.0,
                 logvar_max: float = 2.0):
        super(CVAE_4D_Hybrid, self).__init__()
        
        self.mode = mode
        self.z_dim = z_dim
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.num_trackers = num_trackers
        self.num_targets = num_targets

        self.linear_v_range = linear_v_range  # (100, 400)
        self.angular_v_range = angular_v_range  # (15, 90)
        self.num_linear_v = 4   # (100, 200, 300, 400)
        self.num_angular_v = 4  # (15, 30, 60, 90)
        
        if mode == 'full_encoding':
            num_tasks = num_trackers * num_targets * 4 * 4  # 假设4个速度档位
            self.task_encoder = nn.Parameter(torch.randn(num_tasks, z_dim * 2))  # mean + logvar            
        elif mode == 'cat_all':
            # z % 4 == 0
            self.component_dim = z_dim // 4
            self.tracker_mean = nn.Parameter(torch.randn(num_trackers, self.component_dim))
            self.tracker_logvar = nn.Parameter(torch.zeros(num_trackers, self.component_dim))            
            self.target_mean = nn.Parameter(torch.randn(num_targets, self.component_dim))
            self.target_logvar = nn.Parameter(torch.zeros(num_targets, self.component_dim))
            
            self.linear_v_mean_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.component_dim)
            self.linear_v_logvar_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.component_dim)        
            self.angular_v_mean_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.component_dim)
            self.angular_v_logvar_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.component_dim)
            # mlp_params = sum(p.numel() for p in self.linear_v_mean_encoder.parameters()) * 2 + \
            #             sum(p.numel() for p in self.angular_v_mean_encoder.parameters()) * 2
            # table_params = (num_trackers + num_targets) * self.component_dim * 2           
        elif mode == 'avgpool_fusion':
            # z % 4 == 0
            self.half_dim = z_dim // 2
            self.quarter_dim = z_dim // 4

            self.linear_v_mean_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.quarter_dim)
            self.linear_v_logvar_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.quarter_dim)           
            self.angular_v_mean_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.quarter_dim)
            self.angular_v_logvar_encoder = VelocityEncoder(input_size=1, hidden_size=64, output_size=self.quarter_dim)
            
            self.tracker_mean = nn.Parameter(torch.randn(num_trackers, self.half_dim))
            self.tracker_logvar = nn.Parameter(torch.zeros(num_trackers, self.half_dim))           
            self.target_mean = nn.Parameter(torch.randn(num_targets, self.half_dim))
            self.target_logvar = nn.Parameter(torch.zeros(num_targets, self.half_dim))
            # mlp_params = sum(p.numel() for p in self.linear_v_mean_encoder.parameters()) * 2 + \
            #             sum(p.numel() for p in self.angular_v_mean_encoder.parameters()) * 2
            # table_params = (num_trackers + num_targets) * self.half_dim * 2
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'full_encoding', 'cat_all', or 'avgpool_fusion'")
        
        self.decoder = Decoder(
            z_dim=z_dim,
            state_size=state_size,
            action_size=action_size,
            reward_size=reward_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            output_variance=output_variance,
            predict_state_difference=predict_state_difference,
            merge_reward_next_state=merge_reward_next_state,
            logvar_min=logvar_min,
            logvar_max=logvar_max
        )
    
    def compute_task_id(self, tracker_id, target_id, linear_v_id, angular_v_id):
        # full_encoding use
        T, V, A = self.num_targets, self.num_linear_v, self.num_angular_v
        task_id = tracker_id * (T * V * A) + target_id * (V * A) + linear_v_id * A + angular_v_id
        return task_id
    
    def compute_kl_divergence(self, mean, logvar):
        return (- 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1))
    
    def compute_speed_distance_loss(self, mean, linear_v_max, angular_v_max):
        if self.mode == 'full_encoding':
            return torch.tensor(0.0, device=mean.device), torch.tensor(0.0, device=mean.device)
        component_dim = self.component_dim
        linear_v_z = mean[:, component_dim*2:component_dim*3]   # [batch, component_dim]
        angular_v_z = mean[:, component_dim*3:component_dim*4]  # [batch, component_dim]
        # init max
        linear_v_levels = torch.tensor([100.0, 200.0, 300.0, 400.0], device=mean.device)
        angular_v_levels = torch.tensor([15.0, 30.0, 60.0, 90.0], device=mean.device)
        linear_v_idx = torch.argmin(torch.abs(linear_v_max.unsqueeze(1) - linear_v_levels), dim=1)
        angular_v_idx = torch.argmin(torch.abs(angular_v_max.unsqueeze(1) - angular_v_levels), dim=1)
        
        linear_loss = 0.0
        linear_count = 0
        for i in range(len(linear_v_levels) - 1):
            mask_i = (linear_v_idx == i)
            mask_j = (linear_v_idx == i + 1)            
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                z_i = linear_v_z[mask_i].mean(0)  # [component_dim]
                z_j = linear_v_z[mask_j].mean(0)  # [component_dim]
                dist = torch.norm(z_j - z_i, p=2)
                expected_dist = (linear_v_levels[i+1] - linear_v_levels[i]) / 100.0  # 归一化
                linear_loss += (dist - expected_dist).pow(2)
                linear_count += 1        
        if linear_count > 0:
            linear_loss = linear_loss / linear_count
        else:
            linear_loss = torch.tensor(0.0, device=mean.device)
        
        angular_loss = 0.0
        angular_count = 0
        for i in range(len(angular_v_levels) - 1):
            mask_i = (angular_v_idx == i)
            mask_j = (angular_v_idx == i + 1)            
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                z_i = angular_v_z[mask_i].mean(0)
                z_j = angular_v_z[mask_j].mean(0)
                dist = torch.norm(z_j - z_i, p=2)
                expected_dist = (angular_v_levels[i+1] - angular_v_levels[i]) / 30.0
                angular_loss += (dist - expected_dist).pow(2)
                angular_count += 1        
        if angular_count > 0:
            angular_loss = angular_loss / angular_count
        else:
            angular_loss = torch.tensor(0.0, device=mean.device)
        
        return linear_loss, angular_loss
    
    def compute_total_loss(self, obs, action, reward, next_obs, tracker_id, target_id, linear_v_max, angular_v_max, beta=0.1, lambda_speed=0.1):
        # L = L_recon + β * L_KL + λ * (L_speed + L_ang)
        mean, logvar, z_sample = self.forward_encoder(
            obs, action, reward, next_obs,
            tracker_id, target_id,
            linear_v_max=linear_v_max,
            angular_v_max=angular_v_max
        )
        
        recon_loss_s, recon_loss_r, *_ = self.decoder.losses(obs, action, reward, next_obs, z_sample)
        recon_loss = recon_loss_s + recon_loss_r
        
        kl_loss = self.compute_kl_divergence(mean, logvar).mean()        
        linear_loss, angular_loss = self.compute_speed_distance_loss(mean, linear_v_max, angular_v_max)
        speed_loss = linear_loss + angular_loss
        
        total_loss = recon_loss + beta * kl_loss + lambda_speed * speed_loss        
        loss_dict = {
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'kl_loss': kl_loss.item(),
            'speed_loss': speed_loss.item(),
            'linear_loss': linear_loss.item(),
            'angular_loss': angular_loss.item(),
            'total_loss': total_loss.item()
        }        
        return total_loss, loss_dict
    
    def forward_encoder(self, obs, action, reward, next_obs, tracker_id, target_id, linear_v_id=None, angular_v_id=None, linear_v_max=None, angular_v_max=None):
        if isinstance(tracker_id, torch.Tensor):
            batch_size = tracker_id.shape[0]
            device = tracker_id.device
        else:
            batch_size = 1
            # device = next(self.tracker_embeddings.parameters()).device
            device = self.tracker_mean.device
            tracker_id = torch.tensor([tracker_id], device=device)
            target_id = torch.tensor([target_id], device=device)
        
        # ============= Mode 1: Full Encoding (192 tasks，使用离散ID) =============
        if self.mode == 'full_encoding':            
            if not isinstance(linear_v_id, torch.Tensor):
                linear_v_id = torch.tensor([linear_v_id], device=device)
                angular_v_id = torch.tensor([angular_v_id], device=device)            
            task_id = self.compute_task_id(tracker_id, target_id, linear_v_id, angular_v_id)
            z_full = self.task_encoder[task_id]  # [batch, z_dim*2]           
            if z_full.dim() == 1:
                z_full = z_full.unsqueeze(0)
            if z_full.size(0) == 1 and batch_size > 1:
                z_full = z_full.expand(batch_size, -1)
            
            mean, logvar = torch.split(z_full, self.z_dim, dim=-1)
            z = reparameterize(mean, logvar)
            return mean, logvar, z
        
        # ============= Mode 2: Cat All (分层logvar + MLP无归一化编码) =============
        elif self.mode == 'cat_all':
            if not isinstance(linear_v_max, torch.Tensor):
                linear_v_max = torch.tensor([linear_v_max], device=device, dtype=torch.float)
            if not isinstance(angular_v_max, torch.Tensor):
                angular_v_max = torch.tensor([angular_v_max], device=device, dtype=torch.float)
            
            if linear_v_max.dim() == 0:
                linear_v_max = linear_v_max.unsqueeze(0).unsqueeze(0)
            elif linear_v_max.dim() == 1:
                linear_v_max = linear_v_max.unsqueeze(-1)           
            if angular_v_max.dim() == 0:
                angular_v_max = angular_v_max.unsqueeze(0).unsqueeze(0)
            elif angular_v_max.dim() == 1:
                angular_v_max = angular_v_max.unsqueeze(-1)
            
            tracker_m = self.tracker_mean[tracker_id]      # [batch, component_dim]
            tracker_lv = self.tracker_logvar[tracker_id]
            target_m = self.target_mean[target_id]
            target_lv = self.target_logvar[target_id]
            
            linear_v_m = self.linear_v_mean_encoder(linear_v_max)        # [batch, component_dim]
            linear_v_lv = self.linear_v_logvar_encoder(linear_v_max)
            angular_v_m = self.angular_v_mean_encoder(angular_v_max)
            angular_v_lv = self.angular_v_logvar_encoder(angular_v_max)
            
            if tracker_m.dim() == 1:
                tracker_m = tracker_m.unsqueeze(0)
                tracker_lv = tracker_lv.unsqueeze(0)
            if target_m.dim() == 1:
                target_m = target_m.unsqueeze(0)
                target_lv = target_lv.unsqueeze(0)
            
            mean = torch.cat([tracker_m, target_m, linear_v_m, angular_v_m], dim=-1)
            logvar = torch.cat([tracker_lv, target_lv, linear_v_lv, angular_v_lv], dim=-1)
            z = reparameterize(mean, logvar)
            return mean, logvar, z
        
        # ============= Mode 3: Avgpool Fusion (MLP无归一化编码) =============
        elif self.mode == 'avgpool_fusion':
            if not isinstance(linear_v_max, torch.Tensor):
                linear_v_max = torch.tensor([linear_v_max], device=device, dtype=torch.float)
            if not isinstance(angular_v_max, torch.Tensor):
                angular_v_max = torch.tensor([angular_v_max], device=device, dtype=torch.float)            
            # 确保维度正确 [batch, 1]
            if linear_v_max.dim() == 0:
                linear_v_max = linear_v_max.unsqueeze(0).unsqueeze(0)
            elif linear_v_max.dim() == 1:
                linear_v_max = linear_v_max.unsqueeze(-1)            
            if angular_v_max.dim() == 0:
                angular_v_max = angular_v_max.unsqueeze(0).unsqueeze(0)
            elif angular_v_max.dim() == 1:
                angular_v_max = angular_v_max.unsqueeze(-1)
            
            linear_v_m = self.linear_v_mean_encoder(linear_v_max)        # [batch, quarter_dim]
            linear_v_lv = self.linear_v_logvar_encoder(linear_v_max)            
            angular_v_m = self.angular_v_mean_encoder(angular_v_max)
            angular_v_lv = self.angular_v_logvar_encoder(angular_v_max)
            action_space_m = torch.cat([linear_v_m, angular_v_m], dim=-1)  # [batch, half_dim]
            action_space_lv = torch.cat([linear_v_lv, angular_v_lv], dim=-1)
            tracker_m = self.tracker_mean[tracker_id]          # [batch, half_dim]
            tracker_lv = self.tracker_logvar[tracker_id]
            if tracker_m.dim() == 1:
                tracker_m = tracker_m.unsqueeze(0)
                tracker_lv = tracker_lv.unsqueeze(0)
            # ???? why /2.0 and /4.0 ?
            tracker_policy_m = (action_space_m + tracker_m) / 2.0
            tracker_policy_lv = torch.log((action_space_lv.exp() + tracker_lv.exp()) / 4.0)
            
            target_m = self.target_mean[target_id]             # [batch, half_dim]
            target_lv = self.target_logvar[target_id]
            if target_m.dim() == 1:
                target_m = target_m.unsqueeze(0)
                target_lv = target_lv.unsqueeze(0)
            mean = torch.cat([tracker_policy_m, target_m], dim=-1)
            logvar = torch.cat([tracker_policy_lv, target_lv], dim=-1)
            z = reparameterize(mean, logvar)
            return mean, logvar, z
    
    def forward_decoder(self, obs, action, z=None):
        return self.decoder.forward_decoder(obs, action, z)
    
    def losses(self, obs, action, reward, next_obs, z):
        return self.decoder.losses(obs, action, reward, next_obs, z)
    
    def forward(self, obs, action, reward, next_obs, tracker_id, target_id, linear_v_id, angular_v_id):
        mean, logvar, z = self.forward_encoder(obs, action, reward, next_obs, tracker_id, target_id, linear_v_id, angular_v_id)    
        if self.decoder.variance_mode in ['output', 'output_raw']:
            next_obs_pred, logvar_s_, reward_pred, logvar_r = self.forward_decoder(obs, action, z)
            return mean, logvar, z, next_obs_pred, reward_pred
        else:
            next_obs_pred, reward_pred = self.forward_decoder(obs, action, z)
            return mean, logvar, z, next_obs_pred, reward_pred
    