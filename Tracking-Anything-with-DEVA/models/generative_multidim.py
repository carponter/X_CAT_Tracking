import torch
import torch.nn as nn
from models.networks import FlattenMlp
from models.decoder import FOCALDecoder
import models.pytorch_utils as ptu

# 重参数化技巧：从均值和对数方差中采样隐变量
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


class Decoder(nn.Module):
    def __init__(self, z_dim: int, state_size: int, action_size: int, reward_size: int,
                 hidden_size: int, num_hidden_layers: int, output_variance: str, predict_state_difference: bool,
                 merge_reward_next_state: bool, logvar_min: float, logvar_max: float):
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
        else:
            # Use FOCALDecoder with output variance support.
            self.decoder = FOCALDecoder(
                state_size, action_size, z_dim, False, 
                ptu.device if ptu.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                num_hidden_layers, 1, hidden_size
            )
        if output_variance == 'reference':
            # Model the mixture distribution to serve as a reference for the actual variance
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
        # sampling
        if z is None:
            z = torch.randn(obs.shape[0], self.z_dim, device=obs.device)
        if self.variance_mode in ['output', 'output_raw']:
            if self.variance_mode == 'output':
                mean_s_, logvar_s_, mean_r, logvar_r = self.decoder(z, obs, action)
            else:
                mean_s_, logvar_s_ = self.decoder[0](z, obs, action).chunk(2, dim=-1)
                mean_r, logvar_r = self.decoder[1](z, obs, action).chunk(2, dim=-1)
            if self.predict_state_difference:
                mean_s_ = mean_s_ + obs
            # return mean_s_, mean_r, logvar_s_, logvar_r
            return mean_s_, logvar_s_, mean_r, logvar_r
        out = self.decoder(obs, action, z)
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
                ref_mean, logvar = self.reference_decoder(obs, action).chunk(2, dim=-1)
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
        return (loss.mean() if loss is not None else None for loss in self._losses(obs, action, reward, next_obs, z))

    def unreduced_loss(self, z, obs, action, reward, next_obs):
        loss_s_, loss_r, *_ = self._losses(obs, action, reward, next_obs, z)
        return loss_s_.mean(dim=-1) + loss_r.mean(dim=-1)

    def loss(self, z, obs, action, reward, next_obs):
        loss_s_, loss_r, _, _, ref_s_, ref_r = self.losses(obs, action, reward, next_obs, z)
        total_loss = loss_s_ + loss_r
        if self.variance_mode == 'reference':
            total_loss += ref_s_ + ref_r
        return total_loss


class CVAE_MultiDim(Decoder):
    """
    四维度分解的CVAE模型（只训练target=human的数据）
    将任务分解为：tracker、target、linear_v、angular_v四个维度
    
    两种融合策略可选：
    1. tracker_fusion='cat': (v,a)→cat→action_space, (action_space,tracker)→cat→tracker_policy, (tracker_policy,target)→cat
    2. tracker_fusion='avgpooling': (v,a)→cat→action_space, (action_space,tracker)→avgpool→tracker_policy, (tracker_policy,target)→cat
    """
    def __init__(self, hidden_size=64, num_hidden_layers=1, z_dim=64,
                 action_size=5, state_size=2, reward_size=1,
                 num_trackers=None, num_targets=None,
                 num_linear_velocities=None, num_angular_velocities=None,
                 tracker_fusion='cat',  # 'cat' or 'avgpooling' - how to fuse action_space with tracker
                 predict_state_difference=False,
                 output_variance='output', merge_reward_next_state=False,
                 logvar_min=-10.0, logvar_max=2.0):
        """
        Args:
            z_dim: 最终融合后的embedding维度（必须能被3或4整除，取决于tracker_fusion）
            tracker_fusion: tracker和action_space的融合方式
                - 'cat': (v,a)→cat[z_dim/3], (as,tracker)→cat[2*z_dim/3], (tp,target)→cat[z_dim]
                - 'avgpooling': (v,a)→cat[z_dim/2], (as,tracker)→avgpool[z_dim/2], (tp,target)→cat[z_dim]
            num_trackers: tracker类型数量
            num_targets: target类型数量
            num_linear_velocities: 线速度类型数量
            num_angular_velocities: 角速度类型数量
        """
        super(CVAE_MultiDim, self).__init__(
            z_dim, state_size, action_size, reward_size, hidden_size, 
            num_hidden_layers, output_variance, predict_state_difference, 
            merge_reward_next_state, logvar_min, logvar_max
        )
        
        self.z_dim = z_dim
        self.tracker_fusion = tracker_fusion
        
        # 确保参数完整
        assert num_trackers is not None and num_targets is not None, \
            "num_trackers and num_targets must be provided"
        assert num_linear_velocities is not None and num_angular_velocities is not None, \
            "num_linear_velocities and num_angular_velocities must be provided"
        
        # 计算每个encoder的维度
        if tracker_fusion == 'cat':
            # 方案1: 全部cat
            # v[z/4] + a[z/4] → cat → action_space[z/2]
            # action_space[z/2] + tracker[z/4] → cat → tracker_policy[3z/4]
            # tracker_policy[3z/4] + target[z/4] → cat → task[z]
            assert z_dim % 4 == 0, f"For tracker_fusion='cat', z_dim ({z_dim}) must be divisible by 4"
            self.base_dim = z_dim // 4
            # v, a, tracker, target 各占 z_dim/4
            self.v_dim = self.base_dim
            self.a_dim = self.base_dim
            self.tracker_dim = self.base_dim
            self.target_dim = self.base_dim
        elif tracker_fusion == 'avgpooling':
            # 方案2: action_space和tracker用avgpooling
            # (v,a)→cat[z_dim/2], (as,tracker)→avgpool[z_dim/2], (tp,target)→cat[z_dim]
            assert z_dim % 2 == 0, f"For tracker_fusion='avgpooling', z_dim ({z_dim}) must be divisible by 2"
            self.base_dim = z_dim // 2
            # v和a各输出base_dim/2，cat后为base_dim
            self.v_dim = self.base_dim // 2
            self.a_dim = self.base_dim // 2
            # tracker输出base_dim，和action_space avgpool后仍为base_dim
            self.tracker_dim = self.base_dim
            # target输出base_dim
            self.target_dim = self.base_dim
        else:
            raise ValueError(f"Invalid tracker_fusion: {tracker_fusion}. Must be 'cat' or 'avgpooling'")
        
        # 创建4个基础encoder（只存储mean）
        self.linear_v_encoder = nn.Parameter(torch.randn(num_linear_velocities, self.v_dim))
        self.angular_v_encoder = nn.Parameter(torch.randn(num_angular_velocities, self.a_dim))
        self.tracker_encoder = nn.Parameter(torch.randn(num_trackers, self.tracker_dim))
        self.target_encoder = nn.Parameter(torch.randn(num_targets, self.target_dim))
        
        # 创建logvar参数（为最终的tracker_policy和target）
        if tracker_fusion == 'cat':
            # tracker_policy是action_space[2*base_dim] + tracker[base_dim] = 3*base_dim
            # target是base_dim
            self.tracker_policy_logvar = nn.Parameter(torch.zeros(1, 3 * self.base_dim))
            self.target_logvar = nn.Parameter(torch.zeros(1, self.base_dim))
        else:  # avgpooling
            # tracker_policy和target都是base_dim
            self.tracker_policy_logvar = nn.Parameter(torch.zeros(1, self.base_dim))
            self.target_logvar = nn.Parameter(torch.zeros(1, self.base_dim))
        
        print(f"CVAE_MultiDim initialized (4D with tracker_fusion='{tracker_fusion}'):")
        print(f"  Final z_dim: {z_dim}")
        print(f"  num_trackers: {num_trackers}, num_targets: {num_targets}")
        print(f"  num_linear_velocities: {num_linear_velocities}, num_angular_velocities: {num_angular_velocities}")
        if tracker_fusion == 'cat':
            print(f"  方案1 (全cat):")
            print(f"    v[{self.v_dim}] + a[{self.a_dim}] → cat → action_space[{2*self.base_dim}]")
            print(f"    action_space[{2*self.base_dim}] + tracker[{self.tracker_dim}] → cat → tracker_policy[{3*self.base_dim}]")
            print(f"    tracker_policy[{3*self.base_dim}] + target[{self.target_dim}] → cat → task[{z_dim}]")
        else:
            print(f"  方案2 (action_space+tracker用avgpool):")
            print(f"    v[{self.v_dim}] + a[{self.a_dim}] → cat → action_space[{self.base_dim}]")
            print(f"    action_space[{self.base_dim}] + tracker[{self.tracker_dim}] → avgpool → tracker_policy[{self.base_dim}]")
            print(f"    tracker_policy[{self.base_dim}] + target[{self.target_dim}] → cat → task[{z_dim}]")

    def compute_kl_divergence(self, mean, logvar):
        return (- 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1))

    def forward_encoder(self, obs, action, reward, next_obs, 
                       tracker_id=None, target_id=None,
                       linear_v_id=None, angular_v_id=None):
        """
        4维编码任务信息，两种融合方案可选
        
        方案1 (tracker_fusion='cat'):
            Step 1: (v, a) → cat → action_space [2*base_dim]
            Step 2: (action_space, tracker) → cat → tracker_policy [3*base_dim]
            Step 3: (tracker_policy, target) → cat → task [z_dim=4*base_dim]
        
        方案2 (tracker_fusion='avgpooling'):
            Step 1: (v, a) → cat → action_space [base_dim]
            Step 2: (action_space, tracker) → avgpool → tracker_policy [base_dim]
            Step 3: (tracker_policy, target) → cat → task [z_dim=2*base_dim]
        
        Args:
            tracker_id: tracker索引
            target_id: target索引
            linear_v_id: 线速度索引
            angular_v_id: 角速度索引
        
        Returns:
            mean, logvar, z_sample
        """
        assert tracker_id is not None and target_id is not None, \
            "tracker_id and target_id must be provided"
        assert linear_v_id is not None and angular_v_id is not None, \
            "linear_v_id and angular_v_id must be provided"
        
        batch_size = obs.size(0) if obs is not None else 1
        
        # ========== Step 1: (v, a) → cat → action_space ==========
        linear_v_mean = self.linear_v_encoder[linear_v_id]
        if linear_v_mean.dim() == 1:
            linear_v_mean = linear_v_mean.unsqueeze(0)
        if linear_v_mean.size(0) == 1 and batch_size > 1:
            linear_v_mean = linear_v_mean.expand(batch_size, -1)
        
        angular_v_mean = self.angular_v_encoder[angular_v_id]
        if angular_v_mean.dim() == 1:
            angular_v_mean = angular_v_mean.unsqueeze(0)
        if angular_v_mean.size(0) == 1 and batch_size > 1:
            angular_v_mean = angular_v_mean.expand(batch_size, -1)
        
        # Cat v and a
        action_space_mean = torch.cat([linear_v_mean, angular_v_mean], dim=-1)
        
        # ========== Step 2: (action_space, tracker) → cat or avgpool ==========
        tracker_mean = self.tracker_encoder[tracker_id]
        if tracker_mean.dim() == 1:
            tracker_mean = tracker_mean.unsqueeze(0)
        if tracker_mean.size(0) == 1 and batch_size > 1:
            tracker_mean = tracker_mean.expand(batch_size, -1)
        
        if self.tracker_fusion == 'cat':
            # 方案1: cat
            tracker_policy_mean = torch.cat([action_space_mean, tracker_mean], dim=-1)
        else:  # avgpooling
            # 方案2: avgpool
            tracker_policy_mean = (action_space_mean + tracker_mean) / 2.0
        
        tracker_policy_logvar = self.tracker_policy_logvar.expand(batch_size, -1)
        
        # ========== Step 3: (tracker_policy, target) → cat ==========
        target_mean = self.target_encoder[target_id]
        if target_mean.dim() == 1:
            target_mean = target_mean.unsqueeze(0)
        if target_mean.size(0) == 1 and batch_size > 1:
            target_mean = target_mean.expand(batch_size, -1)
        
        target_logvar = self.target_logvar.expand(batch_size, -1)
        
        # Final cat
        mean = torch.cat([tracker_policy_mean, target_mean], dim=-1)
        logvar = torch.cat([tracker_policy_logvar, target_logvar], dim=-1)
        
        # Reparameterize
        z_sample = reparameterize(mean, logvar)
        
        return mean, logvar, z_sample

    def forward(self, obs, action, reward, next_obs, 
                tracker_id=None, target_id=None,
                linear_v_id=None, angular_v_id=None):
        """完整的前向传播过程（4维）"""
        z, logvar, z_sample = self.forward_encoder(
            obs, action, reward, next_obs, 
            tracker_id, target_id, linear_v_id, angular_v_id
        )
        next_obs_pred, reward_pred = self.forward_decoder(obs, action, z=z_sample)
        return z, logvar, z_sample, next_obs_pred, reward_pred


# 保留原始CVAE类以便向后兼容
class CVAE(Decoder):
    # Subclass Decoder for backward compatibility
    def __init__(self, hidden_size=64, num_hidden_layers=1, z_dim=5,
                 action_size=5, state_size=2, reward_size=1,
                 tabular_encoder_entries=None, predict_state_difference=False,
                 output_variance='output', merge_reward_next_state=False,
                 logvar_min=-10.0, logvar_max=2.0, granularity_mode="coarse",
                 num_trackers=None, num_targets=None, tracker_dim=None, target_dim=None):
        super(CVAE, self).__init__(z_dim, state_size, action_size, reward_size, hidden_size, num_hidden_layers, output_variance, predict_state_difference, merge_reward_next_state, logvar_min, logvar_max)
        self.granularity_mode = granularity_mode
        self.z_dim = z_dim
        self.tabular_encoder = tabular_encoder_entries is not None
        if self.tabular_encoder:
            if granularity_mode == "fine":
                assert (num_trackers is not None and num_targets is not None and tracker_dim is not None and target_dim is not None)
                assert tracker_dim + target_dim == z_dim
                self.tracker_encoder = nn.Parameter(torch.randn(num_trackers, tracker_dim * 2))
                self.target_encoder = nn.Parameter(torch.randn(num_targets, target_dim * 2))
                self.tracker_dim = tracker_dim
                self.target_dim = target_dim
            else:
                self.encoder = nn.Parameter(torch.randn(tabular_encoder_entries, z_dim * 2))
        else:
            if tabular_encoder_entries is None:
                self.encoder = FlattenMlp(input_size=state_size*2+action_size+reward_size,
                                         output_size=z_dim*2,
                                         hidden_sizes=[hidden_size for _ in range(num_hidden_layers)])
            else:
                raise NotImplementedError("非tabular_encoder暂不支持fine granularity")

    def compute_kl_divergence(self, mean, logvar):
        return (- 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1))

    def forward_encoder(self, obs, action, reward, next_obs, task_idx=None, tracker_id=None, target_id=None):
        if self.tabular_encoder:
            if self.granularity_mode == "fine":
                tracker_z = self.tracker_encoder[tracker_id]   # (B, tracker_dim*2)
                target_z = self.target_encoder[target_id]       # (B, target_dim*2)
                if tracker_z.dim() == 1:
                    tracker_z = tracker_z.unsqueeze(0)
                if target_z.dim() == 1:
                    target_z = target_z.unsqueeze(0)
                z = torch.cat([tracker_z, target_z], dim=1)   # (B, z_dim*2)
            else:
                assert task_idx is not None
                z = self.encoder[task_idx]
                if z.dim() == 1:
                    z = z.unsqueeze(0)
            batch_size = obs.size(0) if obs is not None else None
            if batch_size is not None and z.size(0) == 1 and batch_size > 1:
                z = z.expand(batch_size, -1)
            mean, logvar = torch.split(z, self.z_dim, dim=1)
            z_sample = reparameterize(mean, logvar)
            return mean, logvar, z_sample
        else:
            if isinstance(self.encoder, nn.Module):
                z = self.encoder(obs, action, reward, next_obs)
            else:
                assert task_idx is not None
                z = self.encoder[task_idx]
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                if obs is not None:
                    batch_size = obs.size(0)
                    if z.size(0) == 1 and batch_size > 1:
                        z = z.expand(batch_size, -1)
            mean, logvar = torch.split(z, self.z_dim, dim=1)
            z_sample = reparameterize(mean, logvar)
            return mean, logvar, z_sample

    def forward(self, obs, action, reward, next_obs, task_idx=None):
        """完整的前向传播过程，包括编码和解码"""
        mean, logvar, z_sample = self.forward_encoder(obs, action, reward, next_obs, task_idx)
        next_obs_pred, reward_pred = self.forward_decoder(obs, action, z=z_sample)
        return mean, logvar, z_sample, next_obs_pred, reward_pred


# 用于单任务建模的预测器
class Predictor(nn.Module):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=2,
                 # actions, states, rewards
                 action_size=5,
                 state_size=2,
                 reward_size=1
                 ):
        super(Predictor, self).__init__()

        self.action_size=action_size
        self.state_size=state_size
        self.reward_size=reward_size
        self.mlp = FlattenMlp(input_size=state_size+action_size,
                                    output_size=state_size+reward_size,
                                    hidden_sizes=[hidden_size for i in range(num_hidden_layers)])

    def forward(self, obs, action):
        out = self.mlp(obs, action)
        s_, r = out[:,0:self.state_size], out[:, self.state_size:]
        return s_, r
