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


class CVAE(Decoder):
    # Subclass Decoder for backward compatibility
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=1,
                 z_dim=5,
                 # actions, states, rewards
                 action_size=5,
                 state_size=2,
                 reward_size=1,
                 tabular_encoder_entries=None,
                 predict_state_difference=False,
                 output_variance='output',
                 merge_reward_next_state=False,
                 logvar_min=-10.0,
                 logvar_max=2.0,
                 ):
        super(CVAE, self).__init__(
            z_dim, state_size, action_size, reward_size, hidden_size, num_hidden_layers,
            output_variance, predict_state_difference, merge_reward_next_state, logvar_min, logvar_max
        )

        if tabular_encoder_entries is None:
            self.tabular_encoder = False
            self.encoder = FlattenMlp(input_size=state_size * 2 + action_size + reward_size,
                                      output_size=z_dim * 2,
                                      hidden_sizes=[hidden_size for i in range(num_hidden_layers)])
        else:
            assert tabular_encoder_entries > 0
            self.tabular_encoder = True
            self.encoder = nn.Parameter(torch.randn(tabular_encoder_entries, z_dim * 2))
            self.encoder.data[:, z_dim:].zero_()

    def compute_kl_divergence(self, mean, logvar):
        return (- 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1))

    def forward_encoder(self, obs, action, reward, next_obs, task_idx=None):
        if isinstance(self.encoder, nn.Module):
            z = self.encoder(obs, action, reward, next_obs)
        else:
            assert task_idx is not None, "Task index required for tabular encoder"
            z = self.encoder[task_idx]
            # 确保张量至少有2个维度，以便可以沿着维度1拆分
            if z.dim() == 1:
                z = z.unsqueeze(0)  # 添加批次维度

            # 确保批次维度与输入数据匹配 - 当obs为None时跳过此步骤
            if obs is not None:
                batch_size = obs.size(0)
                if z.size(0) == 1 and batch_size > 1:
                    z = z.expand(batch_size, -1)  # 扩展到批次大小，保持第二维不变

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

        self.action_size = action_size
        self.state_size = state_size
        self.reward_size = reward_size
        self.mlp = FlattenMlp(input_size=state_size + action_size,
                              output_size=state_size + reward_size,
                              hidden_sizes=[hidden_size for i in range(num_hidden_layers)])

    def forward(self, obs, action):
        out = self.mlp(obs, action)
        s_, r = out[:, 0:self.state_size], out[:, self.state_size:]
        return s_, r