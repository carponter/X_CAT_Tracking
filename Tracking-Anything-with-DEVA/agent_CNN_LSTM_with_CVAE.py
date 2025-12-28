import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, CNN_LSTM,CNN_simple
import numpy as np
import math
import copy
from torch.autograd import Variable
from utils_basic import weights_init
from networks_with_CVAE import Critic, Actor, CNN_LSTM, CNN_simple, TaskAwareCNN_LSTM
from models.generative import CVAE, reparameterize
import os

class CQLSAC_CNN_LSTM(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device,
                 stack_frames,
                 lstm_seq_len,
                 lstm_layer,
                 lstm_out
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC_CNN_LSTM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.stack_frames=stack_frames
        self.device = device
        self.lstm_seq_len = lstm_seq_len
        self.gamma = torch.FloatTensor([0.99]).to(device)
        # self.gamma = torch.FloatTensor([0.9]).to(device)

        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        # self.alpha=torch.tensor([0.5])

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.lstm_layer = lstm_layer
        self.CNN_LSTM = CNN_LSTM(state_size=self.state_size,
                                action_size=self.action_size,
                                hidden_size=hidden_size,
                                 stack_frames=self.stack_frames,
                                 lstm_out=lstm_out,
                                 lstm_layer=self.lstm_layer
                                ).to(self.device)  # obs_shape,frame_stack
        self.CNN_LSTM_optimizer = optim.Adam(self.CNN_LSTM.parameters(), lr=learning_rate)
        # Actor Network
        self.actor_local = Actor(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()
        self.critic1_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)



    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1, q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q)).mean()
        return actor_loss.cuda(), log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        # with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)

        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs

    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_ritic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        batch_size=states.shape[0]
        actions=np.array(actions.cpu())
        #action space 归一化
        # 定义每个维度的最小和最大值,每个环境不一样
        min_val = np.array([-30, -100]).astype(np.float32)
        max_val = np.array([30, 100]).astype(np.float32)

        # 将数据归一化到0到1的范围
        normalized_data = ((actions - min_val) / (max_val - min_val)).astype(np.float32)

        # 将数据归一化到-1到1的范围
        normalized_data = (2 * normalized_data - 1).astype(np.float32)
        actions = torch.from_numpy(normalized_data).to(self.device)

        #--------------------------------Image processing------------------------#

        states = self.CNN_LSTM(states)
        states=states.reshape(batch_size*self.lstm_seq_len,-1)
        with torch.no_grad():
            next_states = self.CNN_LSTM(next_states)
            next_states = next_states.reshape(batch_size*self.lstm_seq_len,-1)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)

        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()


        # Compute alpha loss
        # alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu().sum(axis=1) + self.target_entropy).detach().cpu()).mean().cuda()
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean().cuda()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()



        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.reshape(batch_size*self.lstm_seq_len,1) + (self.gamma * (1 - dones.reshape(batch_size*self.lstm_seq_len,1)) * Q_target_next)

        #
        # # Compute critic loss
        q1 = self.critic1(states, actions.reshape(batch_size*self.lstm_seq_len,-1))
        q2 = self.critic2(states, actions.reshape(batch_size*self.lstm_seq_len,-1))
        q1_train =q1.mean()
        q2_train =q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # # CQL addon
        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int(random_actions.shape[0] / states.shape[0])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat,
                                                                                  next_states.shape[1])
        current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)

        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0],
                                                                                                        num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0],
                                                                                                        num_repeat, 1)

        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)

        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp,
                                             dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp,
                                             dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # critic 2
        self.CNN_LSTM_optimizer.zero_grad()

        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        self.CNN_LSTM_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


        return (q1_train.item(),q2_train.item(),actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(),cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class CQLSAC_CNN_LSTM_with_CVAE(nn.Module):
    """
    扩展CQLSAC_CNN_LSTM，添加CVAE用于任务感知
    同时保持原有SAC和CQL算法的功能不变
    在CNN提取特征后、LSTM处理前拼接任务嵌入
    使用查表方式获取任务嵌入
    """
    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device,
                 stack_frames,
                 lstm_seq_len,
                 lstm_layer,
                 lstm_out,
                 z_dim=8,              # 任务的潜在空间维度，使用维度8
                 reward_size=1,         # 奖励维度
                 use_task_encoding=True,    # 是否使用任务编码
                 pretrained_cvae_path=None,  # 预训练CVAE模型路径
                 num_tasks=8  # 任务数量参数
                 ):

        # 调用父类的完整初始化方法，使用相同的参数
        super(CQLSAC_CNN_LSTM_with_CVAE, self).__init__()
        self.state_size=state_size
        self.action_size=action_size 
        self.stack_frames=stack_frames
        self.device = device
        self.lstm_seq_len = lstm_seq_len
        self.gamma = torch.FloatTensor([0.99]).to(device)

        self.tau=tau
        self.hidden_size=hidden_size 
        self.learning_rate=learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.lstm_layer = lstm_layer
     
        self.z_dim = z_dim
        self.reward_size = reward_size
        self.use_task_encoding = use_task_encoding
        self.num_tasks = num_tasks 
        
        self.CNN_LSTM = TaskAwareCNN_LSTM(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=hidden_size,
            stack_frames=self.stack_frames,
            lstm_out=lstm_out,
            lstm_layer=lstm_layer,
            z_dim=z_dim,
            reward_size=reward_size,
            use_task_encoding=use_task_encoding,
            tabular_encoder_entries=num_tasks
        ).to(self.device)
        
        if hasattr(self.CNN_LSTM, 'num_tasks'):
            self.CNN_LSTM.num_tasks = num_tasks
        
        if pretrained_cvae_path is not None and os.path.exists(pretrained_cvae_path):
            try:
                print(f"加载预训练CVAE: {pretrained_cvae_path}")
                # 直接加载模型，不使用 add_safe_globals
                import numpy as np
                
                # 尝试加载模型
                try:
                    # 首先尝试简单加载
                    checkpoint = torch.load(pretrained_cvae_path, map_location=device)
                except Exception as load_err:
                    print(f"简单加载失败，错误: {load_err}")
                    # 如果上面的方法失败，尝试使用pickle_module=None
                    checkpoint = torch.load(pretrained_cvae_path, map_location=device, pickle_module=None)
                
                pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
                
                if 'task_ids' in checkpoint:
                    self.task_ids = checkpoint['task_ids']
                if 'num_tasks' in checkpoint:
                    self.num_tasks = checkpoint['num_tasks']
                    self.CNN_LSTM.num_tasks = checkpoint['num_tasks']
                
                if 'encoder' in pretrained_dict and isinstance(pretrained_dict['encoder'], torch.Tensor):
                    self.CNN_LSTM.cvae.tabular_encoder = True
                
                self.CNN_LSTM.cvae.load_state_dict(pretrained_dict, strict=False)
                print("成功加载预训练CVAE权重")
                
            except Exception as e:
                print(f"加载预训练CVAE失败: {e}")
                raise ValueError("加载预训练CVAE失败，请检查路径和模型结构")
        
        # Actor
        self.actor_local = Actor(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic
        self.critic1 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()
        self.critic1_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)        
        self.CNN_LSTM_optimizer = optim.Adam(self.CNN_LSTM.parameters(), lr=learning_rate)

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1, q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q)).mean()
        return actor_loss.cuda(), log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        # with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)

        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs        
    
    # task embedding map
    def load_task_embedding_table(self, embedding_file_path):
        """
        加载预生成的任务嵌入表
        
        Args:
            embedding_file_path: 任务嵌入文件路径
        """
        try:
            # 直接加载模型，不使用 add_safe_globals
            import numpy as np
            
            # 尝试加载任务嵌入表
            try:
                # 首先尝试简单加载
                self.task_embeddings_table = torch.load(embedding_file_path)
            except Exception as load_err:
                print(f"简单加载失败，错误: {load_err}")
                # 尝试其他加载方式
                self.task_embeddings_table = torch.load(embedding_file_path, pickle_module=None)
                
            print(f"成功加载任务嵌入表，包含 {len(self.task_embeddings_table)} 个任务")
            return True
        except Exception as e:
            print(f"加载任务嵌入表失败: {e}")
            self.task_embeddings_table = {}
            return False
    
    def inference(self, raw_states, actions, rewards, raw_next_states):
        """
        从原始状态轨迹推断任务表示，并设置模型使用该任务表示
        调用TaskAwareCNN_LSTM的inference方法
        
        Args:
            raw_states: 输入状态序列 [batch_size, seq_len, c, h, w]
            actions: 动作序列 [batch_size, seq_len, action_dim]
            rewards: 奖励序列 [batch_size, seq_len]
            raw_next_states: 下一状态序列 [batch_size, seq_len, c, h, w]
            
        Returns:
            推断出的任务表示嵌入向量
        """
        # 使用TaskAwareCNN_LSTM的inference方法获取所有需要的信息
        lstm_output, ht, ct, task_id, task_embedding = self.CNN_LSTM.inference(
            raw_states, actions, rewards, raw_next_states
        )
        
        # 记录推断出的任务ID和嵌入
        if task_id >= 0:
            print(f"使用推断出的任务ID: {task_id}")
            
        # 将LSTM输出重新整形以适应后续处理
        batch_size = raw_states.shape[0]
        lstm_features = lstm_output.reshape(batch_size * self.lstm_seq_len, -1)
        
        return lstm_features, task_embedding
       
    def learn(self, experiences):           
        states, actions, rewards, next_states, dones, task_embeddings = experiences
                   
        print(f"z_embedding's shape is {task_embeddings.shape}")
        
        batch_size = states.shape[0]
        actions_np = np.array(actions.cpu())
        min_val = np.array([-30, -100]).astype(np.float32)
        max_val = np.array([30, 100]).astype(np.float32)
        normalized_data = ((actions_np - min_val) / (max_val - min_val)).astype(np.float32)
        normalized_data = (2 * normalized_data - 1).astype(np.float32)
        actions = torch.from_numpy(normalized_data).to(self.device)
        #--------------------------------Image processing------------------------#
        states_features = self.CNN_LSTM(states, task_embeddings)
        states_features = states_features.reshape(batch_size*self.lstm_seq_len, -1)
        with torch.no_grad():
            next_states_features = self.CNN_LSTM(next_states, task_embeddings)
            next_states_features = next_states_features.reshape(batch_size*self.lstm_seq_len, -1)
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states_features, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean().cuda()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states_features)
            Q_target1_next = self.critic1_target(next_states_features, next_action)
            Q_target2_next = self.critic2_target(next_states_features, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.reshape(batch_size*self.lstm_seq_len, 1) + (
                self.gamma * (1 - dones.reshape(batch_size*self.lstm_seq_len, 1)) * Q_target_next
            )

        # Critic loss
        q1 = self.critic1(states_features, actions.reshape(batch_size*self.lstm_seq_len, -1))
        q2 = self.critic2(states_features, actions.reshape(batch_size*self.lstm_seq_len, -1))
        q1_train = q1.mean()
        q2_train = q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # CQL
        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int(random_actions.shape[0] / states_features.shape[0])   
        temp_states = states_features.unsqueeze(1).repeat(1, num_repeat, 1).view(states_features.shape[0] * num_repeat, states_features.shape[1])
        temp_next_states = next_states_features.unsqueeze(1).repeat(1, num_repeat, 1).view(
            next_states_features.shape[0] * num_repeat, next_states_features.shape[1]
        )
        
        current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)

        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(
            states_features.shape[0], num_repeat, 1
        )
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(
            states_features.shape[0], num_repeat, 1
        )

        current_pi_values1 = current_pi_values1.reshape(states_features.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states_features.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(states_features.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states_features.shape[0], num_repeat, 1)

        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

        assert cat_q1.shape == (states_features.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states_features.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Updata Critic
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # critic 2 & CNN_LSTM
        self.CNN_LSTM_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        
        # Ensure CNN_LSTM gradients are properly clipped and updated
        clip_grad_norm_(self.CNN_LSTM.parameters(), self.clip_grad_param)
        self.CNN_LSTM_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return (q1_train.item(), q2_train.item(), actor_loss.item(), alpha_loss.item(), 
                critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), 
                cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())
    
    def soft_update(self, local_model, target_model):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
