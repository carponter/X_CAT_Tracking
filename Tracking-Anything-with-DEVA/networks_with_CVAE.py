import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils_basic import weights_init
from models.generative import CVAE, reparameterize
import os
# import clip
class CNN_simple(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_simple, self).__init__()
        c,w,h = obs_shape
        # self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(c, 32, 5, stride=1, padding=2)

        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, c, w, h))
        out = self.forward(dummy_state)
        self.outshape = out.shape
        out = out.view(stack_frames, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.apply(weights_init)
        self.train()

    def forward(self, x, batch_size=1, fc=False):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        return x

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CNN_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,stack_frames,lstm_out,lstm_layer):
        super(CNN_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.CNN_Simple = CNN_simple(self.input_shape,self.stack_frames)
        self.cnn_dim = self.CNN_Simple.outdim
        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.cnn_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x=self.CNN_Simple(input)
        x=x.reshape(x.shape[0],-1)
        x=x.reshape(batch_size,seq_len,-1)
        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x = self.CNN_Simple(input)
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct
class LSTM(nn.Module):
    def __init__(self, input_dim, action_size, hidden_size,lstm_out,lstm_layer):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size



        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        # x=self.CNN_Simple(input)
        # x=x.reshape(x.shape[0],-1)
        x=input.reshape(batch_size,seq_len,-1)

        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        x = input.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        # log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(2, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    


class TaskAwareCNN_LSTM(nn.Module):
    # cnn_feature+ z
    def __init__(self, state_size, action_size, hidden_size, stack_frames, lstm_out, lstm_layer, 
                 z_dim=8, reward_size=1, use_task_encoding=True, tabular_encoder_entries=None):
        super(TaskAwareCNN_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.CNN_Simple = CNN_simple(state_size, stack_frames)
        self.cnn_dim = self.CNN_Simple.outdim  # 576 for 64x64 input
        self.lstm_layer = lstm_layer
        self.lstm_out = lstm_out
        self.z_dim = z_dim
        self.use_task_encoding = use_task_encoding
        
        # 任务嵌入字典和任务名称映射
        self.task_embeddings_dict = None
        self.task_names_dict = None
        
        # CVAE for task embedding
        self.cvae = CVAE(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            z_dim=z_dim,
            action_size=action_size,
            state_size=np.prod(state_size),
            reward_size=reward_size,
            predict_state_difference=False,  
            merge_reward_next_state=False,   
            output_variance='output',
            logvar_min=-15.0,                
            logvar_max=2.0,               
            tabular_encoder_entries=tabular_encoder_entries  # bool -> whether to use tabular encoder
        ).to(device='cuda' if torch.cuda.is_available() else 'cpu')
                
        lstm_input_dim = self.cnn_dim  # 576
        
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_out, num_layers=lstm_layer, batch_first=True)
        self.ht = None
        self.ct = None
        
        if use_task_encoding:
            self.outdim = lstm_out + z_dim  # 64 + 64 = 128
        else:
            self.outdim = lstm_out  # 64 (baseline without task encoding)

        self.num_tasks = tabular_encoder_entries if tabular_encoder_entries is not None else 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, task_embeddings=None):
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        input_reshaped = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)  # [batch, seq, 576]

        h0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(cnn_features.device)
        c0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(cnn_features.device)
        lstm_output, (ht, ct) = self.lstm(cnn_features, (h0, c0))  # [batch, seq, 64]

        if self.use_task_encoding:
            # Handle shape
            if len(task_embeddings.shape) == 1:
                task_embeddings = task_embeddings.unsqueeze(0)
            if task_embeddings.shape[0] != batch_size:
                task_embeddings = task_embeddings.expand(batch_size, -1)
            
            # Expand to sequence dimension and concatenate
            z_expanded = task_embeddings.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            output = torch.cat([lstm_output, z_expanded], dim=-1)  # [batch, seq, 128]
        else:
            output = lstm_output  # [batch, seq, 64]
        
        return output
    
    def _normalize_image_for_cvae(self, image_tensor):
        if image_tensor.max() <= 1.0 and image_tensor.min() >= 0.0:
            return image_tensor
        
        normalized = image_tensor.clone()          
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
            
        return normalized
    
    def load_task_embeddings(self, embeddings_dir):

        if not os.path.exists(embeddings_dir):
            print(f"任务嵌入目录不存在: {embeddings_dir}")
            return False
        
        try:
            # 直接加载模型，不使用 add_safe_globals
                        
            # 加载任务嵌入表
            embeddings_path = os.path.join(embeddings_dir, 'task_embeddings.pt')
            if os.path.exists(embeddings_path):
                try:
                    self.task_embeddings_dict = torch.load(embeddings_path, map_location=self.device)
                except Exception as load_err:
                    print(f"加载任务嵌入文件时出错: {load_err}")
                    return False
                print(f"成功加载任务嵌入表，包含 {len(self.task_embeddings_dict)} 个任务")
            else:
                print(f"任务嵌入文件不存在: {embeddings_path}")
                return False
            
            # 加载任务名称映射
            task_ids_path = os.path.join(embeddings_dir, 'task_ids.json')
            if os.path.exists(task_ids_path):
                with open(task_ids_path, 'r') as f:
                    import json
                    self.task_names_dict = json.load(f)
                print(f"成功加载任务名称映射，包含 {len(self.task_names_dict)} 个任务")
            else:
                print(f"任务名称映射文件不存在: {task_ids_path}")
                
            # 将NumPy数组转换为PyTorch张量
            for task_id, embedding in self.task_embeddings_dict.items():
                if isinstance(embedding, np.ndarray):
                    self.task_embeddings_dict[task_id] = torch.tensor(embedding, device=self.device)
            
            return True
        except Exception as e:
            print(f"加载任务嵌入表失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_cvae_model(self, model_path):
        """
        加载预训练的CVAE模型
        
        Args:
            model_path: CVAE模型路径
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(model_path):
            print(f"CVAE模型文件不存在: {model_path}")
            return False
        
        try:
            # 直接加载模型，不使用 add_safe_globals
                        
            # 加载模型
            try:
                # 首先尝试简单加载
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as load_err:
                print(f"简单加载失败，错误: {load_err}")
                # 尝试其他加载方式
                checkpoint = torch.load(model_path, map_location=self.device, pickle_module=None)
            
            # 提取模型状态字典
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
                
            # 尝试从checkpoint中读取配置信息，并更新模型参数
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("从checkpoint读取到配置信息：")
                for key in ['predict_state_difference', 'merge_reward_next_state', 'logvar_min', 'logvar_max']:
                    if key in config:
                        print(f"  {key}: {config[key]}")
                        # 更新CVAE的属性
                        if hasattr(self.cvae, key):
                            setattr(self.cvae, key, config[key])
            
            # 加载模型状态
            self.cvae.load_state_dict(model_state_dict, strict=False)
            
            # 确保是表格编码器
            if hasattr(self.cvae, 'tabular_encoder') and not self.cvae.tabular_encoder:
                self.cvae.tabular_encoder = True
                print("已将CVAE的tabular_encoder设置为True")
            
            print(f"成功加载CVAE模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载CVAE模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_task_embedding_by_id(self, task_id):
        """
        根据任务ID获取任务嵌入向量
        
        Args:
            task_id: 任务ID
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if self.task_embeddings_dict is None:
            print("错误: 任务嵌入表未加载，请先调用load_task_embeddings方法")
            return None
        
        # 确保任务ID是整数
        task_id = int(task_id)
        
        if task_id in self.task_embeddings_dict:
            embedding = self.task_embeddings_dict[task_id]
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=self.device)
            return embedding
        else:
            print(f"警告: 任务ID {task_id} 在任务嵌入表中不存在")
            return None
    
    def get_task_embedding_by_name(self, task_name):
        """
        根据任务名称获取任务嵌入向量
        
        Args:
            task_name: 任务名称
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if self.task_names_dict is None:
            print("错误: 任务名称映射未加载，请先调用load_task_embeddings方法")
            return None
        
        if self.task_embeddings_dict is None:
            print("错误: 任务嵌入表未加载，请先调用load_task_embeddings方法")
            return None
        
        # 查找任务ID
        task_id = None
        for name, id_val in self.task_names_dict.items():
            if name == task_name:
                task_id = int(id_val)
                break
        
        if task_id is None:
            print(f"警告: 任务名称 '{task_name}' 在任务名称映射中不存在")
            return None
        
        # 获取任务嵌入
        return self.get_task_embedding_by_id(task_id)
    
    def get_task_embedding_from_cvae(self, task_id):
        """
        从CVAE中获取任务嵌入向量
        
        Args:
            task_id: 任务ID
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if not hasattr(self.cvae, 'tabular_encoder') or not self.cvae.tabular_encoder:
            print("错误: CVAE不是表格编码器，无法根据任务ID获取嵌入")
            return None
        
        with torch.no_grad():
            try:
                mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
                return z_sample
            except Exception as e:
                print(f"从CVAE获取任务ID {task_id} 的嵌入时出错: {e}")
                return None
            

    def inference_with_task_embedding_single_step(self, state, action, reward, next_state, task_id):
        """
        只对单步数据和指定task_id算loss和embedding，返回loss和embedding。
        state: [1, C, H, W]
        action: [1, action_dim]
        reward: [1, 1]
        next_state: [1, C, H, W]
        task_id: int
        """
        device = state.device
        obs = self._normalize_image_for_cvae(state).reshape(1, -1)
        next_obs = self._normalize_image_for_cvae(next_state).reshape(1, -1)
        normalized_action = action.clone().to(device)
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
        decoder_output = self.cvae.forward_decoder(obs, normalized_action, z=z_sample)
        next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
        # print("next_obs:", next_obs_pred)
        # print("logvar_s:",logvar_s)
        # print("reward:", reward_pred)
        # print("logvar_r:", logvar_r)
        state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_action, reward, next_obs, z_sample)
        total_loss = state_loss + reward_loss
        embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
        return total_loss.item(), embedding

    def inference_with_given_embedding(self, input, ht=None, ct=None, embedding=None):
        """
        LSTM inference using a provided embedding tensor (新架构)
        CNN → LSTM → concat(lstm_out, embedding)
        
        Args:
            input: [batch, seq, c, h, w]
            ht, ct: LSTM hidden states
            embedding: [1, z_dim] or [batch, z_dim] task embedding
        
        Returns:
            output: [batch, seq, outdim], ht, ct
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        device = input.device
        
        # Step 1: CNN提取纯视觉特征
        input_reshaped = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)  # [batch, seq, 576]
        
        # Step 2: LSTM处理纯视觉特征
        if ht is None or ct is None:
            lstm_output, (new_ht, new_ct) = self.lstm(cnn_features)
        else:
            lstm_output, (new_ht, new_ct) = self.lstm(cnn_features, (ht, ct))
        
        # Step 3: 在LSTM输出后拼接embedding
        if self.use_task_encoding:
            if embedding is None:
                raise ValueError("Embedding must be provided for inference_with_given_embedding.")
            if embedding.shape[0] != batch_size:
                embedding = embedding.expand(batch_size, -1)
            z_expanded = embedding.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            output = torch.cat([lstm_output, z_expanded], dim=-1)  # [batch, seq, 128]
        else:
            output = lstm_output  # [batch, seq, 64]
        
        return output, new_ht, new_ct

    def inference_with_task_embedding_batch(self, states, actions, rewards, next_states, task_id):
        """
        对一段轨迹（N步）和指定task_id算平均loss和embedding，返回平均loss和embedding。
        states: [N, C, H, W]
        actions: [N, action_dim]
        rewards: [N, 1]
        next_states: [N, C, H, W]
        task_id: int
        """
        device = states.device
        N = states.shape[0]
        obs = self._normalize_image_for_cvae(states).reshape(N, -1)
        next_obs = self._normalize_image_for_cvae(next_states).reshape(N, -1)
        normalized_actions = actions.clone().to(device)
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
        if z_sample.size(0) == 1 and N > 1:
            z_sample = z_sample.expand(N, -1)
        decoder_output = self.cvae.forward_decoder(obs, normalized_actions, z=z_sample)
        next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
        # print("next_obs:", next_obs_pred)
        # print("logvar_s:",logvar_s)
        # print("reward:", reward_pred)
        # print("logvar_r:", logvar_r)
        state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_actions, rewards, next_obs, z_sample)
        total_loss = state_loss + reward_loss
        embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
        return total_loss.item(), embedding

    def inference_with_task_embedding_batch_parallel(self, states, actions, rewards, next_states, task_ids):
        """
        并行对多个任务ID进行批量推断，返回每个任务的loss和embedding。
        states: [N, C, H, W]
        actions: [N, action_dim]
        rewards: [N, 1]
        next_states: [N, C, H, W]
        task_ids: list of int, 要推断的任务ID列表
        Returns: dict with task_id as key, (loss, embedding) as value
        """
        device = states.device
        N = states.shape[0]
        results = {}
        
        # 准备输入数据
        obs = self._normalize_image_for_cvae(states).reshape(N, -1)
        next_obs = self._normalize_image_for_cvae(next_states).reshape(N, -1)
        normalized_actions = actions.clone().to(device)
        
        # 并行处理所有任务ID
        for task_id in task_ids:
            mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
            if z_sample.size(0) == 1 and N > 1:
                z_sample = z_sample.expand(N, -1)
            decoder_output = self.cvae.forward_decoder(obs, normalized_actions, z=z_sample)
            next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
            state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_actions, rewards, next_obs, z_sample)
            total_loss = state_loss + reward_loss
            embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
            results[task_id] = (total_loss.item(), embedding)
        
        return results

    def inference_with_task_embedding_single_step_parallel(self, state, action, reward, next_state, task_ids):
        """
        并行对多个任务ID进行单步推断，返回每个任务的loss和embedding。
        state: [1, C, H, W]
        action: [1, action_dim]
        reward: [1, 1]
        next_state: [1, C, H, W]
        task_ids: list of int, 要推断的任务ID列表
        Returns: dict with task_id as key, (loss, embedding) as value
        """
        device = state.device
        results = {}
        
        # 准备输入数据
        obs = self._normalize_image_for_cvae(state).reshape(1, -1)
        next_obs = self._normalize_image_for_cvae(next_state).reshape(1, -1)
        normalized_action = action.clone().to(device)
        
        # 并行处理所有任务ID
        for task_id in task_ids:
            mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
            decoder_output = self.cvae.forward_decoder(obs, normalized_action, z=z_sample)
            next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
            state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_action, reward, next_obs, z_sample)
            total_loss = state_loss + reward_loss
            embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
            results[task_id] = (total_loss.item(), embedding)
        
        return results

    def inference_with_task_embedding_batch_parallel_batched(self, states, actions, rewards, next_states, task_ids, batch_size=None):
        """
        分批并行对多个任务ID进行批量推断，支持大批量任务ID的情况。
        states: [N, C, H, W]
        actions: [N, action_dim]
        rewards: [N, 1]
        next_states: [N, C, H, W]
        task_ids: list of int, 要推断的任务ID列表
        batch_size: int, 每批处理的任务ID数量，如果为None则一次性处理所有
        Returns: dict with task_id as key, (loss, embedding) as value
        """
        if batch_size is None:
            return self.inference_with_task_embedding_batch_parallel(states, actions, rewards, next_states, task_ids)
        
        results = {}
        # 分批处理任务ID
        for i in range(0, len(task_ids), batch_size):
            batch_task_ids = task_ids[i:i + batch_size]
            batch_results = self.inference_with_task_embedding_batch_parallel(states, actions, rewards, next_states, batch_task_ids)
            results.update(batch_results)
        
        return results

    def inference_with_task_embedding_single_step_parallel_batched(self, state, action, reward, next_state, task_ids, batch_size=None):
        """
        分批并行对多个任务ID进行单步推断，支持大批量任务ID的情况。
        state: [1, C, H, W]
        action: [1, action_dim]
        reward: [1, 1]
        next_state: [1, C, H, W]
        task_ids: list of int, 要推断的任务ID列表
        batch_size: int, 每批处理的任务ID数量，如果为None则一次性处理所有
        Returns: dict with task_id as key, (loss, embedding) as value
        """
        if batch_size is None:
            return self.inference_with_task_embedding_single_step_parallel(state, action, reward, next_state, task_ids)
        
        results = {}
        # 分批处理任务ID
        for i in range(0, len(task_ids), batch_size):
            batch_task_ids = task_ids[i:i + batch_size]
            batch_results = self.inference_with_task_embedding_single_step_parallel(state, action, reward, next_state, batch_task_ids)
            results.update(batch_results)
        
        return results

    # =========================
    # Truly vectorized interfaces
    # =========================
    def _vectorize_tile_io(self, obs, actions, rewards, next_obs, task_ids):
        """
        将同一段(N)轨迹在任务维度(T)上复制，得到 (T*N) 批，以便一次性前向。
        返回: obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled, task_idx_long, T, N
        """
        device = obs.device
        N = obs.shape[0]
        T = len(task_ids)
        # 按任务复制 N 次序列
        obs_tiled = obs.repeat(T, 1)
        next_obs_tiled = next_obs.repeat(T, 1)
        actions_tiled = actions.repeat(T, 1)
        rewards_tiled = rewards.repeat(T, 1)
        task_idx_long = torch.as_tensor(task_ids, device=device, dtype=torch.long).repeat_interleave(N)
        return obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled, task_idx_long, T, N

    def inference_with_task_embedding_batch_vectorized(self, states, actions, rewards, next_states, task_ids):
        """
        真实向量化的批量推断：将 N 步轨迹在 T 个任务上复制，合并为 (T*N) 批次，一次性encoder/decoder前向。
        返回 {task_id: (avg_loss, embedding_mean_over_N)}
        """
        device = states.device
        N = states.shape[0]
        # 展开为 CVAE 需要的平面向量
        obs = self._normalize_image_for_cvae(states).reshape(N, -1)
        next_obs = self._normalize_image_for_cvae(next_states).reshape(N, -1)
        normalized_actions = actions.to(device)
        rewards = rewards.to(device)

        # 拼成 (T*N) 批
        obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled, task_idx_long, T, N = \
            self._vectorize_tile_io(obs, normalized_actions, rewards, next_obs, task_ids)

        # 仅通过任务索引查表获取 z（tabular encoder），一次性采样 z
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_idx_long)

        # 计算逐样本未约简损失，得到 shape: [T*N]
        total_loss_per_sample = self.cvae.unreduced_loss(z_sample, obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled)

        # 每个任务的平均损失，以及按步数对 z 做平均得到任务嵌入
        losses_per_task = total_loss_per_sample.view(T, N).mean(dim=1)
        embeddings_per_task = z_sample.view(T, N, self.z_dim).mean(dim=1)

        results = {}
        for i, task_id in enumerate(task_ids):
            results[int(task_id)] = (losses_per_task[i].item(), embeddings_per_task[i].detach().clone())
        return results

    def inference_with_task_embedding_single_step_vectorized(self, state, action, reward, next_state, task_ids):
        """
        真实向量化的单步推断：将同一 (1 步) 样本在 T 个任务上复制，合并为 (T) 批。
        返回 {task_id: (loss, embedding)}
        """
        device = state.device
        # 展平为向量
        obs = self._normalize_image_for_cvae(state).reshape(1, -1)
        next_obs = self._normalize_image_for_cvae(next_state).reshape(1, -1)
        action = action.to(device)
        reward = reward.to(device)

        # 复制到任务批次维度
        obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled, task_idx_long, T, N = \
            self._vectorize_tile_io(obs, action, reward, next_obs, task_ids)

        # 一次性通过 encoder 采样 z
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_idx_long)

        # 未约简逐样本损失，shape: [T]
        total_loss_per_sample = self.cvae.unreduced_loss(z_sample, obs_tiled, actions_tiled, rewards_tiled, next_obs_tiled)

        # 单步无需对步数取均值，直接取每个任务的损失和 z
        results = {}
        for i, task_id in enumerate(task_ids):
            results[int(task_id)] = (total_loss_per_sample[i].item(), z_sample[i].detach().clone())
        return results

    def inference_with_task_embedding_batch_vectorized_batched(self, states, actions, rewards, next_states, task_ids, batch_size=None):
        """
        向量化 + 分批（按任务分块）处理，避免 OOM。
        返回 {task_id: (avg_loss, embedding_mean_over_N)}
        """
        if batch_size is None or batch_size >= len(task_ids):
            return self.inference_with_task_embedding_batch_vectorized(states, actions, rewards, next_states, task_ids)
        results = {}
        for i in range(0, len(task_ids), batch_size):
            chunk_ids = task_ids[i:i+batch_size]
            chunk_res = self.inference_with_task_embedding_batch_vectorized(states, actions, rewards, next_states, chunk_ids)
            results.update(chunk_res)
        return results

    def inference_with_task_embedding_single_step_vectorized_batched(self, state, action, reward, next_state, task_ids, batch_size=None):
        """
        向量化 + 分批（按任务分块）处理，单步版本，避免 OOM。
        返回 {task_id: (loss, embedding)}
        """
        if batch_size is None or batch_size >= len(task_ids):
            return self.inference_with_task_embedding_single_step_vectorized(state, action, reward, next_state, task_ids)
        results = {}
        for i in range(0, len(task_ids), batch_size):
            chunk_ids = task_ids[i:i+batch_size]
            chunk_res = self.inference_with_task_embedding_single_step_vectorized(state, action, reward, next_state, chunk_ids)
            results.update(chunk_res)
        return results
