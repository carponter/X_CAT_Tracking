import numpy as np
import random
import torch
from collections import deque, namedtuple
import os
import cv2

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples with batch-wise file loading."""

    def __init__(self, buffer_size, batch_size, device, lstm_seq_len, config, load_all_to_memory=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device: device to store tensors
            lstm_seq_len (int): length of LSTM sequence
            config: configuration object
            load_all_to_memory (bool): if True, load all data to memory first
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_id = []
        self.st_id = []
        self.lstm_seq_len = lstm_seq_len
        self.input_type = config.input_type
        self.load_all_to_memory = load_all_to_memory
        
        # 新增：文件管理相关
        self.data_path = None
        self.file_list = []
        self.current_batch_files = []
        self.current_batch_data = []
        self.files_per_batch = batch_size  # 每次加载batch_size个文件
        
        # 全内存模式相关
        self.all_data_loaded = False
        self.all_transitions = []  # 存储所有转换数据

    def set_data_path(self, data_path):
        """设置数据路径并获取文件列表"""
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.pt')]
        self.file_list.sort()
        # print(f"Found {len(self.file_list)} files in {data_path}")
        
        if self.load_all_to_memory:
            self._load_all_data_to_memory()
        else:
            self._load_new_batch()

    def _load_new_batch(self):
        """加载新的一批文件到内存"""
        if not self.file_list:
            # print("No files available to load")
            return
            
        # 随机选择batch_size个文件
        if len(self.file_list) >= self.files_per_batch:
            selected_files = random.sample(self.file_list, self.files_per_batch)
        else:
            selected_files = self.file_list
            
        # print(f"Loading new batch: {len(selected_files)} files")
        self.current_batch_files = selected_files
        self.current_batch_data = []
        
        # 清空之前的buffer
        self.memory.clear()
        
        # 加载选中的文件
        for file_name in selected_files:
            # print(f"  Loading: {file_name}")
            file_path = os.path.join(self.data_path, file_name)
            
            # 加载 .pt 文件
            loaded_data = torch.load(file_path, weights_only=False)
            
            # 如果 loaded_data 为列表，则过滤掉空字典
            if isinstance(loaded_data, list):
                frames = [frame for frame in loaded_data if frame]  
            else:
                frames = [loaded_data]
            
            # 根据 input_type 选择使用 mask 或 image
            if ('deva' in self.input_type.lower() or 'image' in self.input_type.lower() or 'mask' in self.input_type.lower()):
                # 默认使用 mask 的前三个通道
                state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[:-1]
                goal_tmp = np.array([np.array(frame['goal'][:, :, 0:3]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[1:]
            elif 'devadepth' in self.input_type.lower() or 'rgbd' in self.input_type.lower():
                state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[1:]
            
            # 获取动作与奖励信息
            act_tmp = np.array([np.array(frame['action']) for frame in frames])[:-1].squeeze(axis=1)
            
            # 计算IoU奖励
            # re_iou = np.array([
            #     self._reward_cal(state_tmp[i], goal_tmp[i])
            #     for i in range(len(state_tmp))
            # ])
            re_iou = np.array([np.array(frame['reward']) for frame in frames]).squeeze(axis=1)[:-1]
            
            # 确保数据长度一致
            assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_iou.shape[0] == next_state_tmp.shape[0] and \
                next_state_tmp.shape[0] == act_tmp.shape[0], "数据长度不匹配！"
            
            # 遍历所有时间步，将数据加入 buffer（不转移到GPU，保持在CPU）
            for i in range(state_tmp.shape[0]):
                if i % state_tmp.shape[0] == 0 and i > 0:
                    done = True
                else:
                    done = False
                    
                # 保持在CPU，不转移到GPU
                state = torch.from_numpy(np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                action = torch.from_numpy(act_tmp[i]).float()
                reward = torch.from_numpy(np.array(re_iou[i])).float()
                next_state = torch.from_numpy(np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                done_tensor = torch.from_numpy(np.array(done)).float()
                
                self.add(state, action, reward, next_state, done_tensor)
        
        # print(f"Loaded {len(self.memory)} transitions from {len(selected_files)} files")
        # print(f"Current buffer size: {len(self.memory)}")

    def _load_all_data_to_memory(self):
        """一次性加载所有数据到内存"""
        print(f"Loading all {len(self.file_list)} files to memory...")
        self.all_transitions = []
        
        for file_idx, file_name in enumerate(self.file_list):
            if file_idx % 100 == 0:
                print(f"Loading file {file_idx}/{len(self.file_list)}: {file_name}")
                
            file_path = os.path.join(self.data_path, file_name)
            loaded_data = torch.load(file_path, weights_only=False)
            
            # 处理数据格式
            if isinstance(loaded_data, list):
                frames = [frame for frame in loaded_data if frame]  
            else:
                frames = [loaded_data]
            
            # 根据 input_type 选择使用 mask 或 image
            if ('deva' in self.input_type.lower() or 'image' in self.input_type.lower() or 'mask' in self.input_type.lower()):
                state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[:-1]
                goal_tmp = np.array([np.array(frame['goal'][:, :, 0:3]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[1:]
            elif 'devadepth' in self.input_type.lower() or 'rgbd' in self.input_type.lower():
                state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[1:]
            
            # 获取动作与奖励信息
            act_tmp = np.array([np.array(frame['action']) for frame in frames])[:-1].squeeze(axis=1)
            re_iou = np.array([
                self._reward_cal(state_tmp[i], goal_tmp[i])
                for i in range(len(state_tmp))
            ])
            # re_iou = np.array([np.array(frame['reward']) for frame in frames]).squeeze(axis=1)[:-1]
            
            # 确保数据长度一致
            assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_iou.shape[0] == next_state_tmp.shape[0] and \
                next_state_tmp.shape[0] == act_tmp.shape[0], "数据长度不匹配！"
            
            # 将所有转换添加到全内存存储
            for i in range(state_tmp.shape[0]):
                if i % state_tmp.shape[0] == 0 and i > 0:
                    done = True
                else:
                    done = False
                    
                # 保持在CPU，不转移到GPU
                state = torch.from_numpy(np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                action = torch.from_numpy(act_tmp[i]).float()
                reward = torch.from_numpy(np.array(re_iou[i])).float()
                next_state = torch.from_numpy(np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                done_tensor = torch.from_numpy(np.array(done)).float()
                
                self.all_transitions.append((state, action, reward, next_state, done_tensor))
        
        self.all_data_loaded = True
        print(f"Loaded {len(self.all_transitions)} transitions to memory")

    def refresh_batch(self):
        """刷新batch，加载新的文件"""
        if self.load_all_to_memory:
            # 全内存模式下不需要刷新batch
            pass
        else:
            self._load_new_batch()

    def _reward_cal(self, state, goal):
        """计算IoU奖励"""
        if state.max() == 255:
            boxA = self._get_bounding_box(state)
            if boxA is None:
                return -1
            boxB = self._get_bounding_box(goal)
            if boxB is None:
                return 0

            # 计算交集
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            iou_reward = interArea / float(boxAArea + boxBArea - interArea)
            total_reward = np.clip(iou_reward, -1, 1)
        else:
            total_reward = -1

        return total_reward

    def _get_bounding_box(self, mask_image):
        """获取边界框"""
        target_pixels = np.where(np.all(mask_image == [255, 255, 255], axis=-1))

        if len(target_pixels[0]) == 0:
            return None

        y_min = np.min(target_pixels[0])
        y_max = np.max(target_pixels[0])
        x_min = np.min(target_pixels[1])
        x_max = np.max(target_pixels[1])

        return x_min, y_min, x_max, y_max

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(
            state,
            action,
            reward,
            next_state,
            done)
        self.memory.append(e)
    
    def sample(self):
        if self.load_all_to_memory:
            # 全内存模式：直接从所有数据中采样
            if not self.all_data_loaded:
                raise RuntimeError("All data not loaded yet!")
                
            # for lstm,seq frame input
            if ('cnn' in self.input_type.lower() and 'lstm' in self.input_type.lower()) or 'mlp' in self.input_type.lower() or 'clip' in self.input_type.lower():
                states_test = []
                actions_test = []
                rewards_test = []
                next_states_test = []
                done_test = []
                
                for i in range(self.batch_size):
                    # 随机选择起始位置
                    start_idx = random.randint(0, len(self.all_transitions) - self.lstm_seq_len)
                    experiences_test = []
                    
                    for j in range(self.lstm_seq_len):
                        experiences_test.append(self.all_transitions[start_idx + j])
                    
                    states_test.append([e[0] for e in experiences_test])
                    actions_test.append([e[1] for e in experiences_test])
                    rewards_test.append([e[2] for e in experiences_test])
                    next_states_test.append([e[3] for e in experiences_test])
                    done_test.append([e[4] for e in experiences_test])

                states = torch.stack([torch.stack(s) for s in states_test])
                actions = torch.stack([torch.stack(a) for a in actions_test])
                rewards = torch.stack([torch.stack(r) for r in rewards_test])
                next_states = torch.stack([torch.stack(n) for n in next_states_test])
                dones = torch.stack([torch.stack(d) for d in done_test])

                # 在sample时转移到GPU
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

                return (states, actions, rewards, next_states, dones)
            else:
                # 非LSTM模式：随机采样
                batch = random.sample(self.all_transitions, self.batch_size)
                
                states = torch.stack([e[0] for e in batch])
                actions = torch.stack([e[1] for e in batch])
                rewards = torch.stack([e[2] for e in batch])
                next_states = torch.stack([e[3] for e in batch])
                dones = torch.stack([e[4] for e in batch])
                
                # 转移到GPU
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                
                return (states, actions, rewards, next_states, dones)
        else:
            # 原有的batch模式
            if len(self.memory) == 0:
                self._load_new_batch()
            
            # for lstm,seq frame input
            if ('cnn' in self.input_type.lower() and 'lstm' in self.input_type.lower()) or 'mlp' in self.input_type.lower() or 'clip' in self.input_type.lower():
                states_test = []
                actions_test = []
                rewards_test = []
                next_states_test = []
                done_test = []
                self.batch_id = []
                self.st_id = []
                
                for i in range(self.batch_size):
                    self.st_id.append(random.randint(0, 349 - self.lstm_seq_len))
                    self.batch_id.append(random.randint(0, int(len(self.memory)/349) - 1))

                for i in range(self.batch_size):
                    experiences_test = []
                    for j in range(0, self.lstm_seq_len):
                        experiences_test.append(self.memory[self.batch_id[i]*349+self.st_id[i]])
                        self.st_id[i] += 1

                    states_test.append([e.state for e in experiences_test if e is not None])
                    actions_test.append([e.action for e in experiences_test if e is not None])
                    rewards_test.append([e.reward for e in experiences_test if e is not None])
                    next_states_test.append([e.next_state for e in experiences_test if e is not None])
                    done_test.append([e.done for e in experiences_test if e is not None])

                states = torch.stack([torch.stack(s) for s in states_test])
                actions = torch.stack([torch.stack(a) for a in actions_test])
                rewards = torch.stack([torch.stack(r) for r in rewards_test])
                next_states = torch.stack([torch.stack(n) for n in next_states_test])
                dones = torch.stack([torch.stack(d) for d in done_test])

                # 在sample时转移到GPU
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)

            return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)