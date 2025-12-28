import numpy as np
import random
import torch
from collections import deque, namedtuple
import os
import cv2
import re

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
        # 修改Experience类，添加task_embedding字段用于存储任务嵌入向量
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "task_embedding"])
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
        self.task_embeddings_dict = None
        self.name_to_id_map = None
        
        # 全内存模式相关
        self.all_data_loaded = False
        self.all_transitions = []  # 存储所有转换数据
        
        # 🔥 速度档位到实际速度值的映射
        self.velocity_mapping = {
            '1': '100',  # v1 → v100
            '2': '200',  # v2 → v200
            '3': '300',  # v3 → v300
            '4': '400',  # v4 → v400
        }
        self.angular_velocity_mapping = {
            '1': '15',   # a1 → a15
            '2': '30',   # a2 → a30
            '3': '60',   # a3 → a60
            '4': '90',   # a4 → a90
        }
    
    def _convert_velocity_to_actual(self, linear_v_raw, angular_v_raw):
        """
        将档位编号转换为实际速度值
        
        Args:
            linear_v_raw: 原始线速度值（可能是档位编号1-4或实际值100-400）
            angular_v_raw: 原始角速度值（可能是档位编号1-4或实际值15-90）
        
        Returns:
            (linear_v, angular_v): 格式化的速度字符串，如 ('v100', 'a15')
        """
        # 转换线速度
        if linear_v_raw in self.velocity_mapping:
            linear_v = f"v{self.velocity_mapping[linear_v_raw]}"
        else:
            # 已经是实际速度值，直接使用
            linear_v = f"v{linear_v_raw}"
        
        # 转换角速度
        if angular_v_raw in self.angular_velocity_mapping:
            angular_v = f"a{self.angular_velocity_mapping[angular_v_raw]}"
        else:
            # 已经是实际速度值，直接使用
            angular_v = f"a{angular_v_raw}"
        
        return linear_v, angular_v

    def set_data_path(self, data_path, task_embeddings_dict=None, name_to_id_map=None):
        """设置数据路径并获取文件列表"""
        self.data_path = data_path
        self.task_embeddings_dict = task_embeddings_dict
        self.name_to_id_map = name_to_id_map
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
            print("No files available to load")
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
        
        # 记录任务类型和数据
        task_data = {}
        task_ids = {}
        
        # 加载选中的文件
        for file_name in selected_files:
            # print(f"  Loading: {file_name}")
            file_path = os.path.join(self.data_path, file_name)
            
            # 从文件名推断任务类型（4维：tracker2target_v1a2格式）
            # 先提取基本的x2y格式
            match = re.search(r'([a-z0-9]+2[a-z0-9]+)', file_name.lower())
            if not match:
                print(f"无法从文件名提取任务名(期望 X2Y 形式): {file_name}，跳过该文件")
                continue
            base_task_name = match.group(1)
            
            # 检查是否包含速度信息（v1a2格式）
            velocity_match = re.search(r'v(\d+)a(\d+)', file_name.lower())
            if velocity_match:
                # 🔥 使用辅助函数转换档位编号为实际速度值
                linear_v, angular_v = self._convert_velocity_to_actual(
                    velocity_match.group(1), 
                    velocity_match.group(2)
                )
                task_name = f"{base_task_name}_{linear_v}{angular_v}"
            else:
                # 兼容旧格式：level1/level2/level3 映射到实际速度值
                if 'level3' in file_name.lower():
                    task_name = f"{base_task_name}_v300a60"  # level3 → v300a60
                elif 'level2' in file_name.lower():
                    task_name = f"{base_task_name}_v200a30"  # level2 → v200a30
                elif 'level1' in file_name.lower():
                    task_name = f"{base_task_name}_v100a15"  # level1 → v100a15
                else:
                    # 默认为v100a15
                    task_name = f"{base_task_name}_v100a15"
            
            # 使用预生成任务嵌入目录中的任务名->ID映射
            if self.name_to_id_map is None:
                print("错误：未能加载任务名称到ID的映射。请检查 task_embeddings_dir。")
                continue
            if task_name not in self.name_to_id_map:
                print(f"警告：任务名 {task_name} 不在任务嵌入映射中，跳过该文件")
                continue
            current_task_id = int(self.name_to_id_map[task_name])
            # 维护本地task_ids仅用于日志
            task_ids[task_name] = current_task_id
            
            # 加载 .pt 文件
            try:
                loaded_data = torch.load(file_path)
            except Exception as e:
                print(f"标准加载失败: {e}，尝试使用pickle_module=None")
                loaded_data = torch.load(file_path, pickle_module=None)
                
            if isinstance(loaded_data, list):
                frames = [frame for frame in loaded_data if frame]  
            else:
                frames = [loaded_data]
            
            # 处理图像数据
            if ('deva' in self.input_type.lower() or 'image' in self.input_type.lower() or 'mask' in self.input_type.lower()):
                state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[:-1]
                goal_tmp = np.array([np.array(frame['goal'][:, :, 0:3]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[1:]
            elif 'devadepth' in self.input_type.lower() or 'rgbd' in self.input_type.lower():
                state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[1:]
            
            # 获取动作与奖励信息
            act_tmp = np.array([np.array(frame['action']) for frame in frames])[:-1].squeeze(axis=1)
            
            # 计算IoU奖励
            re_iou = np.array([
                self._reward_cal(state_tmp[i], goal_tmp[i])
                for i in range(len(state_tmp))
            ])
            # re_iou = np.array([np.array(frame['reward']) for frame in frames]).squeeze()[:-1]
            
            # 确保数据长度一致
            assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_iou.shape[0] == next_state_tmp.shape[0] and \
                next_state_tmp.shape[0] == act_tmp.shape[0], "数据长度不匹配！"
            
            # 将当前任务的数据添加到字典中
            if task_name not in task_data:
                task_data[task_name] = []
            
            # 遍历所有时间步，将数据加入 buffer（不转移到GPU，保持在CPU）
            for i in range(state_tmp.shape[0]):
                # 设置 done 标志
                if i % state_tmp.shape[0] == 0 and i > 0:
                    done = True
                else:
                    done = False
                
                # 记录任务和相应的训练样本
                task_data[task_name].append({
                    'state': np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1)),
                    'action': act_tmp[i],
                    'reward': np.array(re_iou[i]),
                    'next_state': np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1)),
                    'done': np.array(done),
                    'task_id': current_task_id,
                    'task_name': task_name
                })
                
                # 获取当前任务的嵌入
                task_embedding = None
                if self.task_embeddings_dict is not None and current_task_id in self.task_embeddings_dict:
                    task_embedding = self.task_embeddings_dict[current_task_id]
                
                # 保持在CPU，不转移到GPU
                state = torch.from_numpy(np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                action = torch.from_numpy(act_tmp[i]).float()
                reward = torch.from_numpy(np.array(re_iou[i])).float()
                next_state = torch.from_numpy(np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1))).float()
                done_tensor = torch.from_numpy(np.array(done)).float()
                
                self.add(state, action, reward, next_state, done_tensor, task_embedding)
        
        # print(f"Loaded {len(self.memory)} transitions from {len(selected_files)} files")
        # print(f"Current buffer size: {len(self.memory)}")
        # print(f"识别到的任务类型: {list(task_ids.keys())}")
        # print(f"任务ID映射: {task_ids}")

    def _load_all_data_to_memory(self):
        """一次性加载所有数据到内存"""
        print(f"Loading all {len(self.file_list)} files to memory...")
        self.all_transitions = []
        
        for file_idx, file_name in enumerate(self.file_list):
            if file_idx % 100 == 0:
                print(f"Loading file {file_idx}/{len(self.file_list)}: {file_name}")
                
            file_path = os.path.join(self.data_path, file_name)
            
            # 从文件名推断任务类型（4维：tracker2target_v1a2格式）
            # 先提取基本的x2y格式
            match = re.search(r'([a-z0-9]+2[a-z0-9]+)', file_name.lower())
            if not match:
                print(f"无法从文件名提取任务名(期望 X2Y 形式): {file_name}，跳过该文件")
                continue
            base_task_name = match.group(1)
            
            # 检查是否包含速度信息（v1a2格式）
            velocity_match = re.search(r'v(\d+)a(\d+)', file_name.lower())
            if velocity_match:
                # 🔥 使用辅助函数转换档位编号为实际速度值
                linear_v, angular_v = self._convert_velocity_to_actual(
                    velocity_match.group(1), 
                    velocity_match.group(2)
                )
                task_name = f"{base_task_name}_{linear_v}{angular_v}"
            else:
                # 兼容旧格式：level1/level2/level3 映射到实际速度值
                if 'level3' in file_name.lower():
                    task_name = f"{base_task_name}_v300a60"  # level3 → v300a60
                elif 'level2' in file_name.lower():
                    task_name = f"{base_task_name}_v200a30"  # level2 → v200a30
                elif 'level1' in file_name.lower():
                    task_name = f"{base_task_name}_v100a15"  # level1 → v100a15
                else:
                    # 默认为v100a15
                    task_name = f"{base_task_name}_v100a15"
            
            # 使用预生成任务嵌入目录中的任务名->ID映射
            if self.name_to_id_map is None:
                print("错误：未能加载任务名称到ID的映射。请检查 task_embeddings_dir。")
                continue
            if task_name not in self.name_to_id_map:
                print(f"警告：任务名 {task_name} 不在任务嵌入映射中，跳过该文件")
                continue
            current_task_id = int(self.name_to_id_map[task_name])
            
            # 加载 .pt 文件
            try:
                loaded_data = torch.load(file_path)
            except Exception as e:
                print(f"标准加载失败: {e}，尝试使用pickle_module=None")
                loaded_data = torch.load(file_path, pickle_module=None)
                
            if isinstance(loaded_data, list):
                frames = [frame for frame in loaded_data if frame]  
            else:
                frames = [loaded_data]
            
            # 处理图像数据
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
            # re_iou = np.array([np.array(frame['reward']) for frame in frames]).squeeze()[:-1]
            
            # 确保数据长度一致
            assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_iou.shape[0] == next_state_tmp.shape[0] and \
                next_state_tmp.shape[0] == act_tmp.shape[0], "数据长度不匹配！"
            
            # 获取当前任务的嵌入
            task_embedding = None
            if self.task_embeddings_dict is not None and current_task_id in self.task_embeddings_dict:
                task_embedding = self.task_embeddings_dict[current_task_id]
            
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
                
                self.all_transitions.append((state, action, reward, next_state, done_tensor, task_embedding))
        
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
                return 0
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

            iou = interArea / float(boxAArea + boxBArea - interArea)
            total_reward = iou
        else:
            total_reward = 0

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

    def add(self, state, action, reward, next_state, done, task_embedding=None):
        """Add a new experience to memory.
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            task_embedding: 任务嵌入向量，默认为None
        """
        e = self.experience(
            state,
            action,
            reward,
            next_state,
            done,
            task_embedding)
        self.memory.append(e)
    
    def sample(self):
        """从缓冲区中采样一批数据，无需指定任务ID"""
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
                task_embeddings = []
                
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
                    
                    # 提取任务嵌入信息
                    task_emb = experiences_test[0][5]  # task_embedding is at index 5
                    if task_emb is not None:
                        if not isinstance(task_emb, torch.Tensor):
                            task_emb = torch.tensor(task_emb, device=self.device)
                        task_embeddings.append(task_emb)
                    else:
                        task_embeddings.append(None)

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
                
                # 处理任务嵌入
                if all(emb is None for emb in task_embeddings):
                    return (states, actions, rewards, next_states, dones)
                else:
                    valid_embs = [emb for emb in task_embeddings if emb is not None]
                    if valid_embs:
                        emb_dim = valid_embs[0].shape[-1]
                        batch_task_embeddings = torch.zeros((self.batch_size, emb_dim), device=self.device)
                        for i, emb in enumerate(task_embeddings):
                            if emb is not None:
                                batch_task_embeddings[i] = emb
                        return (states, actions, rewards, next_states, dones, batch_task_embeddings)
                    else:
                        return (states, actions, rewards, next_states, dones)
            else:
                # 非LSTM模式：随机采样
                batch = random.sample(self.all_transitions, self.batch_size)
                
                states = torch.stack([e[0] for e in batch])
                actions = torch.stack([e[1] for e in batch])
                rewards = torch.stack([e[2] for e in batch])
                next_states = torch.stack([e[3] for e in batch])
                dones = torch.stack([e[4] for e in batch])
                
                # 处理任务嵌入
                task_embeddings = [e[5] for e in batch]
                
                # 转移到GPU
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                
                if all(emb is None for emb in task_embeddings):
                    return (states, actions, rewards, next_states, dones)
                else:
                    valid_embs = [emb for emb in task_embeddings if emb is not None]
                    if valid_embs:
                        emb_dim = valid_embs[0].shape[-1]
                        batch_task_embeddings = torch.zeros((self.batch_size, emb_dim), device=self.device)
                        for i, emb in enumerate(task_embeddings):
                            if emb is not None:
                                batch_task_embeddings[i] = emb
                        return (states, actions, rewards, next_states, dones, batch_task_embeddings)
                    else:
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
                    
                    # 提取任务嵌入信息，如果存在的话
                    if hasattr(experiences_test[0], 'task_embedding') and experiences_test[0].task_embedding is not None:
                        # 假设同一序列内任务嵌入相同，只取第一个
                        task_emb = experiences_test[0].task_embedding
                        if not isinstance(task_emb, torch.Tensor):
                            task_emb = torch.tensor(task_emb, device=self.device)
                    else:
                        # 如果没有任务嵌入，创建一个值为None的占位符
                        task_emb = None

                states = torch.stack([torch.stack(s) for s in states_test])
                actions = torch.stack([torch.stack(a) for a in actions_test])
                rewards = torch.stack([torch.stack(r) for r in rewards_test])
                next_states = torch.stack([torch.stack(n) for n in next_states_test])
                dones = torch.stack([torch.stack(d) for d in done_test])
                
                # 处理任务嵌入
                task_embeddings = []
                for i in range(self.batch_size):
                    if hasattr(self.memory[self.batch_id[i]*349+self.st_id[i]-self.lstm_seq_len], 'task_embedding'):
                        emb = self.memory[self.batch_id[i]*349+self.st_id[i]-self.lstm_seq_len].task_embedding
                        if emb is not None:
                            if not isinstance(emb, torch.Tensor):
                                emb = torch.tensor(emb, device=self.device)
                            task_embeddings.append(emb)
                        else:
                            task_embeddings.append(None)
                    else:
                        task_embeddings.append(None)
                
                # 在sample时转移到GPU
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                
                # 检查是否所有的任务嵌入都是None
                if all(emb is None for emb in task_embeddings):
                    # 如果都是None，则不返回任务嵌入信息
                    return (states, actions, rewards, next_states, dones)
                else:
                    # 否则，在结果元组中添加任务嵌入信息
                    # 对于None的任务嵌入，使用零向量替代
                    valid_embs = [emb for emb in task_embeddings if emb is not None]
                    if valid_embs:
                        emb_dim = valid_embs[0].shape[-1]
                        batch_task_embeddings = torch.zeros((self.batch_size, emb_dim), device=self.device)
                        for i, emb in enumerate(task_embeddings):
                            if emb is not None:
                                batch_task_embeddings[i] = emb
                        return (states, actions, rewards, next_states, dones, batch_task_embeddings)
                    else:
                        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)