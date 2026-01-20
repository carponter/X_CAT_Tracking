import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import wandb
import cv2
import traceback
import gc
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset, DataLoader
import re

from models.generative_multidim import CVAE_MultiDim
from networks import CNN_LSTM


class FrozenFeatureExtractor(nn.Module):
    """
    Load CNN+LSTM encoder from CQL-SAC checkpoint and freeze it.
    Only extracts CNN features (576-dim) for this script.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.encoder = CNN_LSTM(
            state_size=(3, 64, 64), 
            action_size=2, 
            hidden_size=256, 
            stack_frames=1, 
            lstm_out=64, 
            lstm_layer=1
        ).to(device)
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        encoder_state = {
            k.replace("CNN_LSTM.", ""): v
            for k, v in ckpt.items()
            if k.startswith("CNN_LSTM.")
        }
        if len(encoder_state) == 0:
            raise RuntimeError("❌ checkpoint 中未找到 CNN_LSTM 权重")
        
        self.encoder.load_state_dict(encoder_state, strict=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.cnn_dim = 576
        self.output_dim = self.cnn_dim
    
    @torch.no_grad()
    def forward(self, x):
        """Extract CNN features"""
        feat = self.encoder.CNN_Simple(x)
        feat = feat.view(feat.size(0), -1)   # [B, 576]
        return feat
    
    def extract_cnn_features(self, x):
        """Alias for forward"""
        return self.forward(x)


def create_save_dirs():
    current_dir = os.getcwd()
    dirs = ["saves", "logs"]
    for dir_name in dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def get_bounding_box(mask_image):
    if len(mask_image.shape) > 2:
        if mask_image.shape[2] == 3:
            white_pixels = np.where(np.all(mask_image == 255, axis=2))
            if len(white_pixels[0]) == 0:
                return None
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
        else:
            mask_binary = mask_image[:, :, 0] > 0
            if not np.any(mask_binary):
                return None
            y_indices, x_indices = np.where(mask_binary)
            if len(y_indices) == 0:
                return None
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
    else:
        mask_binary = mask_image > 0
        if not np.any(mask_binary):
            return None
        y_indices, x_indices = np.where(mask_binary)
        if len(y_indices) == 0:
            return None
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def reward_cal(state_img, goal_img):
    mask_state_bbox = get_bounding_box(state_img)
    mask_goal_bbox = get_bounding_box(goal_img)
    if mask_state_bbox is None or mask_goal_bbox is None:
        return 0.0
    x1, y1, w1, h1 = mask_state_bbox
    x2, y2, w2, h2 = mask_goal_bbox
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def parse_task_from_filename(filename):
    """
    解析文件名，提取tracker、target、linear_v、angular_v等信息
    例如: track_train_colorgoal_human2human_v1a1_0002-350_tag2.pt
    更灵活的匹配模式，支持各种格式
    """
    filename_lower = filename.lower()
    match_tracker_target = re.search(r'([a-z0-9]+)2([a-z0-9]+)', filename_lower) # match label
    if not match_tracker_target:
        return None, None, None, None, None, None
    
    tracker = match_tracker_target.group(1)
    target = match_tracker_target.group(2)
    match_velocities = re.search(r'v(\d+)a(\d+)', filename_lower)
    if not match_velocities:
        linear_v = "v1"
        angular_v = "a1"
    else:
        linear_v = f"v{match_velocities.group(1)}"
        angular_v = f"a{match_velocities.group(2)}"
    v_map = {'v1': 100.0, 'v2': 200.0, 'v3': 300.0, 'v4': 400.0}
    a_map = {'a1': 15.0, 'a2': 30.0, 'a3': 60.0, 'a4': 90.0}
    linear_v_max = v_map.get(linear_v, 100.0)
    angular_v_max = a_map.get(angular_v, 15.0)
    return tracker, target, linear_v, angular_v, linear_v_max, angular_v_max


class AllDataDataset(Dataset):
    def __init__(self, data_path, config, feature_extractor=None):
        self.data_path = data_path
        self.config = config
        self.feature_extractor = feature_extractor
        self.use_frozen_features = feature_extractor is not None

        self.tracker_ids = {}
        self.target_ids = {}
        self.linear_v_ids = {}
        self.angular_v_ids = {}
        print("loading...")

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_tracker_ids = []
        all_target_ids = []
        all_linear_v_ids = []
        all_angular_v_ids = []
        all_linear_v_maxs = []
        all_angular_v_maxs = []
        
        files = sorted(os.listdir(self.data_path))
        print(f"find {len(files)} files")
        
        for idx, fname in enumerate(files):
            if idx % 100 == 0:
                print(f"  processing: {idx}/{len(files)}")
            file_path = os.path.join(self.data_path, fname)
            if not os.path.isfile(file_path):
                continue
            
            tracker, target, linear_v, angular_v, linear_v_max, angular_v_max = parse_task_from_filename(fname)
            if tracker is None or target is None:
                if idx < 5:  # Show first few failed parses for debugging
                    print(f"  [DEBUG] Failed to parse: {fname}")
                continue

            if tracker not in self.tracker_ids:
                self.tracker_ids[tracker] = len(self.tracker_ids)
            if target not in self.target_ids:
                self.target_ids[target] = len(self.target_ids)
            if linear_v not in self.linear_v_ids:
                self.linear_v_ids[linear_v] = len(self.linear_v_ids)
            if angular_v not in self.angular_v_ids:
                self.angular_v_ids[angular_v] = len(self.angular_v_ids)

            try:
                loaded_data = torch.load(file_path, weights_only=False)
                frames = [frame for frame in loaded_data] if isinstance(loaded_data, list) else [loaded_data]
                T = len(frames)
                if T < 2:
                    del loaded_data, frames
                    gc.collect()
                    continue
                
                for i in range(T - 1):
                    src = frames[i]['mask'][:, :, 0:3]  # deva_mask as observation
                    resized = cv2.resize(src, (64, 64))
                    chw = np.transpose(resized, (2, 0, 1)).astype(np.float32)
                    all_states.append(chw)

                    next_src = frames[i + 1]['mask'][:, :, 0:3]
                    next_resized = cv2.resize(next_src, (64, 64))
                    next_chw = np.transpose(next_resized, (2, 0, 1)).astype(np.float32)
                    all_next_states.append(next_chw)
                    
                    action = np.array(frames[i]['action']).squeeze().astype(np.float32)
                    all_actions.append(action)
                    
                    state_img = np.array(frames[i]['mask'][:, :, 0:3])
                    goal_img = np.array(frames[i]['goal'][:, :, 0:3])
                    reward_val = reward_cal(state_img, goal_img)
                    all_rewards.append(np.array([reward_val], dtype=np.float32))
                    
                    all_tracker_ids.append(self.tracker_ids[tracker])
                    all_target_ids.append(self.target_ids[target])
                    all_linear_v_ids.append(self.linear_v_ids[linear_v])
                    all_angular_v_ids.append(self.angular_v_ids[angular_v])
                    all_linear_v_maxs.append(linear_v_max)
                    all_angular_v_maxs.append(angular_v_max)
                
                del loaded_data, frames
                gc.collect()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"\n✅ data loading completed!")
        print(f"numbers: {len(all_states)}, trackers: {sorted(self.tracker_ids.keys())}, targets: {sorted(self.target_ids.keys())}, linear V: {sorted(self.linear_v_ids.keys())}, angular V: {sorted(self.angular_v_ids.keys())}")
        
        # Check if any data was loaded
        if len(all_states) == 0:
            raise RuntimeError(f"❌ No valid data found in {self.data_path}. Please check:\n"
                             f"   1. File path is correct\n"
                             f"   2. Files match the expected naming pattern (e.g., *2* for tracker2target)\n"
                             f"   3. Files contain valid 'mask', 'action', 'goal' keys")
        
        # to numpy arrays
        self.states = np.array(all_states, dtype=np.float32)
        self.next_states = np.array(all_next_states, dtype=np.float32)
        self.actions = np.array(all_actions, dtype=np.float32)
        self.rewards = np.array(all_rewards, dtype=np.float32)
        self.tracker_ids_arr = np.array(all_tracker_ids, dtype=np.int64)
        self.target_ids_arr = np.array(all_target_ids, dtype=np.int64)
        self.linear_v_ids_arr = np.array(all_linear_v_ids, dtype=np.int64)
        self.angular_v_ids_arr = np.array(all_angular_v_ids, dtype=np.int64)
        self.linear_v_maxs_arr = np.array(all_linear_v_maxs, dtype=np.float32)
        self.angular_v_maxs_arr = np.array(all_angular_v_maxs, dtype=np.float32)
        
        # normalize actions to [-1, 1]
        min_val = np.array([-30, -100]).astype(np.float32)
        max_val = np.array([30, 100]).astype(np.float32)
        normalized_actions = ((self.actions - min_val) / (max_val - min_val)).astype(np.float32)
        self.actions = (2 * normalized_actions - 1).astype(np.float32)
        
        # Extract CNN features if needed
        if self.use_frozen_features and self.feature_extractor is not None:
            print(f"\n Extracting CNN features from {len(self.states)} samples...")
            self.state_features = []
            self.next_state_features = []
            batch_size = 256
            num_batches = (len(self.states) + batch_size - 1) // batch_size
            print(f"   Total batches to process: {num_batches}")
            
            for i in range(num_batches):               
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.states))

                states_batch = torch.from_numpy(self.states[start_idx:end_idx]).to(self.feature_extractor.device)
                with torch.no_grad():
                    features_batch = self.feature_extractor.extract_cnn_features(states_batch).cpu().numpy()
                self.state_features.append(features_batch)
                
                next_states_batch = torch.from_numpy(self.next_states[start_idx:end_idx]).to(self.feature_extractor.device)
                with torch.no_grad():
                    next_features_batch = self.feature_extractor.extract_cnn_features(next_states_batch).cpu().numpy()
                self.next_state_features.append(next_features_batch)
                
                if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                    progress = (i + 1) / num_batches * 100
                    print(f"   Progress: {i+1}/{num_batches} ({progress:.1f}%) - Processed {end_idx}/{len(self.states)} samples")
            
            print(f" Concatenating features...")
            self.state_features = np.concatenate(self.state_features, axis=0)
            self.next_state_features = np.concatenate(self.next_state_features, axis=0)
            print(f"✅ Feature extraction completed! Shape: {self.state_features.shape}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        if self.use_frozen_features:
            # use CNN feature
            return (
                torch.from_numpy(self.state_features[idx]).float(),
                torch.from_numpy(self.actions[idx]).float(),
                torch.from_numpy(self.rewards[idx]).float(),
                torch.from_numpy(self.next_state_features[idx]).float(),
                torch.tensor(self.tracker_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.target_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.linear_v_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.angular_v_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.linear_v_maxs_arr[idx], dtype=torch.float),
                torch.tensor(self.angular_v_maxs_arr[idx], dtype=torch.float)
            )
        else:
            # use raw observation
            return (
                torch.from_numpy(self.states[idx]).float() / 255.0,  
                torch.from_numpy(self.actions[idx]).float(),
                torch.from_numpy(self.rewards[idx]).float(),
                torch.from_numpy(self.next_states[idx]).float() / 255.0,
                torch.tensor(self.tracker_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.target_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.linear_v_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.angular_v_ids_arr[idx], dtype=torch.long),
                torch.tensor(self.linear_v_maxs_arr[idx], dtype=torch.float),
                torch.tensor(self.angular_v_maxs_arr[idx], dtype=torch.float)
            )


def get_cvae_loss(model, obs, action, reward, next_obs, 
                  tracker_id, target_id, linear_v_id, angular_v_id,
                  beta=0.1):
    """
    计算CVAE loss (使用 CVAE_MultiDim 的 full encoding 模式)
    """
    # Forward pass
    mean, logvar, z_sample = model.forward_encoder(
        obs, action, reward, next_obs,
        tracker_id=tracker_id, target_id=target_id,
        linear_v_id=linear_v_id, angular_v_id=angular_v_id
    )
    
    # Compute KL divergence
    kl_loss = model.compute_kl_divergence(mean, logvar).mean()
    
    # Compute reconstruction loss
    recon_loss_s, recon_loss_r, unscaled_obs, unscaled_rew, ref_obs, ref_rew = \
        model.losses(obs, action, reward, next_obs, z_sample)
    
    obs_loss = recon_loss_s if recon_loss_s is not None else 0.0
    rew_loss = recon_loss_r if recon_loss_r is not None else 0.0
    
    # Total loss
    total_loss = obs_loss + rew_loss + beta * kl_loss
    
    return (kl_loss, obs_loss, rew_loss, 
            unscaled_obs if unscaled_obs is not None else 0.0,
            unscaled_rew if unscaled_rew is not None else 0.0,
            ref_obs if ref_obs is not None else 0.0,
            ref_rew if ref_rew is not None else 0.0,
            total_loss)


def get_config():
    parser = argparse.ArgumentParser(description='CVAE Training (Full Encoding)')
    parser.add_argument("--run_name", type=str, default="CVAE-full-encoding-", help="run_name")
    parser.add_argument("--data_path", type=str, 
                        default="C:\\Offline_RL_Active_Tracking-master\\Offline_RL_Active_Tracking-master\\data\\train_data", 
                        help="data_path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=16, help="z embedding dimension (must be divisible by 2 or 4)")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--wandb_log_interval", type=int, default=1)
    
    # CVAE settings
    parser.add_argument("--tracker_fusion", type=str, default='avgpooling', 
                        choices=['cat', 'avgpooling'],
                        help="Fusion mode: 'cat' (z_dim divisible by 4) or 'avgpooling' (z_dim divisible by 2)")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence weight")
    
    # Model settings
    parser.add_argument("--predict_state_difference", action='store_true', default=False)
    parser.add_argument("--output_variance", type=str, default='output', 
                        choices=['zero', 'parameter', 'output', 'output_raw', 'reference'])
    parser.add_argument("--logvar_min", type=float, default=-15.0)
    parser.add_argument("--logvar_max", type=float, default=2.0)
    parser.add_argument("--merge_reward_next_state", action='store_true', default=False)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="gradient clipping max norm")
    
    # Feature extraction
    parser.add_argument("--use_cnn_features", action="store_true", default=False, 
                        help="Use CNN features (576-dim) instead of raw obs (12288-dim)")
    parser.add_argument("--frozen_cnn_lstm_path", type=str, 
                        default="C:\\Offline_RL_Active_Tracking-master\\Offline_RL_Active_Tracking-master\\trained_models\\CQL-SAC-base-CQL-SAC1000.pth", 
                        help="pretrained weights path")
    
    parser.add_argument("--use_gpu", action="store_true", default=True)
    args = parser.parse_args()
    return args


def main():
    config = get_config()
    create_save_dirs()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
    print(f"\n️  Device: {device}")
    
    print(f"\n Initializing WandB...")
    try:
        wandb.init(project="cvae-full-encoding", name=config.run_name, config=vars(config))
        print(f"✅ WandB initialized")
    except Exception as e:
        print(f"⚠️  WandB initialization failed: {e}")
        print(f"   Continuing without WandB logging...")
    
    # Load feature extractor if using CNN features
    feature_extractor = None
    if config.use_cnn_features:
        print(f"\n Loading frozen CNN feature extractor...")
        feature_extractor = FrozenFeatureExtractor(
            checkpoint_path=config.frozen_cnn_lstm_path,
            device=device
        )
        print(f"✅ Feature extractor loaded (output_dim={feature_extractor.output_dim})")
    else:
        print(f"\n Using raw observation as state (no feature extractor)")
    
    # Load dataset
    print(f"\n Loading dataset...")
    train_dataset = AllDataDataset(config.data_path, config, feature_extractor)
    
    # Save ID mappings
    current_dir = os.getcwd()
    dim_maps = {
        'tracker_ids': train_dataset.tracker_ids,
        'target_ids': train_dataset.target_ids,
        'linear_v_ids': train_dataset.linear_v_ids,
        'angular_v_ids': train_dataset.angular_v_ids
    }
    dim_maps_file = os.path.join(current_dir, "saves", "maps_full_encoding.pt")
    torch.save(dim_maps, dim_maps_file)
    
    # Determine state size
    if feature_extractor is not None:
        flat_state_size = feature_extractor.output_dim  # 576
        state_repr = "cnn_features_576dim"
    else:
        flat_state_size = 3 * 64 * 64  # 12288
        state_repr = "raw_obs_12288dim"
    
    # Update wandb config
    wandb.config.update({
        "model_type": f"CVAE_MultiDim (tracker_fusion={config.tracker_fusion})",
        "state_representation": state_repr,
        "num_trackers": len(train_dataset.tracker_ids),
        "num_targets": len(train_dataset.target_ids),
        "num_linear_v": len(train_dataset.linear_v_ids),
        "num_angular_v": len(train_dataset.angular_v_ids),
        "total_samples": len(train_dataset),
        "using_frozen_cnn": feature_extractor is not None
    }, allow_val_change=True)
    
    # Create CVAE_MultiDim model
    print(f"\n易 Creating CVAE_MultiDim model (state_size={flat_state_size})...")
    model = CVAE_MultiDim(
        hidden_size=config.hidden_size,
        num_hidden_layers=2,
        z_dim=config.z_dim,
        action_size=2,
        state_size=flat_state_size,
        reward_size=1,
        num_trackers=len(train_dataset.tracker_ids),
        num_targets=len(train_dataset.target_ids),
        num_linear_velocities=len(train_dataset.linear_v_ids),
        num_angular_velocities=len(train_dataset.angular_v_ids),
        tracker_fusion=config.tracker_fusion,
        predict_state_difference=config.predict_state_difference,
        output_variance=config.output_variance,
        merge_reward_next_state=config.merge_reward_next_state,
        logvar_min=config.logvar_min,
        logvar_max=config.logvar_max,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"\n Creating DataLoader (batch_size={config.batch_size})...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False, 
        drop_last=True
    )
    print(f"✅ DataLoader created. Total batches per epoch: {len(train_loader)}")
    
    model.train()
    global_step = 0
    
    print(f"\n Starting training for {config.n_epochs} epochs...")
    for epoch in range(config.n_epochs):
        epoch_start_time = time.time()
        epoch_losses = defaultdict(float)
        n_batches = 0
        
        print(f"\n Epoch {epoch+1}/{config.n_epochs}")
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx == 0:
                print(f"   Processing first batch...")
            
            obs, action, reward, next_obs, tracker_id, target_id, linear_v_id, angular_v_id, linear_v_max, angular_v_max = batch_data
            
            if batch_idx == 0:
                print(f"   Moving data to GPU...")
            
            obs = obs.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_obs = next_obs.to(device)
            tracker_id = tracker_id.to(device)
            target_id = target_id.to(device)
            linear_v_id = linear_v_id.to(device)
            angular_v_id = angular_v_id.to(device)
            
            # Flatten obs and next_obs if using raw images (not frozen features)
            if feature_extractor is None:
                obs = obs.view(obs.size(0), -1)  # [batch, 3, 64, 64] -> [batch, 12288]
                next_obs = next_obs.view(next_obs.size(0), -1)
            
            if batch_idx == 0:
                print(f"   Computing loss...")
            
            # Compute loss
            kl_loss, obs_loss, rew_loss, unscaled_obs, unscaled_rew, ref_obs, ref_rew, total_loss = \
                get_cvae_loss(model, obs, action, reward, next_obs, 
                            tracker_id, target_id, linear_v_id, angular_v_id,
                            beta=config.beta)
            
            if batch_idx < 3:
                print(f"   [Batch {batch_idx}] total_loss: {total_loss.item():.6f}, kl: {kl_loss:.6f}, obs: {obs_loss:.6f}, rew: {rew_loss:.6f}")
            
            # Backward with gradient clipping
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪，避免异常值带偏
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            
            optimizer.step()

            if batch_idx == 0:
                print(f"   First batch completed!")
            
            # Accumulate losses
            epoch_losses["kl"] += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            epoch_losses["obs"] += obs_loss.item() if isinstance(obs_loss, torch.Tensor) else obs_loss
            epoch_losses["rew"] += rew_loss.item() if isinstance(rew_loss, torch.Tensor) else rew_loss
            epoch_losses["total"] += total_loss.item()
            n_batches += 1
            global_step += 1
            
            # Progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)} - loss: {total_loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start_time
        avg_losses = {k: v/n_batches for k, v in epoch_losses.items()}
        
        print(f" Epoch {epoch+1}/{config.n_epochs} Summary:")
        print(f"   kl_loss: {avg_losses['kl']:.6f}, obs_loss: {avg_losses['obs']:.6f}, rew_loss: {avg_losses['rew']:.6f}")
        print(f"   total_loss: {avg_losses['total']:.6f}, time: {epoch_time:.2f}s")
       
        # Log to wandb
        if (epoch + 1) % config.wandb_log_interval == 0:
            wandb.log({
                "epoch": epoch+1,
                "epoch_time": epoch_time,
                "avg_kl_loss": avg_losses["kl"],
                "avg_obs_loss": avg_losses["obs"],
                "avg_rew_loss": avg_losses["rew"],
                "avg_total_loss": avg_losses["total"]
            })
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0 or epoch == config.n_epochs - 1:
            save_path = os.path.join(current_dir, "saves", f"cvae_full_ep{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'dim_maps': dim_maps,
                'avg_losses': avg_losses,
                'config': vars(config),
                'state_size': flat_state_size,
                'model_type': 'CVAE_MultiDim',
                'tracker_fusion': config.tracker_fusion,
                'feature_type': 'cnn_features' if feature_extractor else 'raw_obs'
            }, save_path)
            print(f" Checkpoint saved: {save_path}")
    
    wandb.finish()
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
