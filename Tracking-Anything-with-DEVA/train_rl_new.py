import os
import time
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import random
import cv2
import re
torch.autograd.set_detect_anomaly = True
from agent_CNN_LSTM_with_CVAE_new import CQLSAC_CNN_LSTM_with_CVAE
from task_embedding_utils import load_task_embeddings

# 创建保存模型的目录
def create_save_dirs():
    dirs = ["saves", "logs"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"make dir: {dir_name}")

def get_bounding_box(mask_image):
    target_pixels = np.where(np.all(mask_image == [255, 255, 255], axis=-1))
    if len(target_pixels[0]) == 0:
        return None 
    y_min = np.min(target_pixels[0])
    y_max = np.max(target_pixels[0])
    x_min = np.min(target_pixels[1])
    x_max = np.max(target_pixels[1])
    return x_min, y_min, x_max, y_max

def reward_cal(state, goal):
    if state.max()==255:
        boxA = get_bounding_box(state)
        if boxA is None:
            return 0
        boxB = get_bounding_box(goal)
        if boxB is None:
            return 0
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

def get_config():
    parser = argparse.ArgumentParser(description='RL with CVAE for Task Inference')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-CVAE-full-", help="Run name")
    parser.add_argument("--buffer_path", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Maximal training dataset size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save_every", type=int, default=50, help="Saves the network every x epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for networks")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for RL")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="CQL weight")
    parser.add_argument("--target_action_gap", type=float, default=10, help="Target action gap")
    parser.add_argument("--with_lagrange", type=int, default=0, help="Whether to use Lagrange")
    parser.add_argument("--tau", type=float, default=5e-3, help="Tau parameter")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="LSTM sequence length")
    parser.add_argument("--lstm_out", type=int, default=64, help="LSTM output size")
    parser.add_argument("--lstm_layer", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--stack_frames", type=int, default=1, help="Number of stacked frames")
    parser.add_argument("--input_type", type=str, default='deva_cnn_lstm', help="Input type")
    parser.add_argument("--z_dim", type=int, default=16, help="CVAE latent dimension")

    parser.add_argument("--task_embeddings_dir", type=str, default='/root/autodl-tmp/CVAE/task', help="Path to pre-generated task embeddings")
    parser.add_argument("--use_task_encoding", type=int, default=1, help="Whether to use task encoding (1) or not (0)")
    parser.add_argument("--num_tasks", type=int, default=48, help="Number of tasks for tabular encoder")
    parser.add_argument("--load_all_to_memory", action='store_true', help="Load all data to memory first (requires more RAM but faster training)")
    
    # 🔥 新增：冻结和预训练权重参数
    parser.add_argument("--freeze_encoder", action='store_true', default=True, 
                       help="Freeze CNN+LSTM encoder to maintain feature consistency with CVAE training")
    parser.add_argument("--freeze_cvae", action='store_true', default=True,
                       help="Freeze CVAE to maintain latent variable z semantics")
    parser.add_argument("--pretrained_encoder_path", type=str, default=None,
                       help="Path to pretrained CNN+LSTM checkpoint from train_iou.py (e.g., saves/checkpoint_ep500.pt)")
    
    args = parser.parse_args()
    return args

def train_rl_with_cvae(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    create_save_dirs()

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device,
                          lstm_seq_len=config.lstm_seq_len, config=config, load_all_to_memory=config.load_all_to_memory)  # 设置不按任务ID过滤

    task_embeddings_dict, name_to_id_map = load_task_embeddings(config.task_embeddings_dir, device)
    if task_embeddings_dict is None:
        return
    buffer.set_data_path(config.buffer_path, task_embeddings_dict, name_to_id_map)
    num_tasks = max(len(name_to_id_map), config.num_tasks)
    print(f"mapping: {name_to_id_map}, {num_tasks} tasks")

    with wandb.init(project="CQL", name=config.run_name, config=config) as run:
        wandb.config.update({
            "num_tasks": num_tasks, "tasks": list(name_to_id_map.keys()) if name_to_id_map else "Unknown"})
        
        if 'deva' in config.input_type.lower() or 'image' in config.input_type.lower() or 'mask' in config.input_type.lower():
            state_channels = 3
        if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
            state_channels = 4
                
            if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
                agent = CQLSAC_CNN_LSTM_with_CVAE(
                    state_size=(state_channels, 64, 64),
                    action_size=2,
                    tau=config.tau,
                    hidden_size=config.hidden_size,
                    learning_rate=config.learning_rate,
                    temp=config.temperature,
                    with_lagrange=config.with_lagrange,
                    cql_weight=config.cql_weight,
                    target_action_gap=config.target_action_gap,
                    device=device,
                    stack_frames=config.stack_frames,
                    lstm_seq_len=config.lstm_seq_len,
                    lstm_layer=config.lstm_layer,
                    lstm_out=config.lstm_out,
                    z_dim=config.z_dim,
                    reward_size=1,
                    use_task_encoding=config.use_task_encoding,
                    num_tasks=num_tasks,
                    # 🔥 新增：冻结和预训练权重参数
                    freeze_encoder=config.freeze_encoder,
                    freeze_cvae=config.freeze_cvae,
                    pretrained_encoder_path=config.pretrained_encoder_path
                )
                
                print(f"\n📊 Agent Configuration:")
                print(f"   Freeze encoder: {config.freeze_encoder}")
                print(f"   Freeze CVAE: {config.freeze_cvae}")
                print(f"   Pretrained encoder: {config.pretrained_encoder_path if config.pretrained_encoder_path else 'None (random init)'}")
                
                if hasattr(agent, 'num_tasks'):
                    agent.num_tasks = num_tasks
                elif hasattr(agent, 'CNN_LSTM') and hasattr(agent.CNN_LSTM, 'num_tasks'):
                    agent.CNN_LSTM.num_tasks = num_tasks
                
        agent.to(device)
        wandb.watch(agent, log="gradients", log_freq=10)
        
        all_task_ids = list(name_to_id_map.values())
        print(f"ID: {all_task_ids}")
        
        steps = 0
        average10 = deque(maxlen=10)
        total_steps = 0
        
        checkpoint_dir = "saves"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        for i in range(1, config.episodes + 1):
            episode_start_time = time.time()
            episode_steps = 0
            rewards = 0
            if total_steps > 0:
                buffer.refresh_batch()
            while True:
                experiences = buffer.sample()
                agent.train()  
                results = agent.learn(experiences)

                train_q1, train_q2, policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha, *extra = results
                
                steps += 1
                if steps >= 200: 
                    episode_steps += 1
                    steps = 0
                    break
            
            episode_time = time.time() - episode_start_time
            
            average10.append(rewards)
            total_steps += episode_steps
            print(f"Episode: {i} | Policy Loss: {policy_loss} | Time: {episode_time:.2f}s")

            log_dict = {
                "Steps": total_steps,
                "train_q1": train_q1,
                "train_q2": train_q2,
                "Policy Loss": policy_loss,
                "Alpha Loss": alpha_loss,
                "Lagrange Alpha Loss": lagrange_alpha_loss,
                "CQL1 Loss": cql1_loss,
                "CQL2 Loss": cql2_loss,
                "Bellman error 1": bellmann_error1,
                "Bellman error 2": bellmann_error2,
                "Alpha": current_alpha,
                "Lagrange Alpha": lagrange_alpha,
                "Episode": i,
                "Buffer size": buffer.__len__(),
                "Episode time (s)": episode_time,
            }
            wandb.log(log_dict)
            
            if i % config.save_every == 0 or i == config.episodes:
                checkpoint_path = f"{checkpoint_dir}/CQL-SAC-CVAE_ep{i}.pt"
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'task_ids': name_to_id_map,
                    'num_tasks': num_tasks,
                    'episode': i,
                    'config': vars(config),
                }, checkpoint_path)
                wandb.save(checkpoint_path)
        
        final_model_path = f"{checkpoint_dir}/final_CQL-SAC-CVAE.pt"
        torch.save({
            'model_state_dict': agent.state_dict(),
            'task_ids': name_to_id_map,
            'num_tasks': num_tasks,
            'episode': config.episodes,
            'config': vars(config),
        }, final_model_path)
        wandb.save(final_model_path)

if __name__ == "__main__":
    config = get_config()
    train_rl_with_cvae(config)
