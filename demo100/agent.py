import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
import torch.optim as optim

from const import *


# 设置全局随机数种子
def set_global_seed(seed):
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False
# 在代码的入口处设置种子
set_global_seed(137) 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.tanh_in = torch.tanh

        self.tanh_em = nn.Parameter(torch.ones(state_dim) * TANH_EM)

        # 全连接层（添加层归一化）
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )
        # 应用He初始化（仅针对Linear层）
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh_in(self.tanh_em * x)

        x = self.fc(x)
        
        # 对每个维度进行缩放和平移
        x0 = torch.tanh(x[:, 0]) * FOLLOWER_MAX_SPEED
        x1 = torch.tanh(x[:, 1]) * FOLLOWER_MAX_SPEED
        x2 = torch.tanh(x[:, 2]) * FOLLOWER_MAX_SPEED
        x3 = torch.tanh(x[:, 3]) * FOLLOWER_MAX_YAW_RATE
        
        # 合并为一个张量
        x = torch.stack([x0, x1, x2, x3], dim=1)
        
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.tanh_in = torch.tanh

        self.tanh_em = nn.Parameter(torch.ones(state_dim + action_dim) * TANH_EM)

        # 全连接层（添加层归一化）
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 160),
            nn.SiLU(),
            nn.Linear(160, 320),
            nn.SiLU(),
            nn.Linear(320, 320),
            nn.SiLU(),
            nn.Linear(320, 160),
            nn.SiLU(),
            nn.Linear(160, 1),  # 输出层不添加归一化
        )
        # 应用He初始化
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.tanh_in(self.tanh_em * x)
        x = self.fc(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        (
            state_samples,
            action_samples,
            reward_samples,
            next_state_samples,
            done_samples,
        ) = zip(*samples)
        return (
            state_samples,
            action_samples,
            reward_samples,
            next_state_samples,
            done_samples,
        )


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_delay,
        target_noise,
        noise_clip,
        exploration_noise,
    ):
        set_global_seed(137)
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # 初始化Critic网络
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC)

        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC)

        # 经验回放
        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

        # TD3超参数
        self.policy_delay = policy_delay
        self.target_noise = torch.FloatTensor(target_noise).to(device)
        self.noise_clip = torch.FloatTensor(noise_clip).to(device)
        self.exploration_noise = np.array(exploration_noise)  # 用于探索噪声

        # 更新计数器
        self.total_it = 0

    def get_action(self, state, add_noise):
        # 转化为符合要求的张量
        state = np.array(state)
        state_tensor = torch.FloatTensor(state).to(device=device)
        state_tensor = state_tensor.unsqueeze(0)
        
        # 输入获得结果
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze().cpu().detach().numpy()  # 转换为一位的ndarray返回，元素类型为float32

        if add_noise:
            min_vals = np.array([-FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_YAW_RATE])
            max_vals = np.array([FOLLOWER_MAX_SPEED, FOLLOWER_MAX_SPEED, FOLLOWER_MAX_SPEED, FOLLOWER_MAX_YAW_RATE])
            noise = np.random.normal(0, self.exploration_noise)
            action_with_noise = action + noise
            action = np.clip(action_with_noise, min_vals, max_vals)
        
        return action

    def update_networks(self):
        if len(self.replay_buffer.buffer) < BATCH_SIZE:
            return 0.0, 0.0, 0.0

        samples = self.replay_buffer.sample(BATCH_SIZE)
        (
            state_samples,
            action_samples,
            reward_samples,
            next_state_samples,
            done_samples,
        ) = samples

        state_samples = np.array(state_samples)
        action_samples = np.array(action_samples)
        reward_samples = np.array(reward_samples).reshape(-1, 1)
        next_state_samples = np.array(next_state_samples)
        done_samples = np.array(done_samples).reshape(-1, 1)

        # 转化为张量
        state_samples = torch.FloatTensor(state_samples).to(device=device)
        action_samples = torch.FloatTensor(action_samples).to(device=device)
        reward_samples = torch.FloatTensor(reward_samples).to(device=device)
        next_state_samples = torch.FloatTensor(next_state_samples).to(device=device)
        done_samples = torch.FloatTensor(done_samples).to(device=device)

        # 更新Critic网络
        next_action = self.actor_target(next_state_samples)

        noise = torch.randn_like(next_action) * self.target_noise
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action = next_action + noise

        # 限制每个维度的范围
        min_vals = torch.tensor([-FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_SPEED, -FOLLOWER_MAX_YAW_RATE], device=device)
        max_vals = torch.tensor([FOLLOWER_MAX_SPEED, FOLLOWER_MAX_SPEED, FOLLOWER_MAX_SPEED, FOLLOWER_MAX_YAW_RATE], device=device)
        next_action = torch.clamp(next_action, min_vals, max_vals)

        target_Q1 = self.critic1_target(next_state_samples, next_action)
        target_Q2 = self.critic2_target(next_state_samples, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward_samples + (GAMMA * target_Q * (1 - done_samples))

        current_Q1 = self.critic1(state_samples, action_samples)
        current_Q2 = self.critic2(state_samples, action_samples)

        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optimizer.step()
        self.soft_update(self.critic1, self.critic1_target, TAU)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        self.soft_update(self.critic2, self.critic2_target, TAU)

        # 延迟更新Actor网络和目标网络
        self.total_it += 1
        if self.total_it % self.policy_delay == 0:
            # 更新Actor网络
            pred_action = self.actor(state_samples)
            actor_loss = -self.critic1(state_samples, pred_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target, TAU)

            return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

        return 0.0, critic1_loss.item(), critic2_loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
                # Target networks
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic1_target_state_dict": self.critic1_target.state_dict(),
                "critic2_target_state_dict": self.critic2_target.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
        # Target networks
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
