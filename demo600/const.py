import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_ACTOR = 0.0001  # Actor的学习率
LR_CRITIC = 0.0001  # Critic的学习率
GAMMA = 0.95  # Discount Factor
MEMORY_SIZE = 102400000  # 经验回放缓存的容量
BATCH_SIZE = 512  # Minibatch的大小
TAU = 0.005  # 软更新系数

FOLLOWER_MAX_SPEED = 6
FOLLOWER_MAX_YAW_RATE = 90

LEADER_MAX_SPEED = 4

TANH_EM = 0.1

# 状态维度
STATE_DIM = 17

# 动作维度
ACTION_DIM = 4

COLLID = 4

DURATION = 1
SAFE_DURATION = 2

# 超参数
NUM_EPISODE = 25600
NUM_STEP = 64

# TD3新增超参数
POLICY_DELAY = 1  # 策略更新延迟，用于控制Actor网络的更新频率。

action_max = np.array([
    FOLLOWER_MAX_SPEED,        # r: 8
    FOLLOWER_MAX_SPEED,                    # theta: π
    FOLLOWER_MAX_SPEED,                    # phi: π
    FOLLOWER_MAX_YAW_RATE       # yaw_rate: π/2
])

# 按比例生成独立噪声参数
TARGET_NOISE_SCALE = 0.04
NOISE_CLIP_SCALE = 0.1
EXPLORATION_NOISE_SCALE = 0.04

TARGET_NOISE = TARGET_NOISE_SCALE * action_max
NOISE_CLIP = NOISE_CLIP_SCALE * action_max
EXPLORATION_NOISE = EXPLORATION_NOISE_SCALE * action_max

# 领航者和跟随者的名字
leader_name1 = "leader1"
leader_name2 = "leader2"

follower_name1 = "follower1"
follower_name2 = "follower2"
