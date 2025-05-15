"""
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "Manual",
  "Vehicles": {
    "leader": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": 0,
      "Yaw": 0
    },
    "follower1": {
      "VehicleType": "SimpleFlight",
      "X": 2,
      "Y": 0,
      "Z": 0,
      "Yaw": 0
    },
    "follower2": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 2,
      "Z": 0,
      "Yaw": 0
    },
    "follower3": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": 2,
      "Yaw": 0
    }
  }
}
"""

import airsim
from agent import TD3Agent
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import torch
from utils import *
import joblib
from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep, splev

from const import *

# =========================================================================================================


def generate_3d_b_spline(
    drone_position,
    num_control_points=256,
    num_output_points=6400,
    degree=3,
    x_range=(-25, 25),
    y_range=(-25, 25),
    z_range=(-25, 25),
):
    # 随机生成控制点
    control_points = np.random.rand(num_control_points, 3)
    control_points[:, 0] = control_points[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    control_points[:, 1] = control_points[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    control_points[:, 2] = control_points[:, 2] * (z_range[1] - z_range[0]) + z_range[0]

    # 将第一个控制点调整到无人机当前位置附近
    control_points[0] = drone_position

    # 使用scipy进行B样条插值
    tck, u = splprep(control_points.T, s=0, k=degree, per=True)
    u_new = np.linspace(0, 1, num_output_points)
    curve_points = np.array(splev(u_new, tck)).T

    return curve_points


def generate_max_speed_noise(max_val, min_val):
    # 计算 noise 的合法范围
    upper_bound = max_val - min_val

    # 如果 max 小于等于 min，返回 0
    if upper_bound <= 0:
        return 0.0

    # 在 [0, upper_bound] 范围内生成随机数
    noise = np.random.uniform(0, upper_bound)
    return noise


# =========================================================================================================


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

# 初始化TD3Agent
agent = TD3Agent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    policy_delay=POLICY_DELAY,
    target_noise=TARGET_NOISE,
    noise_clip=NOISE_CLIP,
    exploration_noise=EXPLORATION_NOISE,
)

# 用于记录每个回合的奖励
rewards = []
collids = []
collid_count = 0.0

episode_start = 4250
checkpoint = f"model_ep_{episode_start}.pth"
rewards_file = f"rewards_{episode_start}.txt"
buffer_file = f"buffer_{episode_start}.joblib"
collid_log = f"collids_{episode_start}.txt"

if episode_start == 0:
    print("episode_start = 0")
    # agent.load_checkpoint(checkpoint)
    # print(f'成功加载{checkpoint}')
else:
    agent.load_checkpoint(checkpoint)
    with open(rewards_file, "r") as f:
        for line in f:
            rewards.append(float(line.strip()))

    with open(collid_log, "r") as f:
        for line in f:
            collids.append(float(line.strip()))
            collid_count = collids[-1]

    agent.replay_buffer.buffer = joblib.load(buffer_file)
    print(f"成功加载{checkpoint}")
    print(f"从episode={episode_start}继续训练")

# 初始化AirSim客户端
client = airsim.MultirotorClient()
client.confirmConnection()


# 开始训练
for episode_i in range(episode_start, NUM_EPISODE):
    client.reset()

    # leader
    init_x_leader, init_y_leader, init_z_leader = 0, 0, 0
    init_yaw_leader = random.uniform(-math.pi, math.pi)
    new_pose_leader = airsim.Pose(
        position_val=airsim.Vector3r(init_x_leader, init_y_leader, init_z_leader),
        orientation_val=airsim.to_quaternion(0, 0, init_yaw_leader),
    )
    client.simSetVehiclePose(new_pose_leader, True, vehicle_name=leader_name)

    # follower1
    init_r_follower1 = rand_op(1, 12)
    init_theta_follower1 = random.uniform(-math.pi / 2, math.pi / 2)
    init_phi_follower1 = random.uniform(-math.pi, math.pi)
    init_x_follower1, init_y_follower1, init_z_follower1 = spherical_to_cartesian(
        init_r_follower1, init_theta_follower1, init_phi_follower1
    )
    init_yaw_follower1 = random.uniform(-math.pi, math.pi)
    new_pose_follower1 = airsim.Pose(
        position_val=airsim.Vector3r(
            init_x_follower1, init_y_follower1, init_z_follower1
        ),
        orientation_val=airsim.to_quaternion(0, 0, init_yaw_follower1),
    )
    client.simSetVehiclePose(new_pose_follower1, True, vehicle_name=follower_name1)

    # follower2
    init_r_follower2 = rand_op(1, 12)
    init_theta_follower2 = random.uniform(-math.pi / 2, math.pi / 2)
    init_phi_follower2 = random.uniform(-math.pi, math.pi)
    init_x_follower2, init_y_follower2, init_z_follower2 = spherical_to_cartesian(
        init_r_follower2, init_theta_follower2, init_phi_follower2
    )
    init_yaw_follower2 = random.uniform(-math.pi, math.pi)
    new_pose_follower2 = airsim.Pose(
        position_val=airsim.Vector3r(
            init_x_follower2, init_y_follower2, init_z_follower2
        ),
        orientation_val=airsim.to_quaternion(0, 0, init_yaw_follower2),
    )
    client.simSetVehiclePose(new_pose_follower2, True, vehicle_name=follower_name2)

    # follower3
    init_r_follower3 = rand_op(1, 12)
    init_theta_follower3 = random.uniform(-math.pi / 2, math.pi / 2)
    init_phi_follower3 = random.uniform(-math.pi, math.pi)
    init_x_follower3, init_y_follower3, init_z_follower3 = spherical_to_cartesian(
        init_r_follower3, init_theta_follower3, init_phi_follower3
    )
    init_yaw_follower3 = random.uniform(-math.pi, math.pi)
    new_pose_follower3 = airsim.Pose(
        position_val=airsim.Vector3r(
            init_x_follower3, init_y_follower3, init_z_follower3
        ),
        orientation_val=airsim.to_quaternion(0, 0, init_yaw_follower3),
    )
    client.simSetVehiclePose(new_pose_follower3, True, vehicle_name=follower_name3)

    # 初始化无人机
    client.enableApiControl(True, vehicle_name=leader_name)
    client.armDisarm(True, vehicle_name=leader_name)
    client.enableApiControl(True, vehicle_name=follower_name1)
    client.armDisarm(True, vehicle_name=follower_name1)
    client.enableApiControl(True, vehicle_name=follower_name2)
    client.armDisarm(True, vehicle_name=follower_name2)
    client.enableApiControl(True, vehicle_name=follower_name3)
    client.armDisarm(True, vehicle_name=follower_name3)
    client.takeoffAsync(vehicle_name=leader_name)
    client.takeoffAsync(vehicle_name=follower_name1)
    client.takeoffAsync(vehicle_name=follower_name2)
    client.takeoffAsync(vehicle_name=follower_name3).join()

    leader_state = client.getMultirotorState(vehicle_name=leader_name)
    leader_position = leader_state.kinematics_estimated.position
    _, _, leader_yaw = airsim.to_eularian_angles(
        leader_state.kinematics_estimated.orientation
    )
    leader_linear_velocity = leader_state.kinematics_estimated.linear_velocity
    leader_angular_velocity = leader_state.kinematics_estimated.angular_velocity
    leader_angular_velocity_z = leader_angular_velocity.z_val

    # follower1
    ideal_r_1 = rand_op(2, 6)
    ideal_theta_1 = random.uniform(-math.pi / 2, math.pi / 2)
    ideal_phi_1 = random.uniform(-math.pi, math.pi)
    ideal_x_1, ideal_y_1, ideal_z_1 = spherical_to_cartesian(
        ideal_r_1, ideal_theta_1, ideal_phi_1
    )
    ideal_orientation_difference_1 = random.uniform(-math.pi, math.pi)

    follower1_state = client.getMultirotorState(vehicle_name=follower_name1)
    follower1_position = follower1_state.kinematics_estimated.position
    _, _, follower1_yaw = airsim.to_eularian_angles(
        follower1_state.kinematics_estimated.orientation
    )
    follower1_linear_velocity = follower1_state.kinematics_estimated.linear_velocity
    follower1_angular_velocity = follower1_state.kinematics_estimated.angular_velocity
    follower1_angular_velocity_z = follower1_angular_velocity.z_val
    ideal_follower1_x, ideal_follower1_y, ideal_follower1_z = ideal_position(
        (leader_position.x_val, leader_position.y_val, leader_position.z_val),
        leader_yaw,
        ideal_r_1,
        ideal_theta_1,
        ideal_phi_1,
    )
    ideal_follower1_yaw = compute_follower_yaw(
        leader_yaw, ideal_orientation_difference_1
    )
    state1 = np.array(
        [
            ideal_x_1,
            ideal_y_1,
            ideal_z_1,
            leader_yaw,
            leader_linear_velocity.x_val,
            leader_linear_velocity.y_val,
            leader_linear_velocity.z_val,
            leader_angular_velocity_z,
            follower1_yaw,
            follower1_linear_velocity.x_val,
            follower1_linear_velocity.y_val,
            follower1_linear_velocity.z_val,
            follower1_angular_velocity_z,
            follower1_position.x_val - ideal_follower1_x,
            follower1_position.y_val - ideal_follower1_y,
            follower1_position.z_val - ideal_follower1_z,
            follower1_yaw - ideal_follower1_yaw,
        ],
        dtype=np.float32,
    )

    # follower2
    ideal_r_2 = rand_op(2, 6)
    ideal_theta_2 = random.uniform(-math.pi / 2, math.pi / 2)
    ideal_phi_2 = random.uniform(-math.pi, math.pi)
    ideal_x_2, ideal_y_2, ideal_z_2 = spherical_to_cartesian(
        ideal_r_2, ideal_theta_2, ideal_phi_2
    )
    ideal_orientation_difference_2 = random.uniform(-math.pi, math.pi)

    follower2_state = client.getMultirotorState(vehicle_name=follower_name2)
    follower2_position = follower2_state.kinematics_estimated.position
    _, _, follower2_yaw = airsim.to_eularian_angles(
        follower2_state.kinematics_estimated.orientation
    )
    follower2_linear_velocity = follower2_state.kinematics_estimated.linear_velocity
    follower2_angular_velocity = follower2_state.kinematics_estimated.angular_velocity
    follower2_angular_velocity_z = follower2_angular_velocity.z_val
    ideal_follower2_x, ideal_follower2_y, ideal_follower2_z = ideal_position(
        (leader_position.x_val, leader_position.y_val, leader_position.z_val),
        leader_yaw,
        ideal_r_2,
        ideal_theta_2,
        ideal_phi_2,
    )
    ideal_follower2_yaw = compute_follower_yaw(
        leader_yaw, ideal_orientation_difference_2
    )
    state2 = np.array(
        [
            ideal_x_2,
            ideal_y_2,
            ideal_z_2,
            leader_yaw,
            leader_linear_velocity.x_val,
            leader_linear_velocity.y_val,
            leader_linear_velocity.z_val,
            leader_angular_velocity_z,
            follower2_yaw,
            follower2_linear_velocity.x_val,
            follower2_linear_velocity.y_val,
            follower2_linear_velocity.z_val,
            follower2_angular_velocity_z,
            follower2_position.x_val - ideal_follower2_x,
            follower2_position.y_val - ideal_follower2_y,
            follower2_position.z_val - ideal_follower2_z,
            follower2_yaw - ideal_follower2_yaw,
        ],
        dtype=np.float32,
    )

    # follower3
    ideal_r_3 = rand_op(2, 6)
    ideal_theta_3 = random.uniform(-math.pi / 2, math.pi / 2)
    ideal_phi_3 = random.uniform(-math.pi, math.pi)
    ideal_x_3, ideal_y_3, ideal_z_3 = spherical_to_cartesian(
        ideal_r_3, ideal_theta_3, ideal_phi_3
    )
    ideal_orientation_difference_3 = random.uniform(-math.pi, math.pi)

    follower3_state = client.getMultirotorState(vehicle_name=follower_name3)
    follower3_position = follower3_state.kinematics_estimated.position
    _, _, follower3_yaw = airsim.to_eularian_angles(
        follower3_state.kinematics_estimated.orientation
    )
    follower3_linear_velocity = follower3_state.kinematics_estimated.linear_velocity
    follower3_angular_velocity = follower3_state.kinematics_estimated.angular_velocity
    follower3_angular_velocity_z = follower3_angular_velocity.z_val
    ideal_follower3_x, ideal_follower3_y, ideal_follower3_z = ideal_position(
        (leader_position.x_val, leader_position.y_val, leader_position.z_val),
        leader_yaw,
        ideal_r_3,
        ideal_theta_3,
        ideal_phi_3,
    )
    ideal_follower3_yaw = compute_follower_yaw(
        leader_yaw, ideal_orientation_difference_3
    )
    follower3_linear_velocity.x_val, follower3_linear_velocity.y_val, follower3_linear_velocity.z_val
    state3 = np.array(
        [
            ideal_x_3,
            ideal_y_3,
            ideal_z_3,
            leader_yaw,
            leader_linear_velocity.x_val,
            leader_linear_velocity.y_val,
            leader_linear_velocity.z_val,
            leader_angular_velocity_z,
            follower3_yaw,
            follower3_linear_velocity.x_val,
            follower3_linear_velocity.y_val,
            follower3_linear_velocity.z_val,
            follower3_angular_velocity_z,
            follower3_position.x_val - ideal_follower3_x,
            follower3_position.y_val - ideal_follower3_y,
            follower3_position.z_val - ideal_follower3_z,
            follower3_yaw - ideal_follower3_yaw,
        ],
        dtype=np.float32,
    )

    # 初始化本回合的累积奖励
    episode_reward = 0

    # =========================================================================================================
    # # 领航者悬停
    # client.hoverAsync(vehicle_name=leader_name)

    # 生成领航者航行曲线
    curve_points = generate_3d_b_spline(
        drone_position=[
            leader_position.x_val,
            leader_position.y_val,
            leader_position.z_val,
        ]
    )
    path = [airsim.Vector3r(point[0], point[1], point[2]) for point in curve_points]
    # 绘制曲线
    client.simFlushPersistentMarkers()  # 绘制曲线之前先清除已有轨迹
    plot_traj = [
        airsim.Vector3r(point[0], point[1], point[2]) for point in curve_points
    ]
    client.simPlotLineList(
        plot_traj, color_rgba=[0.0, 1.0, 0.0, 1.0], is_persistent=True
    )
    # 设置飞行参数
    max_speed = random.uniform(0, LEADER_MAX_SPEED)

    timeout = 10000  # 超时时间（秒）

    # 使用moveOnPathAsync方法让领航者沿路径飞行
    client.moveOnPathAsync(
        path,
        max_speed,
        timeout,
        airsim.DrivetrainType.ForwardOnly,
        airsim.YawMode(False, 0),
        vehicle_name=leader_name,
    )
    # =========================================================================================================

    # 本回合的训练步数
    for step_i in range(NUM_STEP):
        client.simPause(True)

        # follower1
        action1 = agent.get_action(state1, add_noise=True)
        vehicle_x_1 = float(action1[0])
        vehicle_y_1 = float(action1[1])
        vehicle_z_1 = float(action1[2])
        yaw_rate_1 = float(action1[3])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 获取当前姿态四元数
        follower1_state = client.getMultirotorState(follower_name1)
        follower1_orientation = follower1_state.kinematics_estimated.orientation
        q = [
            follower1_orientation.x_val,
            follower1_orientation.y_val,
            follower1_orientation.z_val,
            follower1_orientation.w_val,
        ]
        rotation1 = Rotation.from_quat(q)
        rotation_matrix_1 = rotation1.as_matrix()
        # 将全局速度向量转换到机体坐标系
        global_velocity_1 = np.array([vehicle_x_1, vehicle_y_1, vehicle_z_1])
        body_velocity_1 = rotation_matrix_1.T.dot(
            global_velocity_1
        )  # 转置矩阵实现坐标系转换
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # follower2
        action2 = agent.get_action(state2, add_noise=True)
        vehicle_x_2 = float(action2[0])
        vehicle_y_2 = float(action2[1])
        vehicle_z_2 = float(action2[2])
        yaw_rate_2 = float(action2[3])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 获取当前姿态四元数
        follower2_state = client.getMultirotorState(follower_name2)
        follower2_orientation = follower2_state.kinematics_estimated.orientation
        q = [
            follower2_orientation.x_val,
            follower2_orientation.y_val,
            follower2_orientation.z_val,
            follower2_orientation.w_val,
        ]
        rotation2 = Rotation.from_quat(q)
        rotation_matrix_2 = rotation2.as_matrix()
        # 将全局速度向量转换到机体坐标系
        global_velocity_2 = np.array([vehicle_x_2, vehicle_y_2, vehicle_z_2])
        body_velocity_2 = rotation_matrix_2.T.dot(
            global_velocity_2
        )  # 转置矩阵实现坐标系转换
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # follower3
        action3 = agent.get_action(state3, add_noise=True)
        vehicle_x_3 = float(action3[0])
        vehicle_y_3 = float(action3[1])
        vehicle_z_3 = float(action3[2])
        yaw_rate_3 = float(action3[3])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 获取当前姿态四元数
        follower3_state = client.getMultirotorState(follower_name3)
        follower3_orientation = follower3_state.kinematics_estimated.orientation
        q = [
            follower3_orientation.x_val,
            follower3_orientation.y_val,
            follower3_orientation.z_val,
            follower3_orientation.w_val,
        ]
        rotation3 = Rotation.from_quat(q)
        rotation_matrix_3 = rotation3.as_matrix()
        # 将全局速度向量转换到机体坐标系
        global_velocity_3 = np.array([vehicle_x_3, vehicle_y_3, vehicle_z_3])
        body_velocity_3 = rotation_matrix_3.T.dot(
            global_velocity_3
        )  # 转置矩阵实现坐标系转换
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        client.simPause(False)

        # follower1
        client.moveByVelocityBodyFrameAsync(
            body_velocity_1[0],
            body_velocity_1[1],
            body_velocity_1[2],
            duration=SAFE_DURATION,
            yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate_1),
            vehicle_name=follower_name1,
        )

        # follower2
        client.moveByVelocityBodyFrameAsync(
            body_velocity_2[0],
            body_velocity_2[1],
            body_velocity_2[2],
            duration=SAFE_DURATION,
            yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate_2),
            vehicle_name=follower_name2,
        )

        # follower3
        client.moveByVelocityBodyFrameAsync(
            body_velocity_3[0],
            body_velocity_3[1],
            body_velocity_3[2],
            duration=SAFE_DURATION,
            yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate_3),
            vehicle_name=follower_name3,
        )

        time.sleep(DURATION)

        client.simPause(True)

        # 更新状态
        leader_state = client.getMultirotorState(vehicle_name=leader_name)
        leader_position = leader_state.kinematics_estimated.position
        _, _, leader_yaw = airsim.to_eularian_angles(
            leader_state.kinematics_estimated.orientation
        )
        leader_linear_velocity = leader_state.kinematics_estimated.linear_velocity
        leader_angular_velocity = leader_state.kinematics_estimated.angular_velocity
        leader_angular_velocity_z = leader_angular_velocity.z_val
        leader_velocity_r, leader_velocity_theta, leader_velocity_phi = (
            velocity_to_spherical(
                leader_linear_velocity.x_val,
                leader_linear_velocity.y_val,
                leader_linear_velocity.z_val,
            )
        )

        # follower1
        follower1_state = client.getMultirotorState(vehicle_name=follower_name1)
        follower1_position = follower1_state.kinematics_estimated.position
        _, _, follower1_yaw = airsim.to_eularian_angles(
            follower1_state.kinematics_estimated.orientation
        )
        follower1_linear_velocity = follower1_state.kinematics_estimated.linear_velocity
        follower1_angular_velocity = (
            follower1_state.kinematics_estimated.angular_velocity
        )
        follower1_angular_velocity_z = follower1_angular_velocity.z_val
        ideal_follower1_x, ideal_follower1_y, ideal_follower1_z = ideal_position(
            (leader_position.x_val, leader_position.y_val, leader_position.z_val),
            leader_yaw,
            ideal_r_1,
            ideal_theta_1,
            ideal_phi_1,
        )
        ideal_follower1_yaw = compute_follower_yaw(
            leader_yaw, ideal_orientation_difference_1
        )
        next_state1 = np.array(
            [
                ideal_x_1,
                ideal_y_1,
                ideal_z_1,
                leader_yaw,
                leader_linear_velocity.x_val,
                leader_linear_velocity.y_val,
                leader_linear_velocity.z_val,
                leader_angular_velocity_z,
                follower1_yaw,
                follower1_linear_velocity.x_val,
                follower1_linear_velocity.y_val,
                follower1_linear_velocity.z_val,
                follower1_angular_velocity_z,
                follower1_position.x_val - ideal_follower1_x,
                follower1_position.y_val - ideal_follower1_y,
                follower1_position.z_val - ideal_follower1_z,
                follower1_yaw - ideal_follower1_yaw,
            ],
            dtype=np.float32,
        )

        # follower2
        follower2_state = client.getMultirotorState(vehicle_name=follower_name2)
        follower2_position = follower2_state.kinematics_estimated.position
        _, _, follower2_yaw = airsim.to_eularian_angles(
            follower2_state.kinematics_estimated.orientation
        )
        follower2_linear_velocity = follower2_state.kinematics_estimated.linear_velocity
        follower2_angular_velocity = (
            follower2_state.kinematics_estimated.angular_velocity
        )
        follower2_angular_velocity_z = follower2_angular_velocity.z_val
        ideal_follower2_x, ideal_follower2_y, ideal_follower2_z = ideal_position(
            (leader_position.x_val, leader_position.y_val, leader_position.z_val),
            leader_yaw,
            ideal_r_2,
            ideal_theta_2,
            ideal_phi_2,
        )
        ideal_follower2_yaw = compute_follower_yaw(
            leader_yaw, ideal_orientation_difference_2
        )
        next_state2 = np.array(
            [
                ideal_x_2,
                ideal_y_2,
                ideal_z_2,
                leader_yaw,
                leader_linear_velocity.x_val,
                leader_linear_velocity.y_val,
                leader_linear_velocity.z_val,
                leader_angular_velocity_z,
                follower2_yaw,
                follower2_linear_velocity.x_val,
                follower2_linear_velocity.y_val,
                follower2_linear_velocity.z_val,
                follower2_angular_velocity_z,
                follower2_position.x_val - ideal_follower2_x,
                follower2_position.y_val - ideal_follower2_y,
                follower2_position.z_val - ideal_follower2_z,
                follower2_yaw - ideal_follower2_yaw,
            ],
            dtype=np.float32,
        )

        # follower3
        follower3_state = client.getMultirotorState(vehicle_name=follower_name3)
        follower3_position = follower3_state.kinematics_estimated.position
        _, _, follower3_yaw = airsim.to_eularian_angles(
            follower3_state.kinematics_estimated.orientation
        )
        follower3_linear_velocity = follower3_state.kinematics_estimated.linear_velocity
        follower3_angular_velocity = (
            follower3_state.kinematics_estimated.angular_velocity
        )
        follower3_angular_velocity_z = follower3_angular_velocity.z_val
        ideal_follower3_x, ideal_follower3_y, ideal_follower3_z = ideal_position(
            (leader_position.x_val, leader_position.y_val, leader_position.z_val),
            leader_yaw,
            ideal_r_3,
            ideal_theta_3,
            ideal_phi_3,
        )
        ideal_follower3_yaw = compute_follower_yaw(
            leader_yaw, ideal_orientation_difference_3
        )
        follower3_linear_velocity.x_val, follower3_linear_velocity.y_val, follower3_linear_velocity.z_val
        next_state3 = np.array(
            [
                ideal_x_3,
                ideal_y_3,
                ideal_z_3,
                leader_yaw,
                leader_linear_velocity.x_val,
                leader_linear_velocity.y_val,
                leader_linear_velocity.z_val,
                leader_angular_velocity_z,
                follower3_yaw,
                follower3_linear_velocity.x_val,
                follower3_linear_velocity.y_val,
                follower3_linear_velocity.z_val,
                follower3_angular_velocity_z,
                follower3_position.x_val - ideal_follower3_x,
                follower3_position.y_val - ideal_follower3_y,
                follower3_position.z_val - ideal_follower3_z,
                follower3_yaw - ideal_follower3_yaw,
            ],
            dtype=np.float32,
        )

        # 绘制理想跟随者位置
        # follower1
        ideal_follower1_position = airsim.Vector3r(
            ideal_follower1_x, ideal_follower1_y, ideal_follower1_z
        )
        client.simPlotPoints(
            [ideal_follower1_position], color_rgba=[1.0, 0.0, 0.0, 1.0]
        )

        # follower2
        ideal_follower2_position = airsim.Vector3r(
            ideal_follower2_x, ideal_follower2_y, ideal_follower2_z
        )
        client.simPlotPoints(
            [ideal_follower2_position], color_rgba=[0.0, 1.0, 0.0, 1.0]
        )

        # follower3
        ideal_follower3_position = airsim.Vector3r(
            ideal_follower3_x, ideal_follower3_y, ideal_follower3_z
        )
        client.simPlotPoints(
            [ideal_follower3_position], color_rgba=[0.0, 0.0, 1.0, 1.0]
        )

        """计算奖励"""
        flag = False
        # follower1
        reward1 = 0
        reward1 -= binormal(
            follower1_position.x_val - ideal_follower1_x,
            follower1_position.y_val - ideal_follower1_y,
            follower1_position.z_val - ideal_follower1_z,
        )
        reward1 -= abs(follower1_yaw - ideal_follower1_yaw)
        done_1 = False
        collision_info_1 = client.simGetCollisionInfo(vehicle_name=follower_name1)
        if collision_info_1.has_collided:
            if collision_info_1.object_name == leader_name:
                print("COLLID follower1")
                done_1 = True
                collid_count += 1
            else:
                print("跟随者之间碰撞。")
                collid_count += 0.5
                flag = True
        agent.replay_buffer.add_memo(
            state1.copy(),
            action1.copy(),
            reward1,
            next_state1.copy(),
            done_1,
        )
        state1 = next_state1

        # follower2
        reward2 = 0
        reward2 -= binormal(
            follower2_position.x_val - ideal_follower2_x,
            follower2_position.y_val - ideal_follower2_y,
            follower2_position.z_val - ideal_follower2_z,
        )
        reward2 -= abs(follower2_yaw - ideal_follower2_yaw)
        done_2 = False
        collision_info_2 = client.simGetCollisionInfo(vehicle_name=follower_name2)
        if collision_info_2.has_collided:
            if collision_info_2.object_name == leader_name:
                print("COLLID follower2")
                done_2 = True
                collid_count += 1
            else:
                print("跟随者之间碰撞。")
                collid_count += 0.5
                flag = True
        agent.replay_buffer.add_memo(
            state2.copy(),
            action2.copy(),
            reward2,
            next_state2.copy(),
            done_2,
        )
        state2 = next_state2

        # follower3
        reward3 = 0
        reward3 -= binormal(
            follower3_position.x_val - ideal_follower3_x,
            follower3_position.y_val - ideal_follower3_y,
            follower3_position.z_val - ideal_follower3_z,
        )
        reward3 -= abs(follower3_yaw - ideal_follower3_yaw)
        done_3 = False
        collision_info_3 = client.simGetCollisionInfo(vehicle_name=follower_name3)
        if collision_info_3.has_collided:
            if collision_info_3.object_name == leader_name:
                print("COLLID follower3")
                done_3 = True
                collid_count += 1
            else:
                print("跟随者之间碰撞。")
                collid_count += 0.5
                flag = True
        agent.replay_buffer.add_memo(
            state3.copy(),
            action3.copy(),
            reward3,
            next_state3.copy(),
            done_3,
        )
        state3 = next_state3

        episode_reward += (reward1 + reward2 + reward3) / 3

        actor_loss, critic1_loss, critic2_loss = agent.update_networks()
        if (step_i + 1) % 63 == 0:
            print(
                f"Critic1 Loss: {critic1_loss:.4f}, Critic2 Loss: {critic2_loss:.4f}, Actor Loss: {actor_loss:.4f}"
            )

        client.simPause(False)

        # 如果发生碰撞，结束本回合
        if done_1 or done_2 or done_3 or flag:
            break

    # 记录本回合的奖励
    rewards.append(episode_reward)
    collids.append(collid_count)
    print(f"done count: {collid_count}")
    print(f"Episode: {episode_i + 1}/{NUM_EPISODE}, Reward: {episode_reward:.2f}")

    # 定期保存模型
    if (episode_i + 1) % 50 == 0:
        agent.save_checkpoint(f"model_ep_{episode_i + 1}.pth")
        with open(f"rewards_{episode_i + 1}.txt", "w") as f:
            for reward in rewards:
                f.write(f"{reward}\n")

        with open(f"collids_{episode_i + 1}.txt", "w") as f:
            for c in collids:
                f.write(f"{c}\n")

        # 保存数据到文件
        joblib.dump(agent.replay_buffer.buffer, f"buffer_{episode_i + 1}.joblib")

        plt.plot(rewards[-min(300, len(rewards)) :])
        plt.title(f"Training Rewards (Episode {episode_i + 1})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(f"rewards_{episode_i + 1}.png")  # 保存图像到文件
        plt.close()  # 关闭当前画布，避免占用内存

# 绘制训练奖励曲线
plt.plot(rewards)
plt.title(f"Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig(f"rewards.png")  # 保存图像到文件
plt.close()  # 关闭当前画布，避免占用内存

# 保存最终模型
timestamp = time.strftime("%Y%m%d%H%M%S")
agent.save_checkpoint(f"td3_{timestamp}.pth")
