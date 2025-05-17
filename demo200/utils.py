import math
import random

def spherical_to_cartesian(r, theta, phi):
    """
    三维极坐标 → 直角坐标（新定义）
    
    Args:
        r: 半径
        theta: 与xy平面的夹角，向上为正，范围 [-π/2, π/2]
        phi: 方位角（从x轴右偏），范围 [-π, π)
    """
    x = r * math.cos(theta) * math.cos(phi)
    y = r * math.cos(theta) * math.sin(phi)
    z = -r * math.sin(theta)  # 添加负号以适配 NED
    return (x, y, z)

def cartesian_to_spherical(x, y, z):
    """直角坐标 → 新定义的三维极坐标"""
    r_val = math.hypot(math.hypot(x, y), z)
    if r_val < 1e-9:
        return (0.0, 0.0, 0.0)
    theta = math.asin(z / r_val)
    phi = math.atan2(y, x)
    return (r_val, theta, phi)

def ideal_position(leader_pos, yaw, r, theta, phi):
    """
    基于新极角定义的领航者坐标系转换
    
    Args:
        leader_pos: 领航者的全局坐标 (x, y, z)
        yaw: 领航者的偏航角（绕z轴的旋转，弧度）
        r: 跟随者与领航者的距离
        theta: 与领航者xy平面的夹角，向上为正，范围 [-π/2, π/2]
        phi: 方位角（从领航者x轴右偏），范围 [-π, π)
    """
    x_local, y_local, z_local = spherical_to_cartesian(r, theta, phi)
    
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x_global = leader_pos[0] + x_local * cos_yaw - y_local * sin_yaw
    y_global = leader_pos[1] + x_local * sin_yaw + y_local * cos_yaw
    z_global = leader_pos[2] + z_local
    
    return (x_global, y_global, z_global)

def normalize_angle(angle):
    """
    将角度规范化到 [-π, π) 范围内
    """
    normalized = (angle + math.pi) % (2 * math.pi) - math.pi
    return normalized

def compute_follower_yaw(yaw_leader, delta):
    """
    计算跟随者的偏航角
    """
    # AirSim 的 yaw 增加方向为顺时针（Z轴向下）
    return normalize_angle(yaw_leader - delta)  # 改为减法

def binormal(x, y, z):
    """计算三维向量的模长"""
    return math.sqrt(x**2 + y**2 + z**2)

def rand_op(small, big):
    """随机选择一个区间内的数值"""
    if random.random() < 0.5:
        return random.uniform(-big, -small)
    else:
        return random.uniform(small, big)
    
def velocity_to_spherical(vx, vy, vz):
    """速度向量 → 球坐标系分量 (vr, vtheta, vphi)"""
    r = math.sqrt(vx**2 + vy**2 + vz**2)
    if r < 1e-8:
        return (0.0, 0.0, 0.0)
    horizontal = math.hypot(vx, vy)
    if horizontal < 1e-8:
        theta = math.copysign(math.pi/2, -vz)  # 处理纯垂直运动
    else:
        theta = math.atan2(-vz, horizontal)
    phi = math.atan2(vy, vx)
    return (r, theta, phi)


# 示例测试
if __name__ == "__main__":
    # 测试 NED 坐标系下的位置转换
    test_r, test_theta, test_phi = 5, math.pi/4, math.pi/2
    x, y, z = spherical_to_cartesian(test_r, test_theta, test_phi)
    print(f"球坐标(r=5,θ=π/4,φ=π/2) → NED坐标: ({x:.2f}, {y:.2f}, {z:.2f})")
    # 预期输出：≈ (0.0, 3.54, -3.54)

    # 验证偏航角计算
    leader_yaw_test = math.pi/2
    delta_test = math.pi/4
    follower_yaw = compute_follower_yaw(leader_yaw_test, delta_test)
    print(f"领航者偏航π/2，delta=π/4 → 跟随者偏航: {follower_yaw:.2f} (应≈0.785)")
    # 预期输出：π/2 - π/4 = π/4 ≈ 0.785