import numpy as np

# 参数定义
T_env = 25       # 环境温度 (℃)
T_hot = 90       # 大瓶热水温度 (℃)
T_target = 37     # 目标水温 (℃)
k = 0.864        # 冷却速率常数 (h⁻¹)
cup_volume = 0.2  # 杯子容量 (升)
bottle_volume = 1.0  # 大瓶初始水量 (升)
alpha = 0.5       # 每次喝掉杯中水的比例

def cooling_time(T0, T_target, k, T_env):
    """计算从T0冷却到T_target所需时间（小时）"""
    if T0 <= T_target:
        return 0.0
    return (1/k) * np.log((T0 - T_env)/(T_target - T_env))

# 策略一：整杯冷却
N = int(np.ceil(bottle_volume / cup_volume))  # 倒满次数
total_time1 = N * cooling_time(T_hot, T_target, k, T_env)

# 策略二：动态模拟（每次喝掉杯中水的alpha比例，然后续满）
total_time2 = 0.0
remaining_water = bottle_volume  # 大瓶剩余水量
current_temp = T_hot             # 杯中当前水温

while remaining_water > 0:
    # 每次倒满杯子，喝掉alpha比例的水
    drink_volume = min(cup_volume * alpha, remaining_water)
    remaining_water -= drink_volume
    
    # 杯中剩余水量为 (1-alpha)*cup_volume，补充热水至满
    # 混合温度 = (原杯中剩余水的热量 + 补充热水的热量) / 总容量
    mix_temp = ((cup_volume - drink_volume) * current_temp + drink_volume * T_hot) / cup_volume
    
    # 计算冷却时间（从mix_temp到T_target）
    delta_t = cooling_time(mix_temp, T_target, k, T_env)
    total_time2 += delta_t
    
    # 更新杯中水温为混合温度（续满后）
    current_temp = mix_temp

print(f"策略一总时间: {total_time1:.2f} 小时")
print(f"策略二总时间: {total_time2:.2f} 小时")