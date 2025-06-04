import ctypes
import os
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
from gymnasium import spaces
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback

from dll_loader import load_dlls_from_folder
"加载DLL文件"
dll_folder = r"D:\chiken\Stable_train\load_dll"
# 调用函数加载 DLL 文件
all_loaded_dlls = load_dlls_from_folder(dll_folder)


import AirCombat_demo as Env
import reward_gird as envi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

episode = 0
Psi = 0
class AirCombatEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None,max_episode_steps=5000):

        super().__init__()  #起始参数设定
        self.cpp_env = Env
        self.action_space = spaces.Box(-1,1,(3,),np.float32)
        # 定义每个状态量的范围
        self.state_ranges = [
            # (0, 360),  # theta_deg (航向角)
            # (0, 360),  # diff_deg (目标方位角)
            # (0, 100000),  # theta_dis (距离)
            (0, 360),  # yaw (偏航角)
            (0, 1),  # roll (滚转角)
            (-90, 90),  # pitch (俯仰角)
            (-90, 90),  # latitude (纬度)
            (-180, 180),  # longitude (经度)
            (0, 8000),  # altitude(高度)
            (0, 1000),  # TAS (真空速，单位：节)
            # (0, 200)  # Capture_Count (捕获数量)
            (0, 20000),  # grid_num(扫描到的栅格数量)
            (0, 1),  # grid_rate(扫描到的栅格比例)
            (-180,180)#psi(期望角度）
        ]

        # 更新 observation_space 的范围
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10+100,), dtype=np.float32)
        # self.observation_space = spaces.Box(
        #     low=-1e8, high=1e8, shape=(10,), dtype=np.float32
        # )
        self.max_episode_steps = max_episode_steps
        self.current_step = 0



    def reset(self, seed=None, options=None):   #初始化/复位
        super().reset(seed=seed,options=options)
        seed = seed if seed is not None else 0
        global episode
        print("num = ",Env.getCumulativeScannedGridCount())
        print("rate = ",Env.rate())
        print("episode = ",episode)
        episode += 1
        # self.cpp_env.SaveTrajectoryToExcel(episode-1)
        # print("action_num = ",self.current_step)
        self.cpp_env.CEnvInit_test(episode)
        self.cpp_env.GetCState()
        state = self._get_processed_state()
        self.current_step = 0

        return state, {}

    def step(self, action):#刷新动作
        self.current_step += 1
        psi = (action[0]) * 180     # [-1,1] -> [-180,180]
        global Psi
        Psi = psi
        # alt = 5000 + action[1] * 1500  # [-1,1] -> [5000,8000]
        alt = 6000
        ma = 0.45 + (action[2] + 1) * 0.15        # [-1,1] -> [0.45,0.75]
        # Env.high_light()
        # print("action i =",current_action)
        out_done = self.cpp_env.out_done()  #飞机越界
        Reset = self.cpp_env.Reset()    #飞机越界机高度低于2000
        num_done = self.cpp_env.num_done() #判断捕获数量是否>90%
        position_done = self.cpp_env.position_done() #是否完成闭合

        # truncated = (self.current_step > self.max_episode_steps)  #每轮的终止条件
        cumulative_reward = 0
        is_update = True

        # if not Result:#飞机飞出作战区域
        #     psi = self.cpp_env.GetFState().env_state[0].theta_deg
            # print("psi = ",psi)
        # Env.Step()
        # Env.CEnvStep_test([action],is_update)
        current_action = [psi, alt, ma]
        for _ in range(10):
            # is_update = False
            Env.CEnvStep_test(current_action, is_update)
            Env.GetCState()
            current_reward = envi.GetCReward()
            cumulative_reward += current_reward
        reward = cumulative_reward/10
        # print("reward = ",reward)
        if position_done == True:
            print("position_done")
            # Env.SaveTrajectoryToExcel(episode)
            reward+=50
        if out_done == True:
            # reward-=10
            # Env.SaveTrajectoryToExcel(episode)
            print("out_done")
        if self.cpp_env.Get_altitude0() < 4000:
            reward -= 30
            print("altitude < 4000")
        # print("reward = ",reward)
        truncated = ((self.cpp_env.Get_altitude0() < 4000)#高度
                     or (position_done == True)#是否完成闭合
                     or(out_done == True)
                     or (self.current_step > self.max_episode_steps))  # 每轮的终止条件
        # print("reward = ",reward)
        terminated = False

        # reward = envi.GetCReward()
        next_state = self._get_processed_state()
        return next_state, reward, terminated, truncated, {}

    def _get_processed_state(self):
        self.cpp_env.GetCState()
        s = self.cpp_env.GetFState().env_state[0]

        # 更新状态量范围定义
        state_ranges = [
            # (0, 360),  # theta_deg (航向角)
            # (0, 360),  # diff_deg (目标方位角)
            # (0, 100000),  # theta_dis (距离)
            (0, 360),  # yaw (偏航角)
            (0, 1),  # roll (滚转角)
            (-90, 90),  # pitch (俯仰角)
            (-90, 90),  # latitude (纬度)
            (-180, 180),  # longitude (经度)
            (0,8000),  #altitude(高度)
            (0, 1000),  # TAS (真空速，单位：节)
            # (0, 200)  # Capture_Count (捕获次数)
            (0,20000),#grid_num(扫描到的栅格数量)
            (0,1),#grid_rate(扫描到的栅格比例)
            (-180, 180)  # psi(期望角度）
        ]
        if abs(s._roll)<10:
            roll_flag = 1
        else:
            roll_flag = 0
        # 获取原始值
        raw_values = [
            # s.theta_deg, s.diff_deg, s.theta_dis,
            s._yaw, roll_flag, s._pitch,
            s._latitude, s._longitude,
            s._altitude,s.TAS,
            s.grid_num, s.grid_rate,Psi
        ]

        normalized_state = []
        for val, (lower, upper) in zip(raw_values, state_ranges):
            clamped_val = np.clip(val, lower, upper)
            normalized = 2 * (clamped_val - lower) / (upper - lower) - 1
            normalized_state.append(normalized)
        # print("normalized_state = ",normalized_state)

        patch_list = np.array(s.local_patch_flat,dtype=np.float32)

        # return np.array([normalized_state,patch_list], dtype=np.float32)
        return np.concatenate([normalized_state, patch_list]).astype(np.float32)

    def render(self):
        pass

    def close(self):
        pass

# class CustomTensorboardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#
#     def _on_step(self) -> bool:
#         """每一步训练时触发"""
#         # 获取当前环境（假设是单个环境，非向量环境）
#         # 如果是向量环境（VecEnv），使用 self.training_env.get_attr('Capture_Count')[0]
#         self.env = self.training_env.unwrapped  # 解包向量环境（如果有的话）
#
#         # 从环境中获取自定义变量（例如 Capture_Count）
#         # 注意：需要确保环境中该变量可访问（例如在 AirCombatEnv 中暴露该属性）
#         capture_count =  Env.Get_CaptureCount() # 根据你的环境实现调整
#
#         # 记录到 TensorBoard（路径：自定义变量名/Metric）
#         self.logger.record("custom_metrics/capture_count", capture_count)
#         return True  # 保持回调继续执行

    '''回调函数------每轮的奖励和每轮中栅格的扫描数量'''
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.recent_capture_counts = []  # 保存最近10轮的捕获数
        self.max_capture_count = 0  # 记录最大值
    def _on_step(self) -> bool:
        # 获取当前步骤的奖励
        reward = self.locals["rewards"][0]
        self.current_episode_reward += reward
        self.env = self.training_env.unwrapped  # 解包向量环境（如果有的话）
        self.current_capture_count = Env.getCumulativeScannedGridCount()  # 获取当前栅格的扫描数量
        # print("count = ", self.current_capture_count)
        # 保存当前轮的捕获数到最近10轮列表
        self.recent_capture_counts.append(self.current_capture_count)
        # 保持列表仅保留最近5轮的数据
        self.recent_capture_counts = self.recent_capture_counts[-10:]  # 取最后10个元素

        # 计算当前轮与最近5轮的最大值（如果有数据）
        if self.recent_capture_counts:
            self.max_capture_count = max(self.recent_capture_counts)
        else:
            self.max_capture_count = self.current_capture_count  # 首轮情况
        # 检查episode是否结束
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward )

            # 获取捕获次数
            # capture_counts = self.training_env.get_attr('Capture_Count')
            # capture_count = capture_counts[0]
            # self.capture_count = Env.Get_CaptureCount()

            # 记录本轮的奖励和捕获数量
            self.logger.record("episode/reward", self.current_episode_reward)
            # self.logger.record("custom_metrics/current_capture", self.current_capture_count)
            self.logger.record("custom_metrics/gird_count", self.max_capture_count)  # 新增最大值记录

            # # 计算并记录最近10轮的平均奖励
            # avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            # self.logger.record("episode/avg_reward", avg_reward)

            # 将记录写入TensorBoard
            self.logger.dump(self.num_timesteps)

            # 重置当前轮的奖励计数器
            self.current_episode_reward = 0.0

        return True
# 注册环境
register(
    id="AirCombat-v0",
    entry_point="__main__:AirCombatEnv",
    max_episode_steps=1024,
)
def main():

    env = gym.make("AirCombat-v0", max_episode_steps=1024)
    check_env(env.unwrapped)


if __name__ == "__main__":
    main()
