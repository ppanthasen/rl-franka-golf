from stable_baselines3 import SAC, TD3
from sai_rl import SAIClient
import time
import sai_mujoco
import gymnasium as gym
# from utils import RewardShapingWrapper
from wrappers import RewardShapingWrapper, RewardShapingWrapperV2
import numpy as np
import torch as th

def get_manual_actions():
    actions = []
    # actions += [[0, 0, 0, 0, 0, 0, -1]] * 10  # gripper 
    
    # actions += [[0, 0, 0, 0, 0, 0, 1]] * 10  # gripper 
    actions += [[0.5, 0, 0, 0, 0, 0, 0.0]] * 6  # +X & open gripper
    actions += [[0, 0, -0.52, 0, 0, 0, 0.0]] * 5  # down Z
    actions += [[0, 0, 0, 0, 0, 0, 0.1]] * 3  # gripper close
    actions += [[0, 0, 0.5, 0, 0, 0, 0]]  # up Z to lift
    actions += [[0, 0.5, 0, 0, 0, 0, 0]] * 15  # +Y (toward hole)
    actions += [[0, 0, 0, 0, 0, 0, 0]] * 1000 # null for waiting
    actions = np.array(actions)
    return actions

def get_perfect_actions():
    actions = []
    actions += [[1.0, -0.1, 0, 0, 0, 0, -1]] * 1 
    actions += [[1.0, 0, -1, 0, 0, 0, -1]] * 2
    actions += [[0.0, 0, -1, 0, 0, 0, 1]] * 2
    actions += [[0, 0, 0, 0, 0, 0, 1]] * 1
    actions += [[0, 1, 0, 0, 0, 0, 0]] * 10  # +Y (toward hole)
    
    
    actions += [[0, 0, 0, 0, 0, 0, 0]] * 1000
    
    return actions

eval_env = gym.make("FrankaIkGolfCourseEnv-v0", render_mode="human", deterministic_reset=False)
# sai = SAIClient("FrankaIkGolfCourseEnv-v0")
# eval_env = sai.make_env(render_mode="human", hide_overlay=False, deterministic_reset=False)

eval_env = RewardShapingWrapperV2(eval_env, max_steps=300)
model = SAC.load("checkpoint.zip", eval_env)

model.set_env(eval_env)

obs, _ = eval_env.reset()
done = False

actions = get_manual_actions()
actions = get_perfect_actions()
success = 0
ep = 0
t = 0
# for action in actions:
while not done:
    t+=1
    action, _ = model.predict(obs, deterministic=True)
    # action = actions[t-1]
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
    # print(reward, info)
    if done:
        obs, _ = eval_env.reset()
        done = False
        t = 0
        ep += 1
        success += 1 if info['success'] else 0
        print("SR:", success, "/", ep)
    time.sleep(0.01)


    
