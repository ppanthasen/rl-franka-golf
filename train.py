import numpy as np
import gymnasium as gym
from pathlib import Path
import torch as th

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from wrappers import RewardShapingWrapper, RewardShapingWrapperV2
from callbacks import CustomCallback, CustomCallbackV2

from sai_rl import SAIClient 
import sai_mujoco

# -------------------- Config (edit here) --------------------
STAGE = 1
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 4
LOCAL = True

SAVE_DIR = Path("checkpoints/C1")
NAME_PREFIX = "sac_stage"
EVAL_FREQ = 2_000
SAVE_FREQ = 2_000
LOG_INTERVAL = 10

BUFFER_RESET_MODE = "hard"      # "hard" or "soft"
DET_RESET_OFF_FROM_STAGE = -1    # disable deterministic_reset at/after this stage

LOAD_MODEL_PATH = "checkpoints/6/stage4_final.zip"
LOAD_BUFFER_PATH = None
LOAD_PRETRAIN = False
# ------------------------------------------------------------

def make_env(stage: int = 1, local: bool = False, render_mode: str | None = None, deterministic_reset: bool = False):
    if local:
        env = gym.make(
            "FrankaIkGolfCourseEnv-v0",
            render_mode=render_mode,
            hide_overlay=True,
            deterministic_reset=deterministic_reset,
        )
    else:
        sai = SAIClient("FrankaIkGolfCourseEnv-v0")
        env = sai.make_env(
            render_mode=render_mode,
            hide_overlay=True,
            deterministic_reset=deterministic_reset,
        )
        
    return RewardShapingWrapperV2(env)
    return RewardShapingWrapper(env, stage=stage)

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(lambda: make_env(STAGE, LOCAL), n_envs=N_ENVS)

    if LOAD_MODEL_PATH:
        model = SAC.load(
            LOAD_MODEL_PATH,
            env=vec_env,
            custom_objects={
                "gradient_steps": 2,
            }
        )
        if LOAD_BUFFER_PATH:
            model.load_replay_buffer(LOAD_BUFFER_PATH)
        model.set_env(vec_env)
        
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            ent_coef='auto',
            tensorboard_log="logs",
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            verbose=0,
            gradient_steps=2,
            learning_starts=5_000,
            buffer_size=300_000,
        )
        
    if LOAD_PRETRAIN:
        model.actor.load_state_dict(th.load("frankagolf/best_bc_actor.pt"))
        
    cb = CustomCallbackV2(
        eval_freq=EVAL_FREQ,
        save_freq=SAVE_FREQ,
        save_path=str(SAVE_DIR),
        name_prefix=NAME_PREFIX,
        det_reset_off_from_stage=DET_RESET_OFF_FROM_STAGE,
        buffer_reset_mode=BUFFER_RESET_MODE,
        freeze_on_start=bool(LOAD_MODEL_PATH),
        freeze_steps=50_000,
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, log_interval=LOG_INTERVAL)
    model.save(SAVE_DIR / f"{NAME_PREFIX}_final.zip")

if __name__ == "__main__":
    main()


