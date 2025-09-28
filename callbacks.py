from collections import deque
from dataclasses import dataclass, field
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from wrappers import RewardShapingWrapper

SUCCESS_THRESHOLD = 0.90                       # for stages >= 4
WINDOW = 100                                   # rolling window length

@dataclass
class RollingStats:
    rewards: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    successes: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def add(self, avg_step_reward: float, success: bool):
        self.rewards.append(float(avg_step_reward))
        self.successes.append(bool(success))

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    def prop_above(self, thr: float) -> float:
        if not self.rewards:
            return 0.0
        arr = np.asarray(self.rewards, dtype=np.float32)
        return float((arr > thr).mean())

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0

    def __len__(self):
        return len(self.rewards)

class CustomCallback(BaseCallback):
    """
    Curriculum callback aligned with the new RewardShapingWrapper:

    - Consumes per-episode metrics via info:
        info["episode_end"]      -> bool
        info["avg_step_reward"]  -> float
        info["success"]          -> bool (if provided by env)
    - Progresses stages using:
        * Stages 1-3: reward mean + fraction above threshold
        * Stage 4+:   success rate
    - Periodically adjusts entropy (SAC), saves checkpoints/replay buffer,
      freezes/unfreezes actor at stage boundaries, resets replay buffer on stage up.
    """

    def __init__(
        self,
        eval_freq: int,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        freeze_on_start: bool = False,
        freeze_steps: int = 100_000,
        det_reset_off_from_stage: int = 5,
        buffer_reset_mode: str = 'soft',
    ):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.freeze_on_start = freeze_on_start
        self.freeze_steps = freeze_steps
        self.det_reset_off_from_stage = det_reset_off_from_stage
        self.buffer_reset_mode = buffer_reset_mode
        self.actor_frozen = False
        self.best_success_rate = 0.0
        self.stage_reset_step = 0
        self.stage = 0
        self.last_saved_pct = -1
        self.stats = RollingStats()
        self.reset_step = 0
        
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=0,
        )

    # --- SB3 required hooks ---

    def _init_callback(self) -> None:
        self.checkpoint_callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        print("Start training...")
        self.stage = int(self.training_env.get_attr("stage", 0)[0])
        
        self._set_target_entropy(self.stage)
        
        if self.freeze_on_start:
            self._freeze_actor()
        self.training_env.env_method("set_use_guided_traj", False)
            
    def _on_step(self) -> bool:
        self.checkpoint_callback.on_step()
        
        if not self.actor_frozen and self.stage <= 3 and self.model.log_ent_coef:
            self.model.log_ent_coef.data.fill_(np.log(0.1))

        # inject guided traj occasionally
        if self.num_timesteps % 20_000 == 0:
            self.training_env.env_method("set_use_guided_traj", False)
        
        # 2) Collect episode-end metrics from infos (VecEnv-safe)
        self.logger.record("explore/target_entropy", float(getattr(self.model, "target_entropy", np.nan)))

        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if info is None:
                    continue
                if info.get("episode_end", False):
                    avg_step_reward = float(info.get("avg_step_reward", 0.0))
                    success = any(bool(info.get(k)) for k in ("success", "stage_success"))
                    self.stats.add(avg_step_reward, success)
                    self.logger.record("rollout/ep_avg_step_reward", self.stats.mean_reward)
                    self.logger.record("rollout/success_rate", self.stats.success_rate)
                    
        if self.actor_frozen and self.num_timesteps - self.reset_step > self.freeze_steps and len(self.stats) >= WINDOW:
            print("Unfreezing actor")
            self.actor_frozen = False
            for p in self.model.actor.parameters():
                p.requires_grad_(True)
            if self.model.log_ent_coef:
                self.model.log_ent_coef.data.fill_(np.log(0.2))
                
        # # 3) Occasional entropy nudging (SAC only)
        # if hasattr(self.model, "log_ent_coef") and (self.n_calls % 100_000 == 0):
        #     try:
        #         ent = float(torch.exp(self.model.log_ent_coef.detach()).item())
        #         if ent < 0.005:
        #             self.model.log_ent_coef.data.fill_(np.log(0.05))
        #             print("Entropy too low. Resetting to 0.05")
        #         elif ent > 0.3:
        #             self.model.log_ent_coef.data.fill_(np.log(0.1))
        #             print("Entropy too high. Reducing to 0.1")
        #     except Exception:
        #         pass  # if algo isn't SAC-compatible, silently skip

        # 4) Periodic replay buffer save
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(f"{self.save_path}/buffer_latest.pkl")

        # 5) Evaluation & curriculum logic
        if self.n_calls % self.eval_freq == 0:
            self.stage = int(self.training_env.get_attr("stage", 0)[0])


            # Stage-based checks
            sr = self.stats.success_rate
            if self.stage < 4:
                thr = RewardShapingWrapper.REWARD_THRESHOLDS.get(self.stage, 6.0)
                mean_r = self.stats.mean_reward
                frac_above = self.stats.prop_above(thr)
                            
                if self.stage > 1:
                    print(f"Stage {self.stage} | Episodes={len(self.stats)} | "
                        f"MeanR={mean_r:.3f} | >Thr({thr})={frac_above:.3f} | SR={sr:.3f}")
                else:
                    print(f"Stage {self.stage} | Episodes={len(self.stats)} | "
                        f"MeanR={mean_r:.3f} | >Thr({thr})={frac_above:.3f}")
                    
                if len(self.stats) >= WINDOW and mean_r > thr and frac_above >= 0.80 and (self.stage==1 or sr >= 0.80):
                    self._progress_stage()
            else:
                print(f"Stage {self.stage} | Episodes={len(self.stats)} | SuccessRate={sr:.3f}")

                if len(self.stats) >= WINDOW:
                    # Stage-up gate (unchanged)
                    if self.det_reset_off_from_stage != -1 and sr > SUCCESS_THRESHOLD:
                        self._progress_stage()

                    sr_pct = int(sr * 100)  # floor to integer percent
                    should_save = False
                    save_tag = None

                    if sr_pct < 80:
                        milestone = (sr_pct // 10) * 10  # 0,10,20,...,70
                        if milestone >= 10 and milestone > self.last_saved_pct:
                            save_tag = milestone
                            should_save = True
                    else:
                        # at/after 80: save every new integer percent
                        target = max(80, self.last_saved_pct + 1)
                        if sr_pct >= target and sr_pct > self.last_saved_pct:
                            save_tag = sr_pct
                            should_save = True

                    if should_save:
                        self.model.save(f"{self.save_path}/best_stage{self.stage}_sr{save_tag}")
                        self.last_saved_pct = save_tag
                        print(f"Saved milestone model at {save_tag}/{sr}% success (stage {self.stage}).")

                
                # sr = self.stats.success_rate
                # print(f"Stage {self.stage} | Episodes={len(self.stats)} | SuccessRate={sr:.3f}")
                # if (len(self.stats) >= WINDOW and sr > SUCCESS_THRESHOLD):
                #     self._progress_stage()

                # # Save best model from stage 6 onward
                # if self.stage >= 5 and sr > max(0.80, self.best_success_rate):
                #     self.best_success_rate = sr
                #     self.model.save(f"{self.save_path}/best_model_stage{self.stage}_sr{round(sr*100)}")
                #     print(f"New best model saved! Success rate: {sr:.3f}")

            # # Fallback: stagnation guard or stage unset
            # if (self.num_timesteps - self.stage_reset_step) >= 1e7 or self.stage == 0:
            #     self._progress_stage()

        return True

    def _on_training_end(self) -> None:
        self.checkpoint_callback.on_training_end()

    # --- Stage transitions & utilities ---

    def _progress_stage(self):
        prev_stage = int(self.training_env.get_attr("stage", 0)[0])
        next_stage = prev_stage + 1

        self.model.save(f"{self.save_path}/stage{prev_stage}_final")
        self.training_env.env_method("set_stage", next_stage)
        self.stats = RollingStats()
        
        self._apply_buffer_reset(prev_stage)
        self._set_target_entropy(next_stage)
        
        
        if next_stage >= self.det_reset_off_from_stage:
            self.training_env.env_method("set_deterministic_reset", False)
        
        self.last_saved_pct = -1
        
    def _apply_buffer_reset(self, prev_stage):
        if self.buffer_reset_mode == "hard":
            self._freeze_actor()
            self.model.replay_buffer.reset()
            
            for p in self.model.actor.parameters():
                p.requires_grad_(False)
                
        elif self.buffer_reset_mode == "soft":
            self.model.replay_buffer.rewards *= self._stage_scale(prev_stage)
    
    def _stage_scale(self, prev_stage):
        w = dict(RewardShapingWrapper.STAGE_WEIGHTS)
        if prev_stage >= max(w.keys()): return 1
        else:
            w_prev = w[prev_stage]
            w_next = w[prev_stage + 1]
            common = set(w_prev).intersection(w_next)
            
            s_prev = sum(w_prev[k] for k in common)
            s_next = sum(w_next[k] for k in common)
            return (s_next / s_prev) if s_prev != 0 else 1.0
    
    def _set_target_entropy(self, stage):
        stage = min(stage, 4)
        target = {1: -3.0, 2: -5.0, 3: -6.0, 4: -7.0}
        self.model.target_entropy = target[stage]
            
    def _freeze_actor(self):
        print("Freezing Actor")
        self.reset_step = self.num_timesteps
        self.actor_frozen = True
        for param in self.model.actor.parameters():
            param.requires_grad_(False)
            
        if self.model.log_ent_coef:
            self.model.log_ent_coef.data.fill_(np.log(1e-100))



class CustomCallbackV2(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
        freeze_on_start: bool = False,
        freeze_steps: int = 100_000,
        det_reset_off_from_stage: int = 5,
        buffer_reset_mode: str = 'soft',
    ):
        super().__init__(verbose)
        self.eval_freq = int(eval_freq)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.freeze_on_start = freeze_on_start
        self.freeze_steps = freeze_steps
        self.det_reset_off_from_stage = det_reset_off_from_stage
        self.buffer_reset_mode = buffer_reset_mode
        self.actor_frozen = False
        self.best_success_rate = 0.0
        self.stage_reset_step = 0
        self.stage = 0
        self.last_saved_pct = -1
        self.stats = RollingStats()
        self.reset_step = 0
        
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=0,
        )

    def _init_callback(self) -> None:
        self.checkpoint_callback.init_callback(self.model)

        print("Curriculum: switching to Stage 0 (2:2 det:rand resets)")
        # Switch 3 envs to random, 1 to deterministic
        self.training_env.env_method("set_deterministic_reset", False, indices=[2, 3])
        self.training_env.env_method("set_deterministic_reset", True, indices=[0, 1])

    def _on_training_start(self) -> None:
        print("Start training...")
        
        if self.freeze_on_start:
            self._freeze_actor()
        self.training_env.env_method("set_use_guided_traj", True)
        
        
    def _on_step(self) -> bool:
        self.checkpoint_callback.on_step()
        
        # if not self.actor_frozen and self.stage <= 3 and self.model.log_ent_coef:
            # self.model.log_ent_coef.data.fill_(np.log(1e-10))

        # inject guided traj occasionally
        
        # inject guided traj occasionally
        # def should_inject(step: int) -> bool:
        #     if step < 50_000: return step % 200 == 0
        #     if step < 100_000: return step % 1000 == 0
        #     if step < 200_000: return step % 2000 == 0
        #     if step < 500_000: return step % 5000 == 0
        #     if step < 1_000_000: return step % 10_000 == 0
        #     return step % 20_000 == 0
        
        # if should_inject(self.num_timesteps):
        #     self.training_env.env_method("set_use_guided_traj", True)
        
        # 2) Collect episode-end metrics from infos (VecEnv-safe)
        self.logger.record("explore/target_entropy", float(getattr(self.model, "target_entropy", np.nan)))

        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if info is None:
                    continue
                if info.get("episode_end", False):
                    avg_step_reward = float(info.get("avg_step_reward", 0.0))
                    success = any(bool(info.get(k)) for k in ("success", "stage_success"))
                    self.stats.add(avg_step_reward, success)
                    self.logger.record("rollout/ep_avg_step_reward", self.stats.mean_reward)
                    self.logger.record("rollout/success_rate", self.stats.success_rate)
                    
        if self.actor_frozen and self.num_timesteps - self.reset_step > self.freeze_steps and len(self.stats) >= WINDOW:
            print("Unfreezing actor")
            self.actor_frozen = False
            for p in self.model.actor.parameters():
                p.requires_grad_(True)
            if self.model.log_ent_coef:
                self.model.log_ent_coef.data.fill_(np.log(0.2))
                
        # # 3) Occasional entropy nudging (SAC only)
        # if hasattr(self.model, "log_ent_coef") and (self.n_calls % 100_000 == 0):
        #     try:
        #         ent = float(torch.exp(self.model.log_ent_coef.detach()).item())
        #         if ent < 0.005:
        #             self.model.log_ent_coef.data.fill_(np.log(0.05))
        #             print("Entropy too low. Resetting to 0.05")
        #         elif ent > 0.3:
        #             self.model.log_ent_coef.data.fill_(np.log(0.1))
        #             print("Entropy too high. Reducing to 0.1")
        #     except Exception:
        #         pass  # if algo isn't SAC-compatible, silently skip

        # 4) Periodic replay buffer save
        if self.n_calls % self.save_freq == 0:
            self.model.save_replay_buffer(f"{self.save_path}/buffer_latest.pkl")

        # 5) Evaluation & curriculum logic
        if self.n_calls % self.eval_freq == 0:
            # Stage-based checks
            sr = self.stats.success_rate
            mean_r = self.stats.mean_reward
            print(f"MeanR={mean_r:.3f} | SuccessRate={sr:.3f}")

            if len(self.stats) >= WINDOW and self.num_timesteps > 1e5:
                # Stage-up gate (unchanged)
                sr_pct = int(sr * 100)  # floor to integer percent
                should_save = False
                save_tag = None

                if sr_pct < 80:
                    milestone = (sr_pct // 10) * 10  # 0,10,20,...,70
                    if milestone >= 10 and milestone > self.last_saved_pct:
                        save_tag = milestone
                        should_save = True
                else:
                    # at/after 80: save every new integer percent
                    target = max(80, self.last_saved_pct + 1)
                    if sr_pct >= target and sr_pct > self.last_saved_pct:
                        save_tag = sr_pct
                        should_save = True

                if should_save:
                    self.model.save(f"{self.save_path}/best_stage{self.stage}_sr{save_tag}")
                    self.last_saved_pct = save_tag
                    print(f"Saved milestone model at {save_tag}/{sr}% success (stage {self.stage}).")

                if sr_pct >= 80:
                    if self.stage == 0:
                        print("Curriculum: switching to Stage 1 (1:3 det:rand resets)")
                        # Switch 3 envs to random, 1 to deterministic
                        self.training_env.env_method("set_deterministic_reset", False, indices=[1, 2, 3])
                        self.training_env.env_method("set_deterministic_reset", True, indices=[0])
                        self.stage = 1

                    elif self.stage == 1 and sr_pct >= 85:
                        print("Curriculum: switching to Stage 2 (0:4 all random resets)")
                        self.training_env.env_method("set_deterministic_reset", False, indices=[0, 1, 2, 3])
                        self.stage = 2
                
                # sr = self.stats.success_rate
                # print(f"Stage {self.stage} | Episodes={len(self.stats)} | SuccessRate={sr:.3f}")
                # if (len(self.stats) >= WINDOW and sr > SUCCESS_THRESHOLD):
                #     self._progress_stage()

                # # Save best model from stage 6 onward
                # if self.stage >= 5 and sr > max(0.80, self.best_success_rate):
                #     self.best_success_rate = sr
                #     self.model.save(f"{self.save_path}/best_model_stage{self.stage}_sr{round(sr*100)}")
                #     print(f"New best model saved! Success rate: {sr:.3f}")

            # # Fallback: stagnation guard or stage unset
            # if (self.num_timesteps - self.stage_reset_step) >= 1e7 or self.stage == 0:
            #     self._progress_stage()

        return True

    def _on_training_end(self) -> None:
        self.checkpoint_callback.on_training_end()

    

    def _freeze_actor(self):
        print("Freezing Actor")
        self.reset_step = self.num_timesteps
        self.actor_frozen = True
        for param in self.model.actor.parameters():
            param.requires_grad_(False)
            
        if self.model.log_ent_coef:
            self.model.log_ent_coef.data.fill_(np.log(1e-100))
