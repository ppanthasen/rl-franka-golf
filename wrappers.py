import gymnasium as gym
import numpy as np
import types


class RewardShapingWrapper(gym.Wrapper):
    STAGE_WEIGHTS = {
        1: {"ee_club_dist": 9.0,
            "open_gripper": 1.0},
        2: {"ee_club_dist": 4.0,
            "align_ee_handle": 5.0,
            "open_gripper": 1.0},
        3: {"ee_club_dist": 1.0,
            "align_ee_handle": 2.0,
            "fingers_club_grasp": 6.0,
            "close_gripper": 1.0},
        4: {"ee_club_dist": 1.0,
            "align_ee_handle": 2.0,
            "fingers_club_grasp": 5.0,
            "ball_hole_dist": 10.0,
            "ball_in_hole": 20.0,
            "club_ball_dist": 2.0,
            "ball_moved": 5.0,
            }
    }

    PENALTY_WEIGHTS = {
        "club_dropped": -2.0,
        "ball_passed_hole": -4.0,
        "joint_vel": -0.005,
    }

    REWARD_THRESHOLDS = {1: 7.0, 2: 8.5, 3: 7.2, 4: 6.0}
    SUCCESS_GATE_THRESHOLDS = {2: 0.97, 3: 0.70}
    
    def __init__(self, env, stage: int, max_steps: int = 300):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.step_count = 0
        self.max_steps = max_steps
        self.ep_rew = 0.0
        self.gate_rew = 0.0
        self.gate_best = 0.0
        self.prev_rew = 0.0
        self.set_stage(stage)
        self.use_guided_traj = False
        self.start_guided = False
        self.guided_actions = get_manual_actions()
        # self.ep_success = False
        
        self.base_env._ee_club_dist = types.MethodType(self._ee_club_dist_offset, self.base_env)
        
        self.base_env._approach_fingers_handle = types.MethodType(self._approach_fingers_handle_offset, self.base_env)

        
    # ---- Stage setters for VecEnv ----
    def set_stage(self, stage: int):
        self.stage = stage
        self.set_reward_weights()
        
        offset_th = {1: 0.18}
        self.base_env._approach_offset_m = offset_th.get(stage, 0.12)
        
    def set_deterministic_reset(self, deterministic_reset: bool):
        self.base_env.deterministic_reset = deterministic_reset

    def set_reward_weights(self):
        key = 4 if self.stage >= 4 else max(1, self.stage)
        weights = dict(self.STAGE_WEIGHTS[key])
        weights.update(self.PENALTY_WEIGHTS)
        self.base_env.reward_config = weights

    def set_use_guided_traj(self, use: bool = True):
        if use: self.start_guided = True
        
    
    # ---- Core Gym methods ----
    def reset(self, **kwargs):
        self.step_count = 0
        self.ep_rew = 0.0
        self.gate_rew = 0.0
        self.gate_best = 0.0
        self.prev_rew = 0.0
        
        if self.use_guided_traj:
            self.use_guided_traj = False
            
        if self.start_guided:
            self.use_guided_traj = True
            self.start_guided = False
        
        # self.ep_success = False
        return self.env.reset(**kwargs)
    def step(self, action):
        if self.use_guided_traj:
            action = self.guided_actions[self.step_count]
        
        obs, _, term, trunc, info = self.env.step(action)
        self.step_count += 1
        
        raw_reward = self._compute_reward(info)
        total_reward = self.base_env._total_reward(raw_reward)
        # total_reward += self._compute_progress_reward(total_reward)
        
        self.ep_rew += total_reward

        if self._should_truncate(total_reward):
            trunc = True
            
        if term or trunc:
            info = self._update_dict(info)
              
        # for k, v in raw_reward.items():
        #     if k in self.STAGE_WEIGHTS[self.stage]:
        # #         # if k=="fingers_club_grasp" :print(k,v)      
        #         print(k,v)      
                
        
        return obs, total_reward, term, trunc, info

    def _compute_reward(self, info):
        env = self.base_env
        data = env.robot_model.data

        raw_reward = dict(env.compute_reward()) 
            
        # Add custom terms
        club_head_pos = data.xpos[env.club_head_id]
        ball_pos = data.xpos[env.golf_ball_id]
        club_ball = self._club_ball_dist(club_head_pos, ball_pos)
        
        if "fingers_club_grasp" in raw_reward:
            g = raw_reward["fingers_club_grasp"]
            raw_reward["fingers_club_grasp"] = min(1, g/3)
        
        if self.stage <= 2:
            raw_reward['open_gripper'] = self._compute_open_gripper(0.1,0.1,1.0)
        
        if self.stage == 2:
            self.gate_rew += raw_reward['align_ee_handle']
            if raw_reward['align_ee_handle'] < 0.97:
                raw_reward["ee_club_dist"] = 0.0
            raw_reward['align_ee_handle'] = self._alignment_adjust(raw_reward['align_ee_handle'])
            
        if self.stage == 3:
            raw_reward["fingers_club_grasp"] = self._gate_steady(raw_reward["fingers_club_grasp"])
            self.gate_rew += raw_reward['fingers_club_grasp']
            
        if self.stage >= 3:
            gr = self._compute_open_gripper() if raw_reward["ee_club_dist"] > 0.8 else 0
            
            raw_reward['close_gripper'] = gr
            if gr < 0.3 or raw_reward.get("align_ee_handle", 0.0) < 0.97:
                raw_reward["fingers_club_grasp"] = 0.0
        

        ball_moved = 1.0 if float(ball_pos[0]) < 0.4 else 0.0
        success = 1.0 if info.get("success", False) else 0.0

        raw_reward.update({
            "ball_in_hole": success,
            "club_ball_dist": club_ball,
            "ball_moved": ball_moved,
        })

        return raw_reward
        
    def _should_truncate(self, reward: float) -> bool:
        if self.step_count >= 100:
            thr = self.REWARD_THRESHOLDS.get(self.stage, 6.0)
            if reward < thr:
                return True
        return self.step_count >= self.max_steps
    
    def _gate_steady(self, curr, epsilon=0.01):
        # Only reward if still close to the best record
        if curr >= self.gate_best - epsilon:
            reward = curr
            self.gate_best = max(self.gate_best, curr)
        else:
            reward = -0.5 * (self.gate_best - curr)
        return reward
    
    def _compute_open_gripper(self, target=0.048, max_range=0.03, offset=1.4):
        # gripper dist range (0.0182, 0.0952) 
        # lift dist 0.037
        env = self.base_env
        left_finger_pos = env.robot_model.data.xpos[env.left_finger_body_id]
        right_finger_pos = env.robot_model.data.xpos[env.right_finger_body_id]
        d = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        diff = abs(d - target)
        reward = offset - diff / max_range
        return max(0.0, min(1.0, reward))
        
    def _compute_progress_reward(self, reward, c=5, cap=0.2):
        rew = np.clip(c * (reward - self.prev_rew), -cap, cap)
        self.prev_rew = reward
        return rew

    def _update_dict(self, info):
        info = dict(info)
        info.update({"episode_end": True, 
                     "avg_step_reward": self.ep_rew / max(1, self.step_count)
        })
        if self.stage in [2, 3]:
            avg_gate = self.gate_rew / max(1, self.step_count)
            thr = self.SUCCESS_GATE_THRESHOLDS[self.stage]
            info['stage_success'] = bool(avg_gate > thr)
        return info

    @staticmethod
    def _club_ball_dist(club_head_pos, ball_pos):
        distance = np.linalg.norm(club_head_pos - ball_pos)
        return np.exp(-10.0 * distance)
    
    @staticmethod
    def _ee_club_dist_offset(self_, ee_pos, club_grip_pos, alpha=0.5):
        offset = getattr(self_, "_approach_offset_m", 0.12)
        target_pos = club_grip_pos.copy()
        target_pos[2] += offset  # Add offset for z axis
        
        diff = ee_pos - target_pos
        dxy = np.linalg.norm(diff[:2])
        xy_rew = np.exp(-20 * dxy)
        rew = alpha * xy_rew
        
        if dxy < 0.03:
            dz = max(0, diff[2])
            rew += (1-alpha) * (1-dz/0.15)
        
        return rew
    
        
    @staticmethod
    def _approach_fingers_handle_offset(self_, ee_pos, handle_pos, offset=0.04):
        left_finger_pos = self_.robot_model.data.xpos[self_.left_finger_body_id]
        right_finger_pos = self_.robot_model.data.xpos[self_.right_finger_body_id]
        
        # Check if hand is in a graspable pose
        is_graspable = (right_finger_pos[1] < handle_pos[1]) & (
            left_finger_pos[1] > handle_pos[1]
        )

        is_graspable = (
            is_graspable
            & (ee_pos[2] < handle_pos[2] + 0.14)
            & (ee_pos[0] - handle_pos[0] < 0.02)
        )

        if not is_graspable:
            return 0.0

        # Compute the distance of each finger from the handle
        lfinger_dist = np.abs(left_finger_pos[1] - handle_pos[1])
        rfinger_dist = np.abs(handle_pos[1] - right_finger_pos[1])

        # Reward is proportional to how close the fingers are to the handle when in a graspable pose
        reward = is_graspable * (1 / (lfinger_dist + rfinger_dist)) / 10.0
        return reward

    
    @staticmethod
    def _alignment_adjust(s, k=4, alpha=0.1):
        """
        Map original alignment score s (0..1) to shaped reward.
        
        Parameters:
            s     : float or array, original score in [0,1]
            k     : steepness of exponential ramp
            alpha : fraction of linear blend to keep gradients alive

        Returns:
            float or array in [0,1]
        """
        exp_part = (np.exp(k*s) - 1.0) / (np.exp(k) - 1.0)
        return (1 - alpha) * exp_part + alpha * s
    

def get_manual_actions():
    actions = []
    # actions += [[0, 0, 0, 0, 0, 0, 1]] * 10  # gripper 
    actions += [[0.5, 0, 0, 0, 0, 0, -1]] * 6  # +X & open gripper
    actions += [[0, 0, -0.52, 0, 0, 0, -1]] * 5  # down Z
    actions += [[0, 0, 0, 0, 0, 0, 1]] * 3  # gripper close
    actions += [[0, 0, 0.5, 0, 0, 0, 1]]  # up Z to lift
    actions += [[0, 0.5, 0, 0, 0, 0, 1]] * 15  # +Y (toward hole)
    actions += [[0, 0, 0, 0, 0, 0, 1]] * 300 # null for waiting
    return actions

class RewardShapingWrapperV2(gym.Wrapper):
    reward_config = {
        "ee_club_dist": 1.0,
        "align_ee_handle": 2.0,
        "fingers_club_grasp": 5.0,
        "ball_hole_dist": 10.0,
        "ball_in_hole": 1.0,
        "bih_val": 300,
        "club_dropped": -2.0,
        "ball_passed_hole": -4.0,
        "joint_vel": -0.005,
        "club_ball_dist": 2.0,
        "ball_moved": 1.0,
        "gripper_control": 0.3,
        "steps": -0.05, 
    }

    def __init__(self, env, max_steps: int = 300):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.step_count = 0
        self.max_steps = max_steps
        self.ep_rew = 0.0
        self.prev_rew = 0.0
        self.ep_hit = False
        self.use_guided_traj = False
        self.start_guided = False
        self.guided_actions = get_manual_actions()
        self.ep_count = 0
        self.base_env.reward_config = self.reward_config
        self.base_env._ee_club_dist = types.MethodType(self._ee_club_dist_offset, self.base_env)
        
        # self.base_env._approach_fingers_handle = types.MethodType(self._approach_fingers_handle_offset, self.base_env)

    def set_deterministic_reset(self, deterministic_reset: bool):
        self.base_env.deterministic_reset = deterministic_reset
        
    def set_use_guided_traj(self, use: bool = True):
        if use: self.start_guided = True

        
    def reset(self, **kwargs):
        self.ep_count += 1
        # print(self.ep_count, self.step_count)
        self.ep_rew = 0.0
        self.ep_hit = False
        if self.use_guided_traj:
            # print(self.ep_count-1, self.step_count, "End Guide", self.step_count)
            self.use_guided_traj = False
            
        if self.start_guided:
            # print(self.ep_count, self.step_count, "Start Guide")
            self.use_guided_traj = True
            self.start_guided = False
        
        self.step_count = 0
        
        return self.env.reset(**kwargs) 
        
    def step(self, action):
        if self.use_guided_traj:
            action = self.guided_actions[self.step_count]
        
        obs, _, term, trunc, info = self.env.step(action)
        self.step_count += 1  
        
        raw_reward = self._compute_reward(info)
        total_reward = self.base_env._total_reward(raw_reward)
        total_reward += self._compute_progress_reward(total_reward)
        
        
        self.ep_rew += total_reward
        
        if self._should_truncate(total_reward):
            trunc = True
            
        if term or trunc:
            info = self._update_dict(info)
              
        # for k, v in raw_reward.items():
        #     if k in self.reward_config and v!=0:
        #         # if k=="fingers_club_grasp" :print(k,v)      
            
        #         print(k,round(v,3))      
        # # # print()
        return obs, total_reward, term, trunc, info

        
    def _compute_reward(self, info):
        env = self.base_env
        data = env.robot_model.data
                
        r_rew = dict(env.compute_reward())
        r_rew['steps'] = self.step_count
        # if r_rew['align_ee_handle'] < 0.97:
            # r_rew['ee_club_dist'] = 0
        
        # if r_rew['ee_club_dist'] > 0.5:
            # gr = self._compute_control_gripper()
        # else: 
            # gr = 0
            
        # r_rew['gripper_control'] = gr
        # if r_rew['align_ee_handle'] < 0.97 or gr < 0.8:
            # r_rew['fingers_club_grasp'] = 0
        # else:
            # r_rew['fingers_club_grasp'] = min(1, r_rew['fingers_club_grasp']/3)
        
        # if r_rew['fingers_club_grasp'] > 0.5 or self.ep_hit:
            
        #     club_head_pos = data.xpos[env.club_head_id]
        #     ball_pos = data.xpos[env.golf_ball_id]
        #     club_ball = self._club_ball_dist(club_head_pos, ball_pos)

            # ball_moved = 1.0 if float(ball_pos[0]) < 0.4 else 0.0   
            # if ball_moved and not self.ep_hit:
        #         # print("Hit!")
        #         self.ep_hit = True
        #     success = 1.0 if info.get("success", False) else 0.0
        #     if success: print("Success!")

        #     r_rew.update({
        #         "ball_in_hole": success,
        #         "club_ball_dist": club_ball,
        #         "ball_moved": ball_moved,
        #     })
        
        # else:
        #     r_rew['ball_hole_dist'] = 0
        club_head_pos = data.xpos[env.club_head_id]
        ball_pos = data.xpos[env.golf_ball_id]
        club_ball = self._club_ball_dist(club_head_pos, ball_pos)
        
        r_rew.update({
            'club_ball_dist': club_ball,
            'fingers_club_grasp': min(r_rew['fingers_club_grasp']/2.7, 1),
            'ball_moved': 1.0 if float(ball_pos[0]) < 0.4 else 0.0,
            'ball_in_hole': self.reward_config['bih_val'] - self.step_count if info.get("success", False) else 0.0
        })
        
        return r_rew
    
    def _should_truncate(self, reward: float) -> bool:
        if self.step_count >= 100 and reward < 3:
            return True
        return self.step_count >= self.max_steps       
    
    def _update_dict(self, info):
        info = dict(info)
        info.update({"episode_end": True, 
                     "avg_step_reward": self.ep_rew / max(1, self.step_count)
        })
        return info
    
    def _compute_control_gripper(self, target=0.048, max_range=0.03, offset=1.4):
        # gripper dist range (0.0182, 0.0952) 
        # lift dist 0.037
        env = self.base_env
        left_finger_pos = env.robot_model.data.xpos[env.left_finger_body_id]
        right_finger_pos = env.robot_model.data.xpos[env.right_finger_body_id]
        d = np.linalg.norm(left_finger_pos - right_finger_pos)
        
        diff = abs(d - target)
        reward = offset - diff / max_range
        return max(0.0, min(1.0, reward))

    def _compute_progress_reward(self, reward, c=1, cap=1):
        rew = np.clip(c * (reward - self.prev_rew), 0, cap)
        self.prev_rew = reward
        return rew

    
    
    @staticmethod
    def _club_ball_dist(club_head_pos, ball_pos):
        distance = np.linalg.norm(club_head_pos - ball_pos)
        return np.exp(-10.0 * distance)
    
    @staticmethod
    def _ee_club_dist_offset(self_, ee_pos, club_grip_pos, alpha=0.5):
        offset = getattr(self_, "_approach_offset_m", 0.12)
        target_pos = club_grip_pos.copy()
        target_pos[2] += offset  # Add offset for z axis
        
        diff = ee_pos - target_pos
        dxy = np.linalg.norm(diff[:2])
        xy_rew = np.exp(-20 * dxy)
        rew = alpha * xy_rew
        
        if dxy < 0.03:
            dz = max(0, diff[2])
            rew += (1-alpha) * (1-dz/0.15)
        
        return rew
    
        
    @staticmethod
    def _approach_fingers_handle_offset(self_, ee_pos, handle_pos, offset=0.04):
        left_finger_pos = self_.robot_model.data.xpos[self_.left_finger_body_id]
        right_finger_pos = self_.robot_model.data.xpos[self_.right_finger_body_id]
        
        # Check if hand is in a graspable pose
        is_graspable = (right_finger_pos[1] < handle_pos[1]) & (
            left_finger_pos[1] > handle_pos[1]
        )

        is_graspable = (
            is_graspable
            & (ee_pos[2] < handle_pos[2] + 0.14)
            & (ee_pos[0] - handle_pos[0] < 0.02)
        )

        if not is_graspable:
            return 0.0

        # Compute the distance of each finger from the handle
        lfinger_dist = np.abs(left_finger_pos[1] - handle_pos[1])
        rfinger_dist = np.abs(handle_pos[1] - right_finger_pos[1])

        # Reward is proportional to how close the fingers are to the handle when in a graspable pose
        reward = is_graspable * (1 / (lfinger_dist + rfinger_dist)) / 10.0
        return reward

    
    