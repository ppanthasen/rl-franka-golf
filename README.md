# Franka Golf — Reinforcement Learning Project

This project trains a Franka Emika Panda robot arm to play a simplified golf task using reinforcement learning.  
It was developed as part of the [CompeteSAI](https://competesai.com/) challenge, where my solution ranked **6th out of 59 competitors**.  
My official submission can be found here: [Competition Submission](https://competesai.com/submissions/sb_jPR1AlAH9ik5).

## Project Overview
- **Algorithm**: Soft Actor-Critic (SAC)  
- **Environment**: Custom Gymnasium wrapper for the Franka Golf task  
- **Approach**:  
  - Curriculum learning with four stages:  
    1. Orientation alignment  
    2. Approach with alignment  
    3. Grasping  
    4. Full task (hit-to-hole)  
  - Reward shaping with truncation thresholds to encourage stable progress  
- **Result**: Successfully trained an agent to complete the golf task within competition constraints

https://github.com/user-attachments/assets/aab3b34d-3ece-4e25-b714-abde6f39d81d




## Repo Structure
- `train.py` — main training script with environment setup, model initialization, and training loop  
- `eval.py` — evaluation script to test a trained model checkpoint  
- `wrappers.py` — custom environment wrappers (e.g., reward shaping for staged curriculum)  
- `callbacks.py` — custom training callbacks (evaluation, checkpointing, buffer resets, logging)
- `checkpoint.zip` — trained model ready for evaluation   

## Quick Start
```bash
# clone repo
git clone https://github.com/<your-username>/franka-golf.git
cd franka-golf

# install dependencies
pip install -r requirements.txt

# run evaluation
python eval.py
```

## Notes
- Code is currently unpolished; uploaded primarily to show results and methodology.
- Refactoring and documentation improvements are in progress.


