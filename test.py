import time

import numpy as np
import rospy
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


from rl_velocity_env import UAVEnv, UAVEnv_PX4

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":

    print("Testing the model")

    rospy.init_node("RL")
    env = UAVEnv_PX4()

    model = PPO.load("./model_3_checkpoints/rl_model_1300000_steps.zip")
    
    # run the trained model to see how it performs
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        time.sleep(0.1)

    env.close()

    print("Done testing the model")
