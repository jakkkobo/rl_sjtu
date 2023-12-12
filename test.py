import time

import numpy as np
import rospy
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


from rl_velocity_env import UAVEnv, UAVEnv_PX4, UAVEnv_Sentinel

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





if __name__ == "__main__":

    print("Testing the model")

    rospy.init_node("RL")
    env = UAVEnv_Sentinel()

    model = PPO.load("ppo_quadrotor_4.zip")
    
    # run the trained model to see how it performs
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("done")
        time.sleep(0.1)

    env.close()

    print("Done testing the model")
