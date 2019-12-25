import multiprocessing
from os import path
from PIL import Image
import cv2
import gym
import time
from statistics import mean
from lib.env.USStockEnv import USStockEnv
from lib.env.IndianStockEnv import IndianStockEnv
from lib.env.TraderRenkoEnv_v3_lite import StockTradingEnv

env_config = {
    "initial_balance": 10000,
    "look_back_window_size": 375 * 10,
    "enable_env_logging": True,
    "observation_window": 32,
    "frame_stack_size": 1,
    "use_leverage": False,
    "hold_reward": False,
}

env = StockTradingEnv(env_config)

observation = env.reset()

max_env_steps = 0

time_obs = []

frames = []

while True:
    # env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)

    # frames.append(Image.fromarray(observation[-1]))
    # path = '../output/'
    #
    # img = Image.fromarray(observation[:, :, -1])
    # img.save(path + str(env.current_step) + '.png')

    # env.plot_renko(path=path)

    start = time.time()
    observation, reward, done, info = env.step(action)
    # print(len(observation), observation.shape)
    end = time.time()
    max_env_steps += 1
    time_obs.append(end - start)

    # print("###############################")
    # print("Step:", env.current_step)
    # print(action)
    # print(str(observation) + str(reward) + str(done) + str(info))
    # print(len(observation))
    # print("###############################")

    if done:
        # observation = env.reset()
        break
env.close()

# path = '../output/'
# frames[0].save('plot.gif',
#                save_all=True,
#                append_images=frames[1:],
#                duration=1,
#                loop=0)

print("Avg Response Time: ", mean(time_obs))
print("Theoretical Traversal Time: {} Min".format((mean(time_obs) * max_env_steps) / 60))
print("Total days", len(env.daily_profit_per))
print(len(time_obs))
