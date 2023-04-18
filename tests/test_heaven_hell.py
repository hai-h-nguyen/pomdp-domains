from pdomains import *
import matplotlib.pyplot as plt
import gym
import numpy as np
import time
import keyboard

env=gym.make('pdomains-heaven-hell-v1', rendering=False)
obs = env.reset()
env.render()

cnt = 0

for i in range(1000):
    while True:
        if keyboard.is_pressed('left'):
            action = 2
            break
        elif keyboard.is_pressed('right'):
            action = 3
            break
        elif keyboard.is_pressed('up'):
            action = 0
            break
        elif keyboard.is_pressed('down'):
            action = 1
            break
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.2)
    cnt += 1
    if done:
        print(reward, "Done")
        cnt = 0
        env.reset()
