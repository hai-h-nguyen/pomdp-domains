from stable_baselines3_latest import SAC
import pdomains
import gym

env = gym.make("pdomains-peg-insertion-state-v0")

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("new")

# del model # remove to demonstrate saving and loading

model = SAC.load("new")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
    env.render()
    if terminated:
        obs = env.reset()