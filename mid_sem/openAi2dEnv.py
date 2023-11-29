import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CarRacing-v2"
env = gym.make(environment_name,render_mode="human")
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        # env.render()
        env.render()
        action = env.action_space.sample()
        # result = env.step(action)
        # print(result)
        n_state, reward, done, info, _ = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
print(env.action_space.sample())

log_path = os.path.join('Training', 'Logs')

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=40000)
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')
model.save(ppo_path)
