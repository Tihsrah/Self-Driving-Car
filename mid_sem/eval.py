
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()