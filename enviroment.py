import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset(options={"sutton_barto_reward": True})

while True:
    
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    done = terminated or truncated

    if done:
        done = False
        env.reset(options={"sutton_barto_reward": True})
        
env.close()
