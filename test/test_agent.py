import gymnasium as gym
import torch
import numpy as np

def test_agent(env:gym.Env, model, n_episodes=10, render=False):
    
    model.eval()
    scores = []

    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            if render:
                env.render()

        scores.append(total_reward)
        print(f"Score = {total_reward:.2f}")

    model.train()
    avg_score = np.mean(scores)
    print(f"Evaluación del modelo: promedio = {avg_score:.2f} sobre {n_episodes} episodios")
    return avg_score
