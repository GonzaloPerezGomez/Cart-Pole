import gymnasium as gym
import torch
from agent.model import CartPoleRL
from agent.replay_buffer import ReplayBuffer
import torch.optim as optim
import torch.nn as nn
from test.test_agent import test_agent
from train.train import train

    

if __name__ == "__main__":
    
    LR = 2e-4
    # Initialize the env
    env = gym.make("CartPole-v1")
    # Initialize the principal CNN
    policy_model = CartPoleRL(4, 128, 2)
    # Initialize the target CNN
    target_model = CartPoleRL(4, 128, 2)
    # Initialize the optimizer
    optimizer = optim.Adam(policy_model.parameters(),
                          lr=LR)
    # Initialize the replay buffer
    buffer = ReplayBuffer(capacity=50_000)
    # Initialize the loss funcion
    loss_fn = nn.MSELoss()
    # Use the trained model
    trained_model = train(env, policy_model, target_model, optimizer, buffer, loss_fn)
    # Use the saved model
    trained_model = CartPoleRL(4, 128, 2)
    trained_model.load_state_dict(torch.load("models/top_modelo.pth"))
    
    env = gym.make("CartPole-v1", render_mode="human")
    test_agent(env, trained_model, render=True)
    
    

            