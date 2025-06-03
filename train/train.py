import random
import torch
from train.plot_results import init_live_plot, plot_training, update_live_plot


def train(env, policy_model, target_model, optimizer, buffer, loss_fn):
    
    EPSILON_START = 0.99
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.997  # se aplica al final de cada episodio
    GAMES_LIMIT = 2000
    STEPS_LIMIT = 200_000
    GAMMA = 0.99

    state = env.reset()[0]
    
    # Fill the buffer with random steps
    while len(buffer) < buffer.maxlen:
        # Random action
        action = env.action_space.sample()
        # Perform the action
        next_state, reward, truncated, terminated, _ = env.step(action)
        # Fill the buffer
        buffer.push((state, action, reward, next_state, truncated or terminated))
        
        if terminated or truncated:
            state = env.reset()[0]
        else:
            state = next_state
            
    # Go through the episodes
    n_games = 1
    steps = 1
    running_loss = 0
    top_score = 0
    scores = []
    epsilons = []
    losses = []
    fig, ax = init_live_plot()


    while n_games < GAMES_LIMIT or steps < STEPS_LIMIT:
        
        # Reset
        state = env.reset()[0]
        truncated, terminated = False, False
        done = truncated or terminated
        
        total_reward = 0
        
        while not done:
            
            # Update the epsilon value
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** n_games))
            # Use epsilon-greedy
            random_float = random.random()
            if random_float < epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Act greedy (explotation)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = torch.argmax(policy_model(state_tensor)).item()
                
            # We perform the new action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                reward = -1
            
            # Save the data into the buffer
            buffer.push((state, action, reward, next_state, truncated or terminated))
            
            # Take a sample of the buffer
            sample = buffer.sample(64)
            states, actions, rewards, next_states, dones = zip(*sample)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            
            # Calculate the target q_value
            with torch.no_grad():
                max_next_Q = target_model(next_states).max(1)[0]
                target_Q = rewards + GAMMA * max_next_Q * (1 - dones)

            
            # Calculate the current q_value
            all_q_values = policy_model(states)
            current_Q = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Calculate the loss
            loss = loss_fn(current_Q, target_Q)
            
            # Optimier zero grad
            optimizer.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Update the target model with policy model every 100 steps
            if steps % 1000 == 0:
                target_model.load_state_dict(policy_model.state_dict())
                
            state = next_state
            running_loss += loss.item()
            total_reward += reward
            steps += 1
        
            
        n_games += 1

        # Guarda el puntaje del episodio
        scores.append(total_reward)
        epsilons.append(epsilon)
        losses.append(running_loss / steps)  # si calculas la pérdida acumulada
        update_live_plot(ax, scores)
        
        if total_reward >= top_score:
            top_score = total_reward
            torch.save(policy_model.state_dict(), "models/top_modelo2.pth")
            print(f"Game: {n_games} | Total_Reward: {top_score}")
            
            
            
    plot_training(scores, epsilons, losses)
    #torch.save(policy_model.state_dict(), "models/modelo.pth")
    return policy_model