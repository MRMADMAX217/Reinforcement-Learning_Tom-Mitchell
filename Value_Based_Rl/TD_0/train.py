import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def train(episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
    """
    Trains an agent on FrozenLake-v1 using Q-Learning (TD(0)).
    """
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # Initialize Q-table 
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    
    rewards_history = []
    
    print("Starting training...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update rule
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action] * (not done)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} - Mean Reward (last 100): {np.mean(rewards_history[-100:]):.4f} - Epsilon: {epsilon:.4f}")
            
    print("Training finished.")
    
    # Save Q-table
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
    print("Q-table saved to q_table.pkl")
    
    # Plot results
    rolling_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avg)
    plt.title("Rolling Average Reward (Window 100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_plot.png")
    print("Training plot saved to training_plot.png")
    
    env.close()

if __name__ == "__main__":
    train()
