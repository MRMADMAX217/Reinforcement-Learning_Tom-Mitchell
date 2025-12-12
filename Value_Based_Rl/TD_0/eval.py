import gymnasium as gym
import numpy as np
import pickle
import time

def evaluate(episodes=10, render=True):
    """
    Evaluates the trained agent on FrozenLake-v1 using the saved Q-table.
    """
    # Load Q-table
    try:
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("Error: q_table.pkl not found. Train the agent first.")
        return

    # Create environment
    render_mode = "human" if render else None
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=render_mode)
    
    total_rewards = 0
    
    print(f"Starting evaluation for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        while not done:
            # Greedy action selection
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if render:
                time.sleep(0.5) # Slow down rendering for visibility
        
        total_rewards += episode_reward
        result = "Success" if episode_reward > 0 else "Fail"
        print(f"Result: {result}")
        if render:
            time.sleep(1)

    mean_reward = total_rewards / episodes
    print(f"\nEvaluation finished.")
    print(f"Mean Reward (Success Rate): {mean_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # You can change render to False to just check metrics quickly
    evaluate(episodes=5, render=True)
