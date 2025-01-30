import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.first_order_dev import FirstOrderAgent
from agents.random_agent import RandomAgent  # noqa: F401
from agents.second_order import SecondOrderAgent
from envs.bluff_env import env
from utils import print_strategy_analysis
from agents.zero_order import ZeroOrderAgent
import csv

seed = 1
random.seed(seed)
np.random.seed(seed)

def play_bluff_game(num_players: int = 2, episodes: int = 8) -> None:
    """Play a game of Bluff with the specified number of players."""

    game_env = env(num_players=num_players, render_mode="huma")

    agent_0 = SecondOrderAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)

    agent_1 = FirstOrderAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)

    wins_agent_0 = 0
    wins_agent_1 = 0

    draws = 0

    for episode in range(episodes):
        agents = [agent_0, agent_1]

        players = ["player_0", "player_1"]

        np.random.shuffle(agents)

        agents = dict(zip(players, agents))

        game_env.reset()
        obs, info = game_env.get_initial_observation()
        mask = np.array(info["action_mask"])

        # Needed as you do not need to update in the first step, need both players to actually play something
        prev_rewards = {"player_0": None, "player_1": None}

        play = 0

        while True:
            play += 1
            if play >= 1000:
                draws += 1
                break
            current_agent = game_env.agent_selection

            if game_env.check_victory(current_agent):
                final_rewards = {}
                for pos in agents.keys():
                    final_rewards[pos] = game_env.rewards[
                        pos
                    ]  # Get the actual final rewards

                # Final update for both agents with terminal state and correct final rewards
                for pos, agent_obj in agents.items():
                    agent_obj.update(final_rewards[pos], None)

                # Track wins for the actual agents
                winning_agent = agents[game_env.agent_selection]

                if winning_agent == agent_0:
                    wins_agent_0 += 1
                else:
                    wins_agent_1 += 1
                break

            agent = agents[current_agent]

            if prev_rewards[current_agent] is not None:
                agent.update(prev_rewards[current_agent], obs)

            action = agent.select_action(obs, mask)
            # print(f"Agent {current_agent} plays {action}")

            game_env.step(action)
            next_obs, reward, termination, truncation, info = game_env.last()

            mask = np.array(info["action_mask"])

            prev_rewards[current_agent] = reward

            obs = next_obs

        if episode % 500 == 0:
            print(f"Episode {episode}")
            print(f"Agent 0 wins: {wins_agent_0}")
            print(f"Agent 1 wins: {wins_agent_1}")
            print(f"Draws: {draws}")

    return agent_0, agent_1, wins_agent_0, wins_agent_1, draws


def run_multiple_games(num_iterations: int = 2, episodes_per_game: int = 1000):
    """Run multiple iterations of the game and collect comprehensive results."""
    results = []
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        _, _, wins_0, wins_1, draws = play_bluff_game(num_players=2, episodes=episodes_per_game)
        results.append((wins_0, wins_1, draws))
        
        # Print interim results
        total_games = wins_0 + wins_1 + draws
        print(f"Agent 0 Win Rate: {(wins_0/total_games)*100:.1f}%")
        print(f"Agent 1 Win Rate: {(wins_1/total_games)*100:.1f}%")
        print(f"Draw Rate: {(draws/total_games)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_multiple_games(num_iterations=10, episodes_per_game=1500)
    
    print("\nFinal Results:")
    for wins_0, wins_1, draws in results:
        total_games = wins_0 + wins_1 + draws
        print(f"Agent 0 Win Rate: {(wins_0/total_games)*100:.1f}%")
        print(f"Agent 1 Win Rate: {(wins_1/total_games)*100:.1f}%")
        print(f"Draw Rate: {(draws/total_games)*100:.1f}%")
        print()
    
    # save resutls as a csv file
    with open("first_second3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Agent 0 Win Rate", "Agent 1 Win Rate", "Draw Rate"])
        for wins_0, wins_1, draws in results:
            total_games = wins_0 + wins_1 + draws
            writer.writerow([(wins_0/total_games)*100, (wins_1/total_games)*100, (draws/total_games)*100])  