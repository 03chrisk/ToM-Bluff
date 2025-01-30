import functools
import random
from typing import Tuple

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# CONSTANTS
RANKS = ["ACE", "JACK", "QUEEN", "KING"]
NUM_CARDS_PER_RANK = 4
ACTION_CHALLENGE = [0, 0, 0, 0]  # Not necessary but not to hardcode it

ACTION_SPACE = MultiDiscrete([5, 5, 5, 5])

NUM_ACTIONS = 70


def env(num_players: int = 2, render_mode: str = None) -> AECEnv:
    """Wrapper for the Bluff environment."""
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    base_env = BluffEnv(num_players=num_players, render_mode=internal_render_mode)
    if render_mode == "ansi":
        base_env = wrappers.CaptureStdoutWrapper(base_env)
    base_env = wrappers.AssertOutOfBoundsWrapper(base_env)
    base_env = wrappers.OrderEnforcingWrapper(base_env)
    return base_env


class BluffEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "bluff_v1"}

    def __init__(self, num_players: int = 2, render_mode: str = None) -> None:
        """Initialize the Bluff environment with the specified number of players."""
        self.num_players = num_players
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        self._action_spaces = {agent: ACTION_SPACE for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Discrete:
        """
        Return the observation space for the specified agent.
        """
        # We never need the observation space
        raise NotImplementedError

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        """
        Return the action space for the specified agent.
        """
        return self._action_spaces[agent]

    def reset(self, seed: int = None, options: dict = None) -> Tuple:
        """Reset the environment to start a new game."""

        # Deck and hand initialization
        deck = RANKS * NUM_CARDS_PER_RANK
        random.shuffle(deck)
        cards_per_player = len(deck) // self.num_players

        # Assign cards to players put the rest in the central pile
        self.player_hands = {
            agent: np.array(
                self._list_to_frequency_vector(
                    deck[i * cards_per_player : (i + 1) * cards_per_player]
                )
            )
            for i, agent in enumerate(self.possible_agents)
        }
        self.central_pile = deck[self.num_players * cards_per_player :]

        # Reset piles
        self._first_action_played = False
        self._cards_played_from_rank = 0

        # Game state variables
        self.current_rank = 0  # Start with "ACE"
        self.current_claim = []
        self.last_played_agent = None
        self._last_step = None
        self.current_player_index = 0
        self._is_truthful = None

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.infos[self.agent_selection]["action_mask"] = self._get_action_mask(
            self.agent_selection
        )
        for agent in self.agents:
            self.infos[agent]["cards_other_agent_played"] = 0

    def _list_to_frequency_vector(self, hand_list: list) -> list:
        """Convert a list of cards to a frequency vector."""
        freq_vector = [0] * len(RANKS)
        for card in hand_list:
            freq_vector[RANKS.index(card)] += 1
        return freq_vector

    def _frequency_vector_to_card_list(self, freq_vector: list) -> list:
        """Convert a frequency vector back to a list of cards."""
        card_list = []
        for i, count in enumerate(freq_vector):
            card_list.extend([RANKS[i]] * count)
        return card_list

    def get_initial_observation(self) -> dict:
        """
        Return the initial observation for the specified agent.
        """
        return (self.observe(self.agent_selection), self.infos[self.agent_selection])

    def observe(self, agent: str) -> dict:
        """Return the current observation for the specified agent."""
        # Only one other agent for now, change later for 3 agents.
        previous_agent = self.last_played_agent
        how_many_last_played = (
            0
            if previous_agent is None
            else self.infos[previous_agent]["cards_other_agent_played"]
        )

        cards_in_other_agent_hand = (
            0 if previous_agent is None else sum(self.player_hands[previous_agent])
        )

        return {
            "current_rank": self.current_rank,
            "central_pile_size": len(self.central_pile),
            "hand": np.array(self.player_hands[agent]).tolist(),
            "cards_other_agent_played": how_many_last_played,
            "cards_in_other_agent_hand": cards_in_other_agent_hand,
        }

    def _validate_action(self, action: str) -> None:
        """Validate the action."""
        if action not in self._get_action_mask(self.agent_selection):
            raise ValueError(f"Invalid action: {action}")

    def step(self, action: str) -> Tuple[dict, dict, dict, dict]:
        """Take a step in the game."""

        agent = self.agent_selection

        self._validate_action(action)

        if np.array_equal(action, ACTION_CHALLENGE):
            self._handle_challenge(agent)
        else:
            self._handle_play(agent, action)

        if self.render_mode == "human":
            self.render(action)

        # THIS ACTUALLY NEEDS TO BE HERE DON'T TOUCH
        self._cumulative_rewards[agent] = self.rewards[agent]

        if not self.terminations[agent]:
            # Iterate agent selection once in any case

            self.agent_selection = self._agent_selector.next()

            # If challenger wins, they play again, so skip everyone internally

            if np.array_equal(action, ACTION_CHALLENGE) and not self._is_truthful:
                for _ in range(self.num_players - 1):
                    self.agent_selection = self._agent_selector.next()

            # LEGACY: This was the code for 2 agents

            # self.agent_selection = self._agent_selector.next()
            # if np.array_equal(action, ACTION_CHALLENGE) and not self._is_truthful:
            #     self.agent_selection = self._agent_selector.next()

        self.infos[self.agent_selection]["action_mask"] = self._get_action_mask(
            self.agent_selection
        )

        self._last_step = self.last()

    def last(self) -> Tuple:
        """Return the last step information."""
        return (
            self.observe(self.agent_selection),
            self.rewards[self.agent_selection],
            self.terminations[self.agent_selection],
            self.truncations[self.agent_selection],
            self.infos[self.agent_selection],
        )

    def _get_action_mask(self, agent: str) -> list:
        """Return the valid actions for the given agent."""
        previous_agent = self.last_played_agent
        if previous_agent and sum(self.player_hands[previous_agent]) == 0:
            valid_combinations = np.array([ACTION_CHALLENGE])
        else:
            cards_left_to_play = 4 - self._cards_played_from_rank
            current_agent_hand = self.player_hands[agent]
            cards_left_to_play = min(cards_left_to_play, sum(current_agent_hand))

            num_of_aces_in_hand = current_agent_hand[0]
            num_of_jacks_in_hand = current_agent_hand[1]
            num_of_queens_in_hand = current_agent_hand[2]
            num_of_kings_in_hand = current_agent_hand[3]
            a, b, c, d = np.indices(
                (
                    min(cards_left_to_play + 1, num_of_aces_in_hand + 1),
                    min(cards_left_to_play + 1, num_of_jacks_in_hand + 1),
                    min(cards_left_to_play + 1, num_of_queens_in_hand + 1),
                    min(cards_left_to_play + 1, num_of_kings_in_hand + 1),
                )
            )
            combinations = np.stack(
                [a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=-1
            )
            valid_combinations = combinations[
                np.sum(combinations, axis=1) <= cards_left_to_play
            ]

            if self.last_played_agent is None:
                valid_combinations = np.delete(
                    valid_combinations, ACTION_CHALLENGE, axis=0
                )

        return valid_combinations

    def _handle_play(self, agent: str, action: list) -> None:
        """Handle the play action."""
        action = np.array(action)
        self.player_hands[agent] -= action

        number_of_cards = np.sum(action)
        self.infos[agent]["cards_other_agent_played"] = number_of_cards
        cards_to_play = self._frequency_vector_to_card_list(action)

        # Add cards to the central pile
        self.central_pile.extend(cards_to_play)
        self.current_claim = cards_to_play
        self.last_played_agent = agent
        self._cards_played_from_rank += number_of_cards
        # print(f"Cards played from rank: {self._cards_played_from_rank}")

        if self._cards_played_from_rank > 4:
            raise ValueError("Too many cards played from the same rank.")

        if self._cards_played_from_rank == 4:
            self._cards_played_from_rank = 0
            self.current_rank = (self.current_rank + 1) % len(RANKS)

        # give agent rewards based on how many cards they played
        self.rewards[agent] = number_of_cards

    def check_victory(self, agent: str) -> None:
        """
        Check if the agent has won the game.
        """
        if sum(self.player_hands[agent]) == 0:
            self.terminations[agent] = True
            self.rewards[agent] = 100
            for other_agent in self.agents:
                if other_agent != agent:
                    self.terminations[other_agent] = True
                    self.rewards[other_agent] = -100
            return True
        return False

    def _handle_challenge(self, agent: str) -> None:
        """Handle the challenge action."""
        if self.last_played_agent is None:
            raise RuntimeError("No play to challenge.")

        self.infos[agent]["cards_other_agent_played"] = 0
        
        if self._cards_played_from_rank == 0:
            temp_rank = (self.current_rank -1) % len(RANKS)
        else:
            temp_rank = self.current_rank

        # Check if the last play was truthful

        is_truthful = all(
            card == RANKS[temp_rank] for card in self.current_claim
        )

        if is_truthful:
            # Challenger takes all cards in the central pile
            challenger_hand_list = self._frequency_vector_to_card_list(
                self.player_hands[agent]
            )
            challenger_hand_list.extend(self.central_pile)
            self.player_hands[agent] = self._list_to_frequency_vector(
                challenger_hand_list
            )
            self.rewards[agent] = -len(self.central_pile)
        else:
            # Last player takes all cards in the central pile
            last_player_hand_list = self._frequency_vector_to_card_list(
                self.player_hands[self.last_played_agent]
            )
            last_player_hand_list.extend(self.central_pile)
            self.player_hands[self.last_played_agent] = self._list_to_frequency_vector(
                last_player_hand_list
            )
            self.rewards[agent] = len(self.central_pile)

        # Reset the central pile and move to the next rank
        self.central_pile = []
        self.current_rank = (temp_rank + 1) % len(RANKS)
        self._cards_played_from_rank = 0

        # Handle action masks so that you cant challenge after challenge
        previous_agent = self.last_played_agent
        self.last_played_agent = None
        self.infos[previous_agent]["action_mask"] = self._get_action_mask(agent)

        self._is_truthful = is_truthful

    def render(self, action: list) -> None:
        """Render the current game state."""
        current_card_from_rank = RANKS[self.current_rank]
        cards_played = self._frequency_vector_to_card_list(action)
        print("\n")
        if len(cards_played) != 0:
            print(
                f"Player {self.agent_selection} plays {cards_played}, claiming they are {current_card_from_rank}."
            )
        else:
            print(f"Player {self.agent_selection} challenges.")
            print(f"The other player was truthful: {self._is_truthful}")

        print(f"Cards from current rank played so far: {self._cards_played_from_rank}")
        for agent in self.agents:
            print(f"{agent}: {sum(self.player_hands[agent])} cards")

        print(f"Central pile: {len(self.central_pile)} cards")
        print(f"Reward: {self.rewards[self.agent_selection]}")
