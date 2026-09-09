"""
config/settings.py
------------------
Dataclass-based configuration for the Game Economy Simulator.
All numeric defaults represent a balanced baseline economy.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Economy Configuration
# ---------------------------------------------------------------------------

@dataclass
class EconomyConfig:
    """
    Defines the per-player daily gold sources and sinks.

    Sources
    -------
    quest_reward        : Gold earned per quest completed.
    quests_per_day      : Average number of quests a player completes per day.
    enemy_reward        : Gold earned per enemy defeated.
    enemies_per_day     : Average number of enemies a player defeats per day.

    Sinks
    -----
    potion_cost         : Gold cost of a single potion.
    potions_per_day     : Average number of potions a player purchases per day.
    weapon_cost         : Gold cost of a weapon upgrade.
    weapon_interval_days: How often (in days) a player attempts to buy a weapon.
                          Purchase only occurs if the player can afford it.
    """

    # --- Sources ---
    quest_reward: float = 50.0
    quests_per_day: float = 3.0
    enemy_reward: float = 20.0
    enemies_per_day: float = 5.0

    # --- Sinks ---
    potion_cost: float = 10.0
    potions_per_day: float = 5.0
    weapon_cost: float = 300.0
    weapon_interval_days: int = 7

    # --- Derived convenience properties ---
    @property
    def daily_income(self) -> float:
        """Deterministic daily gold income (no randomness)."""
        return self.quest_reward * self.quests_per_day + self.enemy_reward * self.enemies_per_day

    @property
    def daily_potion_cost(self) -> float:
        """Deterministic daily potion expenditure."""
        return self.potion_cost * self.potions_per_day

    @property
    def daily_net(self) -> float:
        """Expected net gold per day (excluding weapon purchases)."""
        return self.daily_income - self.daily_potion_cost


# ---------------------------------------------------------------------------
# Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """
    Controls the simulation scale and initial conditions.

    Attributes
    ----------
    num_players     : Total number of simulated players (supports up to 10 000).
    days            : Simulation duration in days (30–90 recommended).
    starting_gold   : Each player's initial gold balance at Day 0.
    random_seed     : NumPy RNG seed for reproducible results.
                      Set to None for a non-deterministic run.
    noise_std_pct   : Std-dev of per-player daily income noise as a fraction
                      of daily_income (e.g. 0.15 = ±15 %).  Introduces
                      natural spread without loop-based branching.
    """

    num_players: int = 1_000
    days: int = 30
    starting_gold: float = 500.0
    random_seed: int | None = 42
    noise_std_pct: float = 0.15          # 15 % income variance across players
