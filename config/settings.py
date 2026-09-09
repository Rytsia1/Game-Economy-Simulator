"""
config/settings.py
------------------
Dataclass-based configuration for the Game Economy Simulator.
v0.2 additions: ArchetypeProfile, gear milestone costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Archetype Profile
# ---------------------------------------------------------------------------

@dataclass
class ArchetypeProfile:
    """
    Defines one player archetype's behavioral multipliers and population share.

    Attributes
    ----------
    name                : Human-readable archetype label.
    population_share    : Fraction of total players assigned to this archetype.
                          All shares across all archetypes must sum to 1.0.
    quest_mult          : Multiplier on the base ``quests_per_day`` rate.
    enemy_mult          : Multiplier on the base ``enemies_per_day`` rate.
    potion_mult         : Multiplier on the base ``potions_per_day`` rate.
    saving_buffer_mult  : The player only attempts to buy a weapon when their
                          balance ≥ weapon_cost * saving_buffer_mult.
                          1.0 = buy as soon as affordable.
                          2.0 = hoard until balance is 2× the cost first.
    color               : Hex color used in Plotly charts for this archetype.
    """

    name: str
    population_share: float
    quest_mult: float
    enemy_mult: float
    potion_mult: float
    saving_buffer_mult: float
    color: str = "#FFFFFF"


# ---------------------------------------------------------------------------
# Default Archetype Roster  (Casual 50, Grinder 25, Collector 15, Optimizer 10)
# ---------------------------------------------------------------------------

DEFAULT_ARCHETYPES: List[ArchetypeProfile] = [
    ArchetypeProfile(
        name="Casual",
        population_share=0.50,
        quest_mult=0.6,
        enemy_mult=0.4,
        potion_mult=0.8,
        saving_buffer_mult=1.05,   # buys almost as soon as they can afford it
        color="#4F8EF7",           # blue
    ),
    ArchetypeProfile(
        name="Grinder",
        population_share=0.25,
        quest_mult=1.2,
        enemy_mult=2.0,
        potion_mult=1.5,
        saving_buffer_mult=1.2,    # comfortable safety margin
        color="#00C9A7",           # teal
    ),
    ArchetypeProfile(
        name="Collector",
        population_share=0.15,
        quest_mult=0.8,
        enemy_mult=0.7,
        potion_mult=1.2,
        saving_buffer_mult=1.0,    # aggressive — buys the moment they can
        color="#A78BFA",           # purple
    ),
    ArchetypeProfile(
        name="Optimizer",
        population_share=0.10,
        quest_mult=1.0,
        enemy_mult=1.3,
        potion_mult=0.3,           # efficiency-first, minimal potion spend
        saving_buffer_mult=2.0,    # strategic hoarder
        color="#FBBF24",           # amber
    ),
]


# ---------------------------------------------------------------------------
# Economy Configuration
# ---------------------------------------------------------------------------

@dataclass
class EconomyConfig:
    """
    Defines the per-player daily gold sources and sinks (base rates).

    Sources
    -------
    quest_reward        : Gold earned per quest completed.
    quests_per_day      : Average number of quests completed per day (base).
    enemy_reward        : Gold earned per enemy defeated (μ for Normal draw).
    enemies_per_day     : Average number of enemies defeated per day (base).
    enemy_reward_std_pct: Std-dev of per-enemy gold drop as a fraction of μ.

    Sinks
    -----
    potion_cost         : Gold cost of a single potion.
    potions_per_day     : Average number of potions purchased per day (base).
    weapon_cost         : Gold cost of a tier-0 weapon upgrade (periodic sink).
    weapon_interval_days: How often (in days) a weapon purchase is attempted.

    Gear Milestones
    ---------------
    tier_1_weapon_cost  : Gold threshold for "Time-to-Afford Tier 1" tracking.
    tier_2_weapon_cost  : Gold threshold for "Time-to-Afford Tier 2" tracking.

    Archetypes
    ----------
    archetypes          : List of ArchetypeProfile objects defining the player
                          population mix.  Defaults to DEFAULT_ARCHETYPES.
    """

    # --- Sources ---
    quest_reward: float = 50.0
    quests_per_day: float = 3.0
    enemy_reward: float = 20.0
    enemies_per_day: float = 5.0
    enemy_reward_std_pct: float = 0.25   # 25 % std-dev on per-enemy gold drop

    # --- Sinks ---
    potion_cost: float = 10.0
    potions_per_day: float = 5.0
    weapon_cost: float = 300.0
    weapon_interval_days: int = 7

    # --- Gear milestones (for Time-to-Afford analytics) ---
    tier_1_weapon_cost: int = 500
    tier_2_weapon_cost: int = 2_500

    # --- Archetypes ---
    archetypes: List[ArchetypeProfile] = field(
        default_factory=lambda: list(DEFAULT_ARCHETYPES)
    )

    # --- Derived convenience properties ---
    @property
    def daily_income(self) -> float:
        """Deterministic base daily gold income (no archetype or randomness)."""
        return self.quest_reward * self.quests_per_day + self.enemy_reward * self.enemies_per_day

    @property
    def daily_potion_cost(self) -> float:
        """Deterministic base daily potion expenditure."""
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
    noise_std_pct   : Retained for backward-compatibility.  In v0.2 the
                      stochastic income variance is driven by archetype
                      multipliers + Poisson/Normal draws rather than this
                      scalar noise field.  Setting it to 0 has no effect
                      when stochastic_mode=True.
    stochastic_mode : If True (default), use Poisson quest draws and clipped
                      Normal enemy-drop draws.  If False, fall back to the
                      v0.1 Gaussian-noise model for deterministic comparisons.
    """

    num_players: int = 1_000
    days: int = 30
    starting_gold: float = 500.0
    random_seed: int | None = 42
    noise_std_pct: float = 0.15          # retained for v0.1 fallback mode
    stochastic_mode: bool = True
