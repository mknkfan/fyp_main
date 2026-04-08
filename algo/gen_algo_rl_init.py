from func.datastruct import Point, Machine
from algo.agent import QLearningAgent
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import time
import copy
from algo.gen_algo_init import GeneticAlgorithm   # note: adjust import to your actual module name

class RLGeneticAlgorithm(GeneticAlgorithm):
    def __init__(
        self,
        machines: List[Machine],
        sequence: List[int],
        robot_position: Point,
        workspace_bounds: Tuple[float, float, float, float],
        rl_agent: Optional[QLearningAgent] = None,
        control_interval: int = 1,   # how often RL can change params
    ):
        super().__init__(machines, sequence, robot_position, workspace_bounds)

        # RL-controlled mutation strategy
        self.current_mutation_strategy = self.mutate

        # Allow reuse of a trained agent
        self.rl_agent = rl_agent or QLearningAgent(state_size=9, action_size=8)

        # Frequency of control decisions
        self.control_interval = control_interval

        # GA hyperparameter bounds/config
        self.param_config = {
            "mutation_rate": {"min": 0.01, "max": 0.8, "delta": 0.02},
            "crossover_rate": {"min": 0.2, "max": 1.0, "delta": 0.05},
            "elitism_rate": {"min": 0.0, "max": 0.5, "delta": 0.02},
        }

        # Probability to apply local search on elites each generation
        self.local_search_prob = 0.1
        self.local_search_delta = 0.1  # step for RL

        # For RL reward computation
        self.best_fitness_prev = float("inf")

    # ---------- State handling ----------

    def discretize_state(self, diversity: float, improvement: float) -> int:
        """Map continuous values into discrete RL states."""
        # Diversity bins
        if diversity < 300:
            div_bin = 0  # Low diversity, converging
        elif diversity < 400:
            div_bin = 1 # Medium diversity, stable
        else:
            div_bin = 2  # Exploration, diverging

        # Improvement bins
        if improvement <= 0:
            imp_bin = 0 # No improvement / worse
        elif improvement < 0.01:
            imp_bin = 1 # Small improvement
        else:
            imp_bin = 2 # Significant improvement
        return div_bin * 3 + imp_bin  # 0..8

    # ---------- Bounds / mutation operators ----------

    def apply_bounds(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply workspace bounds to chromosome."""
        min_x, max_x, min_y, max_y = self.workspace_bounds

        for i in range(0, len(chromosome), 3):
            machine_idx = i // 3
            machine_width = self.machines[machine_idx].width
            machine_height = self.machines[machine_idx].height

            chromosome[i] = np.clip(
                chromosome[i],
                min_x + machine_width / 2,
                max_x - machine_width / 2,
            )
            chromosome[i + 1] = np.clip(
                chromosome[i + 1],
                min_y + machine_height / 2,
                max_y - machine_height / 2,
            )
            chromosome[i + 2] = chromosome[i + 2] % 360

        return chromosome

    def gaussian_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Standard Gaussian mutation (delegates to base GA)."""
        return self.mutate(chromosome)

    def cauchy_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Cauchy mutation (heavy-tailed distribution)."""
        mutated = chromosome.copy()

        for i in range(0, len(chromosome), 3):
            if random.random() < self.mutation_rate:
                # x, y
                mutated[i] += np.random.standard_cauchy() * 2.0
                mutated[i + 1] += np.random.standard_cauchy() * 2.0
                # angle
                mutated[i + 2] += np.random.standard_cauchy() * 20.0

        return self.apply_bounds(mutated)

    def polynomial_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        mutated = chromosome.copy()
        eta = 20.0  # distribution index

        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                u = random.random()
                if u <= 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

                # scale with a small factor
                mutated[i] += delta * 0.1 * (abs(chromosome[i]) + 1.0)

        return self.apply_bounds(mutated)

    # ---------- Hyperparameter adjustment helpers ----------

    def _adjust_param(self, name: str, direction: int):
        """
        direction: +1 for increase, -1 for decrease
        """
        cfg = self.param_config[name]
        val = getattr(self, name)
        val += direction * cfg["delta"]
        val = max(cfg["min"], min(cfg["max"], val))
        setattr(self, name, val)

    def _adjust_local_search_prob(self, direction: int):
        self.local_search_prob += direction * self.local_search_delta
        self.local_search_prob = max(0.0, min(1.0, self.local_search_prob))

    # ---------- RL action application ----------

    def apply_action(self, action: int, population: List[np.ndarray]):
        """
        Apply RL-chosen action.

        Example mapping (8 actions):
          0: increase mutation rate
          1: decrease mutation rate
          2: increase crossover rate
          3: decrease crossover rate
          4: increase local search probability
          5: decrease local search probability
          6: cycle mutation strategy (Gaussian -> Cauchy -> Polynomial -> Gaussian)
          7: soft reset towards defaults
        """

        if action == 0:
            self._adjust_param("mutation_rate", +1)

        elif action == 1:
            self._adjust_param("mutation_rate", -1)

        elif action == 2:
            self._adjust_param("crossover_rate", +1)

        elif action == 3:
            self._adjust_param("crossover_rate", -1)

        elif action == 4:
            self._adjust_local_search_prob(+1)

        elif action == 5:
            self._adjust_local_search_prob(-1)

        elif action == 6:
            # cycle mutation strategy
            if self.current_mutation_strategy is self.gaussian_mutation:
                self.current_mutation_strategy = self.cauchy_mutation
            elif self.current_mutation_strategy is self.cauchy_mutation:
                self.current_mutation_strategy = self.polynomial_mutation
            else:
                self.current_mutation_strategy = self.gaussian_mutation

        elif action == 7:
            # move params slightly back toward defaults
            self.mutation_rate = 0.5 * self.mutation_rate + 0.5 * 0.1
            self.crossover_rate = 0.5 * self.crossover_rate + 0.5 * 0.8
            self.local_search_prob = 0.5 * self.local_search_prob

    # ---------- Main optimization loop ----------

    def optimize(self, initial_population: Optional[List[np.ndarray]] = None) -> Tuple[List[Machine], float, Dict]:
        print("Starting Reinforcement Learning - Genetic Algorithm")

        # Initialize population (shared or new)
        if initial_population is None:
            population = self.create_initial_population()
        else:
            population = [chrom.copy() for chrom in initial_population]
            if len(population) != self.population_size:
                raise ValueError(
                    f"Initial population size {len(population)} != GA population_size {self.population_size}"
                )
        decoded_evolution = []
        best_solution = None
        best_fitness = float("inf")
        self.best_fitness_history = []
        self.avg_fitness_history = []
        rl_log = []
        start_time = time.time()

        prev_state = None
        prev_action = None
        prev_best_fitness = float("inf")
        snapshot_interval = 20  # generations

        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = [self.fitness_function(ch) for ch in population]
            best_idx = np.argmin(fitness_scores)
            best_fitness_curr = fitness_scores[best_idx]
            if snapshot_interval and (generation % snapshot_interval == 0):
                snap_machines = self.decode_chromosome(population[best_idx])
                decoded_evolution.append(copy.deepcopy(snap_machines))

            diversity = self.calculate_diversity(population)
            improvement = prev_best_fitness - best_fitness_curr
            state = self.discretize_state(diversity, improvement)

            # RL update from previous transition
            if generation > 0 and prev_action is not None:
                if prev_best_fitness > best_fitness_curr:
                    reward = (prev_best_fitness - best_fitness_curr) / (abs(prev_best_fitness))
                else:
                    reward = -0.1
                    
                self.rl_agent.update(prev_state, prev_action, reward, state)
            else:
                reward = 0

            # RL decides whether to change GA params (at intervals)
            if generation % self.control_interval == 0:
                action = self.rl_agent.get_action(state)
                self.apply_action(action, population)
            else:
                action = None  # no change this generation

            # Logging RL info
            rl_log.append(
                {
                    "generation": generation,
                    "state": state,
                    "action": int(action) if action is not None else -1,
                    "reward": float(reward),
                    "diversity": float(diversity),
                    "improvement": float(improvement),
                    "best_fitness": float(best_fitness_curr),
                    "mutation_rate": float(self.mutation_rate),
                    "crossover_rate": float(self.crossover_rate),
                    "local_search_prob": float(self.local_search_prob),
                }
            )

            # Track best solution
            if best_fitness_curr < best_fitness:
                best_fitness = best_fitness_curr
                best_solution = population[best_idx].copy()

            avg_fitness = np.mean([f for f in fitness_scores if f != float("inf")])
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Create new population
            new_population = []

            # Elitism (rate is also controllable)
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                elite_chrom = population[idx].copy()

                # Optional: apply local search probabilistically
                if random.random() < self.local_search_prob:
                    elite_chrom = self.local_search(elite_chrom, max_iterations=5)
                new_population.append(elite_chrom)

            # Fill the rest
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(population, fitness_scores)
                p2 = self.tournament_selection(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.current_mutation_strategy(c1)
                c2 = self.current_mutation_strategy(c2)
                new_population.extend([c1, c2])

            population = new_population[: self.population_size]

            # Store for next iteration
            prev_state = state
            prev_action = action
            prev_best_fitness = best_fitness_curr

            if generation % 1 == 0:
                print(
                    f"Gen {generation}: Best={best_fitness:.2f}, "
                    f"Avg={avg_fitness:.2f}, Diversity={diversity:.2f}"
                )
                recent = rl_log[-1]
                print(
                    f"[RL] Gen {recent['generation']}: Action={recent['action']}, "
                    f"Reward={recent['reward']:.3f}, MutRate={recent['mutation_rate']:.3f}, "
                    f"CrossRate={recent['crossover_rate']:.3f}, "
                    f"LS_Prob={recent['local_search_prob']:.2f}"
                )

        end_time = time.time()

        final_machines = self.decode_chromosome(best_solution)
        results = {
            "total_distance": self.calculate_total_distance(final_machines),
            "generations": self.generations,
            "execution_time": end_time - start_time,
            "decoded_evolution": decoded_evolution,
            "snapshot_interval": snapshot_interval,
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "final_fitness": best_fitness,
            "q_table": self.rl_agent.q_table,
            "rl_log": rl_log,
        }

        return final_machines, best_fitness, results
