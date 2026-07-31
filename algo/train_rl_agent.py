"""
train_rl_agent.py

Train a single Q-learning agent that controls GA parameters across
multiple machine-layout problems, then save its Q-table to disk.

The GA used for training is RLGeneticAlgorithmRand (algo.gen_algo_rl_init_rand),
which applies local search to a random crossover offspring instead of to the
elites. This mirrors the plain GeneticAlgorithm in algo.gen_algo_init, so the
trained agent is compared against the baseline on equal footing.

Runs are scheduled as `runs_per_problem` shuffled passes over the problem set:
every problem is solved exactly `runs_per_problem` times, but never twice in a
row, so the agent does not overfit to one layout in a consecutive block.

Training writes three artefacts alongside the Q-table: an .npz of the reward /
fitness history, a four-panel training-progress figure, and a Q-table heat map.
Use plot_saved_training_history() to redraw the figures from the .npz without
retraining.

Usage (from project root):
    python -m algo.train_rl_agent

Or:
    python algo/train_rl_agent.py
"""

import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from func.datastruct import Point, Machine
from algo.agent import QLearningAgent
from algo.gen_algo_rl_init_rand import RLGeneticAlgorithmRand
from visual.visualize import plot_rl_training_progress, plot_q_table


# ============================================================
# 1. Define training problems 
# ============================================================

def build_training_problems() -> List[Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]]:
    problems: List[Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]] = []

    # ============================================================
    # PROBLEM SET A
    # ============================================================
    machines_A = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
                access_point=Point(1.5, 0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=6, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=7, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
    ]

    seq_A = [1,2,3,4,5,6,7,8]
    problems.append((machines_A, seq_A, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET B
    # ============================================================
    machines_B = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
                access_point=Point(1.5, 0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=6, shape='l_shape', width=5, height=3, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=2, l_cutout_height=2),
        Machine(id=7, shape='l_shape', width=3.5, height=2.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, 0))
    ]

    seq_B = [1,2,3,4,5,6,7,8]
    problems.append((machines_B, seq_B, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET C — More rectangles, wider machines
    # ============================================================
    machines_C = [
        Machine(id=1, shape="rectangle", width=6.0, height=3.0,
                access_point=Point(2.0, 0)),
        Machine(id=2, shape="rectangle", width=5.5, height=2.5,
                access_point=Point(-1.0, 0)),
        Machine(id=3, shape="rectangle", width=4.5, height=3.0,
                access_point=Point(0, -1.0)),
        Machine(id=4, shape="rectangle", width=3.5, height=4.0,
                access_point=Point(0, 1.5)),
        Machine(id=5, shape="rectangle", width=4.0, height=4.0,
                access_point=Point(1.0, -1.0)),
        Machine(id=6, shape="rectangle", width=5.0, height=3.5,
                access_point=Point(-1.5, 0)),
        Machine(id=7, shape="rectangle", width=6.0, height=2.5,
                access_point=Point(0, 1.0)),
        Machine(id=8, shape="rectangle", width=4.0, height=2.0,
                access_point=Point(0.5, 0)),
    ]

    seq_C = [1,2,3,4,5,6,7,8]
    problems.append((machines_C, seq_C, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET D — Small machines, mixed shapes
    # ============================================================
    machines_D = [
        Machine(id=1, shape="rectangle", width=2.5, height=2.5,
                access_point=Point(0.5, 0)),
        Machine(id=2, shape="l_shape", width=3.0, height=3.0,
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.0, l_cutout_height=1.0),
        Machine(id=3, shape="rectangle", width=2.0, height=2.0,
                access_point=Point(0, 1.0)),
        Machine(id=4, shape="l_shape", width=3.5, height=3.5,
                access_point=Point(-0.5, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.2),
        Machine(id=5, shape="rectangle", width=2.0, height=3.0,
                access_point=Point(0.5, -1.0)),
        Machine(id=6, shape="rectangle", width=3.0, height=2.0,
                access_point=Point(1.0, 0)),
        Machine(id=7, shape="l_shape", width=3.0, height=3.5,
                access_point=Point(1.0, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.0),
        Machine(id=8, shape="rectangle", width=2.5, height=2.0,
                access_point=Point(0, 0)),
    ]
    
    seq_D = [1,2,3,4,5,6,7,8]
    problems.append((machines_D, seq_D, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET E — Big machines, heavy L-shapes
    # ============================================================
    machines_E = [
        Machine(id=1, shape="l_shape", width=6.0, height=6.0,
                access_point=Point(1.0, -1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=2, shape="l_shape", width=5.5, height=5.0,
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape="rectangle", width=6.0, height=4.0,
                access_point=Point(0, -1.5)),
        Machine(id=4, shape="l_shape", width=6.5, height=5.5,
                access_point=Point(1.0, 1.0),
                l_cutout_width=2.5, l_cutout_height=2.0),
        Machine(id=5, shape="rectangle", width=5.0, height=5.0,
                access_point=Point(1.5, 0)),
        Machine(id=6, shape="l_shape", width=5.0, height=4.5,
                access_point=Point(0, -1.0),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=7, shape="rectangle", width=5.5, height=4.0,
                access_point=Point(-1, 0)),
        Machine(id=8, shape="rectangle", width=4.5, height=3.5,
                access_point=Point(1.0, 0)),
    ]

    seq_E = [1,2,3,4,5,6,7,8]
    problems.append((machines_E, seq_E, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET F — Vertical and slim machines
    # ============================================================
    machines_F = [
        Machine(id=1, shape="rectangle", width=2.0, height=6.0,
                access_point=Point(0.8, 0)),
        Machine(id=2, shape="rectangle", width=2.5, height=5.5,
                access_point=Point(-0.5, 1.0)),
        Machine(id=3, shape="l_shape", width=3.0, height=6.0,
                access_point=Point(0, -1.0),
                l_cutout_width=1.0, l_cutout_height=2.0),
        Machine(id=4, shape="rectangle", width=2.2, height=6.5,
                access_point=Point(1.0, 0)),
        Machine(id=5, shape="rectangle", width=1.8, height=5.0,
                access_point=Point(1.0, -1.0)),
        Machine(id=6, shape="l_shape", width=3.0, height=5.5,
                access_point=Point(-1.0, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.5),
        Machine(id=7, shape="rectangle", width=2.5, height=6.0,
                access_point=Point(0, 1.0)),
        Machine(id=8, shape="l_shape", width=3.5, height=5.0,
                access_point=Point(0.5, 0),
                l_cutout_width=1.0, l_cutout_height=2.0),
    ]

    seq_F = [1,2,3,4,5,6,7,8]
    problems.append((machines_F, seq_F, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET G — Random mixed medium machines
    # ============================================================
    machines_G = [
        Machine(id=1, shape="rectangle", width=4.0, height=2.5,
                access_point=Point(1, 0)),
        Machine(id=2, shape="rectangle", width=5.0, height=3.5,
                access_point=Point(-1, 0)),
        Machine(id=3, shape="l_shape", width=4.0, height=4.0,
                access_point=Point(1, 1),
                l_cutout_width=1.5, l_cutout_height=1.0),
        Machine(id=4, shape="rectangle", width=3.0, height=3.0,
                access_point=Point(0, -1)),
        Machine(id=5, shape="l_shape", width=4.5, height=3.0,
                access_point=Point(1, -1),
                l_cutout_width=1.0, l_cutout_height=1.5),
        Machine(id=6, shape="rectangle", width=4.0, height=3.0,
                access_point=Point(-1, 1)),
        Machine(id=7, shape="rectangle", width=3.0, height=2.0,
                access_point=Point(0.5, 0)),
        Machine(id=8, shape="l_shape", width=4.0, height=3.0,
                access_point=Point(0, -1),
                l_cutout_width=1.0, l_cutout_height=1.0),
    ]

    seq_G = [1,2,3,4,5,6,7,8]
    problems.append((machines_G, seq_G, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET H — Very mixed shapes + rotating access point directions
    # ============================================================
    machines_H = [
        Machine(id=1, shape="rectangle", width=3.5, height=4.0,
                access_point=Point(1, 0)),
        Machine(id=2, shape="l_shape", width=4.5, height=4.0,
                access_point=Point(0, 1),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=3, shape="rectangle", width=5.0, height=3.0,
                access_point=Point(-1, 0)),
        Machine(id=4, shape="l_shape", width=5.0, height=5.0,
                access_point=Point(0, -1),
                l_cutout_width=2.0, l_cutout_height=1.5),
        Machine(id=5, shape="rectangle", width=3.0, height=3.5,
                access_point=Point(1, 1)),
        Machine(id=6, shape="rectangle", width=4.0, height=3.0,
                access_point=Point(-1, -1)),
        Machine(id=7, shape="l_shape", width=4.5, height=3.5,
                access_point=Point(1, 0),
                l_cutout_width=1.5, l_cutout_height=1.0),
        Machine(id=8, shape="rectangle", width=3.5, height=2.0,
                access_point=Point(0, -0.5)),
    ]
    
    # ============================================================
    # PROBLEM SET I — Symmetric rectangles (tests symmetry breaking)
    # ============================================================

    seq_H = [1,2,3,4,5,6,7,8]
    problems.append((machines_H, seq_H, Point(0,0), (-15,15,-15,15)))

    machines_I = [
        Machine(1, "rectangle", 4.0, 3.0, Point(1, 0)),
        Machine(2, "rectangle", 4.0, 3.0, Point(-1, 0)),
        Machine(3, "rectangle", 4.0, 3.0, Point(0, 1)),
        Machine(4, "rectangle", 4.0, 3.0, Point(0, -1)),
        Machine(5, "rectangle", 3.5, 2.5, Point(1, 1)),
        Machine(6, "rectangle", 3.5, 2.5, Point(-1, 1)),
        Machine(7, "rectangle", 3.5, 2.5, Point(1, -1)),
        Machine(8, "rectangle", 3.5, 2.5, Point(-1, -1)),
    ]

    seq_I = [1,2,3,4,5,6,7,8]
    problems.append((machines_I, seq_I, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET J — Clustered access points (path-dominated)
    # ============================================================
    
    machines_J = [
        Machine(1, "rectangle", 4.5, 3.0, Point(1.2, 0.8)),
        Machine(2, "rectangle", 4.0, 3.5, Point(1.0, 0.9)),
        Machine(3, "l_shape", 5.0, 4.0, Point(1.1, 1.0)),
        Machine(4, "rectangle", 3.5, 3.0, Point(0.9, 1.1)),
        Machine(5, "l_shape", 4.5, 3.5, Point(1.0, 0.7)),
        Machine(6, "rectangle", 3.0, 2.5, Point(1.2, 1.2)),
        Machine(7, "rectangle", 3.0, 3.0, Point(0.8, 0.9)),
        Machine(8, "rectangle", 3.5, 2.5, Point(1.1, 0.6)),
    ]

    seq_J = [1,2,3,4,5,6,7,8]
    problems.append((machines_J, seq_J, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET K  — Narrow corridor packing
    # ============================================================

    machines_K = [
        Machine(1, "rectangle", 2.0, 6.0, Point(0, 1)),
        Machine(2, "rectangle", 2.2, 5.8, Point(0, -1)),
        Machine(3, "rectangle", 2.0, 6.2, Point(0.5, 0)),
        Machine(4, "rectangle", 2.3, 5.5, Point(-0.5, 0)),
        Machine(5, "l_shape", 3.0, 6.0, Point(0, 1)),
        Machine(6, "rectangle", 2.0, 5.0, Point(0, -1)),
        Machine(7, "rectangle", 2.1, 6.1, Point(0.3, 0)),
        Machine(8, "rectangle", 2.4, 5.6, Point(-0.3, 0)),
    ]

    seq_K = [1,2,3,4,5,6,7,8]
    problems.append((machines_K, seq_K, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET L — Large machines near workspace limits
    # ============================================================

    machines_L = [
        Machine(1, "rectangle", 6.5, 4.0, Point(1.5, 0)),
        Machine(2, "rectangle", 6.0, 4.5, Point(-1.5, 0)),
        Machine(3, "l_shape", 6.0, 5.0, Point(0, 1)),
        Machine(4, "rectangle", 5.5, 4.0, Point(0, -1)),
        Machine(5, "l_shape", 5.5, 5.5, Point(1, 1)),
        Machine(6, "rectangle", 5.0, 4.5, Point(-1, 1)),
        Machine(7, "rectangle", 5.8, 3.8, Point(1, -1)),
        Machine(8, "rectangle", 5.2, 4.2, Point(-1, -1)),
    ]
    seq_L = [1,2,3,4,5,6,7,8]
    problems.append((machines_L, seq_L, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET M — Alternating small / large machines
    # ============================================================

    machines_M = [
        Machine(1, "rectangle", 2.0, 2.0, Point(0.5, 0)),
        Machine(2, "rectangle", 6.0, 4.0, Point(-1.5, 0)),
        Machine(3, "rectangle", 2.2, 2.5, Point(0, 0.5)),
        Machine(4, "l_shape", 6.0, 5.0, Point(1, 1)),
        Machine(5, "rectangle", 2.0, 3.0, Point(0, -0.5)),
        Machine(6, "rectangle", 5.5, 4.5, Point(-1, 1)),
        Machine(7, "rectangle", 2.5, 2.0, Point(1, 0)),
        Machine(8, "l_shape", 5.5, 4.0, Point(0, -1)),
    ]
    seq_M = [1,2,3,4,5,6,7,8]
    problems.append((machines_M, seq_M, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET N — L-shape dominant, irregular cutouts
    # ============================================================

    machines_N = [
        Machine(1, "l_shape", 4.5, 5.5, Point(1, 0)),
        Machine(2, "l_shape", 5.0, 4.5, Point(-1, 1)),
        Machine(3, "l_shape", 4.0, 5.0, Point(0, -1)),
        Machine(4, "l_shape", 5.5, 4.0, Point(1, -1)),
        Machine(5, "rectangle", 3.0, 3.0, Point(0.5, 0)),
        Machine(6, "rectangle", 3.5, 2.5, Point(-0.5, 0)),
        Machine(7, "rectangle", 3.0, 2.0, Point(0, 1)),
        Machine(8, "rectangle", 2.5, 3.0, Point(0, -1)),
    ]
    seq_N = [1,2,3,4,5,6,7,8]
    problems.append((machines_N, seq_N, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET O - Asymmetric layout + skewed access points
    # ============================================================

    machines_O = [
        Machine(1, "rectangle", 4.5, 3.0, Point(1.8, 0.2)),
        Machine(2, "rectangle", 4.0, 3.5, Point(1.6, -0.4)),
        Machine(3, "l_shape", 5.0, 4.0, Point(1.4, 1.0)),
        Machine(4, "rectangle", 3.5, 3.0, Point(1.2, -1.0)),
        Machine(5, "l_shape", 4.0, 3.5, Point(1.0, 0.6)),
        Machine(6, "rectangle", 3.0, 2.5, Point(0.8, -0.8)),
        Machine(7, "rectangle", 3.5, 2.8, Point(1.3, 0.4)),
        Machine(8, "rectangle", 3.0, 3.0, Point(0.9, 1.2)),
    ]
    seq_O = [1,2,3,4,5,6,7,8]
    problems.append((machines_O, seq_O, Point(0,0), (-15,15,-15,15)))

    return problems


# ============================================================
# 2. Run schedule: shuffled, but balanced across problems
# ============================================================

def build_training_schedule(
    num_problems: int,
    runs_per_problem: int,
    rng: random.Random,
) -> List[int]:
    """
    Build the order in which problems are trained on.

    The schedule is `runs_per_problem` independently shuffled passes over the
    problem set, so:
      - every problem appears exactly `runs_per_problem` times
        (total runs = num_problems * runs_per_problem),
      - the order within each pass is random,
      - a problem is never scheduled twice in a row, including across the
        boundary between two passes.
    """
    schedule: List[int] = []

    for _ in range(runs_per_problem):
        block = list(range(num_problems))
        rng.shuffle(block)

        # Avoid a back-to-back repeat at the pass boundary by swapping the
        # offending first element with a random later one.
        if schedule and len(block) > 1 and block[0] == schedule[-1]:
            swap_idx = rng.randrange(1, len(block))
            block[0], block[swap_idx] = block[swap_idx], block[0]

        schedule.extend(block)

    return schedule


# ============================================================
# 3. Training loop: run GA over the shuffled schedule
# ============================================================

def extract_credited_rewards(rl_log: List[Dict]) -> List[float]:
    """
    Pull out the rewards that actually drove a Q-update.

    The GA only picks an action every `control_interval` generations, and the
    reward at generation g is credited to the action taken at g-1. Generations
    with no preceding action log a placeholder reward of 0, which would swamp
    the plot with zeros, so they are dropped here.
    """
    rewards = []
    for g in range(1, len(rl_log)):
        if rl_log[g - 1].get("action", -1) != -1:
            rewards.append(float(rl_log[g]["reward"]))
    return rewards


def train_agent(
    runs_per_problem: int = 10,
    q_table_path: str = "rl_agent/rl_ga_q_table_rand_200_300_15.npy",
    state_size: int = 9,
    action_size: int = 8,
    seed: int = 42,
    history_path: str = "rl_agent/rl_training_history_rand_200_300_15.npz",
    plot_path: str = "rl_agent/rl_training_progress_rand_200_300_15.png",
    show_plot: bool = False,
) -> Tuple[QLearningAgent, Dict]:
    """
    Train a single QLearningAgent across multiple problems and save its Q-table.

    - runs_per_problem: how many GA runs each problem gets (10 runs x 15
      problems = 150 runs in total), interleaved so the same problem is never
      solved twice consecutively
    - q_table_path: output .npy file path for the Q-table
    - seed: seed for the run-order shuffle, for reproducible schedules
    - history_path: .npz file recording the reward/fitness evolution
    - plot_path: .png file for the training-progress figure
    - show_plot: open the figure interactively as well as saving it

    Returns (agent, history).
    """
    problems = build_training_problems()
    if not problems:
        raise ValueError(
            "No training problems defined. "
            "Fill in build_training_problems() with your machine layouts."
        )

    # Shared RL agent across all problems
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    rng = random.Random(seed)
    schedule = build_training_schedule(len(problems), runs_per_problem, rng)
    total_runs = len(schedule)

    print(
        f"Starting training on {len(problems)} problems x {runs_per_problem} "
        f"runs = {total_runs} GA runs (shuffled order, seed={seed})..."
    )

    # How many times each problem has been run so far, for logging only
    run_counts = [0] * len(problems)

    # Training history, for the reward/convergence plots
    run_problem: List[int] = []
    run_reward_sum: List[float] = []
    run_reward_mean: List[float] = []
    run_best_fitness: List[float] = []
    run_q_delta: List[float] = []
    gen_rewards: List[float] = []

    for run_no, problem_idx in enumerate(schedule, start=1):
        machines, sequence, robot_position, workspace_bounds = problems[problem_idx]
        run_counts[problem_idx] += 1

        print(
            f"\n=== Run {run_no}/{total_runs} "
            f"-> Problem {problem_idx + 1}/{len(problems)} "
            f"(run {run_counts[problem_idx]}/{runs_per_problem} for this problem) ==="
        )

        q_before = agent.q_table.copy()

        # Local search is applied to a random crossover offspring here (same as
        # the plain GeneticAlgorithm in algo.gen_algo_init), not to the elites.
        ga = RLGeneticAlgorithmRand(
            machines=machines,
            sequence=sequence,
            robot_position=robot_position,
            workspace_bounds=workspace_bounds,
            rl_agent=agent,    # reuse same agent
        )

        final_layout, best_fitness, results = ga.optimize()

        # Record this run's reward signal and how much it moved the Q-table
        rewards = extract_credited_rewards(results.get("rl_log", []))
        gen_rewards.extend(rewards)
        run_problem.append(problem_idx)
        run_reward_sum.append(float(np.sum(rewards)) if rewards else 0.0)
        run_reward_mean.append(float(np.mean(rewards)) if rewards else 0.0)
        run_best_fitness.append(float(best_fitness))
        run_q_delta.append(float(np.linalg.norm(agent.q_table - q_before)))

        print(
            f"     Finished GA run. Best fitness: {best_fitness:.4f}, "
            f"total_distance: {results.get('total_distance', float('nan')):.4f}"
        )
        print(
            f"     Reward: total={run_reward_sum[-1]:.4f}, "
            f"mean={run_reward_mean[-1]:.4f} over {len(rewards)} decisions, "
            f"|dQ|={run_q_delta[-1]:.5f}"
        )

        # Checkpoint after every full pass over the problem set, so a long
        # training session is not lost if it is interrupted.
        if run_no % len(problems) == 0:
            np.save(q_table_path, agent.q_table)
            print(f"     [checkpoint] Q-table saved to: {q_table_path}")

    # Save the learned Q-table
    np.save(q_table_path, agent.q_table)
    print(f"\nTraining complete ({total_runs} runs). Saved Q-table to: {q_table_path}")

    history = {
        "run_problem": np.array(run_problem, dtype=int),
        "run_reward_sum": np.array(run_reward_sum, dtype=float),
        "run_reward_mean": np.array(run_reward_mean, dtype=float),
        "run_best_fitness": np.array(run_best_fitness, dtype=float),
        "run_q_delta": np.array(run_q_delta, dtype=float),
        "gen_rewards": np.array(gen_rewards, dtype=float),
        "num_problems": len(problems),
        "runs_per_problem": runs_per_problem,
        "seed": seed,
        "q_table": agent.q_table,
    }

    if history_path:
        np.savez(history_path, **history)
        print(f"Saved training history to: {history_path}")

    if plot_path or show_plot:
        plot_rl_training_progress(history, save_path=plot_path, show=show_plot)
        if plot_path:
            q_plot_path = plot_path.replace(".png", "_q_table.png")
            plot_q_table(agent.q_table, save_path=q_plot_path, show=show_plot)

    return agent, history


def plot_saved_training_history(
    history_path: str = "rl_agent/rl_training_history_rand_200_300_15.npz",
    plot_path: Optional[str] = None,
    show: bool = True,
):
    """
    Re-draw the training curves from a saved .npz without retraining.

    Handy for tweaking figures for the report after a long training session.
    """
    data = np.load(history_path)
    history = {key: data[key] for key in data.files}
    plot_rl_training_progress(history, save_path=plot_path, show=show)
    if "q_table" in history:
        q_plot_path = plot_path.replace(".png", "_q_table.png") if plot_path else None
        plot_q_table(history["q_table"], save_path=q_plot_path, show=show)
    return history


# ============================================================
# 4. Helper: load a trained agent
# ============================================================

def load_trained_agent(
    q_table_path: str = "rl_agent/rl_ga_q_table_rand_200_300_15.npy",
    state_size: int = 9,
    action_size: int = 8,
    epsilon: float = 0.0,
) -> QLearningAgent:
    """
    Load a trained QLearningAgent from a saved Q-table.

    - epsilon: set to 0.0 for fully greedy behaviour (no exploration),
               or a small value (e.g. 0.05) if you still want some exploration.
    """
    q_table = np.load(q_table_path)
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    agent.q_table = q_table
    agent.epsilon = epsilon
    return agent


# ============================================================
# 5. (Optional) Quick demo on a NEW problem using a trained agent
# ============================================================

def solve_new_problem_with_trained_agent(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    q_table_path: str = "rl_agent/rl_ga_q_table_rand_200_300_15.npy",
):
    """
    Example of how to use a trained agent on a fresh machine layout.
    """
    agent = load_trained_agent(q_table_path=q_table_path, epsilon=0.0)

    ga = RLGeneticAlgorithmRand(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
        rl_agent=agent,
    )

    final_layout, best_fitness, results = ga.optimize()
    print("Solved new problem with trained agent.")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Total distance: {results.get('total_distance', float('nan')):.4f}")
    return final_layout, best_fitness, results


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    # 15 problems x 10 runs each = 150 GA runs, shuffled
    train_agent(
        runs_per_problem=10,
        q_table_path="rl_agent/rl_ga_q_table_rand_200_300_15.npy",
        history_path="rl_agent/rl_training_history_rand_200_300_15.npz",
        plot_path="rl_agent/rl_training_progress_rand_200_300_15.png",
        show_plot=False,
    )